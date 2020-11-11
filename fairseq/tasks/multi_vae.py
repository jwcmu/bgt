# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
import os

import torch

from fairseq import options, utils
from fairseq.data import (
    Dictionary,
    LanguagePairDataset,
    LanguagePairDataset2,
    RoundRobinZipDatasets,
    TransformEosLangPairDataset,
)
from fairseq.models import FairseqMultiModel
from fairseq.tasks.translation_vae import load_langpair_dataset


from . import FairseqTask, register_task

"python -u train.py /projects/tir5/users/jwieting/data/multilingual-os-data/data-bin/500k-en-de:/projects/tir5/users/jwieting/data/multilingual-os-data/data-bin/500k-en-de -a bgt1 --setting 5 --optimizer adam --lr 0.0005 -s en -t fr --label-smoothing 0.1 --dropout 0.3 --max-tokens 500 --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 --criterion bgt_loss --max-epoch 20 --warmup-updates 4000 --warmup-init-lr '1e-07' --adam-betas '(0.9, 0.98)' --save-dir checkpoints/bgt5-de-/projects/tir5/users/jwieting/data/multilingual-os-data/all.500k.sp.20k.model-65536-25-0.0 --distributed-world-size 2 --latent-size 1024 --update-freq 25 --task multi_vae --save-interval-updates 0 --freeze 0 --sentencepiece /projects/tir5/users/jwieting/data/multilingual-os-data/all.500k.sp.20k.model --x0 65536 --translation-loss 1.0 --sentence-avg --num-workers 0 --consistent 0 --translation-type sample --reconstruction-loss 0.0 --lv-dim 1024"

def _lang_token(lang: str):
    return '__{}__'.format(lang)

def _lang_decoder_token(lang: str):
    return '__decoder_{}__'.format(lang)

def _lang_token_index(dic: Dictionary, lang: str):
    """Return language token index."""
    idx = dic.index(_lang_token(lang))
    assert idx != dic.unk_index, \
        'cannot find language token for lang {}'.format(lang)
    return idx


@register_task('multi_vae')
class MultiVAETask(FairseqTask):
    """A task for training multiple translation models simultaneously.

    We iterate round-robin over batches from multiple language pairs, ordered
    according to the `--lang-pairs` argument.

    The training loop is roughly:

        for i in range(len(epoch)):
            for lang_pair in args.lang_pairs:
                batch = next_batch_for_lang_pair(lang_pair)
                loss = criterion(model_for_lang_pair(lang_pair), batch)
                loss.backward()
            optimizer.step()

    In practice, `next_batch_for_lang_pair` is abstracted in a FairseqDataset
    (e.g., `RoundRobinZipDatasets`) and `model_for_lang_pair` is a model that
    implements the `FairseqMultiModel` interface.

    During inference it is required to specify a single `--source-lang` and
    `--target-lang`, which indicates the inference langauge direction.
    `--lang-pairs`, `--encoder-langtok`, `--decoder-langtok` have to be set to
    the same value as training.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', metavar='DIR', help='path to data directory')
        parser.add_argument('--lang-pairs', default=None, metavar='PAIRS',
                            help='comma-separated list of language pairs (in training order): en-de,en-fr,de-fr')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language (only needed for inference)')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language (only needed for inference)')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', default=False, action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left (default: True)')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left (default: False)')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--encoder-langtok', default=None, type=str, choices=['src', 'tgt'],
                            metavar='SRCTGT',
                            help='replace beginning-of-sentence in source sentence with source or target '
                                 'language token. (src/tgt)')
        parser.add_argument('--decoder-langtok', action='store_true',
                            help='replace beginning-of-sentence in target sentence with target language token')
        parser.add_argument('--add-lang-tokens', type=int, default=0,
                            help='share decoder input and output embeddings')
        parser.add_argument('--add-encoder-tokens', type=int, default=0,
                            help='share decoder input and output embeddings')
        parser.add_argument('--compressed', type=int, default=0,
                            help='share decoder input and output embeddings')
        parser.add_argument('--debug', action='store_true',
                            help='share decoder input and output embeddings')
        # fmt: on

    def __init__(self, args, dicts, training):
        super().__init__(args)
        self.dicts = dicts
        self.training = training
        if training:
            self.lang_pairs = args.lang_pairs
            args.source_lang, args.target_lang = args.lang_pairs[0].split('-')
        else:
            self.lang_pairs = ['{}-{}'.format(args.source_lang, args.target_lang)]
        # eval_lang_pairs for multilingual translation is usually all of the
        # lang_pairs. However for other multitask settings or when we want to
        # optimize for certain languages we want to use a different subset. Thus
        # the eval_lang_pairs class variable is provided for classes that extend
        # this class.
        self.eval_lang_pairs = self.lang_pairs
        # model_lang_pairs will be used to build encoder-decoder model pairs in
        # models.build_model(). This allows multitask type of sub-class can
        # build models other than the input lang_pairs
        self.model_lang_pairs = self.lang_pairs
        self.langs = list(dicts.keys())

    @classmethod
    def setup_task(cls, args, **kwargs):
        dicts, training = cls.prepare(args, **kwargs)
        return cls(args, dicts, training)

    @classmethod
    def prepare(cls, args, **kargs):
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)
        if getattr(args, 'raw_text', False):
            utils.deprecation_warning('--raw-text is deprecated, please use --dataset-impl=raw')
            args.dataset_impl = 'raw'
        elif getattr(args, 'lazy_load', False):
            utils.deprecation_warning('--lazy-load is deprecated, please use --dataset-impl=lazy')
            args.dataset_impl = 'lazy'

        if args.lang_pairs is None:
            raise ValueError('--lang-pairs is required. List all the language pairs in the training objective.')
        args.lang_pairs = args.lang_pairs.split(',')
        sorted_langs = sorted(list({x for lang_pair in args.lang_pairs for x in lang_pair.split('-')}))
        if args.source_lang is not None or args.target_lang is not None:
            training = False
        else:
            training = True

        from os import path
        # load dictionaries
        dicts = OrderedDict()
        for lang in sorted_langs:
            paths = args.data.split(':')
            assert len(paths) > 0
            for i in paths:
                if path.exists(os.path.join(i, 'dict.{}.txt'.format(lang))):
                    dicts[lang] = Dictionary.load(os.path.join(i, 'dict.{}.txt'.format(lang)))
            if len(dicts) > 0:
                assert dicts[lang].pad() == dicts[sorted_langs[0]].pad()
                assert dicts[lang].eos() == dicts[sorted_langs[0]].eos()
                assert dicts[lang].unk() == dicts[sorted_langs[0]].unk()
            #if args.encoder_langtok is not None or args.decoder_langtok:
            #    for lang_to_add in sorted_langs:
            #        dicts[lang].add_symbol(_lang_token(lang_to_add))
            # add extra tokens
            if args.add_lang_tokens:

                for i in sorted_langs:
                    dicts[lang].add_symbol(_lang_token(i))
                    if args.compressed:
                        dicts[lang].add_symbol(_lang_decoder_token(i))

            if args.compressed:
                for i in sorted_langs:
                    dicts[lang].add_symbol(_lang_decoder_token(i))

            if args.add_encoder_tokens:
                dicts[lang].add_symbol(_lang_token("lang"))
                dicts[lang].add_symbol(_lang_token("lang"))

                dicts[lang].add_symbol(_lang_token("sem"))
                dicts[lang].add_symbol(_lang_token("sem"))
            print('| [{}] dictionary: {} types'.format(lang, len(dicts[lang])))
        return dicts, training
    """
    def get_encoder_langtok(self, src_lang, tgt_lang):
        if self.args.encoder_langtok is None:
            return self.dicts[src_lang].eos()
        if self.args.encoder_langtok == 'src':
            return _lang_token_index(self.dicts[src_lang], src_lang)
        else:
            return _lang_token_index(self.dicts[src_lang], tgt_lang)

    def get_decoder_langtok(self, tgt_lang):
        if not self.args.decoder_langtok:
            return self.dicts[tgt_lang].eos()
        return _lang_token_index(self.dicts[tgt_lang], tgt_lang)
    """
    def alter_dataset_langtok(self, lang_pair_dataset,
                              src_eos=None, src_lang=None, tgt_eos=None, tgt_lang=None):
        if self.args.encoder_langtok is None and not self.args.decoder_langtok:
            return lang_pair_dataset

        """
        new_src_eos = None
        if self.args.encoder_langtok is not None and src_eos is not None \
           and src_lang is not None and tgt_lang is not None:
            new_src_eos = self.get_encoder_langtok(src_lang, tgt_lang)
        else:
            src_eos = None

        new_tgt_bos = None
        if self.args.decoder_langtok and tgt_eos is not None and tgt_lang is not None:
            new_tgt_bos = self.get_decoder_langtok(tgt_lang)
        else:
            tgt_eos = None

        return TransformEosLangPairDataset(
            lang_pair_dataset,
            src_eos=src_eos,
            new_src_eos=new_src_eos,
            tgt_bos=tgt_eos,
            new_tgt_bos=new_tgt_bos,
        )
        """

    def load_dataset(self, split, epoch=0, **kwargs):
        """Load a dataset split."""

        paths = self.args.data.split(':')
        assert len(paths) > 0

        def language_pair_dataset(lang_pair, data_path):
            src, tgt = lang_pair.split('-')
            if self.args.debug:
                print("DEBUG: multi_vae src: ", src)
                print("DEBUG: multi_vae tgt: ", tgt)
                print("DEBUG: multi_vae data_path: ", data_path)
            langpair_dataset = load_langpair_dataset(
                data_path, split, src, self.dicts[src], tgt, self.dicts[tgt],
                combine=True, dataset_impl=self.args.dataset_impl,
                upsample_primary=self.args.upsample_primary,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
            )
            return self.alter_dataset_langtok(
                langpair_dataset,
                src_eos=self.dicts[src].eos(),
                src_lang=src,
                tgt_eos=self.dicts[tgt].eos(),
                tgt_lang=tgt,
            )

        self.datasets[split] = RoundRobinZipDatasets(
            OrderedDict([
                (lang_pair, language_pair_dataset(lang_pair, paths[n]))
                for n, lang_pair in enumerate(self.lang_pairs)
            ]),
            eval_key=None if self.training else "%s-%s" % (self.args.source_lang, self.args.target_lang),
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        lang_pair = "%s-%s" % (self.args.source_lang, self.args.target_lang)
        return RoundRobinZipDatasets(
            OrderedDict([(
                lang_pair,
                self.alter_dataset_langtok(
                    LanguagePairDataset(
                        src_tokens, src_lengths,
                        self.source_dictionary,
                        left_pad_source=self.args.left_pad_source
                    ),
                    src_eos=self.source_dictionary.eos(),
                    src_lang=self.args.source_lang,
                    tgt_eos=self.target_dictionary.eos(),
                    tgt_lang=self.args.target_lang,
                ),
            )]),
            eval_key=lang_pair,
        )

    def build_model(self, args):
        def check_args():
            messages = []
            if len(set(self.args.lang_pairs).symmetric_difference(args.lang_pairs)) != 0:
                messages.append('--lang-pairs should include all the language pairs {}.'.format(args.lang_pairs))
            if self.args.encoder_langtok != args.encoder_langtok:
                messages.append('--encoder-langtok should be {}.'.format(args.encoder_langtok))
            if self.args.decoder_langtok != args.decoder_langtok:
                messages.append('--decoder-langtok should {} be set.'.format("" if args.decoder_langtok else "not"))

            if len(messages) > 0:
                raise ValueError(' '.join(messages))

        # Check if task args are consistant with model args
        check_args()

        from fairseq import models
        model = models.build_model(args, self)
        #if not isinstance(model, FairseqMultiModel):
        #    raise ValueError('MultilingualTranslationTask requires a FairseqMultiModel architecture')
        return model

    def init_logging_output(self, sample):
        return {
            'ntokens': sum(
                sample_lang.get('ntokens', 0)
                for sample_lang in sample.values()
            ) if sample is not None else 0,
            'nsentences': sum(
                sample_lang['target'].size(0) if 'target' in sample_lang else 0
                for sample_lang in sample.values()
            ) if sample is not None else 0,
        }

    def grad_denom(self, sample_sizes, criterion):
        return criterion.__class__.grad_denom(sample_sizes)

    @property
    def source_dictionary(self):
        return self.dicts[self.args.source_lang]

    @property
    def target_dictionary(self):
        return self.dicts[self.args.target_lang]

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        if len(self.datasets.values()) == 0:
            return {'%s-%s' % (self.args.source_lang, self.args.target_lang):
                    (self.args.max_source_positions, self.args.max_target_positions)}
        return OrderedDict([
            (key, (self.args.max_source_positions, self.args.max_target_positions))
            for split in self.datasets.keys()
            for key in self.datasets[split].datasets.keys()
        ])
