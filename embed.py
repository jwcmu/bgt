#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

from collections import namedtuple
import fileinput

import torch
import numpy as np

from fairseq import checkpoint_utils, options, tasks, utils

Batch = namedtuple('Batch', 'ids src_tokens src_lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')


def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, args, task, max_positions, encode_fn):
    tokens = [
        task.source_dictionary.encode_line(
            encode_fn(src_str), add_if_not_exist=False
        ).long()
        for src_str in lines
    ]
    lengths = torch.LongTensor([t.numel() for t in tokens])
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens, lengths),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            ids=batch['id'],
            src_tokens=batch['net_input']['src_tokens'], src_lengths=batch['net_input']['src_lengths'],
        )

class Embedder():

    def __init__(self, args, model=None, task=None):

        if model is None:
            utils.import_user_module(args)

            if args.buffer_size < 1:
                args.buffer_size = 1
            if args.max_tokens is None and args.max_sentences is None:
                args.max_sentences = 1

            assert not args.sampling or args.nbest == args.beam, \
                '--sampling requires --nbest to be equal to --beam'
            assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
                '--max-sentences/--batch-size cannot be larger than --buffer-size'

            print(args)

            self.use_cuda = torch.cuda.is_available() and not args.cpu

            # Setup task, e.g., translation
            self.task = tasks.setup_task(args)

            # Load ensemble
            print('| loading model(s) from {}'.format(args.path))
            models, _model_args = checkpoint_utils.load_model_ensemble(
                args.path.split(':'),
                arg_overrides=eval(args.model_overrides),
                task=self.task,
            )

            self.model = models[0]

            if self.use_cuda:
                self.model.cuda()

            self.max_positions = utils.resolve_max_positions(
                self.task.max_positions(),
                *[model.max_positions() for model in models]
            )

            self.args = args
        else:
            self.args = args
            self.model = model
            self.task = task
            self.max_positions = task.max_positions()
            self.use_cuda = torch.cuda.is_available() and not args.cpu

    def add_extra_tokens(self, toks, lens, add_lang_tokens, add_encoder_tokens, vocab, lang="en", lang_emb=False):

        if lang_emb:
            if add_lang_tokens:
                if "__{0}__".format(lang) in vocab.indices:
                    lang_tok = vocab.indices["__{0}__".format(lang)]
                else:
                    lang_tok = vocab.indices["__en__"]

                toks_to_add = toks.clone()[:,0] * 0 + lang_tok
                toks = torch.cat((toks_to_add[:,None], toks), dim=1)
                lens = lens + 1

            if add_encoder_tokens:
                toks_to_add = toks.clone()[:, 0] * 0 + vocab.indices["__lang__"]
                toks = torch.cat((toks_to_add[:, None], toks), dim=1)
                lens = lens + 1


        elif add_encoder_tokens:
            toks_to_add = toks.clone()[:,0] * 0 + vocab.indices["__sem__"]
            toks = torch.cat((toks_to_add[:,None], toks), dim=1)
            lens = lens + 1

        return toks, lens

    def embed(self, inputs, encoder, lang="en", lang_emb=False):

        self.model.eval()

        encoder = getattr(self.model, encoder)

        results = []
        for batch in make_batches(inputs, self.args, self.task, self.max_positions, lambda x: x):
            if self.use_cuda:
                toks, lens = self.add_extra_tokens(batch[1].cuda(), batch[2].cuda(), self.model.args.add_lang_tokens,
                                                   self.model.args.add_encoder_tokens, self.model.dict, lang=lang, lang_emb=lang_emb)
                vecs = encoder(toks, lens, False, sem=not lang_emb)
            else:
                toks, lens = self.add_extra_tokens(batch[1], batch[2], self.model.args.add_lang_tokens,
                                                   self.model.args.add_encoder_tokens, self.model.dict, lang=lang, lang_emb=lang_emb)
                vecs = encoder(toks, lens, False, sem=not lang_emb)
            results.append((batch.ids, vecs['mean'].detach().cpu().numpy()))

        vecs = np.vstack([i[1] for i in results])
        ids = np.hstack([i[0] for i in results])
        vecs = vecs[np.argsort(ids)]

        self.model.train()

        return vecs

def cli_main():
    parser = options.get_generation_parser(interactive=True)
    args = options.parse_args_and_arch(parser)

    embed = Embedder(args)

    data = ['I have a dog.', 'How are you?', 'What!', 'I want something to eat.', 'How is youuuu?']
    vecs = embed.embed(data, 'encoder_sem')

    return vecs

if __name__ == '__main__':
    cli_main()
