# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch

from fairseq import utils

from . import FairseqCriterion, register_criterion


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        non_pad_mask = target.ne(ignore_index)
        nll_loss = nll_loss[non_pad_mask]
        smooth_loss = smooth_loss[non_pad_mask]
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('bgt_mono_loss')
class BGTMonoLossCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.args = args
        self.eps = args.label_smoothing
        self.x0 = args.x0
        self.kl_threshold = args.kl_threshold
        self.translation_loss = args.translation_loss

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--x0', default=0., type=float, metavar='D',
                            help='annealing rate')
        parser.add_argument('--kl-threshold', default=0., type=float, metavar='D',
                            help='target value for KL')
        parser.add_argument('--translation-loss', default=1., type=float, metavar='D',
                            help='weight on translation loss')
        # fmt: on

    def kl_anneal_function(self, step, x0):
        return min(1, step/x0)

    def get_au(self, means):
        delta = 0.01
        au_var = means.var(dim=0)
        return (au_var >= delta).sum().data

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if self.args.setting == 9:
            assert self.args.sentence_avg
        net_output = model(sample['en_net_input']['src_tokens'], sample['en_net_input']['src_lengths'],
                           sample['en_net_input']['prev_output_tokens'], sample['fr_net_input']['src_tokens'],
                           sample['fr_net_input']['src_lengths'],
                           sample['fr_net_input']['prev_output_tokens'])

        #get targets
        target = sample['fr_target'].view(-1, 1)

        #compute loss on lv en
        label_smoothed_loss, nll_loss_lv_en = self.compute_loss(model, [net_output['lv_logits']], target, reduce=True)

        #compute KL term
        KL_loss = torch.tensor(0)
        KL_weight = 0
        if self.kl_threshold > 0:
            if 'logv' in net_output:
                KL_loss = -0.5 * (1 + net_output['logv'] - net_output['mean'].pow(2) -
                                                       net_output['logv'].exp())
                KL_loss = torch.sum(torch.clamp(KL_loss, min=self.kl_threshold / KL_loss.size()[1]))
        else:
            if 'logv' in net_output:
                KL_loss = -0.5 * torch.sum(
                    1 + net_output['logv'] - net_output['mean'].pow(2) - net_output['logv'].exp())

        #compute loss
        if 'logv' in net_output:
            KL_weight = self.kl_anneal_function(model.num_updates, self.x0)
            loss = nll_loss_lv_en + KL_weight*KL_loss
        else:
            loss = label_smoothed_loss

        if self.translation_loss:
            en_trans_loss, nll_loss = self.compute_loss(model, [net_output['trans_logits']], target, reduce=reduce)
            loss += self.translation_loss*en_trans_loss
        else:
            nll_loss = torch.tensor(0)

        sample_size = sample['en_target'].size(0)

        au = self.get_au(net_output['mean'])

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['en_ntokens'],
            'nsentences': sample['en_target'].size(0),
            'sample_size': sample_size,
            'au': utils.item(au * sample['en_target'].size(0)),
            'kl_weight': KL_weight * sample['en_target'].size(0),
            'kl_loss': utils.item(KL_loss.data),
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, target, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size if sample_size > 0 else 0.,
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
            'au': sum(log.get('au', 0) for log in logging_outputs) / nsentences,
            'kl_weight': sum(log.get('kl_weight', 0) for log in logging_outputs) / nsentences,
            'kl_loss': sum(log.get('kl_loss', 0) for log in logging_outputs) / nsentences,
        }
