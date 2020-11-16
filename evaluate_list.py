import sentencepiece as spm
from sacremoses import MosesTokenizer
import numpy as np

#python -u evaluate_list.py training/fr/data-joint-bin/ -s en -t fr --path checkpoints/bgt5-fr-65536-25-0-sample-0.5-1.0/checkpoint20.pt  --sentencepiece training/fr/fr-en.1m.tok.all.sp.20k.model --cpu --model-overrides "{'cpu': 1}" --sim-file /projects/tir5/users/jwieting/data/analysis/sentences.txt

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

class FileSim(object):

    def __init__(self):
        self.similarity = lambda s1, s2: np.nan_to_num(cosine(np.nan_to_num(s1), np.nan_to_num(s2)))

    def score(self, params, batcher, f):
        f = open(f, 'r')
        lines = f.readlines()
        input1 = []
        input2 = []
        for i in lines:
            i = i.strip().split("\t")
            s1 = i[0].strip()
            s2 = i[1].strip()
            input1.append(s1)
            input2.append(s2)
        sys_scores = []
        for ii in range(0, len(input1), params.batch_size):
            batch1 = input1[ii:ii + params.batch_size]
            batch2 = input2[ii:ii + params.batch_size]

            # we assume get_batch already throws out the faulty ones
            if len(batch1) == len(batch2) and len(batch1) > 0:
                enc1 = batcher(params, batch1)
                enc2 = batcher(params, batch2)

                for kk in range(enc2.shape[0]):
                    sys_score = self.similarity(enc1[kk], enc2[kk])
                    sys_scores.append(sys_score)

        return sys_scores

def batcher(params, batch):
    batch = [" ".join(s) for s in batch]
    new_batch = []
    for i in batch:
        tok = params.entok.tokenize(i, escape=False)
        p = " ".join(tok).lower()
        p = params.sp.EncodeAsPieces(p)
        p = " ".join(p)
        new_batch.append(p)
    vecs = params.embedder.embed(new_batch, params.encoder)
    return vecs

def evaluate(embedder, args):

    sp = spm.SentencePieceProcessor()
    sp.Load(args.sentencepiece)

    entok = MosesTokenizer(lang='en')

    from argparse import Namespace

    new_args = Namespace(batch_size=32, entok=entok, sp=sp, embedder=embedder,
                     encoder=args.eval_encoder)

    s = FileSim()
    scores = s.score(new_args, batcher, args.sim_file)

    f = open(args.sim_file, 'r')
    lines = f.readlines()

    for i in range(len(scores)):
        print(lines[i].strip() + "\t{0}".format(scores[i]))


if __name__ == '__main__':

    from embed import Embedder
    from fairseq import options

    parser = options.get_generation_parser(interactive=True)
    args = options.parse_args_and_arch(parser)

    embedder = Embedder(args)

    evaluate(embedder, args)