import os
import numpy as np
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from glob import glob

CODE_DIR = "/projects/tir5/users/jwieting/multi-single-crosslingual3"
TATOEBA_DATA_DIR = "/projects/tir5/users/jwieting/LASER/data/tatoeba/v1/*"

def embed_load(fname, dim=1024, verbose=False):
    x = np.fromfile(fname, dtype=np.float32, count=-1)
    x.resize(x.shape[0] // dim, dim)
    if verbose:
        print(' - Embeddings: {:s}, {:d}x{:d}'.format(fname, x.shape[0], dim))
    return x

def compute_accuracy(scores):
    n = scores.shape[0]
    correct = 0
    for i in range(n):
        _scores = scores[i]
        max_idx = np.argmax(_scores)
        if max_idx == i:
            correct += 1
    return correct / n * 100

parser = argparse.ArgumentParser()

parser.add_argument("--data", help="training data")
parser.add_argument("--path", help="training data")
parser.add_argument("--sentencepiece", help="")
parser.add_argument("--lang-pairs", help="")
parser.add_argument("--task", help="")
parser.add_argument("--compressed", type=int, help="")
parser.add_argument("--tokenize", type=int, help="")
parser.add_argument("--add-lang-tokens", type=int, default=0, help="")
parser.add_argument("--add-encoder-tokens", type=int, default=0, help="")
parser.add_argument("--eval-type", choices=["tri", "multi"], help="")

args = parser.parse_args()

args.model = args.path.split("/")[-1]

flis = glob(TATOEBA_DATA_DIR)
flis.sort()
lang_lis = []
total = 0
num = 0

lo = []
hi = []

for n in range(0, len(flis), 2):
    i = flis[n]
    if "README" not in i:
        key = i[-3:]
        eng = None
        foreign = None
        lang = None

        if key == "eng":
            eng = i
            foreign = flis[n - 1]
        else:
            eng = flis[n - 1]
            foreign = i

        lang = foreign[-3:]

        if args.eval_type == "tri":
            if lang == "glg" or lang == "aze" or lang == "ara" or lang == "spa" or lang == "tur":
                pass
            else:
                continue
        elif args.eval_type == "multi":
            if lang == "glg" or lang == "aze" or lang == "ukr" or lang == "ara" or lang == "spa" or \
                    lang == "tur" or lang == "fra" or lang == "deu" or lang == "rus":
                pass
            else:
                continue

        os.chdir(CODE_DIR)

        cmd = "python -u embed_bucc.py {0} --path {1} --sentencepiece {2} --task multi_vae --lang-pairs {3} " \
              "--compressed {4}  --add-lang-tokens {5} --add-encoder-tokens {6} --tokenize {9} --bucc-file {7} " \
              "--output-file analysis/embeddings-english-{8}.np".format(args.data,
                                                               args.path, args.sentencepiece, args.lang_pairs,
                                                               args.compressed, args.add_lang_tokens,
                                                               args.add_encoder_tokens, eng, args.model, args.tokenize)
        os.system(cmd)

        cmd = "python -u embed_bucc.py {0} --path {1} --sentencepiece {2} --task multi_vae --lang-pairs {3} " \
              "--compressed {4}  --add-lang-tokens {5} --add-encoder-tokens {6} --tokenize {9} --bucc-file {7} " \
              "--output-file analysis/embeddings-foreign-{8}.np".format(args.data,
                                                               args.path, args.sentencepiece, args.lang_pairs,
                                                               args.compressed, args.add_lang_tokens,
                                                               args.add_encoder_tokens, foreign, args.model, args.tokenize)
        os.system(cmd)

        eng_emb = embed_load("analysis/embeddings-english-{0}.np".format(args.model))
        foreign_emb = embed_load("analysis/embeddings-foreign-{0}.np".format(args.model))
        scores = cosine_similarity(eng_emb, foreign_emb)

        english_acc = compute_accuracy(scores)
        foreign_acc = compute_accuracy(np.transpose(scores))

        total += english_acc + foreign_acc
        num += 2

        print(lang, english_acc, foreign_acc)

        if lang == "glg" or lang == "aze" or lang == "ukr":
            lo.append(english_acc)
            lo.append(foreign_acc)
        else:
            hi.append(english_acc)
            hi.append(foreign_acc)

print("low resource: ", np.mean(lo))
print("high resource: ", np.mean(hi))

print(total / num)
