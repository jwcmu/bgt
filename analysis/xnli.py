import os
import sys
import argparse
import pdb
import faiss
import numpy as np

"""
python -u jieba_tok.py /projects/tir5/users/jwieting/LASER/tasks/xnli/embed/xnli.dev.hyp.zh
python -u jieba_tok.py /projects/tir5/users/jwieting/LASER/tasks/xnli/embed/xnli.dev.prem.zh
python -u jieba_tok.py /projects/tir5/users/jwieting/LASER/tasks/xnli/embed/xnli.test.hyp.zh
python -u jieba_tok.py /projects/tir5/users/jwieting/LASER/tasks/xnli/embed/xnli.test.prem.zh
"""

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

CURR_DIR = os.getcwd()

CODE_DIR = "/projects/tir5/users/jwieting/multi-single-crosslingual3"
XNLI_DIR = "/projects/tir5/users/jwieting/LASER/tasks/xnli"
LASER_DIR = "/projects/tir5/users/jwieting/LASER"
data_dir = XNLI_DIR + "/embed"
model = args.path.split('/')[-1]

os.makedirs(data_dir, exist_ok=True)

languages_train = ('en',)
#languages = ('en', 'ar', 'bg', 'de', 'el', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh')
languages = {'en','fr', 'es', 'de', 'ru', 'tr', 'ar', 'zh'}

print('\nProcessing train:')
for lang in languages_train:
    for part in ('prem', 'hyp'):
        cfname = os.path.join(data_dir, 'xnli.train.' + part + '.')
        print(cfname + lang)

        os.chdir(CODE_DIR)

        cmd = f"python -u embed_bucc.py {args.data} --path {args.path} --sentencepiece {args.sentencepiece} " \
              f"--task multi_vae --lang-pairs {args.lang_pairs} --compressed {args.compressed}  " \
              f"--add-lang-tokens {args.add_lang_tokens} --add-encoder-tokens {args.add_encoder_tokens} --tokenize 0 " \
              f"--bucc-file {cfname + lang} " \
              f"--output-file {cfname + 'enc.{0}'.format(model) + '.' + lang}"
        print(cmd)
        os.system(cmd)

        os.chdir(XNLI_DIR)

for corpus in ('xnli.dev', 'xnli.test'):
    print('\nProcessing {}:'.format(corpus))
    for part in ('prem', 'hyp'):
        cfname = os.path.join(data_dir, corpus + '.' + part + '.')
        for lang in languages:
            print(cfname + lang)

            os.chdir(CODE_DIR)

            cmd = f"python -u embed_bucc.py {args.data} --path {args.path} --sentencepiece {args.sentencepiece} " \
                  f"--task multi_vae --lang-pairs {args.lang_pairs} --compressed {args.compressed}  " \
                  f"--add-lang-tokens {args.add_lang_tokens} --add-encoder-tokens {args.add_encoder_tokens} --tokenize 0 " \
                  f"--bucc-file {cfname + lang} " \
                  f"--output-file {cfname + 'enc.{0}'.format(model) + '.' + lang}"
            print(cmd)
            os.system(cmd)

            os.chdir(XNLI_DIR)

fr = 1.0
N = 200
nhid = "512 384"
drop = 0.3
seed = 159753
bsize = 128
lr = 0.001
#langs = "en fr es de el bg ru tr ar vi th zh hi sw ur"
langs = "en fr es de ru tr ar zh"

os.chdir(CURR_DIR)

print("\nTraining the classifier (see {0}/xnli.frac-{1}.{2}.log)".format(data_dir, fr, model))

cmd = f"python {LASER_DIR}/source/nli.py -b {data_dir} " \
      f"--train xnli.train.%s.enc.{model}.en --train-labels xnli.train.cl.en " \
      f"--dev xnli.dev.%s.enc.{model}.en --dev-labels xnli.dev.cl.en --test xnli.test.%s.enc.{model} " \
      f"--test-labels xnli.test.cl --lang en fr es de ru tr ar zh --nhid 512 384 --dropout {drop} " \
      f"--bsize {bsize} --seed {seed} --lr {lr} --nepoch {N} --fraction {fr} " \
      f"--save-outputs {data_dir}/xnli.fract-{fr}.{model}.outputs --gpu 0"

print(cmd)
os.system(cmd)
