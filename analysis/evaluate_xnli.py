import os
import argparse

"""
run bash /projects/tir5/users/jwieting/LASER/tasks/xnli/xnli2.sh to download and preprocess files first.
python -u jieba_tok.py /projects/tir5/users/jwieting/LASER/tasks/xnli/embed/xnli.dev.hyp.zh
python -u jieba_tok.py /projects/tir5/users/jwieting/LASER/tasks/xnli/embed/xnli.dev.prem.zh
python -u jieba_tok.py /projects/tir5/users/jwieting/LASER/tasks/xnli/embed/xnli.test.hyp.zh
python -u jieba_tok.py /projects/tir5/users/jwieting/LASER/tasks/xnli/embed/xnli.test.prem.zh
"""

parser = argparse.ArgumentParser()
parser.add_argument("--list", nargs='+', type=int)
args = parser.parse_args()

files = args.list

CODE_DIR = "/projects/tir5/users/jwieting/multi-single-crosslingual3"

multidata = "multilingual-zero-shot/ar-de-es-fr-ru-tr-zh/ar:multilingual-zero-shot/ar-de-es-fr-ru-tr-zh/de:" \
            "multilingual-zero-shot/ar-de-es-fr-ru-tr-zh/es:multilingual-zero-shot/ar-de-es-fr-ru-tr-zh/fr:" \
            "multilingual-zero-shot/ar-de-es-fr-ru-tr-zh/ru:multilingual-zero-shot/ar-de-es-fr-ru-tr-zh/tr:" \
            "multilingual-zero-shot/ar-de-es-fr-ru-tr-zh/zh"
multilangs = "en-ar,en-de,en-es,en-fr,en-ru,en-tr,en-zh"
multisp = "multilingual-zero-shot/ar-de-es-fr-ru-tr-zh/multi.zero.all.model"


tridata = "multilingual-zero-shot/ar-es-tr/ar:" \
          "multilingual-zero-shot/ar-es-tr/es:" \
          "multilingual-zero-shot/ar-es-tr/tr"
trilangs = "en-ar,en-es,en-tr"
trisp = "multilingual-zero-shot/ar-es-tr/multi.zero.all.model"

enesdata = "multilingual-zero-shot-en-es/data-bin/ar:" \
           "multilingual-zero-shot-en-es/data-bin/ar:" \
           "multilingual-zero-shot-en-es/data-bin/es:" \
           "multilingual-zero-shot-en-es/data-bin/es:" \
           "multilingual-zero-shot-en-es/data-bin/tr:" \
           "multilingual-zero-shot-en-es/data-bin/tr"
eneslangs = "en-ar,es-ar,en-es,es-en,en-tr,es-tr"
enessp = "multilingual-zero-shot-en-es/multi.zero.all.model"

f = open('list_of_models.txt', 'r')
lines = f.readlines()

for n,i in enumerate(lines):
    i = i.strip()
    i = i.split()
    if len(i) == 0:
        continue
    if n + 1 not in files:
        continue
    print(i)
    data = None
    langs = None
    sp = None
    eval_type = None
    compressed = i[3]
    model = i[0][3:]
    if i[2] == "tri":
        data = tridata
        langs = trilangs
        sp = trisp
        eval_type = "tri"
    elif i[2] == "multi":
        data = multidata
        langs = multilangs
        sp = multisp
        eval_type = "multi"
    elif i[2] == "enes":
        data = enesdata
        langs = eneslangs
        sp = enessp
        eval_type = "tri"
    cmd = None
    if i[1] == "mt":
        cmd = "python -u xnli.py --data {0} --path {1} --sentencepiece {2} --task multi_vae " \
          "--lang-pairs {3} --compressed {4} --tokenize 0 --eval-type {5}".format(data, model, sp, langs, compressed, eval_type)
    elif i[1] == "mgt":
        cmd = "python -u xnli.py --data {0} --path {1} --sentencepiece {2} --task multi_vae " \
          "--lang-pairs {3} --compressed {4} --add-lang-tokens 1 --add-encoder-tokens 1 --tokenize 0 --eval-type {5}".format(data,
                                                                                    model, sp, langs, compressed, eval_type)
    print(cmd)
    os.system(cmd)
