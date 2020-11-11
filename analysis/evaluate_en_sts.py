import os

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

for i in lines:
    i = i.strip()
    i = i.split()
    if len(i) == 0:
        continue
    print(i)
    data = None
    langs = None
    sp = None
    compressed = i[3]
    model = i[0][3:]
    if i[2] == "tri":
        data = tridata
        langs = trilangs
        sp = trisp
    elif i[2] == "multi":
        data = multidata
        langs = multilangs
        sp = multisp
    elif i[2] == "enes":
        data = enesdata
        langs = eneslangs
        sp = enessp
    cmd = None
    if i[1] == "mt":
        cmd = "python -u evaluate.py {0} --path {1} --sentencepiece {2} --task multi_vae " \
          "--lang-pairs {3} --compressed {4} --tokenize 0".format(data, model, sp, langs, compressed)
    elif i[1] == "mgt":
        cmd = "python -u evaluate.py {0} --path {1} --sentencepiece {2} --task multi_vae " \
          "--lang-pairs {3} --compressed 1 --add-lang-tokens 1 --add-encoder-tokens 1 --tokenize 0".format(data, model, sp, langs)
    os.chdir("..")
    print(cmd)
    os.system(cmd)
    os.chdir("analysis")
