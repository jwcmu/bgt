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
    if "all-lang" not in i:
        continue
    i = i.strip()
    i = i.split()
    if len(i) == 0:
        continue
    print(i)
    model = i[0][3:]
    cmd = f"bash bucc.sh {i[1]} {model}"
    print(cmd)
    os.system(cmd)

"""
bash bucc.sh mt ../mgt-paper-models/mgt-paper-models-2/all-langs-mt.pt
bash bucc.sh mgt ../mgt-paper-models/mgt-paper-models-2/all-langs-mgt.pt
"""
