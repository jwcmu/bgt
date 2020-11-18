import io
import numpy as np
from sacremoses import MosesTokenizer
from glob import glob
from scipy import stats
from wer import wer

def get_wer(s1, s2):
    s1 = s1.split()                                                                                                                                                                                                        
    s2 = s2.split()
    return 0.5 * wer(s1,s2) + 0.5 * wer(s2,s1)

entok = MosesTokenizer(lang='en')

textfiles = glob("../STS/*-en-test/*input*txt")

def make_dataset(f, gs):
    sent1, sent2 = zip(*[l.split("\t") for l in
                               io.open(f,
                    encoding='utf8').read().splitlines()])
    raw_scores = np.array([x for x in
                           io.open(gs,
                                           encoding='utf8')
                        .read().splitlines()])
    not_empty_idx = raw_scores != ''

    def process(s):
        tok = entok.tokenize(s, escape=False)
        return " ".join(tok).lower()
    gs_scores = [float(x) for x in raw_scores[not_empty_idx]]
    sent1 = np.array([s for s in sent1])[not_empty_idx]
    sent2 = np.array([s for s in sent2])[not_empty_idx]

    wers = []
    for i in range(len(sent1)):
        wers.append(get_wer(process(sent1[i]), process(sent2[i])))

    return list(zip(sent1, sent2, gs_scores, wers))

for i in textfiles:
    f = open(i, 'r')
    lines = f.readlines()

all_data = []
all_wers = []
for f in textfiles:
    gs = f.replace("input", "gs")
    data = make_dataset(f, gs)
    all_data.extend(data)
    all_wers.extend([d[-1] for d in data])

wers = np.array(all_wers)    
percs = []

for i in all_data:
    percs.append(stats.percentileofscore(wers, i[-1]))

fpos = open('hard-pos.txt', 'w')
fneg = open('hard-neg.txt', 'w')

for i in range(len(all_data)):
    if all_data[i][2] >= 4 and percs[i] >= 80:
        fpos.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(all_data[i][0], all_data[i][1], all_data[i][2], all_data[i][3], percs[i]))
    elif all_data[i][2] <= 1 and percs[i] <= 20:
        fneg.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(all_data[i][0], all_data[i][1], all_data[i][2], all_data[i][3], percs[i]))
