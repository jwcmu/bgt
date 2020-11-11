"""
get sts
"""

f = open('slurm-287552_7.out', 'r')
sts = f.readlines()

cross = []
foreign = []
en = []
for i in sts:
    i = i.strip()
    if "(comps)" in i:
        s_en = i.split()[2]
        en.append(float(s_en))
    if "(cross)" in i:
        s_cross = i.split()[2]
        cross.append(float(s_cross))
    if "(foreign)" in i:
        s_foreign = i.split()[2]
        foreign.append(float(s_foreign))

print(en)
print(cross)
print(foreign)
print(len(en))
print(len(cross))
print(len(foreign))

en.pop(-3)
en.pop(-3)
cross.pop(-3)
cross.pop(-3)
foreign.pop(-3)
foreign.pop(-3)
print(len(en))
print(len(cross))
print(len(foreign))

"""
get tatoeba
"""

f = open('slurm-287552_5.out', 'r')
tatoeba = f.readlines()

high = []
low = []
for i in tatoeba:
    i = i.strip()
    if "high resource:" in i:
        acc = i.split()[2]
        high.append(float(acc))
    if "low resource:" in i:
        acc = i.split()[2]
        low.append(float(acc))

print(high)
print(low)
print(len(high))
print(len(low))
high.pop(-3)
high.pop(-3)
low.pop(-3)
low.pop(-3)
print(len(high))
print(len(low))

"""
get xnli
"""

xnli = []
f = open('slurm-287552_1.out', 'r')
xnli.extend(f.readlines())

f = open('slurm-287552_2.out', 'r')
xnli.extend(f.readlines())

f = open('slurm-287552_3.out', 'r')
xnli.extend(f.readlines())

f = open('slurm-287552_4.out', 'r')
xnli.extend(f.readlines())

xnlien = []
xnliar = []
xnlies = []
xnlitr = []
xnlifr = []
xnlide = []
xnliru = []
xnlizh = []
for i in xnli:
    i = i.strip()
    if "Eval Test lang en" in i:
        xnlien.append(float(i.split()[9].replace("%","")))
    if "Eval Test lang fr" in i:
        xnlifr.append(float(i.split()[9].replace("%","")))
    if "Eval Test lang es" in i:
        xnlies.append(float(i.split()[9].replace("%","")))
    if "Eval Test lang de" in i:
        xnlide.append(float(i.split()[9].replace("%","")))
    if "Eval Test lang ru" in i:
        xnliru.append(float(i.split()[9].replace("%","")))
    if "Eval Test lang tr" in i:
        xnlitr.append(float(i.split()[9].replace("%","")))
    if "Eval Test lang ar" in i:
        xnliar.append(float(i.split()[9].replace("%","")))
    if "Eval Test lang zh" in i:
        xnlizh.append(float(i.split()[9].replace("%","")))

print(en)
print(len(en))

"""
get bucc
"""

f = open('slurm-287552_8.out', 'r')
bucc = f.readlines()

f1s = []
for i in bucc:
    i = i.strip()
    if "F1=" in i:
        f1 = i.split()[5].replace("F1=", "")
        f1s.append(float(f1))

print(f1s)
print(len(f1s))

print()
print()

for i in range(12):
    print(f"{en[i]*100:.2f}\t{cross[i]*100:.2f}\t{foreign[i]*100:.2f}\t{low[i]:.2f}\t{high[i]:.2f}\t{xnlien[i]:.2f}\t{xnliar[i]:.2f}\t{xnlies[i]:.2f}\t{xnlitr[i]:.2f}")

print()

for i in range(12,19,1):
    print(f"{en[i]*100:.2f}\t{cross[i]*100:.2f}\t{foreign[i]*100:.2f}\t{low[i]:.2f}\t{high[i]:.2f}\t{xnlien[i]:.2f}\t{xnliar[i]:.2f}\t{xnlies[i]:.2f}\t{xnlitr[i]:.2f}")

print()

for i in range(19,21,1):
    print(f"{en[i]*100:.2f}\t{cross[i]*100:.2f}\t{foreign[i]*100:.2f}\t{low[i]:.2f}\t{high[i]:.2f}\t{xnlien[i]:.2f}\t{xnliar[i]:.2f}\t{xnlies[i]:.2f}\t{xnlitr[i]:.2f}\t{xnlifr[i]:.2f}\t{xnlifr[i]:.2f}\t{xnlizh[i]:.2f}")

#print bucc
print(f"{f1s[0]:.2f}\t{f1s[1]:.2f}\t{f1s[2]:.2f}\t{f1s[3]:.2f}")
print(f"{f1s[4]:.2f}\t{f1s[5]:.2f}\t{f1s[6]:.2f}\t{f1s[7]:.2f}")

print()

#probe
f = open('slurm-287552_6.out', 'r')
probe = f.readlines()

d = {}
ct = 0
for i in probe:
    i = i.strip()
    if "Test acc :" in i:
        name = i.split()[9]
        score = float(i.split()[7])
        #print(name, score)
        if name in d:
            d[name] += [score]
        else:
            d[name] = [score]
    if "comps" in i:
        name = "STS"
        score = float(i.split()[2])*100
        if name in d:
            d[name] += [score]
        else:
            d[name] = [score]
        ct += 1

#print(d)

for i in range(ct):
    print(f"{d['STS'][i]:.2f}\t{d['POSCOUNT'][i]:.2f}\t{d['POSFIRST'][i]:.2f}\t{d['LENGTH'][i]:.2f}\t{d['WORDCONTENT'][i]:.2f}\t{d['DEPTH'][i]:.2f}\t{d['TOPCONSTITUENTS'][i]:.2f}\t{d['BIGRAMSHIFT'][i]:.2f}\t{d['TENSE'][i]:.2f}\t{d['SUBJNUMBER'][i]:.2f}\t{d['OBJNUMBER'][i]:.2f}")
