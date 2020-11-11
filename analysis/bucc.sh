#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# LASER  Language-Agnostic SEntence Representations
# is a toolkit to calculate multilingual sentence embeddings
# and to use them for document classification, bitext filtering
# and mining
#
# --------------------------------------------------------
#
# bash script to mine for bitexts in the BUCC corpus

export LASER=/projects/tir5/users/jwieting/LASER

CODE_DIR=/projects/tir5/users/jwieting/multi-single-crosslingual3
BUCC_DIR=/projects/tir5/users/jwieting/LASER/tasks/bucc2

multidata="multilingual-zero-shot/ar-de-es-fr-ru-tr-zh/ar:multilingual-zero-shot/ar-de-es-fr-ru-tr-zh/de:multilingual-zero-shot/ar-de-es-fr-ru-tr-zh/es:multilingual-zero-shot/ar-de-es-fr-ru-tr-zh/fr:multilingual-zero-shot/ar-de-es-fr-ru-tr-zh/ru:multilingual-zero-shot/ar-de-es-fr-ru-tr-zh/tr:multilingual-zero-shot/ar-de-es-fr-ru-tr-zh/zh"
multilangs="en-ar,en-de,en-es,en-fr,en-ru,en-tr,en-zh"
multisp="/projects/tir5/users/jwieting/multi-single-crosslingual3/multilingual-zero-shot/ar-de-es-fr-ru-tr-zh/multi.zero.all.model"

#python -u train.py multilingual-zero-shot/ar-de-es-fr-ru-tr/ar:multilingual-zero-shot/ar-de-es-fr-ru-tr/de:multilingual-zero-shot/ar-de-es-fr-ru-tr/es:multilingual-zero-shot/ar-de-es-fr-ru-tr/fr:multilingual-zero-shot/ar-de-es-fr-ru-tr/ru:multilingual-zero-shot/ar-de-es-fr-ru-tr/tr  -a mgtsepcompressed1024-6layer2 --setting 5 --optimizer adam --lr 0.0005 --label-smoothing 0.1 --dropout 0.3 --max-tokens 1250 --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 --criterion mgt_loss --max-epoch 15 --warmup-updates 4000 --warmup-init-lr '1e-07' --adam-betas '(0.9, 0.98)' --save-dir checkpoints/mgt-compressed-rl=0.5-tl=1-sl=1-lang/sem=1-kl=1 --distributed-world-size 4 --latent-size 1024 --update-freq 5 --task multi_vae --save-interval-updates 0 --freeze 0 --sentencepiece multilingual-zero-shot/ar-de-es-fr-ru-tr/multi.zero.all.model --x0 72741464 --translation-loss 1 --sentence-avg --num-workers 0 --translation-type sample --reconstruction-loss 0.5 --add-lang-tokens 1 --lang-pairs en-ar,en-de,en-es,en-fr,en-ru,en-tr --add-encoder-tokens 1 --left-pad-source False --separate-layers 1 --uniform-language-weight 1 --tokenize 0  --compressed 1 --kl-weight 1 --save-sts 1 --no-epoch-checkpoints

cd $BUCC_DIR
rm embed.$1/*

if [ -z ${LASER+x} ] ; then
  echo "Please set the environment variable 'LASER'"
  exit
fi

# general config
bucc="bucc2018"
idx=$1
model=$2
data="."
xdir=downloaded	# tar files as distrubuted by the BUCC evaluation
ddir=${bucc}	# raw texts of BUCC
edir=embed.$1	# normalized texts and embeddings
langs=("fr" "de" "ru" "zh")
#langs=("fr" "de" "ru")
#langs=("zh")
ltrg="en"	# English is always the 2nd language

# encoder
model_dir="${LASER}/models"
encoder="${model_dir}/bilstm.93langs.2018-12-26.pt"
bpe_codes="${model_dir}/93langs.fcodes"


###################################################################
#
# Extract files with labels and texts from the BUCC corpus
#
###################################################################

GetData () {
  fn1=$1; fn2=$2; lang=$3
  outf="${edir}/${bucc}.${lang}-${ltrg}.${fn2}"
  for ll  in ${ltrg} ${lang} ; do
    inf="${ddir}/${fn1}.${ll}"
    if [ ! -f ${outf}.txt.${ll} ] ; then
      echo " - extract files ${outf} in ${ll}"
      cat ${inf} | cut -f1 > ${outf}.id.${ll}
      cat ${inf} | cut -f2 > ${outf}.txt.${ll}
    fi
  done
}

ExtractBUCC () {
  slang=$1
  tlang=${ltrg}

  pushd ${data} > /dev/null
  if [ ! -d ${ddir}/${slang}-${tlang} ] ; then
    for tf in ${xdir}/${bucc}-${slang}-${tlang}.*.tar.bz2 ; do
      echo " - extract from tar `basename ${tf}`"
      tar jxf $tf
    done
  fi

  GetData "${slang}-${tlang}/${slang}-${tlang}.sample" "dev" ${slang}
  GetData "${slang}-${tlang}/${slang}-${tlang}.training" "train" ${slang}
  GetData "${slang}-${tlang}/${slang}-${tlang}.test" "test" ${slang}
  popd > /dev/null
}


###################################################################
#
# Tokenize and Embed
#
###################################################################

Embed () {
  ll=$2
  txt="$1.txt.${ll}"
  txt=$LASER/tasks/bucc2/${txt}
  enc="$1.enc.${ll}"
  enc=$LASER/tasks/bucc2/${enc}
  echo ${txt}
  echo ${enc}
  if [ ${ll} == "zh" ] ; then
      echo python jieba_tok.py ${txt} 
      python jieba_tok.py ${txt}
  fi
  if [ ! -s ${enc} ] ; then
    cd ${CODE_DIR}
    if [ "$idx" = "mt" ]; then
      python -u embed_bucc.py ${multidata} --path ${model} --sentencepiece ${multisp} --bucc-file ${txt} --task multi_vae --lang-pairs ${multilangs}  --add-lang-tokens 0 --add-encoder-tokens 0 --compressed 1 --output-file ${enc} --tokenize 0
    elif [ "$idx" = "mgt" ]; then
      python -u embed_bucc.py ${multidata} --path ${model} --sentencepiece ${multisp} --bucc-file ${txt} --task multi_vae --lang-pairs ${multilangs}  --add-lang-tokens 1 --add-encoder-tokens 1 --compressed 1 --output-file ${enc} --tokenize 0
    fi
    cd $BUCC_DIR
  fi
}


###################################################################
#
# Mine for bitexts
#
###################################################################

Mine () {
  bn=$1
  l1=$2
  l2=$3
  cand="${bn}.candidates.tsv"
  if [ ! -s ${cand} ] ; then
    python3 ${LASER}/source/mine_bitexts.py \
       ${bn}.txt.${l1} ${bn}.txt.${l2} \
       --src-lang ${l1} --trg-lang ${l2} \
       --src-embeddings ${bn}.enc.${l1} --trg-embeddings ${bn}.enc.${l2} \
       --unify --mode mine --retrieval max --margin ratio -k 4  \
       --output ${cand} \
       --verbose --gpu
  fi
}


###################################################################
#
# Main loop
#
###################################################################

echo -e "\nProcessing BUCC data in ${data}"

# create output directories
for d in ${ddir} ${edir} ; do
  mkdir -p ${d}
done

for lsrc in ${langs[@]} ; do
  ExtractBUCC ${lsrc}

  # Tokenize and embed train 
  bname="${bucc}.${lsrc}-${ltrg}"
  rm ${bname}.*
  part="${bname}.${1}.train"
  part2="${bname}.train"
  Embed ${edir}/${part2} ${lsrc} ${encoder} ${bpe_codes}
  Embed ${edir}/${part2} ${ltrg} ${encoder} ${bpe_codes}

  # mine for texts in train
  Mine ${edir}/${part2} ${lsrc} ${ltrg}

  # optimize threshold on BUCC training data and provided gold alignments
  if [ ! -s ${part}.log ] ; then
    python3 bucc.py \
      --src-lang ${lsrc} --trg-lang ${ltrg} \
      --bucc-texts ${edir}/${part2}.txt \
      --bucc-ids ${edir}/${part2}.id \
      --candidates ${edir}/${part2}.candidates.tsv \
      --gold ${ddir}/${lsrc}-${ltrg}/${lsrc}-${ltrg}.training.gold \
      --verbose \
      | tee ${part}.log
  fi

  # Tokenize and embed test 
  #part="${bname}.test"
  #Embed ${edir}/${part2} ${lsrc} ${encoder} ${bpe_codes}
  #Embed ${edir}/${part2} ${ltrg} ${encoder} ${bpe_codes}

  # mine for texts in test
  #Mine ${edir}/${part2} ${lsrc} ${ltrg}

  # extract test bitexts for treshhold optimized on train
  #th=`grep 'best threshold' ${bname}.train.log | sed -e 's/[=:]/ /g' | awk '{print $4}'`
  #extracted="${edir}/${part}.extracted.tsv"
  #if [ ! -s ${extracted} ] ; then
  #  python3 bucc.py \
  #    --src-lang ${lsrc} --trg-lang ${ltrg} \
  #    --bucc-texts ${edir}/${part2}.txt \
  #    --bucc-ids ${edir}/${part2}.id \
  #    --candidates ${edir}/${part2}.candidates.tsv \
  #    --threshold ${th} --output ${extracted} \
  #    --verbose
  #fi
done

# Bonus: extract bitexts with English alignments
# using a (conservative) threshold of 1.1
# All the data is supposed to be already tokenized

#th=1.1
#for lsrc in ${langs[@]} ; do
#  for ltrg in ${langs[@]} ; do
#    if [ ${lsrc} != 'en' -a ${ltrg} != "en" -a ${lsrc} != ${ltrg} ] ; then
#      bitext="${bucc}.${lsrc}-${ltrg}.train.extracted.th${th}.csv"
#      if [ ! -s ${bitext} ] ; then
#        echo "Extracting bitexts for ${lsrc}-${ltrg}"
#        python3 ${LASER}/source/mine_bitexts.py \
#          ${edir}/${bucc}.${lsrc}-en.train.txt.${lsrc} \
#          ${edir}/${bucc}.${ltrg}-en.train.txt.${ltrg} \
#          --src-lang ${lsrc} --trg-lang ${ltrg} \
#          --src-embeddings ${edir}/${bucc}.${lsrc}-en.train.enc.${lsrc} \
#          --trg-embeddings ${edir}/${bucc}.${ltrg}-en.train.enc.${ltrg} \
#          --unify --mode mine --retrieval max --margin ratio -k 4  \
#          --output ${bitext} --threshold ${th} \
#          --verbose --gpu
#      fi
#    fi
#  done
#done
