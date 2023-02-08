#!/bin/bash
set -Eeuo pipefail
source `dirname $0`/../config

REF=$1
HYP=$2

EVAL_DIR=`dirname ${HYP}`
P2G_DIR=${ROOT}/models/phoneme_to_grapheme

if [[ ! -s ${DATA_HOME}/de-text-without-durations-en-text-without-durations/test.en ]]; then
    echo "Please generate ${DATA_HOME}/de-text-without-durations-en-text-without-durations first. This is required for the reference text."
    exit 1
else
    REF_TEXT=${DATA_HOME}/de-text-without-durations-en-text-without-durations/test.debpe.en
    SRC_TEXT=${DATA_HOME}/de-text-without-durations-en-text-without-durations/test.debpe.de
    if [[ ! -s ${REF_TEXT} ]]; then
        sed -r 's/(@@ )|(@@ ?$)//g' ${DATA_HOME}/de-text-without-durations-en-text-without-durations/test.en > ${REF_TEXT}
        sed -r 's/(@@ )|(@@ ?$)//g' ${DATA_HOME}/de-text-without-durations-en-text-without-durations/test.de > ${SRC_TEXT}
    fi
fi

if [[ ! -s ${HYP}.words ]]; then
    # Phoneme-to-grapheme conversion
    echo "Converting phonemes to graphemes for translation quality evaluation"

    # Separate phonemes and durations
    python `dirname $0`/separate-hyp-factors.py ${HYP} --shift

    # Preprocess to prepare output for phoneme-to-grapheme conversion
    sed "s/\[pause\]//g" ${HYP}.phonemes > ${HYP}.nopause
    python ${ROOT}/phonemes-eow-to-phoneticwords.py ${HYP}.nopause withoutdurations

    subword-nmt apply-bpe \
        -c ${P2G_DIR}/phoneme_to_grapheme_bpe_10k \
        -i ${HYP}.nopause.phoneticwords \
        | cut -d' ' -f1-1023 \
        > ${HYP}.nopause.phoneticwords.phonebpe

    # fairseq expects specific filenames
    ln -sf `realpath ${HYP}.nopause.phoneticwords.phonebpe` ${EVAL_DIR}/test.en
    ln -sf `realpath ${EVAL_DIR}/test.en` ${EVAL_DIR}/test.txt

    fairseq-preprocess \
        --testpref ${EVAL_DIR}/test \
        --source-lang en \
        --target-lang txt \
        --srcdict ${P2G_DIR}/dict.en.txt \
        --tgtdict ${P2G_DIR}/dict.txt.txt \
        --destdir ${EVAL_DIR}/data-bin

    # Reduce --max-tokens if your GPU runs out of memory
    fairseq-generate \
        ${EVAL_DIR}/data-bin \
        --path ${ROOT}/models/phoneme_to_grapheme/checkpoint_best.pt \
        --gen-subset test \
        --source-lang en \
        --target-lang txt \
        --max-tokens 16384 \
        --data-buffer-size 100 \
        --beam 5 \
        --remove-bpe \
        | grep -P "^H" | sort -V | cut -f 3- | sed 's/\[en\]//g' \
        > ${HYP}.words
else
    echo "${HYP}.words already exists and will not be re-generated."
fi

echo "Calculating translation quality metrics:"
sacrebleu ${REF_TEXT} -m bleu -f text -lc --tokenize none < ${HYP}.words
echo "Prism:"
# Prism uses PyTorch 1.4.0 which expects CUDA 10.1. Sockeye expects a much more modern CUDA (e.g. 11.7). So run prism on CPU (takes ~8min)
CUDA_VISIBLE_DEVICES=-1 `dirname ${CONDA_PREFIX}`/prism/bin/python ${ROOT}/third_party/prism/prism.py --cand ${HYP}.words --ref ${REF_TEXT} --lang en --model-dir ${ROOT}/third_party/prism/m39v1
echo "COMET:"
comet-score --gpus 1 --quiet --batch_size 128 \
    --model wmt21-comet-da \
    -s ${SRC_TEXT} -t ${HYP}.words -r ${REF_TEXT} \
    | tail -n1

echo -e "\nSpeech overlap metrics:"
python `dirname $0`/format-phonemes-durations.py ${HYP}
python count_durations.py ${REF} ${HYP}.altformat
