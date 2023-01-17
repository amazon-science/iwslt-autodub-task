
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


BPEROOT=../subword-nmt/subword_nmt
modelname=$2
subset=$3
chckp=$(echo "$1" | sed 's:.*/::')

name=test_phones_hyp_"$modelname"_"$subset".txt

perl -pe "s/\[pause\]//g" $name >  "$name".hyp.nopause
python phonemes-eow-to-phoneticwords.py "$name".hyp.nopause $4

mkdir $modelname-subset-test-using-6-$chckp
mv  "$name".hyp.nopause.phoneticwords $modelname-subset-test-using-6-$chckp/test.en

cd $modelname-subset-test-using-6-$chckp
python $BPEROOT/apply_bpe.py -c ../en_phonetic_transcr_codes_10k < test.en > bpe.test.en
cp bpe.test.en bpe.test.txt
fairseq-preprocess --source-lang en --target-lang txt --testpref bpe.test --srcdict ../data-bin/model6-en-phoneticwords-en-txt/dict.en.txt --tgtdict ../data-bin/model6-en-phoneticwords-en-txt/dict.txt.txt
CUDA_VISIBLE_DEVICES=$4 fairseq-generate ./data-bin/ --path ../trained_models/model6-en-phoneticwords-en-txt/checkpoint_best.pt --gen-subset test --batch-size 256 --source-lang en --target-lang txt --remove-bpe --beam 5  > $modelname_gen_with_6_chckp_$chckp.txt

cat $modelname_gen_with_6_chckp_$chckp.txt | grep -P "^H" |sort -V |cut -f 3- | sed 's/\[en\]//g' > $modelname_gen_with_6_chckp_$chckp.txt.hyp

# to compute the bleu, the reference should only contain the sentences that were actually used from each subset #1, #2 (the ones we have videos for)

if [ $subset = "subset1" ]; then
   subset="subset1_1to40_150to200"
else
 subset="subset2_1to101"
fi

sacrebleu ../test_txt_sentences_"$subset".en -i $modelname_gen_with_6_chckp_$chckp.txt.hyp -m bleu -lc --tokenize none
