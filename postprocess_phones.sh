
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

for chckppath in "$1"/*;
 do
  cd ~/alexandra_internship
  echo "$chckppath"
  chckp=$(echo "$chckppath" | sed 's:.*/::')

  CUDA_VISIBLE_DEVICES=$3 fairseq-generate data-bin/$modelname --path $chckppath --source-lang de --target-lang en --batch-size 256  --remove-bpe --beam 5  --gen-subset valid >"$modelname"gpu"$3".en
  cat "$modelname"gpu"$3".en | grep -P "^H" |sort -V |cut -f 3- | sed 's/\[en\]//g' > "$modelname"gpu"$3".en.hyp

  perl -pe "s/\[pause\]//g" "$modelname"gpu"$3".en.hyp > "$modelname"gpu"$3".en.hyp.nopause
  python ./phonemes-eow-to-phoneticwords.py "$modelname"gpu"$3".en.hyp.nopause $4

  mkdir "$modelname"gpu"$3"-using-6
  mv "$modelname"gpu"$3".en.hyp.nopause.phoneticwords "$modelname"gpu"$3"-using-6/valid.en

  cd "$modelname"gpu"$3"-using-6
  python $BPEROOT/apply_bpe.py -c ../en_phonetic_transcr_codes_10k < valid.en > bpe.valid.en
  cp bpe.valid.en bpe.valid.txt
  fairseq-preprocess --source-lang en --target-lang txt --validpref bpe.valid --srcdict ../data-bin/model6-en-phoneticwords-en-txt/dict.en.txt --tgtdict ../data-bin/model6-en-phoneticwords-en-txt/dict.txt.txt
  CUDA_VISIBLE_DEVICES=$3 fairseq-generate ./data-bin/ --path ../trained_models/model6-en-phoneticwords-en-txt/checkpoint_best.pt --gen-subset valid --batch-size 256 --source-lang en --target-lang txt --remove-bpe --beam 5  > "$modelname"gpu"$3"_gen_with_6.txt

  cat "$modelname"gpu"$3"_gen_with_6.txt | grep -P "^H" |sort -V |cut -f 3- | sed 's/\[en\]//g' > "$modelname"gpu"$3"_gen_with_6.txt.hyp

  echo "$chckp" >> ../results_valid_$modelname.txt
  sacrebleu ../valid_txt.en -i  "$modelname"gpu"$3"_gen_with_6.txt.hyp -m bleu -lc --tokenize none >> ../results_valid_$modelname.txt
done

# argument 1: path where all checkpoints of this model are saved
# argument 2: binarized data dir name
# argument 3: in which GPU should this run
# argument 4: if this model has durations on the target side (English) (values: durations or withoutdurations)

#bash postprocess_phones.sh trained_models/model7 model7 1 durations

