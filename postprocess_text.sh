
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


checkpoint=$1
modelname=$2
name=test$modelname

CUDA_VISIBLE_DEVICES=$3 fairseq-generate data-bin/$modelname --path $checkpoint --source-lang de --target-lang en --batch-size 256  --remove-bpe --beam 5 > $name

cat $name | grep -P "^H" |sort -V |cut -f 3- | sed 's/\[en\]//g' | ./mosesdecoder/scripts/tokenizer/detokenizer.perl -l en > $name.hyp.detok
cat $name | grep -P "^T" |sort -V |cut -f 2- | sed 's/\[en\]//g' | ./mosesdecoder/scripts/tokenizer/detokenizer.perl -l en > $name.ref.detok

perl -pe "s/\[pause\]//g" $name.hyp.detok > $name.hyp.detok.nopause
perl -pe "s/\[pause\]//g" $name.ref.detok > $name.ref.detok.nopause

python3 remove-punc.py $name.hyp.detok.nopause
python3 remove-punc.py $name.ref.detok.nopause

sacrebleu $name.ref.detok.nopause.no-punc -i $name.hyp.detok.nopause.no-punc -m bleu -lc --tokenize none

# args: trained_models/model7/checkpoint180.pt model7 0
