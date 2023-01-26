
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
  
Licensed under the Apache License, Version 2.0 (the "License").
You may not use this file except in compliance with the License.
You may obtain a copy of the License at
  
    http://www.apache.org/licenses/LICENSE-2.0
  
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


# Introduction


# Setting up the environment 

```
conda create --name iwslt-autodub --file requirements.txt
conda activate iwslt-autodub

cd <TODO_repo_dir>
git clone https://github.com/rsennrich/subword-nmt.git subword-nmt
git clone https://github.com/moses-smt/mosesdecoder.git mosesdecoder

aws s3 cp <TODO_dir1>/covost-2/covost_v2.en_de.train.tsv ./covost_tsv
aws s3 cp <TODO_dir1>/covost-2/covost_v2.en_de.dev.tsv ./covost_tsv
aws s3 cp <TODO_dir1>/covost-2/covost_v2.en_de.test.tsv ./covost_tsv
aws s3 cp <TODO_dir2>/model6-phoneticwords-to-text/checkpoint_best.pt trained_models/model6-en-phoneticwords-en-txt

pushd data/training/ ; tar -xvf covost2_mfa.tz ; popd
python3 extract_all_jsons.py
```
Now, all the json files should be in `covost_mfa/data/`.

# Compute the distribution of speech durations
``` 
python3 get_durations_frequencies.py ./covost_mfa/data 
```

This computes how often each speech duration is observed in our training data, so that we do the binning correctly. 

# Build dataset

Depending on which model we want to run, we can create the corresponding dataset: 

```
python3 build_datasets.py ./covost_mfa/data processed_datasets en-text-without-durations de-text-without-durations # model 1b
python3 build_datasets.py ./covost_mfa/data processed_datasets en-phones-without-durations de-text-without-durations # model 2b
python3 build_datasets.py ./covost_mfa/data processed_datasets en-phones-durations de-text-without-durations  # model 4a 
python3 build_datasets.py ./covost_mfa/data processed_datasets en-phones-durations de-text-clean-durations # model 7
python3 build_datasets.py ./covost_mfa/data processed_datasets en-phones-durations de-text-noisy-durations 0.5 # model 7f 
python3 build_datasets.py ./covost_mfa/data processed_datasets en-phones-durations de-text-dummy-durations # model 8 
```

# Example 1: Model 7 w/o noise (input: text and speech durations; output: phones and phone durations)

### Train 

```
cd processed_datasets/de-text-clean-durations-en-phones-durations
fairseq-preprocess --source-lang de --target-lang en --trainpref train --validpref valid  --testpref test --destdir ../../data-bin/model7 --workers 20

cd ../../

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 fairseq-train ./data-bin/model7 \
    --arch transformer_wmt_en_de --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe --eval-bleu-print-samples --save-dir trained_models/model7 --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --save-interval 10 --max-epoch 200 > ./model7.log
```

### Evaluate in terms of BLEU

```
bash postprocess_phones.sh trained_models/model7 model7 1 durations
```
argument 1: path where all checkpoints of this model are saved

argument 2: binarized data dir name

argument 3: in which GPU should this run

argument 4: if this model has durations on the target side (English) (values: `durations` or `withoutdurations`)

The results will be saved in `results_valid_$modelname.txt`, where `$modelname` is argument 2. Select the model with highest validation BLEU to compute the test set BLEU score.


```
bash postprocess_phones_test.sh trained_models/model7/checkpoint100.pt model7 0 durations
```

argument 1: path where the best valid checkpoint is saved

argument 2: binarized data dir name

argument 3: in which GPU should this run

argument 4: if this model has durations on the target side (English) (values: `durations` or `withoutdurations`)

Optionally delete temp files afterwards: `rm -r model7gpu0*`

### Evaluate in terms of speech overlap

After you have run the `postprocess_phones` and `postprocess_phones_test` scripts, run:

```
 python3 count_durations.py test_phones.en testmodel7.hyp
 ```

### Synthesize speech and dubs from our German videos (new dataset)

```
git clone https://github.com/snakers4/silero-vad.git silero_vad
git clone https://github.com/ming024/FastSpeech2.git FastSpeech2
mkdir FastSpeech2/output/ckpt/LJSpeech
mkdir FastSpeech2/output/result/LJSpeech 
```

Download the [pretrained FastSpeech2 model](https://drive.google.com/file/d/1r3fYhnblBJ8hDKDSUDtidJ-BN-xAM9pe/view?usp=share_link) and place it in `FastSpeech2/output/ckpt/LJSpeech`.

Then please execute the following commands:

``` 
unzip FastSpeech2/hifigan/generator_LJSpeech.pth.tar.zip
pip install -r FastSpeech2/requirements.txt
git apply patch_to_use_mfa_durations.patch
mkdir -p for_human_eval
```
If you want to generate dubs for subsets 1,2 (please see description in the paper), you need to download the following files from s3:

```
aws s3 cp <TODO_dir2>/original_91_videos_subset1/ for_human_eval/original_videos --recursive
aws s3 cp <TODO_dir2>/original_101_videos_subset2/ for_human_eval/original_videos --recursive
aws s3 cp <TODO_dir1>/covost-2/custom-test-set/ for_human_eval/ --recursive 
```

Assuming you have trained a model (`model7` for this example), please run the following script to get dubbed videos, compute BLEU scores + speech overlap:

``` 
bash generate_dub.sh model7 test_set trained_models/model7/checkpoint170.pt
``` 
arg 1: pretrained model 

arg 2: specifies the subset used and can be either `test_set` (subset 1) or `test_set_with_pauses.txt` (subset 2)

arg 3: path to checkpoint 

To embed the wav files back to video (and finally get the dubs), run:

``` 
bash embed_wav_to_videos.sh
``` 
Please adjust the `$modelname` and `$subset` in `embed_wav_to_videos.sh`, as well as the indices if you want to use this script for other pretrained models and/or subsets.

# Example 2: Model 7 with noise with standard deviation 0.1 (input: text and speech durations; output: phones and phone durations)

### Train 

```
cd processed_datasets/de-text-noisy-durations0.1-en-phones-durations
fairseq-preprocess --source-lang de --target-lang en --trainpref train --validpref valid  --testpref test --destdir ../../data-bin/model7-sd0.1 --workers 20

cd ../../

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 fairseq-train ./data-bin/model7-sd0.1 \
    --arch transformer_wmt_en_de --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe --eval-bleu-print-samples --save-dir trained_models/model7-sd0.1 --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --save-interval 1 --max-epoch 20 > ./model7-sd0.1.log
```


Notice that we train the model for 20 epochs (while model 7 was trained for 200 epochs). This happens because the dataset we are using now is 10x bigger than the dataset of `model 7` (for each sentence of the initial dataset, 10 noisy versions were added).

Therefore, an "epoch" of `model7-sd0.1` is actually equal to 10 x epoch of model 7. The command above makes sure that we will train the model for the same number of instances as `model 7`.

The rest of the steps are the same as in Example 1. 


# Example 3: Model 1b (text-to-text baseline)

### Train 

```
cd processed_datasets/de-text-without-durations-en-text-without-durations
fairseq-preprocess --source-lang de --target-lang en --trainpref train --validpref valid  --testpref test --destdir ../../data-bin/model1b --workers 20

cd ../../

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 fairseq-train ./data-bin/model1b \
    --arch transformer_wmt_en_de --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe --eval-bleu-print-samples --save-dir trained_models/model1b --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --save-interval 10 --max-epoch 200 > ./model1b.log
```
### Evaluate in terms of BLEU

Assuming the checkpoint with the best valid BLEU is checkpoint_best.pt:

```
bash postprocess_text.sh trained_models/model1b/checkpoint_best.pt model1b 0
```
