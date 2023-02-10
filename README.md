
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

This repo contains both the data and code to train and run **automatic dubbing**: translating the speech in a video into a new language such that the new speech is natural when overlayed on the original video. 

Our system takes in German videos like this, along with their transcripts (e.g. "Sein Hauptgebiet waren es, romantische und gotische Poesie aus dem Englischen ins Japanische zu übersetzen."):

https://user-images.githubusercontent.com/3534106/217985339-fb31a3a5-7845-4d52-b651-0ab93e426c70.mp4

And produce videos like this, dubbed into English:

https://user-images.githubusercontent.com/3534106/217978682-d74d35b8-3a5f-4e46-82c2-94269e56b3b4.mp4

# Setting up the environment 

Install [Miniconda/Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) if needed.
For training models, it is assumed that you have at least 1 GPU, with CUDA drivers set up.
This has been tested on 1 NVIDIA V100 GPU with [CUDA 11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive), on Ubuntu 20.04.

```bash
sudo apt install git-lfs awscli ffmpeg
git lfs install --skip-repo
git clone https://github.com/amazon-science/iwslt-autodub-task.git --recursive
cd iwslt-autodub-task

# Create a conda environment
conda env create --file environment.yml
conda activate iwslt-autodub

# Download Prism model for evaluation
cd third_party/prism
conda create -n prism python=3.7 -y
conda activate prism
pip install -r requirements.txt
conda deactivate
wget http://data.statmt.org/prism/m39v1.tar
tar xf m39v1.tar
rm m39v1.tar
cd ../..
```

# Download and extract data

Download the CoVoST2 en-de dataset following these steps, or directly follow instructions at https://github.com/facebookresearch/covost#covost-2.

* First, download [Common Voice audio clips and transcripts](https://commonvoice.mozilla.org/en/datasets) (English, version 4). Note that after filling out the form you can copy the url to download from the download button and download it with wget.
* Next, extract `validated.tsv` from it:
```bash
mkdir covost_tsv
tar -xvf en.tar.gz validated.tsv
mv validated.tsv covost_tsv/
```
* Then extract the required TSV files:
```bash
# Download and split CoVoST2 TSV files
pushd covost_tsv
wget https://dl.fbaipublicfiles.com/covost/covost_v2.en_de.tsv.tar.gz https://raw.githubusercontent.com/facebookresearch/covost/main/get_covost_splits.py
tar -xzf covost_v2.en_de.tsv.tar.gz
python3 get_covost_splits.py -v 2 --src-lang en --tgt-lang de --root . --cv-tsv validated.tsv
popd
```
You should now have `covost_v2.en_de.dev.tsv`, `covost_v2.en_de.test.tsv`, and `covost_v2.en_de.train.tsv` in the `covost_tsv` directory.
* Then extract MFA files:
```bash
mkdir covost_mfa
tar -xf data/training/covost2_mfa.tz -C covost_mfa
mv covost_mfa/covost2_mfa covost_mfa/data
```
Now, all the json files should be in `covost_mfa/data`.

## Compute the distribution of speech durations
```bash
python3 get_durations_frequencies.py ./covost_mfa/data 
```

This computes how often each speech duration is observed in our training data, so that we do the binning correctly. 

## Build dataset

Create the processed datasets for training/evaluation ("text and noised binned segments -> phones and durations" is the provided baseline): 
```bash
# text -> text. Used to generate references for automatic evaluation of translation quality.
python3 build_datasets.py --en en-text-without-durations --de de-text-without-durations

# text and noised binned segments -> phones and durations. For the baseline model.
python3 build_datasets.py --en en-phones-durations --de de-text-noisy-durations --noise-std 0.1 --upsampling 10 --write-segments-to-file
```
For full usage options, run `build_datasets.py -h`.  
For use with factored models, make sure you use the `--write-segments-to-file` option, since that will generate some files required for generating the factored data.

## Prepare target factor files
For the factored baselines, you need to prepare the datasets in the factored formats and generate the auxiliary factors.
```bash
python3 separate_factors.py -i processed_datasets/de-text-noisy-durations0.1-en-phones-durations -o multi_factored
```
This will generate target factor input files in `processed_datasets/de-text-noisy-durations0.1-en-phones-durations/multi_factored/`.
* `*.en.text`: Main output containing the original text, with `<shift>` tokens to account for internal factor shifts so that the factors are conditioned on the main output.
* `*.en.duration`: Main target factor to predict durations. Contains the durations (number of frames) corresponding to each phoneme in `*.en.text`.
* `*.en.total_duration_remaining`: Auxiliary factor to count down the number of frames remaining in each **line**. This is calculated from the (noised) segment durations, and counts down by the number of frames generated at each time step. Note that this may not count down to 0 due to the noise added to the segment durations.
* `*.en.segment_duration_remaining`: Auxiliary factor to count down the number of frames remaining in each **segment**, i.e. until a `[pause]` token is encountered. Similar to the previous factor, but initialized by the corresponding target segment duration for each segment within a line.
* `*.en.pauses_remaining`: Auxiliary factor that counts down the number of `[pause]` tokens remaining in a line.

This is what they should look like:
```bash
$ head -2 processed_datasets/de-text-noisy-durations0.1-en-phones-durations/multi_factored/test.en.{text,duration,total_duration_remaining,segment_duration_remaining,pauses_remaining}
==> processed_datasets/de-text-noisy-durations0.1-en-phones-durations/multi_factored/test.en.text <==
<shift> F AO1 R CH AH0 N AH0 T L IY0 <eow> DH AH1 <eow> R EY1 T S <eow> AH1 V <eow> D EH1 TH S <eow> IH1 N <eow> DH IY0 <eow> Y UW0 N AY1 T IH0 D <eow> K IH1 NG D AH0 M <eow> HH AE1 V <eow> R IH0 D UW1 S T <eow> sp
<shift> AY1 <eow> D IH1 D <eow> HH AE1 V <eow> sp T AH0 <eow> K AH1 T <eow> sp AH0 <eow> sp F Y UW1 <eow> sp K AO1 R N ER0 Z <eow>

==> processed_datasets/de-text-noisy-durations0.1-en-phones-durations/multi_factored/test.en.duration <==
0 12 3 8 12 4 5 5 9 8 14 0 5 7 0 7 17 5 5 0 10 10 0 3 12 13 3 0 13 13 0 2 3 0 3 5 5 13 8 6 10 0 5 8 12 6 8 7 0 5 8 7 0 3 12 8 16 12 14 0 24
0 11 0 17 9 15 0 1 4 5 0 9 3 5 0 20 6 7 0 8 7 0 7 5 4 16 0 3 11 13 4 6 26 24 0

==> processed_datasets/de-text-noisy-durations0.1-en-phones-durations/multi_factored/test.en.total_duration_remaining <==
413 401 398 390 378 374 369 364 355 347 333 333 328 321 321 314 297 292 287 287 277 267 267 264 252 239 236 236 223 210 210 208 205 205 202 197 192 179 171 165 155 155 150 142 130 124 116 109 109 104 96 89 89 86 74 66 50 38 24 24 0
246 235 235 218 209 194 194 193 189 184 184 175 172 167 167 147 141 134 134 126 119 119 112 107 103 87 87 84 73 60 56 50 24 0 0

==> processed_datasets/de-text-noisy-durations0.1-en-phones-durations/multi_factored/test.en.segment_duration_remaining <==
413 401 398 390 378 374 369 364 355 347 333 333 328 321 321 314 297 292 287 287 277 267 267 264 252 239 236 236 223 210 210 208 205 205 202 197 192 179 171 165 155 155 150 142 130 124 116 109 109 104 96 89 89 86 74 66 50 38 24 24 0
246 235 235 218 209 194 194 193 189 184 184 175 172 167 167 147 141 134 134 126 119 119 112 107 103 87 87 84 73 60 56 50 24 0 0

==> processed_datasets/de-text-noisy-durations0.1-en-phones-durations/multi_factored/test.en.pauses_remaining <==
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
```

# Decode test set and evaluate using provided Sockeye baseline model

NOTE: Test set here does not refer to the specific subsets in `data/test/`; rather it refers to the full test sets generated from CoVoST2.

We provide a baseline model checkpoint in **models/sockeye/trained_baselines/baseline_factored_noised0.1**. This uses a target factor to predict durations and additional target factors to help the model keep track of time. The training segment durations have Gaussian noise (std. dev. 0.1) added to teach the model to be flexible about timing in hopes of striking a balance between speech overlap, speech naturalness, and translation quality. (Note that the speech overlap in real human dubs is [only about 70%.](https://arxiv.org/abs/2212.12137))

Before you proceed, in `sockeye_scripts/config`, set ROOT as the path of this repo. For example, `ROOT=~/iwslt-autodub-task`.

Before decoding, please make sure you have run the data and factor preparation steps, so that you have at least `processed_datasets/de-text-noised-durations0.1-en-phones-durations` prepared with the `multi_factored` subdirectory, `processed_datasets/de-text-without-durations-en-text-without-durations` for the translation reference text files. If you ran the steps in the previous section, you will have these already.

## Decoding using **baseline_factored_noised0.1**
The input format for decoding is a specific JSON format that can be prepared using:
```bash
$ python3 sockeye_scripts/decoding/create-json-inputs.py -d processed_datasets/de-text-noisy-durations0.1-en-phones-durations --subset test --output-segment-durations -o processed_datasets/de-text-noisy-durations0.1-en-phones-durations/test.de.json

# Check JSON file looks like this
$ head -2 ~/iwslt-autodub-task/processed_datasets/de-text-noisy-durations0.1-en-phones-durations/test.de.json | jq
{
  "text": "Glück@@ licherweise sind die Ster@@ ber@@ aten im Vereinigten Königreich ges@@ unken <||> <bin87>",
  "target_prefix": "<shift>",
  "target_prefix_factors": [
    "0",
    "413",
    "413",
    "0"
  ],
  "target_segment_durations": [
    413
  ],
  "use_target_prefix_all_chunks": false
}
{
  "text": "Ich musste einige Ab@@ str@@ ich@@ e mach@@ en@@ . <||> <bin44>",
  "target_prefix": "<shift>",
  "target_prefix_factors": [
    "0",
    "246",
    "246",
    "0"
  ],
  "target_segment_durations": [
    246
  ],
  "use_target_prefix_all_chunks": false
}
```

To decode using **baseline_factored_noised0.1**, run
```bash
mkdir -p models/sockeye/trained_baselines/baseline_factored_noised0.1/eval
sockeye-translate \
    -i processed_datasets/de-text-noisy-durations0.1-en-phones-durations/test.de.json \
    -o models/sockeye/trained_baselines/baseline_factored_noised0.1/eval/test.en.output \
    --models models/sockeye/trained_baselines/baseline_factored_noised0.1/model \
    --checkpoints 78 \
    -b 5 \
    --batch-size 32 \
    --chunk-size 20000 \
    --output-type translation_with_factors \
    --max-output-length 768 \
    --force-factors-stepwise frames total_remaining segment_remaining pauses_remaining \
    --json-input \
    --quiet
```
This should take around 5 minutes on 1 V100 GPU, or around an hour without a GPU.

## Evaluate **baseline_factored_noised0.1** output
```bash
./sockeye_scripts/evaluation/evaluate-factored.sh processed_datasets/de-text-noisy-durations0.1-en-phones-durations/test.en models/sockeye/trained_baselines/baseline_factored_noised0.1/eval/test.en.output
```

This will print:
* Translation quality metrics
    - BLEU
    - Prism
    - COMET
* Speech overlap metrics

# Reproduce Sockeye baseline models

Scripts are included to reproduce the Sockeye baselines included here. Before launching training for the factored models, you need specially created vocab files which can be generated using
```bash
cd sockeye_scripts/training
wget https://raw.githubusercontent.com/Proyag/sockeye/factor-pe/sockeye_contrib/create_seq_vocab.py
python create_seq_vocab.py --min-val -4000 --max-val 5000  --output seq_vocab_expanded.json
```

And now, the training can be launched using
```bash
./train_factored_noised0.1.sh
```
If you're using >1 GPU, adjust the following settings in the script first
```bash
# Set the number of GPUs for distributed training
# Adjust BATCH_SIZE and UPDATE_INTERVAL according to your GPU situation.
# For example, if you change N_GPU to 2, you should set update-interval to 8 to have the same effective batch size
N_GPU=1
BATCH_SIZE=4096
UPDATE_INTERVAL=16
```
The trained models will be in `models/sockeye`.

# Generate test set dubs with Sockeye models
There are German videos for two subsets of the test set in `data/test/subset{1,2}`. We want to generate English dubbed videos for these.

Extract the test set audio/video in `data/test`
```bash
pushd data/test
tar -xzf subset1/subset1.tgz -C subset1
tar -xzf subset2/subset2.tgz -C subset2
popd
```

Set up FastSpeech2 (only for the first usage)
```bash
sudo apt-get install ffmpeg
cd third_party/FastSpeech2
# Create separate environment for FastSpeech2 dependencies
conda create -n fastspeech2 python=3.8 -y
conda activate fastspeech2
pip install -r requirements.txt
pip install gdown
# Download and extract pretrained model
mkdir -p output/ckpt/LJSpeech output/result/LJSpeech preprocessed_data/LJSpeech/duration
cd output/ckpt/LJSpeech
gdown https://drive.google.com/uc?id=1r3fYhnblBJ8hDKDSUDtidJ-BN-xAM9pe
unzip LJSpeech_900000.zip
cd ../../../hifigan
unzip generator_LJSpeech.pth.tar.zip
conda deactivate
```

Generate dubbed videos for the test set subsets using FastSpeech2
```bash
cd ~/iwslt-autodub-task  # Or the path to the repo home, if different
python synthesize_speech.py --subset 1
python synthesize_speech.py --subset 2
```
Corresponding to each file `*.X.mov` or `*.X.mp4` in `data/test/subset{1,2}`, a dubbed video `data/test/subset{1,2}/dubbed/*.X.mp4` will be created.  
