
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


import pickle
import re
import argparse
import os

import torch
import yaml
import numpy as np
from pydub import AudioSegment
import codecs
from subword_nmt.apply_bpe import BPE

from FastSpeech2.utils.model import get_model, get_vocoder
from FastSpeech2.utils.tools import to_device, synth_samples
from FastSpeech2.text import text_to_sequence
from extract_timestamps import SileroVad
from preprocessing_scripts import Bin

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_generated_file(line):

    durations, phone = [], []

    temp = line.split()[1:][::2]
    for _duration in temp:
        durations.append(int(_duration))
    phone = line.split()[::2]

    phones = "{" + "}{".join(phone) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")
    sequence = np.array(
        text_to_sequence(phones, preprocess_config["preprocessing"]["text"]["text_cleaners"])
    )
    return np.array(sequence), durations, " ".join(phone)


def synthesize(model, step, configs, vocoder, batchs, control_values, basename=None):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
                basename
            )
    duration_rounded = output[5]
    return duration_rounded


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=900000)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        default="single",
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        default='FastSpeech2/config/LJSpeech/preprocess.yaml',
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, default='FastSpeech2/config/LJSpeech/model_MFAphones_MFAdurations.yaml',
        help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, default='FastSpeech2/config/LJSpeech/train.yaml', help="path to train.yaml"
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    parser.add_argument(
        "--wav_file",
        type=str,
        required=True,
        help="provide the reference wav file",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="provide the name of the model used",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="provide the name of the model used",
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    if model_config["phones"] == "MFA":
        if model_config["FS2prediction"] is False:
            suffix = "MFAphones_MFAdurations"
        else:
            suffix = "MFAphones_FS2durations"
    elif model_config["phones"] == "FS2":
        suffix = "FS2phones_FS2durations"
    configs = (preprocess_config, model_config, train_config)

    # Get FS2 model and pretrained model we will use (text + speech durations to phones + phone durations)
    model = get_model(args, configs, device, train=False)
    control_values = args.pitch_control, args.energy_control, args.duration_control
    wav_file = args.wav_file
    model_name = args.model
    checkp_path = args.checkpoint_path

    index = int(wav_file.split(".")[-2])
    name = ".".join(wav_file.split(".")[:-2])
    if name == "test_set":
        name = "test_subset_raw.de.txt"

    # Read the source text data and get correct sentence (using the provided index of wav file) + pauses
    with open(os.path.join('./for_human_eval', name), 'r') as f:
        lines_de_text = f.readlines()
        sentence = lines_de_text[index-1]
        sentence_segments = sentence.split('[pause]')

    # Load durations (the binning computed using the train set of covost, which we used for all our experiments)
    durations_path = './durations_freq_all.pkl'
    num_bins = 100
    if os.path.exists(durations_path):
        with open(durations_path, 'rb') as f:
            durations_freq = pickle.load(f)
    else:
        print("Please run get_durations_frequencies.py!")

    # Load correct BPE codes for German and English

    codes_de = codecs.open("./de_codes_10k", encoding='utf-8')
    bpe_de = BPE(codes_de)
    codes_en = codecs.open("./en_codes_10k_mfa", encoding='utf-8')
    bpe_en = BPE(codes_en)

    bin_instance = Bin(durations_freq, n=num_bins)

    # Use VAD to get speech durations and pauses

    silero_vad = SileroVad()
    sampling_rate, hop_length = 22050, 256

    duration_frames = []
    speech_timestamps, pauses = silero_vad.get_timestamps("./for_human_eval/original_videos/" + wav_file)

    for timestamp in speech_timestamps:
        duration_frames.append(str(np.round(timestamp["end"] * sampling_rate / hop_length) - np.round(timestamp["start"] * sampling_rate / hop_length)))

    with open("durations_{}.txt".format(model_name), 'a') as f:
        f.write(wav_file + '\n')
        f.write('Durations from VAD (our reference) \n')
        f.write('{}\n'.format(duration_frames))

    # Bin the detected durations to 100 bins using the durations_freq_all.pkl file (to match training binning)

    bins = bin_instance.find_bin(speech_durations=duration_frames)

    sentence_bped = []
    for sentence_seg in sentence_segments:
        sentence_bped.append(bpe_de.process_line(sentence_seg))
    if model_name == 'model8':
        temp = []
        for i in range(len(bins)):
            temp.append(' <X>')
        sentence_bped_str = " ".join(sentence_bped).rstrip() + " <||> " + " ".join(temp)
    elif "7" in model_name:
        sentence_bped_str = " ".join(sentence_bped).rstrip() + " <||> " + " ".join(bins)
    else:
        print("please provide a correct model name")
        exit()

    with open('sentence.de', 'w') as f:
        f.write('{}\n'.format(sentence_bped_str))
    with open('sentence.en', 'w') as f:
        f.write('{}\n'.format(sentence_bped_str))

    # Generate translation of the transcript of this video

    if model_name == 'model8':
        if not os.path.exists("dict.en.txt.model8"):
            os.system("fairseq-preprocess --source-lang de --target-lang en  --testpref sentence"
                      " --destdir ./data-bin/sentence_to_generate "
                      "--trainpref processed_datasets/de-text-dummy-durations-en-phones-durations/train --workers 20")
            os.system("cp ./data-bin/sentence_to_generate/dict.en.txt dict.en.txt.model8")
            os.system("cp ./data-bin/sentence_to_generate/dict.de.txt dict.de.txt.model8")
        else:
            os.system("fairseq-preprocess --source-lang de --target-lang en "
                      "--srcdict dict.de.txt.model8 --tgtdict dict.en.txt.model8 --testpref sentence"
                      " --destdir ./data-bin/sentence_to_generate --workers 20")

    elif model_name == 'model7':
        if not os.path.exists("dict.en.txt.model7"):
            os.system("fairseq-preprocess --source-lang de --target-lang en "
                      "--trainpref processed_datasets/de-text-clean-durations-en-phones-durations/train  --testpref sentence"
                      " --destdir ./data-bin/sentence_to_generate --workers 20")
            os.system("cp ./data-bin/sentence_to_generate/dict.en.txt dict.en.txt.model7")
            os.system("cp ./data-bin/sentence_to_generate/dict.de.txt dict.de.txt.model7")
        else:
            os.system("fairseq-preprocess --source-lang de --target-lang en "
                      "--srcdict dict.de.txt.model7 --tgtdict dict.en.txt.model7 --testpref sentence"
                      " --destdir ./data-bin/sentence_to_generate --workers 20")

    elif model_name.startswith('model7-sd'):
        sd_value = model_name.split("sd")[1]
        if not os.path.exists("dict.en.txt.model7-sd0.1"):
            os.system("fairseq-preprocess --source-lang de --target-lang en "
                      "--trainpref processed_datasets/de-text-noisy-durations{}-en-phones-durations/train "
                      " --testpref sentence --destdir ./data-bin/sentence_to_generate --workers 20".format(sd_value))
            os.system("cp ./data-bin/sentence_to_generate/dict.en.txt dict.en.txt.model7-sd{}".format(sd_value))
            os.system("cp ./data-bin/sentence_to_generate/dict.de.txt dict.de.txt.model7-sd{}".format(sd_value))
        else:
            os.system("fairseq-preprocess --source-lang de --target-lang en "
                      "--srcdict dict.de.txt.model7-sd{} --tgtdict dict.en.txt.model7-sd{} --testpref sentence"
                      " --destdir ./data-bin/sentence_to_generate --workers 20".format(sd_value, sd_value))
    else:
        print("please provide a correct model name")
        exit()

    os.system("CUDA_VISIBLE_DEVICES=0 fairseq-generate data-bin/sentence_to_generate --path {} --source-lang de "
              "--target-lang en --batch-size 256  --remove-bpe --beam 5 --gen-subset test > sentence_translated.txt".format(checkp_path))

    os.system("cat sentence_translated.txt | grep -P '^H' |sort -V |cut -f 3- | sed 's/\[en\]//g' > "
              "sentence_translated.txt.hyp")

    with open('sentence_translated.txt.hyp') as f:
        line_hyp = f.readlines()[0]
        line = " ".join(line_hyp.split("<eow>"))
        line_segments = line.split('[pause]')

    with open('test_phones_hyp_{}_{}.txt'.format(model_name, name), 'a') as f:
        f.write(line_hyp)

    # We generated the translation given the durations detected by VAD, now let's see if there is silence in the
    # beginning of the video

    if len(pauses) == 0:
        pauses = [0]
    if speech_timestamps[0]['start'] > 0.0:
        pauses_start = speech_timestamps[0]['start']
    else:
        pauses_start = 0.0

    num_pauses_hyp = len(line_segments) - 1
    basename_list = []

    # Add silence in the beginning (if VAD detected speech after 0.0s in the beginning of the video)
    audio = [AudioSegment.silent(duration=pauses_start * 1000)]
    silence = []
    durations_list = []

    # Init vocoder of FS2
    vocoder = get_vocoder(model_config, device)
    speakers = np.array([args.speaker_id])

    for j, line in enumerate(line_segments):
        if line != '\n' and line != "" and line != " ":
            texts, duration, phones = parse_generated_file(line)
            texts = np.array([texts])
            ids = [phones]
            text_lens = np.array([len(texts[0])])
            batches = [(ids, ids, speakers, texts, text_lens,  max(text_lens), np.array([duration]))]
            basename_basic = "{}.{}.using{}".format(name, index, model_name)
            basename = basename_basic + "_" + str(j)
            basename_list.append(basename)
            sum_dur = 0
            for dur in duration:
                sum_dur += dur
            durations_list.append(str(sum_dur))

            # Generate speech using FS2
            duration_rounded = synthesize(model, args.restore_step, configs, vocoder, batches, control_values, basename)

    with open("log_generate_{}_{}.txt".format(name, model_name), 'a') as f:
        f.write('Durations from model\n')
        f.write('{}\n'.format(" ".join(durations_list)))
        f.write("Pauses detected from VAD and pauses predicted from model: \n")
        f.write('{} {}\n'.format(len(pauses), num_pauses_hyp))

    with open("durations_hyp_{}.txt".format(model_name), 'a') as f:
        f.write('{}\n'.format(" ".join(durations_list)))
    with open("durations_vad_{}.txt".format(model_name), 'a') as f:
        f.write('{}\n'.format(" ".join(duration_frames)))

    # If needed, add pauses between audio segments
    more_pauses = True
    for i in range(len(line_segments)):
        audio.append(AudioSegment.from_file(os.path.join(train_config["path"]["result_path"], "{}_{}.wav".
                                                     format(basename_list[i], suffix)), format="wav"))
        if num_pauses_hyp > 0 and num_pauses_hyp == len(pauses) and more_pauses:
            pause_mseconds = pauses[i] * 1000
            audio.append(AudioSegment.silent(duration=pause_mseconds))
            if len(pauses) == i + 1:
                more_pauses = False

    if pauses[0] != 0 and num_pauses_hyp == 0:
        print("{}: The model did not predict pauses but VAD detected pauses, we will not add pauses\n".format(basename_basic))
    elif pauses[0] == 0 and num_pauses_hyp > 0:
        print("{}: The model predicted pauses but VAD did not detect pauses, we will not add pauses\n".format(basename_basic))
    else:
        print("{}: The model predicted pauses and VAD detected them, adding pauses!\n".format(basename_basic))

    # Concatenate all audio segments together
    audio_final = 0
    for i in range(len(audio)):
        audio_final += audio[i]

    file_handle = audio_final.export(os.path.join('./for_human_eval', "{}_{}_joined.wav".
                                                  format(basename_basic, suffix)), format="wav")

    os.system("rm -r data-bin/sentence_to_generate/")
