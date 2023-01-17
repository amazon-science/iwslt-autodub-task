
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


"""
To create the dataset for model 1b, please use:
python3 build_datasets.py ./covost_mfa/data processed_datasets en-text-without-durations de-text-without-durations

For model 2b:
python3 build_datasets.py ./covost_mfa/data processed_datasets en-phones-without-durations de-text-without-durations

For model 4a:
python3 build_datasets.py ./covost_mfa/data processed_datasets en-phones-durations de-text-without-durations

For model 7 without noise:
python3 build_datasets.py ./covost_mfa/data processed_datasets en-phones-durations de-text-clean-durations

For model 7 with noise:
python3 build_datasets.py ./covost_mfa/data processed_datasets en-phones-durations de-text-noisy-durations 0.5 (model 7f)

For model 8 (dummy noise symbols added,
 <X>)
python3 build_datasets.py ./covost_mfa/data processed_datasets en-phones-durations de-text-dummy-durations (model 8)
"""

import codecs
import json
import os
import pickle
import sys
from preprocessing_scripts import load_tsv, add_noise_to_durations, get_speech_durations, Bin
from subword_nmt.apply_bpe import BPE


def build_datasets(data_path, duration_freq, num_bins=100, upsampling=None, sd=None, args=None):
    bin_instance = Bin(duration_freq, n=num_bins)
    count_jsons_with_silences = 0
    counter = 0
    train_de, dev_de, test_de = [], [], []
    train_en, dev_en, test_en = [], [], []
    return_durations = False
    return_text = False

    for file in os.listdir(data_path):
        name = file.split(".")[0]
        if os.path.isfile(os.path.join(data_path, name + ".json")):
            data = json.load(open(os.path.join(data_path, name + ".json")))
        else:
            print(file)
            print("ignored")
            continue

        if args[3] == 'en-phones-durations':
            return_durations = True
        if args[3] == 'en-text-without-durations':
            return_text = True

        if name in train_tsv.keys() or name in dev_tsv.keys() or name in test_tsv.keys():
            counter += 1
            if counter % 100000 == 0:
                print(counter)

            phones, duration_freq, count_jsons_with_silences, durations, pause_durations, text = get_speech_durations(data,
                                                                                                                      duration_freq,
                                                                                                                      count_jsons_with_silences,
                                                                                                                      return_durations=return_durations,
                                                                                                                      return_text=return_text)
            pauses_count = 0
            for i in phones:
                if i == '[pause]':
                    pauses_count += 1

            if return_durations:
                assert len(durations) >= 1

            if args[4] in ['de-text-clean-durations', 'de-text-noisy-durations', 'de-text-dummy-durations']:
                bins = bin_instance.find_bin(speech_durations=durations)

                # noisy or dummy durations for De
                if args[4] == 'de-text-noisy-durations':
                    noisy_durations = add_noise_to_durations(durations, sd, upsampling)
                    noisy_bins = [[] for x in range(upsampling)]
                    for dur in range(len(noisy_durations)):
                        noisy_bins_temp = bin_instance.find_bin(speech_durations=noisy_durations[dur])
                        for i in range(upsampling):
                            noisy_bins[i].append(noisy_bins_temp[i])
                elif args[4] == 'de-text-dummy-durations':
                    temp = []
                    for i in range(len(bins)):
                        temp.append(' <X>')

            if name in train_tsv.keys():
                if args[3] == 'en-phones-durations':
                    if args[4] in ['de-text-clean-durations', 'de-text-noisy-durations', 'de-text-dummy-durations']:
                        assert pauses_count == len(durations) - 1

                # Source side (German)
                if args[4] == 'de-text-clean-durations':
                    sentence = bpe_de.process_line(train_tsv[name][1]) + " <||> " + " ".join(bins)
                elif args[4] == 'de-text-noisy-durations':
                    sentence = []
                    for i in range(upsampling):
                        sentence.append(bpe_de.process_line(train_tsv[name][1]) + " <||> " + " ".join(noisy_bins[i]))
                elif args[4] == 'de-text-dummy-durations':
                    sentence = bpe_de.process_line(train_tsv[name][1]) + " <||> " + " ".join(temp)
                elif args[4] == 'de-text-without-durations':
                    sentence = bpe_de.process_line(train_tsv[name][1])
                else:
                    print("Error, check argument 4! (De)")
                    exit()

                if isinstance(sentence, list):
                    train_de.extend(sentence)
                else:
                    train_de.append(sentence)

                # Target side (English)
                if args[3] == 'en-text-without-durations':
                    train_en.append(bpe_en.process_line(text))
                elif args[3].startswith('en-phones'):
                    if args[4] != 'de-text-noisy-durations':
                        train_en.append(" ".join(phones))
                    else:
                        for i in range(upsampling):
                            train_en.append(" ".join(phones))
                else:
                    print("Error, check argument 3! (En)")
                    exit()

            elif name in dev_tsv.keys():
                if args[3] == 'en-phones-durations':
                    if args[4] in ['de-text-clean-durations', 'de-text-noisy-durations', 'de-text-dummy-durations']:
                        assert pauses_count == len(durations) - 1

                # Source side (German)
                if args[4] == 'de-text-noisy-durations' or args[4] == 'de-text-clean-durations':
                    sentence = bpe_de.process_line(dev_tsv[name][1]) + " <||> " + " ".join(bins)
                elif args[4] == 'de-text-dummy-durations':
                    sentence = bpe_de.process_line(dev_tsv[name][1]) + " <||> " + " ".join(temp)
                elif args[4] == 'de-text-without-durations':
                    sentence = bpe_de.process_line(dev_tsv[name][1])

                dev_de.append(sentence)

                # Target side (English)
                if args[3] == 'en-text-without-durations':
                    dev_en.append(bpe_en.process_line(text))
                elif args[3].startswith('en-phones'):
                    dev_en.append(" ".join(phones))

            elif name in test_tsv.keys():
                if args[3] == 'en-phones-durations':
                    if args[4] in ['de-text-clean-durations', 'de-text-noisy-durations', 'de-text-dummy-durations']:
                        assert pauses_count == len(durations) - 1

                # Source side (German)
                if args[4] == 'de-text-noisy-durations' or args[4] == 'de-text-clean-durations':
                    sentence = bpe_de.process_line(test_tsv[name][1]) + " <||> " + " ".join(bins)
                elif args[4] == 'de-text-dummy-durations':
                    sentence = bpe_de.process_line(test_tsv[name][1]) + " <||> " + " ".join(temp)
                elif args[4] == 'de-text-without-durations':
                    sentence = bpe_de.process_line(test_tsv[name][1])

                test_de.append(sentence)

                # Target side (English)
                if args[3] == 'en-text-without-durations':
                    test_en.append(bpe_en.process_line(text))
                elif args[3].startswith('en-phones'):
                    test_en.append(" ".join(phones))

    if args[4] == 'de-text-noisy-durations':
        new_path = os.path.join(dir_name, args[4] + str(sd) + '-' + args[3])
    else:
        new_path = os.path.join(dir_name, args[4] + '-' + args[3])

    if not os.path.exists(new_path):
        os.makedirs(new_path)
        with open(os.path.join(new_path, 'train.de'), 'w') as f:
            for line in train_de:
                f.write('{}\n'.format(line))
        with open(os.path.join(new_path, 'valid.de'), 'w') as f:
            for line in dev_de:
                f.write('{}\n'.format(line))
        with open(os.path.join(new_path, 'test.de'), 'w') as f:
            for line in test_de:
                f.write('{}\n'.format(line))

        with open(os.path.join(new_path, 'train.en'), 'w') as f:
            for line in train_en:
                f.write('{}\n'.format(line))
        with open(os.path.join(new_path, 'valid.en'), 'w') as f:
            for line in dev_en:
                f.write('{}\n'.format(line))
        with open(os.path.join(new_path, 'test.en'), 'w') as f:
            for line in test_en:
                f.write('{}\n'.format(line))
        print("Wrote new dataset to {}!".format(new_path))

    else:
        print("Path {} already exists.".format(new_path))


if __name__ == "__main__":

    covost_dir = './covost_tsv'
    train_tsv, dev_tsv, test_tsv = load_tsv(covost_dir)
    codes_de = codecs.open("./de_codes_10k", encoding='utf-8')
    bpe_de = BPE(codes_de)
    codes_en = codecs.open("./en_codes_10k_mfa", encoding='utf-8')
    bpe_en = BPE(codes_en)

    durations_path = './durations_freq_all.pkl'
    if os.path.exists(durations_path):
        with open(durations_path, 'rb') as f:
            durations_pkl = pickle.load(f)
            print("loaded durations' freq!")
    else:
        print("Run get_durations_frequencies.py first to get the dictionary of durations"
              " and how many times each is observed in our data!")
        exit()

    upsampling_value, sd_value = None, None
    dir_name = sys.argv[2]

    if sys.argv[4] == 'de-text-noisy-durations':
        sd_value = float(sys.argv[5])
        upsampling_value = 10
        print("Will add noise to speech durations in De and upsample by {}.".format(upsampling_value))

    if not os.path.exists(os.path.join('./', dir_name)):
        os.makedirs(dir_name)

    build_datasets(data_path=sys.argv[1], duration_freq=durations_pkl, num_bins=100, upsampling=upsampling_value,
                   sd=sd_value, args=sys.argv)
