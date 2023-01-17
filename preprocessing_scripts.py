
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

import os
import numpy as np

import numpy
from matplotlib import pyplot as plt
from scipy import stats


class Bin:
    def __init__(self, durations_freq, n=200):
        list_durations = []
        for key in durations_freq.keys():
            for _ in range(durations_freq[key]):
                list_durations.append(key)
        durations = numpy.array(list_durations)
        bins = stats.mstats.mquantiles(durations, [i/n for i in range(0, n + 1)])
        self.bins = numpy.array(bins)

    def find_bin(self, speech_durations, plot=False):
        assigned_bins = []

        if plot:
            plt.ylabel("# times this duration is observed in our data")
            plt.xlabel("Durations")
            plt.hist(speech_durations, self.bins, edgecolor="k")
            plt.show()
        ind_bins = numpy.digitize(speech_durations, self.bins)
        for ind in ind_bins:
            assigned_bins.append('<bin{}>'.format(ind))
        return assigned_bins


def load_tsv(path):
    dict_audio = {}
    for i, split in enumerate(["train", "dev", "test"]):
        with open(os.path.join(path, "covost_v2.en_de.{}.tsv".format(split))) as f:
            lines = f.readlines()
            dict_audio[split] = {}
            for line in lines:
                fields = line.split("\t")
                name = fields[0].split(".")[0]
                # dict_audio[split][name] = fields[1].strip('\"')
                # fields[1] -> English, fields[2] -> German
                dict_audio[split][name] = [fields[1].strip('\"'), fields[2].strip('\"')]

    return dict_audio["train"], dict_audio["dev"], dict_audio["test"]


def get_speech_durations(tier, duration_freq=None, count_jsons_with_silences=0, return_durations=False, return_text=False):
    sampling_rate = 22050
    hop_length = 256
    sil_phones = ["sil", "sp", "spn", '']
    phones = []
    # print("We consider as silence everything that has silent phonemes for > {} frames".format(silence_duration))
    # 26 frames
    end_of_word_sec = []
    pause_durations = []
    text = []
    counter_dur = 0
    durations_list = []
    for i, k in enumerate(tier['tiers']['words']['entries']):
        s, e, p = k[0], k[1], k[2]
        end_of_word_sec.append(e)
        if return_text:
            text.append(p)

    for i, k in enumerate(tier['tiers']['phones']['entries']):
        s, e, p = k[0], k[1], k[2]
        # Trim leading silences
        if phones == []:
            if p in sil_phones:
                continue

        phone_duration = (int(np.round(e * sampling_rate / hop_length) - np.round(s * sampling_rate / hop_length)))

        if p in sil_phones:
            if e - s >= 0.3 and return_durations:
                if phones[-1] != '[pause]':
                    phones.append('[pause]')
                if counter_dur > 0:
                    durations_list.append(counter_dur)
                counter_dur = 0

                pause_durations.append(str(int(np.round(e * sampling_rate / hop_length) - np.round(s * sampling_rate / hop_length))))
            else:
                phones.append('sp')
                if return_durations:
                    phones.append(str(phone_duration))
                    counter_dur += phone_duration
        else:
            phones.append(p)
            if return_durations:
                phones.append(str(phone_duration))
                counter_dur += phone_duration
        if e in end_of_word_sec:
            phones.append('<eow>')
    if counter_dur != 0:
        durations_list.append(counter_dur)
    # trim trailing silences
    for i in range(5):
        if phones[-1] == '[pause]' and return_durations:
            pause_durations = pause_durations[:-1]
        if phones[-1] in ['[pause]', 'sp']:
            phones = phones[:-1]

    if len(durations_list) > 1 and return_durations:
        count_jsons_with_silences += 1

    if return_durations:
        for duration in durations_list:
            duration_freq[duration] += 1
        if not pause_durations:
            pause_durations = [str(0)]

    return phones, duration_freq, count_jsons_with_silences, durations_list, pause_durations, " ".join(text)


def add_noise_to_durations(durations, sd, upsampling):
    noise = np.random.normal(0, sd, upsampling * len(durations))
    noisy_durations = []

    k = 0
    for duration in durations:
        noisy_duration_temp = []
        for i in range(upsampling):
            noisy = duration + noise[k + i] * duration
            noisy_duration_temp.append(noisy)
            # print(noisy)
        noisy_durations.append(noisy_duration_temp)
        k += upsampling
    return noisy_durations
