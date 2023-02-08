
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


# python3 get_durations_frequencies.py ./covost_mfa/data (wherever the covost .json files are saved)

import json
import os
import pickle
import sys

from collections import defaultdict
from build_datasets import get_speech_durations
from preprocessing_scripts import load_tsv

counter = 0
count_jsons_with_silences = 0
data_path = sys.argv[1]
durations_path = './durations_freq_all.pkl'
covost_dir = './covost_tsv'
train_tsv, dev_tsv, test_tsv = load_tsv(covost_dir)

if not os.path.exists(durations_path):
    duration_freq = defaultdict(int)

    for file in os.listdir(data_path):
        name = file.split(".")[0]
        if os.path.isfile(os.path.join(data_path, name + ".json")):
            data = json.load(open(os.path.join(data_path, name + ".json")))
        else:
            #print(file, "ignored")
            continue
        if name in train_tsv.keys():
            phones, duration_freq, count_jsons_with_silences, durations, pause_durations, _ = get_speech_durations(data, duration_freq,
                                                                                               count_jsons_with_silences, return_durations=True)

            assert len(durations) >= 1
            counter += 1
            if counter % 50000 == 0:
                print('done:', counter)

    with open(durations_path, 'wb') as f:
        pickle.dump(duration_freq, f)
        print("Wrote durations to {}".format(durations_path))
else:
    print("The dictionary of speech durations has already been computed and stored in {}".format(durations_path))
