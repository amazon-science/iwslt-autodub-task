
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
import sys

file = sys.argv[1]
path = "./"

with open(os.path.join(path, file)) as f:
    lines = f.readlines()
    sentences = []
    for i, line in enumerate(lines):
        phonetic_words = []
        phones = line.split('<eow>')
        for phone in phones:
            if phone != '\n':
                # if we don't have durations comment out next line
                if sys.argv[2] == 'durations':
                    phone = phone.split()[::2]
                else:
                    phone = phone.split()
                cleaned_phones = []
                for item in phone:
                    if item != 'sp':
                        cleaned_phones.append(item)
                phonetic_word = "-".join(cleaned_phones)
                phonetic_words.append(phonetic_word)

        sentences.append(" ".join(phonetic_words))


with open(os.path.join(path, file + '.phoneticwords'), 'w') as f:
    for sentence in sentences:
        f.write('{}\n'.format(sentence))



