
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
import pickle

from matplotlib import pyplot as plt

durations_path = './durations_freq_all.pkl'
if os.path.exists(durations_path):
    with open(durations_path, 'rb') as f:
        durations_pkl = pickle.load(f)
        print("loaded durations' freq!")


plt.ylabel("# times this duration observed in train set")
plt.xlabel("Durations")
plt.bar(durations_pkl.keys(), durations_pkl.values())
plt.show()
