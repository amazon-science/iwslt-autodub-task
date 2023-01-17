
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


import matplotlib.pyplot as plt
import numpy as np

bleu = np.array([30.6,  30.8, 31.7, 32.4,  32.8, 33.2])
overlap1 = np.array([0.92, 0.90, 0.87, 0.81, 0.73, 0.56])

sd = [" 0  ", "  0.02", "   0.05", "  0.1", "  0.2", "   1.5"]

plt.xlabel('Test BLEU score', fontsize=13)
plt.ylabel('Speech overlap metric 1', fontsize=13)

dummy_bleu = [33.3]
dummy_overlap = [0.52]
plt.legend(fontsize=25, bbox_to_anchor=(0, 1),loc="upper right") # using a size in points

plt.scatter(dummy_bleu, dummy_overlap, label="Text2phonemes", color="red", marker='o', linewidths=1)
plt.scatter(bleu, overlap1, label="Ours (w/ source dur.)", color="blue", marker='o', linewidths=1)

for i, txt in enumerate(sd):
    plt.annotate(txt, (bleu[i], overlap1[i]))

plt.plot(bleu, overlap1, ls="--")
plt.plot(dummy_bleu, dummy_overlap, ls="--")

plt.legend()
# plt.show()
plt.savefig('bleu_overlap.pdf',  bbox_inches='tight')
