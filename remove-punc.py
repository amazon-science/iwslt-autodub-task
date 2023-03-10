
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


import re
import os
import sys
file = sys.argv[1]

path = "./"
lines_nopunc = []

with open(os.path.join(path, file)) as f:
    lines = f.readlines()
    for line in lines:
        line_new = re.sub(r"[\.,\?:;!\"\(\)]", "", line)
        lines_nopunc.append(line_new)

with open(os.path.join(path, file + '.no-punc'), 'w') as the_file:
    for line in lines_nopunc:
        the_file.write('{}'.format(line))
