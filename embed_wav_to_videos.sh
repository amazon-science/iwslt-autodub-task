
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


modelname=model7
subset=test_set # subset 1, use "test_set_with_pauses.txt" for subset 2

mkdir -p for_human_eval/dubbed

# Here are the indices of the videos we have in German for subset #2:
# #for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101;

# The indices below are the ones for which we have videos available in German (subset #1)
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200;
do
  name="$subset".$i

  ffmpeg -i for_human_eval/original_videos/$name.mov -i for_human_eval/$name.using"$modelname"_MFAphones_MFAdurations_joined.wav -map 0:v:0 -map 1:a:0 -c:v copy  for_human_eval/dubbed/dubbed_$name.using"$modelname".mp4 -hide_banner -loglevel error
  ffmpeg -i for_human_eval/original_videos/$name.mp4 -i for_human_eval/$name.using"$modelname"_MFAphones_MFAdurations_joined.wav -map 0:v:0 -map 1:a:0 -c:v copy  for_human_eval/dubbed/dubbed_$name.using"$modelname".mp4 -hide_banner -loglevel error

done
