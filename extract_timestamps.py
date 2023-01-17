
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


# @title Install and Import Dependencies
from silero_vad.utils_vad import get_speech_timestamps, read_audio

# this assumes that you have a relevant version of PyTorch installed

import torch


class SileroVad:
    def __init__(self):
        sampling_rate = 16000
        USE_ONNX = False  # change this to True if you want to test onnx model
        self.model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad',
                                           force_reload=False, onnx=USE_ONNX)
        self.sampling_rate = sampling_rate

    def get_timestamps(self, wav_file, visualize_probs=False):
        wav = read_audio(wav_file, sampling_rate=self.sampling_rate)
        speech_timestamps = get_speech_timestamps(wav, self.model, sampling_rate=self.sampling_rate, min_silence_duration_ms=300,
                                                  visualize_probs=visualize_probs, threshold=0.3, return_seconds=True)

        pauses = []
        if len(speech_timestamps) > 1:
            for i, pair in enumerate(speech_timestamps):
                if i == 0:
                    previous_start, previous_end = pair["start"], pair["end"]
                else:
                    current_start, current_end = pair["start"], pair["end"]
                    pause = current_start - previous_end
                    pauses.append(round(pause, 3))
                    previous_start, previous_end = pair["start"], pair["end"]

        return speech_timestamps, pauses

