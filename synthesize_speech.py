import os
import argparse
import pickle
import json
import re
from subprocess import Popen, PIPE, DEVNULL, check_call
import logging
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s][%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

from tqdm import tqdm
import numpy as np
import torch
from subword_nmt.apply_bpe import BPE
from pydub import AudioSegment

from preprocessing_scripts import Bin


SEGMENT_DURATION_SEPARATOR = ' <||> '
FACTOR_DELIMITER = '|'
EOW = '<eow>'
PAUSE = '[pause]'
SHIFT= '<shift>'
SAMPLING_RATE = 22050
HOP_LENGTH = 256


def get_sorted_audio_files(data_dir):
    """
    Get all the wav files in the directory named `*.Y.wav` and return them sorted numerically by `Y`
    """
    files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
    files = sorted(files, key=lambda f: int(f.split('.')[-2]))
    return [os.path.join(args.data_dir, "subset" + args.subset, f) for f in files]


class SileroVad:
    """
    Wrapper around Silero voice activity detection
    """
    def __init__(self):
        self.sampling_rate = 16000
        self.model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                           model='silero_vad',
                                           force_reload=False,
                                           onnx=False)
        (self.get_speech_timestamps, _, self.read_audio, _, _) = utils

    def get_timestamps(self, wav_file):
        """
        Get list of start and end timestamps of speech segments and lengths of pauses
        """
        wav = self.read_audio(wav_file, sampling_rate=self.sampling_rate)
        speech_timestamps = self.get_speech_timestamps(wav, self.model, sampling_rate=self.sampling_rate,
                                                       min_silence_duration_ms=300, visualize_probs=False,
                                                       threshold=0.3, return_seconds=True)

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


class SockeyeTranslator:
    """
    Wrapper around sockeye-translate command line to translate lines one at a time without reloading model
    """
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Specified Sockeye model checkpoint {model_path} does not exist")
        sockeye_command = ['python', '-u', '-m', 'sockeye.translate',
                           '--models', os.path.dirname(model_path),
                           '--checkpoints', os.path.basename(model_path).split('.')[-1],
                           '-b', '5',
                           '--batch-size', '1',
                           '--output-type', 'translation_with_factors',
                           '--max-output-length', '768',
                           '--force-factors-stepwise', 'frames', 'total_remaining', 'segment_remaining', 'pauses_remaining',
                           '--json-input'
                          ]

        logging.info(f"Running Sockeye command: {' '.join(sockeye_command)}")
        self.sockeye_process = Popen(sockeye_command, stdin=PIPE, stdout=PIPE, stderr=DEVNULL, env=os.environ,
                                     text=True, encoding='utf-8', universal_newlines=True, bufsize=1)

    def translate_line(self, line, segments):
        """
        Send one line to sockeye-translate and get back the translation
        """
        json_line = self.make_json_input(line, segments)
        logging.debug(f"Sending input to sockeye-translate: {json_line}")
        self.sockeye_process.stdin.write(json_line + '\n')
        self.sockeye_process.stdin.flush()
        return self.sockeye_process.stdout.readline()

    def make_json_input(self, line, segment_durations):
        """
        Create the JSON-formatted input for target factor prefixes etc.
        """
        input_dict = {
            'text': line,
            'target_prefix': SHIFT,
            'target_prefix_factors': ['0',
                                      str(sum(segment_durations)),
                                      str(segment_durations[0]),
                                      str(len(segment_durations) - 1)
                                     ],
            'target_segment_durations': segment_durations,
            'use_target_prefix_all_chunks': 'false'
        }
        return json.dumps(input_dict, ensure_ascii=False).replace('"false"', 'false')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--data-dir", type=str,
                        default=os.path.join(os.path.expanduser('~'), "iwslt-autodub-task", "data", "test"),
                        help="Directory containing the audio files. Inside this directory, the files should be in subsetX/*.Y.wav, "
                             "where sorting numerically by the Y field will give us the files in the same order as the transcript file. "
                             "This is already true for the test set subsets.")
    parser.add_argument("--source-text", type=str,
                        help="File containing the source German text. Defaults to using subsetX.de for the test set subsets.")
    parser.add_argument("--subset", choices=['1', '2'], type=str, required=True,
                        help="Which test set subset to generate dubs for.")
    parser.add_argument("--sockeye-model", type=str,
                        default=os.path.join(os.path.expanduser('~'), "iwslt-autodub-task", "models", "sockeye", "trained_baselines", "baseline_factored_noised0.1", "model", "params.00078"),
                        help="Path to a Sockeye model checkpoint.")
    parser.add_argument("--fastspeech-dir", type=str,
                        default=os.path.join(os.path.expanduser('~'), "iwslt-autodub-task", "third_party", "FastSpeech2"),
                        help="Path to the FastSpeech2 directory.")
    parser.add_argument("--bpe-de", type=str,
                        default=os.path.join(os.path.expanduser('~'), "iwslt-autodub-task", "data", "training", "de_codes_10k"),
                        help="BPE codes for German source text.")
    parser.add_argument("--durations-freq", type=str,
                        default=os.path.join(os.path.expanduser('~'), "iwslt-autodub-task", "durations_freq_all.pkl"),
                        help="Path to durations_freq_all.pkl")
    parser.add_argument("--output-video-dir", type=str,
                        help="Directory to write final dubbed videos.")
    parser.add_argument("--join-mode", type=str, choices=['match_pause', 'match_start'], default='match_start',
                        help="When joining segments together to create final clip:\n"
                             "match_pause: Pause lengths match source.\n"
                             "match_start: Try to match segment start times. May not match exactly if segments are too long.\n")

    args = parser.parse_args()

    # Default source text is `subsetX.de`
    if args.source_text is None:
        args.source_text = os.path.join(args.data_dir, "subset" + args.subset + '.de')

    # Do not change: These directories are fixed for FastSpeech2 trained on LJSpeech data
    output_dir = os.path.join(args.fastspeech_dir, 'output', 'result', 'LJSpeech')
    durations_dir = os.path.join(args.fastspeech_dir, 'preprocessed_data', 'LJSpeech', 'duration')

    # Default directory is a subdirectory of the input audio directory called `dubbed`
    if args.output_video_dir is None:
        args.output_video_dir = os.path.join(args.data_dir, "subset" + args.subset, 'dubbed')
    os.makedirs(args.output_video_dir, exist_ok=True)

    # Get audio files and lines of text - aligned with each other
    audio_files = get_sorted_audio_files(os.path.join(args.data_dir, "subset" + args.subset))
    with open(args.source_text) as f_src:
        src_text = f_src.readlines()
    assert len(audio_files) == len(src_text), "Number of audio files and number of lines in source text did not match."

    # Create BPE processor
    bpe_de = BPE(open(args.bpe_de))

    # Load duration frequencies for binning
    with open(args.durations_freq, 'rb') as f:
        durations_freq = pickle.load(f)
    bin_instance = Bin(durations_freq, n=100)

    silero_vad = SileroVad()

    sockeye_translator = SockeyeTranslator(args.sockeye_model)

    speech_timestamps = []
    pauses = []
    hyp_segments = []
    logging.info(f"Generating translated phoneme and duration outputs")
    with open(os.path.join(output_dir, 'subset' + args.subset + '.en.output'), 'w') as f_out, \
         open(os.path.join(output_dir, 'subset' + args.subset + '.en.fs2_inp'), 'w') as f_fs2_inp:
        for idx, audio_file in tqdm(enumerate(audio_files)):
            duration_frames = []
            vad = silero_vad.get_timestamps(audio_file)
            speech_timestamps.append(vad[0])
            pauses.append(vad[1])
            for timestamp in speech_timestamps[idx]:
                duration_frames.append(int(np.round(timestamp["end"] * SAMPLING_RATE / HOP_LENGTH) - np.round(timestamp["start"] * SAMPLING_RATE / HOP_LENGTH)))

            # BPE each segment and append segment durations bins
            bins = bin_instance.find_bin(speech_durations=duration_frames)
            sentence_segments = src_text[idx].split('[pause]')
            sentence_bpe = [bpe_de.process_line(sentence_seg.strip()) for sentence_seg in sentence_segments]
            sentence_bped_str = " ".join(sentence_bpe) + SEGMENT_DURATION_SEPARATOR + " ".join(bins)

            # Get translation from Sockeye
            hyp = sockeye_translator.translate_line(sentence_bped_str, duration_frames)
            f_out.write(hyp)
            # Remove `<eow>` and `<shift>` tokens
            hyp = " ".join([t for t in hyp.split() if t.split(FACTOR_DELIMITER)[0] not in [EOW, SHIFT]])
            # Split upon `[pause]`
            hyp_segments.append(re.split(r"\s*" + re.escape(PAUSE) + r"\|[^\s]+\s*", hyp))

            # Process each segment separately. Will later be joined with pauses again
            for seg_idx, hyp_segment in enumerate(hyp_segments[idx]):
                seg_fs2_id = f"subset{args.subset}-{idx+1}-{seg_idx+1}"
                # Write input in FastSpeech2 format
                f_fs2_inp.write(seg_fs2_id + '|LJSpeech|{')
                f_fs2_inp.write(' '.join([t.split(FACTOR_DELIMITER)[0] for t in hyp_segment.split()]))
                f_fs2_inp.write('}|\n')
                # Save durations to file for FastSpeech2 to read
                np.save(os.path.join(durations_dir, "LJSpeech-duration-" + seg_fs2_id + '.npy'),
                        np.array([int(t.split(FACTOR_DELIMITER)[1]) for t in hyp_segment.split()]))

    # FastSpeech2 doesn't work unless you're in the right directory due to relative paths in their configs.
    os.chdir(args.fastspeech_dir)
    logging.info("Running FastSpeech2 on phoneme and duration outputs")
    check_call(f"`dirname ${{CONDA_PREFIX}}`/fastspeech2/bin/python {os.path.join(args.fastspeech_dir, 'synthesize.py')} --mode batch "
               f"--source {os.path.join(output_dir, 'subset' + args.subset + '.en.fs2_inp')} --restore_step 900000 "
               f"-p {os.path.join(args.fastspeech_dir, 'config/LJSpeech/preprocess.yaml')} "
               f"-m {os.path.join(args.fastspeech_dir, 'config/LJSpeech/model.yaml')} "
               f"-t {os.path.join(args.fastspeech_dir, 'config/LJSpeech/train.yaml')} >/dev/null",
               shell=True)

    logging.info("Reconstructing final audio segments")
    # Re-construct audio from the pieces and add pauses
    for idx, audio_file in tqdm(enumerate(audio_files)):
        # Counting pauses for re-insertion
        num_pauses_hyp = len(hyp_segments[idx]) - 1

        # Add silence in the beginning (if VAD detected speech after 0.0s in the beginning of the video)
        if speech_timestamps[idx][0]['start'] > 0.0:
            pauses_start = speech_timestamps[idx][0]['start']
        else:
            pauses_start = 0.0
        audio = [AudioSegment.silent(duration=pauses_start * 1000)]

        for seg_idx, hyp_segment in enumerate(hyp_segments[idx]):
            # Join audio segments, adding pauses if needed
            seg_fs2_id = f"subset{args.subset}-{idx+1}-{seg_idx+1}"
            audio.append(AudioSegment.from_file(os.path.join(output_dir, seg_fs2_id + '.wav'), format="wav"))
            if seg_idx < num_pauses_hyp and seg_idx < len(pauses[idx]):
                pause_mseconds = pauses[idx][seg_idx] * 1000
                if args.join_mode == 'match_start':
                    # Adjust the pause by the difference between original and generated audio (without going below zero)
                    orig_seg_mseconds = (speech_timestamps[idx][seg_idx]['end'] - speech_timestamps[idx][seg_idx]['start']) * 1000
                    pause_mseconds -= len(audio[-1]) - orig_seg_mseconds
                    pause_mseconds = max(0, pause_mseconds)
                audio.append(AudioSegment.silent(duration=pause_mseconds))

        # Concatenate all audio segments together
        audio_final = sum(audio)
        audio_path = os.path.join(args.output_video_dir, os.path.basename(audio_file).replace('.wav', '.en.wav'))
        audio_final.export(audio_path, format="wav")

        # Embed wav onto video
        video_path = audio_path.replace('.wav', '.mp4')
        if os.path.exists(audio_file.replace('.wav', '.mp4')):
            check_call(f"ffmpeg -i {audio_file.replace('.wav', '.mp4')} -i {audio_path} -map 0:v:0 -map 1:a:0 -c:v copy {video_path} -hide_banner -loglevel error -y", shell=True)
        elif os.path.exists(audio_file.replace('.wav', '.mov')):
            check_call(f"ffmpeg -i {audio_file.replace('.wav', '.mov')} -i {audio_path} -map 0:v:0 -map 1:a:0 -c:v copy {video_path} -hide_banner -loglevel error -y", shell=True)
        else:
            logging.error(f"Could not find video at {audio_file.replace('.wav', '.{mp4,mov}')}")

    logging.info("Cleaning up intermediate files")
    # Remove intermediate files
    check_call(f"rm -f {output_dir}/*.wav", shell=True)
    check_call(f"rm -f {output_dir}/*.png", shell=True)
    check_call(f"rm -f {args.output_video_dir}/*.wav", shell=True)
    check_call(f"rm -f {durations_dir}/*", shell=True)

    logging.info(f"Dub generation complete. Output videos can be found in {args.output_video_dir}")
