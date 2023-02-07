import codecs
import json
import os
import pickle
import sys
import argparse
import logging
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s][%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
from preprocessing_scripts import load_tsv, add_noise_to_durations, get_speech_durations, Bin
from subword_nmt.apply_bpe import BPE


DE_OUTPUT_CHOICES_WITH_DURATIONS = {
    'de-text-clean-durations',
    'de-text-noisy-durations',
    'de-text-dummy-durations'
}
DE_OUTPUT_CHOICES = DE_OUTPUT_CHOICES_WITH_DURATIONS.add('de-text-without-durations')

EN_OUTPUT_CHOICES = {
    'en-text-without-durations',
    'en-phones-without-durations',
    'en-phones-durations'
}


def build_datasets(data_path,
                   duration_freq,
                   de_output_type,
                   en_output_type,
                   output_dir,
                   bpe_de,
                   bpe_en,
                   tsvs,
                   num_bins=100,
                   upsampling=None,
                   sd=None,
                   write_segments_to_file=False):
    if num_bins > 0:
        bin_instance = Bin(duration_freq, n=num_bins)
    counter = 0
    train_tsv, dev_tsv, test_tsv = tsvs
    train_de, dev_de, test_de = [], [], []
    train_en, dev_en, test_en = [], [], []
    train_segments, dev_segments, test_segments = [], [], []
    return_durations = False
    return_text = False

    all_included_keys = set().union(train_tsv.keys(), dev_tsv.keys(), test_tsv.keys())

    for file in os.listdir(data_path):
        # We want only JSON files
        name = file.split(".")[0]
        if os.path.isfile(os.path.join(data_path, name + ".json")):
            data = json.load(open(os.path.join(data_path, name + ".json")))
        else:
            logging.debug(f"{file} ignored")
            continue

        # Data that is not in the covost_tsv TSV files is not used
        if name not in all_included_keys:
            continue

        counter += 1

        if en_output_type == 'en-phones-durations':
            return_durations = True
        if en_output_type == 'en-text-without-durations':
            return_text = True

        phones, duration_freq, _, durations, _, text = get_speech_durations(data,
                                                                            duration_freq,
                                                                            return_durations=return_durations,
                                                                            return_text=return_text)
        pauses_count = phones.count('[pause]')

        if return_durations:
            assert len(durations) >= 1

        if de_output_type in DE_OUTPUT_CHOICES_WITH_DURATIONS:
            if num_bins > 0:
                bins = bin_instance.find_bin(speech_durations=durations)

            # noisy or dummy durations for De
            if de_output_type == 'de-text-noisy-durations':
                noisy_durations = add_noise_to_durations(durations, sd, upsampling)
                if num_bins > 0:
                    noisy_bins = [[] for _ in range(upsampling)]
                    for dur in noisy_durations:
                        noisy_bins_temp = bin_instance.find_bin(speech_durations=dur)
                        for i in range(upsampling):
                            noisy_bins[i].append(noisy_bins_temp[i])
                noisy_durations_rearrange_int = [[] for _ in range(upsampling)]
                for dur in noisy_durations:
                    for i in range(upsampling):
                        noisy_durations_rearrange_int[i].append(round(dur[i]))
            elif de_output_type == 'de-text-dummy-durations':
                temp = []
                for _ in range(len(bins)):
                    temp.append(' <X>')

        if en_output_type == 'en-phones-durations':
            if de_output_type in DE_OUTPUT_CHOICES_WITH_DURATIONS:
                assert pauses_count == len(durations) - 1

        if name in train_tsv.keys():
            # Source side (German)
            sentence_segments = []
            if de_output_type == 'de-text-clean-durations':
                if num_bins > 0:
                    sentence = [bpe_de.process_line(train_tsv[name][1]) + " <||> " + " ".join(bins)]
                else:
                    sentence = [bpe_de.process_line(train_tsv[name][1]) + " <||> " + " ".join(map(str, durations))]
                if return_durations and write_segments_to_file:
                    sentence_segments = [" ".join(map(str, durations))]
            elif de_output_type == 'de-text-noisy-durations':
                sentence = []
                for i in range(upsampling):
                    if num_bins > 0:
                        sentence.append(bpe_de.process_line(train_tsv[name][1]) + " <||> " + " ".join(noisy_bins[i]))
                    else:
                        sentence.append(bpe_de.process_line(train_tsv[name][1]) + " <||> " + " ".join(map(str, noisy_durations_rearrange_int[i])))
                    if return_durations and write_segments_to_file:
                        sentence_segments.append(" ".join(map(str, noisy_durations_rearrange_int[i])))
            elif de_output_type == 'de-text-dummy-durations':
                sentence = [bpe_de.process_line(train_tsv[name][1]) + " <||> " + " ".join(temp)]
                if return_durations and write_segments_to_file:
                    sentence_segments = [" ".join(map(str, durations))]
            elif de_output_type == 'de-text-without-durations':
                sentence = [bpe_de.process_line(train_tsv[name][1])]
                if return_durations and write_segments_to_file:
                    sentence_segments = [" ".join(map(str, durations))]

            train_de.extend(sentence)
            train_segments.extend(sentence_segments)

            # Target side (English)
            if en_output_type == 'en-text-without-durations':
                train_en.append(bpe_en.process_line(text))
            elif en_output_type.startswith('en-phones'):
                if de_output_type != 'de-text-noisy-durations':
                    train_en.append(" ".join(phones))
                else:
                    for _ in range(upsampling):
                        train_en.append(" ".join(phones))

        elif name in dev_tsv.keys() or name in test_tsv.keys():
            if name in dev_tsv.keys():
                curr_tsv = dev_tsv
                curr_de = dev_de
                curr_en = dev_en
                curr_segments = dev_segments
            else:
                curr_tsv = test_tsv
                curr_de = test_de
                curr_en = test_en
                curr_segments = test_segments

            # Source side (German)
            if de_output_type == 'de-text-noisy-durations' or de_output_type == 'de-text-clean-durations':
                if num_bins > 0:
                    sentence = bpe_de.process_line(curr_tsv[name][1]) + " <||> " + " ".join(bins)
                else:
                    sentence = bpe_de.process_line(curr_tsv[name][1]) + " <||> " + " ".join(map(str, durations))
            elif de_output_type == 'de-text-dummy-durations':
                sentence = bpe_de.process_line(curr_tsv[name][1]) + " <||> " + " ".join(temp)
            elif de_output_type == 'de-text-without-durations':
                sentence = bpe_de.process_line(curr_tsv[name][1])
            if return_durations and write_segments_to_file:
                curr_segments.append(" ".join(map(str, durations)))

            curr_de.append(sentence)

            # Target side (English)
            if en_output_type == 'en-text-without-durations':
                curr_en.append(bpe_en.process_line(text))
            elif en_output_type.startswith('en-phones'):
                curr_en.append(" ".join(phones))
            
        if counter % 20000 == 0:
            logging.info(f"{counter} files processed")

    write_to_file(train_de, os.path.join(output_dir, 'train.de'))
    write_to_file(dev_de, os.path.join(output_dir, 'valid.de'))
    write_to_file(test_de, os.path.join(output_dir, 'test.de'))
    write_to_file(train_en, os.path.join(output_dir, 'train.en'))
    write_to_file(dev_en, os.path.join(output_dir, 'valid.en'))
    write_to_file(test_en, os.path.join(output_dir, 'test.en'))
    if train_segments != [] and write_segments_to_file:
        write_to_file(train_segments, os.path.join(output_dir, 'train.segments'))
        write_to_file(dev_segments, os.path.join(output_dir, 'valid.segments'))
        write_to_file(test_segments, os.path.join(output_dir, 'test.segments'))

    logging.info("Wrote new dataset to {}".format(output_dir))

def write_to_file(data, path):
    with open(path, 'w') as f:
        for line in data:
            f.write('{}\n'.format(line))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Required arguments
    parser.add_argument("--de-output-type", "--de", required=True,
                        choices=DE_OUTPUT_CHOICES)
    parser.add_argument("--en-output-type", "--en", required=True,
                        choices=EN_OUTPUT_CHOICES)

    # Paths
    parser.add_argument("-i", "--input-mfa-dir", default='covost_mfa/data',
                        help="Directory containing MFA JSON files")
    parser.add_argument("-o", "--processed-output-dir", default='processed_datasets',
                        help="Parent directory for output data")
    parser.add_argument("--covost-dir", default='./covost_tsv',
                        help="Directory containing covost TSV files")
    parser.add_argument("--durations-path", default='durations_freq_all.pkl',
                        help="Pickle file containing dictionary of durations"
                        " and corresponding frequencies")
    parser.add_argument("--bpe-de", default='data/training/de_codes_10k',
                        help="BPE codes for de side")
    parser.add_argument("--bpe-en", default='data/training/en_codes_10k_mfa',
                        help="BPE codes for en side")
    parser.add_argument("--force-redo", "-f", action='store_true',
                        help="Redo datasets even if the output directory already exists")
    parser.add_argument("--write-segments-to-file", action='store_true',
                        help="Write unnoise and unbinned segment durations to a separate file")

    # Other arguments
    parser.add_argument("--upsampling", type=int, default=1,
                        help="Upsample examples by this factor (for noisy outputs)")
    parser.add_argument("--noise-std", type=float, default=0.0,
                        help="Standard deviation for noise added to durations")
    parser.add_argument("--num-bins", type=int, default=100,
                        help="Number of bins. 0 means no binning.")

    args = parser.parse_args()

    # Read data
    train_tsv, dev_tsv, test_tsv = load_tsv(args.covost_dir)
    codes_de = codecs.open(args.bpe_de, encoding='utf-8')
    bpe_de = BPE(codes_de)
    codes_en = codecs.open(args.bpe_en, encoding='utf-8')
    bpe_en = BPE(codes_en)

    assert os.path.exists(args.durations_path), \
        "Run get_durations_frequencies.py first to get the dictionary of durations" \
        " and how many times each is observed in our data!"
    with open(args.durations_path, 'rb') as f:
        logging.info("Loading durations' frequencies")
        durations_pkl = pickle.load(f)
    if not os.path.exists(args.processed_output_dir):
        os.makedirs(args.processed_output_dir)

    output_path = os.path.join(args.processed_output_dir, args.de_output_type)
    if args.num_bins == 0:
        logging.warning("Binning of source segment durations is turned off. "
                        "This is not expected for any of the default models. "
                        "Run with --num-bins > 0 if this was not intentional.")
        output_path += '-unbinned'
    if args.de_output_type == 'de-text-noisy-durations':
        if args.noise_std == 0.0:
            logging.error(f"You probably want non-zero noise with {args.de_output_type}")
            sys.exit(1)
        output_path += str(args.noise_std)
        logging.info(f"Will add noise to speech durations in De and upsample by {args.upsampling}.")
    output_path += '-' + args.en_output_type
    logging.info(f"Setting output directory to {output_path}")

    if not args.force_redo and os.path.exists(output_path):
        logging.error(f"Path {output_path} already exists. Run with --force-redo/-f to force overwrite.")
        sys.exit(1)
    else:
        os.makedirs(output_path, exist_ok=True)

    logging.info("Building datasets")
    build_datasets(data_path=args.input_mfa_dir,
                   duration_freq=durations_pkl,
                   de_output_type=args.de_output_type,
                   en_output_type=args.en_output_type,
                   output_dir=output_path,
                   bpe_de=bpe_de,
                   bpe_en=bpe_en,
                   tsvs=[train_tsv, dev_tsv, test_tsv],
                   num_bins=args.num_bins,
                   upsampling=args.upsampling,
                   sd=args.noise_std,
                   write_segments_to_file=args.write_segments_to_file)
