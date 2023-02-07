import argparse

EOW_TOKEN = '<eow>'
DUMMY_DURATION = 0

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("output_file")
parser.add_argument("--shift", action='store_true',
                    help="Remove the first shifted tokens (from multi-factored model outputs)")
parser.add_argument("--zero-non-numeric", "-z", action='store_true',
                    help="Replace any non-numeric durations with zeros")
parser.add_argument("--reinsert-eow", action='store_true',
                    help="For models with EOW factors, reinsert the <eow> tags into the output")
args = parser.parse_args()

with open(args.output_file) as f_in, \
     open(args.output_file + '.phonemes', 'w') as out_txt, \
     open(args.output_file + '.durations', 'w') as out_dur:
    for line in f_in:
        line = line.strip().split(' ')
        curr_txt = []
        curr_dur = []
        for token in line:
            factors = token.split('|')
            curr_txt.append(factors[0])
            curr_dur.append(factors[1])
            if args.reinsert_eow and factors[5] == EOW_TOKEN:
                curr_txt.append(EOW_TOKEN)
                curr_dur.append(str(DUMMY_DURATION))
        if args.shift:
            curr_txt = curr_txt[1:]
            curr_dur = curr_dur[1:]
        if args.zero_non_numeric:
            # Replace non-numeric elements with 0
            curr_dur = [d if d.isnumeric() else '0' for d in curr_dur]
        out_txt.write(' '.join(curr_txt) + '\n')
        out_dur.write(' '.join(curr_dur) + '\n')
