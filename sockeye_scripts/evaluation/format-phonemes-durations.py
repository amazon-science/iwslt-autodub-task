import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("output_file")
parser.add_argument("--zero-non-numeric", "-z", action='store_true',
                    help="Replace any non-numeric durations with zeros")
parser.add_argument("--reinsert-eow", action='store_true',
                    help="For models with EOW factors, reinsert the <eow> tags into the output")
args = parser.parse_args()

PAUSE = '[pause]'
EOW = '<eow>'
SHIFT = '<shift>'

with open(args.output_file) as f_in, \
     open(args.output_file + '.altformat', 'w') as f_out:
    for line in f_in:
        line = line.strip().split()
        new_line = []
        for token in line:
            token = token.split('|')
            if token[0] == SHIFT:
                continue
            new_line.append(token[0])
            if token[0] != PAUSE and token[0] != EOW:
                if token[1].isnumeric() or not args.zero_non_numeric:
                    new_line.append(token[1])
                else:
                    # Replace non-numeric durations with 0
                    new_line.append('0')
                if args.reinsert_eow and token[5] == EOW:
                    # EOW factor. Add <eow> tag back into sequence
                    new_line.append(EOW)
        f_out.write(' '.join(new_line) + '\n')
