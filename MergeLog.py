#!/usr/env/python python
# _*_ coding: utf-8 _*_

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="MergeLog.py", fromfile_prefix_chars='@')

    parser.add_argument('--infiles', action='store', nargs='+', required=True, help='input files')
    parser.add_argument('--outfile', action='store', required=True, help='output file')
    parser.add_argument('--fp_prefix', action='store', default='./', help='file path prefix')
    parser.add_argument('--remove_repeated_header', action='store_true', default=True, help='remove repeated header')

    args = parser.parse_args()

    in_files = args.infiles
    outfile = args.outfile
    fp_prefix = args.fp_prefix
    remove_repeated_header = args.remove_repeated_header

    header_saved = False
    with open(fp_prefix+outfile, 'w') as fout:
        for filename in in_files:
            with open(fp_prefix+filename) as fin:
                header = next(fin)
                if not header_saved:
                    fout.write(header)  # you may need to work here. The writerows require an iterable.
                    header_saved = True
                for line in fin:
                    if line:
                        fout.write(line)

