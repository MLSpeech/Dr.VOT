import sys

sys.path.append('..')
from tqdm import tqdm
import os
import argparse

import glob
import wave
import contextlib
from process_data import pitch_process
import time
__author__ = 'YosiShrem'

VOICE_STARTS_FILENAME = "voice_starts.txt"
PRAAT_PATH = '/home/yosi/custom_commands/praat'

"""
for every wav dir, generate voice_starts.txt file which includes the offset for each file.
the offset is when the speach detector activates.
"""


def get_wav_duration(wav_path):
    with contextlib.closing(wave.open(wav_path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    return round(duration, 4) - 0.0001  # make sure to 'floor' last digit


def main(args):
    global VOICE_STARTS_FILENAME, PRAAT_PATH

    parser = argparse.ArgumentParser(description='Get timing form TextGrid')
    parser.add_argument('input_dir', type=str, help='Path to TextGrid dir')
    parser.add_argument('--output_dir', type=str, help='Path to output dir')
    parser.add_argument('--out_filename', default=VOICE_STARTS_FILENAME, type=str,
                        help='requested output path(default is VOICE_STARTS_FILENAME')
    parser.add_argument('--praat_path', default=PRAAT_PATH, type=str, help='path to praat executable')
    parser.add_argument('--skip_pi', action='store_true', help='skip pi extraction')

    args = parser.parse_args(args)
    args.input_dir = os.path.abspath(args.input_dir)
    args.output_dir = os.path.abspath(args.output_dir) if args.output_dir else args.input_dir

    PRAAT_PATH = args.praat_path
    # mis_aligned = [line.strip() for line in open(os.path.join(args.input_dir,'misaligned_files.txt'),'r').readlines()]
    #####check for wrong inputs#####
    assert os.path.isdir(args.input_dir), "Couldn't find input dir [{}]".format(args.input_dir)
    assert os.path.isdir(args.output_dir), "Couldn't find output dir [{}]".format(args.output_dir)
    assert os.path.exists(PRAAT_PATH), "Couldn't find praat path : {}".format(args.praat_path)
    ####################

    wav_files = [glob.glob(os.path.join(args.input_dir, '*{}'.format(ext))) for ext in ['.WAV', '.wav']]
    wav_files = [item for sublist in wav_files for item in sublist]  # flatten list

    offsets_list = []
    is_skip_PI = args.skip_pi
    if not is_skip_PI:
        pitch_process.run_PI_for_dir(args.input_dir, PRAAT_PATH)
        time.sleep(1)
        is_skip_PI=True
    for wav_file in tqdm(wav_files):
        # if not wav_file.__contains__("VI007-3_43.wav"):
        #     continue
        # if not os.path.basename(wav_file) in mis_aligned:
        #     continue
        # if len(offsets_list ) > 20:
        #     break
        try:
            wav_path = os.path.join(args.input_dir, wav_file)
            out_path = os.path.join(args.output_dir, wav_file)
            voice_start = pitch_process.get_voice_start(wav_path, out_path, PRAAT_PATH, skip_extract_PI=is_skip_PI)
            wav_duration = get_wav_duration(wav_path)
            offsets_list += [(wav_path, voice_start, wav_duration)]
        except Exception as e:
            print ("Failed to process file [{}]\n error:{}".format(wav_file,e))

    if args.out_filename == VOICE_STARTS_FILENAME:  # if default value, write file in input dir
        VOICE_STARTS_FILENAME = [os.path.join(args.input_dir, VOICE_STARTS_FILENAME)]
                                 # os.path.join(args.output_dir, VOICE_STARTS_FILENAME)]
    # else:
    #     VOICE_STARTS_FILENAME = [VOICE_STARTS_FILENAME]

    for voice_start_file in VOICE_STARTS_FILENAME:
        with open(voice_start_file, 'w') as f:
            for (file, voice_start, duration) in offsets_list:
                f.write("{} : {}  :{}\n".format(file, voice_start, duration))


if __name__ == '__main__':
    main(sys.argv[1:])
