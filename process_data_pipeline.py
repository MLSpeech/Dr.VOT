import os
import argparse
import sys

proj_path = os.path.abspath(os.path.join(__file__, '..', '..'))
sys.path.append(proj_path)

import time

from process_data import extract_voice_starts, feature_extractor

__author__ = 'YosiShrem'
# VOT_FONT_END_PATH = 'process_data/VotFrontEnd2'
mac_VOT_FONT_END_PATH = 'process_data/mac_VotFrontEnd2'
linux_VOT_FONT_END_PATH = 'process_data/linux_VotFrontEnd2'

"""
going to run the following 4 scripts:
    -extract_voice_starts.py
    -validate_voice_starts.py(for debug to make sure alignment is correct)
    -textgrid_to_timing.py
    -feature_extractor.py



extract_voice_starts:
    parser.add_argument('input_dir', type=str, help='Path to TextGrid dir')
    parser.add_argument('--out_filename', default=None, type=str,help='requested output path(default is VOICE_STARTS_FILENAME')
    parser.add_argument('--praat_path', default=None, type=str, help='path to praat executable')


validate:
    parser.add_argument('input_dir', type=str, help='Path to TextGrid dir')
    parser.add_argument('--parser', default=None, type=str,
                        help='using this parser to get the vot labels')
    parser.add_argument('--window_size', type=float, default=0.25, help='window size in sec')
    parser.add_argument('--pre', type=float, default=0.05, help='window size in sec')


textgrid_to_timing:
    parser.add_argument('input_dir', type=str, help='Path to TextGrid dir')
    parser.add_argument('--parser', default=None, help='Path to TextGrid parser', required=True)


feature_extractor:
    parser.add_argument('input_dir', type=str, help='Path to TextGrid dir')
    parser.add_argument('output_dir', default=None, help='Path to output dir')
    parser.add_argument('--window_size', type=float, default=0.25, help='window size in sec')
    parser.add_argument('--pre', type=float, default=0.05, help='window begins --pre sec before voice starts')
    parser.add_argument('--neg_word', default="prevoiced", help='negative vot files are named with this word')
    parser.add_argument('--prefix', type=str, required=True,
                        help='prefix for new files, labels depends on it, prevoiced if negative,voiced otherwise')
    parser.add_argument('--test', default=False, help='no labels, get only features')
"""
"""
 
                                            ###dmitrieva###
python process_data_pipeline.py /media/yosi/data/vot/raw/dmitrieva/voiceless /media/yosi/data/vot/processed/dmitrieva/pos --parser dmitrieva_timing.py --prefix voiced
python process_data_pipeline.py /media/yosi/data/vot/raw/dmitrieva/voiced /media/yosi/data/vot/processed/dmitrieva/pos --parser dmitrieva_timing.py --prefix voiced
python process_data_pipeline.py /media/yosi/data/vot/raw/dmitrieva/prevoiced /media/yosi/data/vot/processed/dmitrieva/neg --parser dmitrieva_timing.py --prefix prevoiced

                                            ###natalia###
python process_data_pipeline.py /media/yosi/data/vot/raw/natalia/cropped/voiced /media/yosi/data/vot/processed/natalia/pos --parser natalia_timing.py --prefix voiced
python process_data_pipeline.py /media/yosi/data/vot/raw/natalia/cropped/prevoiced /media/yosi/data/vot/processed/natalia/neg --parser natalia_timing.py --prefix prevoiced



                                            ###murphy###
python process_data_pipeline.py /media/yosi/data/vot/raw/murphy/cropped/voiced /media/yosi/data/vot/processed/murphy/pos --parser murphy_timing.py --prefix voiced
python process_data_pipeline.py /media/yosi/data/vot/raw/murphy/cropped/prevoiced /media/yosi/data/vot/processed/murphy/neg --parser murphy_timing.py --prefix prevoiced


                                            ###shultz###
python process_data_pipeline.py /media/yosi/data/vot/raw/shultz/cropped/voiced /media/yosi/data/vot/processed/shultz/pos --parser shultz_timing.py --prefix voiced
python process_data_pipeline.py /media/yosi/data/vot/raw/shultz/cropped/prevoiced /media/yosi/data/vot/processed/shultz/neg --parser shultz_timing.py --prefix prevoiced



"""
parser = argparse.ArgumentParser(description='Get timing form TextGrid')
parser.add_argument('--input_dir', default="wav_files/", help='Path to TextGrid dir')
parser.add_argument('--output_dir', default="features/", help='Path to output dir')
parser.add_argument('--skip_pi', action='store_true', help='skip_pi in the extract_voice script')
parser.add_argument('--parser', default=None, help='Path to TextGrid parser')
parser.add_argument('--window_size', type=float, default=0.25, help='window size in sec')
parser.add_argument('--pre', type=float, default=0.05, help='window begins --pre sec before voice starts')
parser.add_argument('--neg_word', default="prevoiced", help='negative vot files are named with this word')
parser.add_argument('--prefix', type=str,
                    help='prefix for new files, labels depends on it, prevoiced if negative,voiced otherwise')
parser.add_argument('--test', action="store_true",default=True, help='no labels, get only features')
# parser.add_argument('--praat', type=str, default="linux_praat", help='path to praat executable')#TODO linux
parser.add_argument('--praat', type=str, default="/Applications/Praat.app/Contents/MacOS/Praat", help='path to praat executable')
parser.add_argument('--features', action="store_true", help='skip to extract features')
# parser.add_argument('--timing', action="store_true", help='skip to textgrid top timing')
# parser.add_argument('--validate', action="store_true", help='skip to textgrid top validate')
parser.add_argument('--no_labels', action="store_true", help='skip validate and timing scripts(use on new unlabeled data')
args = parser.parse_args()

skip_to = 0  # for debugging

if args.features:
    skip_to = 3
# if args.timing:
#     skip_to = 2
# if args.validate:
#     skip_to = 1

start_time = int(time.time())
prev_time = time.time()
try:
    if skip_to <= 0:
        print("---------------\nRunning extract_voice_starts...(phase 1/4)\n---------------\n")
        command = f"{args.input_dir} --output_dir {args.output_dir} {'--skip_pi' if args.skip_pi else ''}"
        if os.popen('uname -a').read().lower().__contains__("darwin"):
            command += " --praat {}".format("/Applications/Praat.app/Contents/MacOS/Praat")
        if os.popen('uname -a').read().lower().__contains__("linux"):
            command += " --praat {}".format(os.path.join(os.getcwd(),"linux_praat"))
        extract_voice_starts.main(command.split())

        print("Voice_Extract - {} sec \n".format(time.time() - start_time))
        prev_time = int(time.time())

        if args.no_labels:
            skip_to=3
    # if skip_to <= 1 :
    #     print("\n\n\n\n\n---------------\nvalidate_voice_starts...(phase 2/4)\n---------------\n")
    #     command = "{} --parser {} --pre {} --window_size {}".format(args.input_dir, args.parser, args.pre,
    #                                                                 args.window_size)
    #     validate_voice_starts.main(command.split())
    #
    #     print("validation voice start - {} sec , overall {} sec\n".format(int(time.time() - prev_time),
    #                                                                       int(time.time() - start_time)))
    #     prev_time = int(time.time())
    #
    # if skip_to <= 2:
    #     print("\n\n\n\n\n---------------\ntextgrid_to_timing(phase 3/4)\n---------------\n")
    #     command = "{} --parser {}".format(args.input_dir, args.parser)
    #     textgrid_to_timing.main(command.split())
    #
    #     print("textgrid_to_timing - {} sec , overall {} sec\n".format(int(time.time() - prev_time),
    #                                                                   int(time.time() - start_time)))
    #     prev_time = int(time.time())

    if skip_to <= 3:
        print("\n\n\n\n\n---------------\nfeature_extractor...(phase 4/4)\n---------------\n")
        if os.popen('uname -a').read().lower().__contains__("darwin"):
            VotFrontEnd_path= mac_VOT_FONT_END_PATH
        elif os.popen('uname -a').read().lower().__contains__("linux"):
            VotFrontEnd_path = linux_VOT_FONT_END_PATH
        else: raise Exception("unknown operating system! couldn't match VotFrontEnd2 version."
                              "")
        command = "{} {} --pre {} --window_size {} --neg_word {} --prefix {} --features_file {}".format(args.input_dir, args.output_dir,
                                                                                     args.pre, args.window_size,
                                                                                     args.neg_word, args.prefix,os.path.join(os.getcwd(),VotFrontEnd_path))
        if args.test: command += " --test"
        feature_extractor.main(command.split())

        print("feature_extractor - {} sec , overall {} sec\n".format(int(time.time() - prev_time),
                                                                     int(time.time() - start_time)))

except Exception as e:
    print("Failed to run the pipeline process \n exception msg: [{}]".format(e))
    exit(1)

"""
Server

                                            ###dmitrieva###
python process_data_pipeline.py /data/shremjo/vot/raw/dmitrieva/voiceless /data/shremjo/vot/processed/dmitrieva/pos --parser dmitrieva_timing.py --prefix voiceless --validate
python process_data_pipeline.py /data/shremjo/vot/raw/dmitrieva/voiced /data/shremjo/vot/processed/dmitrieva/pos --parser dmitrieva_timing.py --prefix voiced --validate
python process_data_pipeline.py /data/shremjo/vot/raw/dmitrieva/prevoiced /data/shremjo/vot/processed/dmitrieva/neg --parser dmitrieva_timing.py --prefix prevoiced --validate

                                            ###natalia###
python process_data_pipeline.py /data/shremjo/vot/raw/natalia/cropped/voiced /data/shremjo/vot/processed/natalia/pos --parser natalia_timing.py --prefix voiced --validate
python process_data_pipeline.py /data/shremjo/vot/raw/natalia/cropped/prevoiced /data/shremjo/vot/processed/natalia/neg --parser natalia_timing.py --prefix prevoiced --validate



                                            ###murphy###
python process_data_pipeline.py /data/shremjo/vot/raw/murphy/cropped/voiced /data/shremjo/vot/processed/murphy/pos --parser murphy_timing.py --prefix voiced --validate
python process_data_pipeline.py /data/shremjo/vot/raw/murphy/cropped/prevoiced /data/shremjo/vot/processed/murphy/neg --parser murphy_timing.py --prefix prevoiced --validate


                                            ###shultz###
python process_data_pipeline.py /data/shremjo/vot/raw/shultz/cropped/voiced /data/shremjo/vot/processed/shultz/pos --parser shultz_timing.py --prefix voiced --validate
python process_data_pipeline.py /data/shremjo/vot/raw/shultz/cropped/prevoiced /data/shremjo/vot/processed/shultz/neg --parser shultz_timing.py --prefix prevoiced --validate


"""
