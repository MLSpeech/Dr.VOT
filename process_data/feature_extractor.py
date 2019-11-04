import sys
import glob

sys.path.append('..')
import argparse
from helpers import utilities
import os
import wave
import random
import tqdm
from process_data.extract_voice_starts import VOICE_STARTS_FILENAME
from helpers import textgrid as tg

__author__ = 'YosiShrem'

base_all_files_input_info_list = "files_extraction_info_list.txt"
base_all_files_out_filenames_list = "out_ft_filenames_list.txt"
all_files_input_info_list = None
all_files_out_filenames_list = None
# VOT_FONT_END_PATH = 'linux_VotFrontEnd2'

# temps
temp_input_list = "temp_filename_list.txt"
temp_ft_out_filename_list = "temp_ft_filename_list.txt"

rename_mapping = {"prevoiced": "prevoiced_",
                  "voiced": "voiced_",
                  "voiceless": "voiceless_"}

#
def extract_ft(input_filename, features_filename, features_exec_filename, logging_level="ERROR"):
    # print(os.getcwd())

    if not os.path.exists(input_filename):
        assert False, "Couldn't find wav file :{}".format(input_filename)
    cmd_vot_front_end = '{} -verbose {} {} {} {}'.format(features_exec_filename,
                                                         logging_level, input_filename, features_filename, "null")

    # print(cmd_vot_front_end)
    utilities.easy_call(cmd_vot_front_end)


def init_processing_lists(pref, output_dir):
    global all_files_input_info_list, all_files_out_filenames_list
    all_files_input_info_list = pref + '_' + base_all_files_input_info_list
    all_files_out_filenames_list = pref + '_' + base_all_files_out_filenames_list
    # clean previous lists that were used for the processing , the 2 lists that were used in 'linux_VotFrontEnd2' script
    if os.path.exists(os.path.join(output_dir, all_files_input_info_list)):
        os.remove(os.path.join(output_dir, all_files_input_info_list))
    if os.path.exists(os.path.join(output_dir, all_files_out_filenames_list)):
        os.remove(os.path.join(output_dir, all_files_out_filenames_list))


def remove_timing_files(path):
    print("Deleting all '.timing' files from output dir......")
    current_path = os.getcwd()
    os.chdir(path)
    os.system("rm *.timing")
    os.chdir(current_path)


def fix_extention(path):
    if path.endswith(".WAV"):
        os.system("mv {} {}".format(path, path.replace(".WAV", ".wav")))


def get_voice_starts_dict(in_dir):
    """
    reads the voice_starts file and return a dict of {filename: voice starts}

    """
    voice_start_file_path = os.path.join(in_dir, VOICE_STARTS_FILENAME)
    assert os.path.exists(voice_start_file_path), 'Couldn\'t find the voice_starts file, no such file :{}\n'.format(
        voice_start_file_path)

    file_to_timing = {}
    with open(voice_start_file_path, 'r') as f:
        for line in f:
            file, onset, duration = line.strip().split(':')
            file_to_timing[os.path.basename(file).strip().replace(".WAV", ".wav")] = (float(onset), float(duration))
    return file_to_timing


#
# def get_file_to_label_dict(input_dir):
#     """
#     reads the voice_starts file and return a dict of {filename: vot_onset,vot_offset,wav_duration}
#
#     """
#     voice_start_file_path = os.path.join(input_dir, RESULTS_FILE)
#     assert os.path.exists(voice_start_file_path), 'Couldn\'t find the labels_file, no such file :{}\n'.format(
#         voice_start_file_path)
#
#     file_to_label = {}
#     with open(voice_start_file_path, 'r') as f:
#         for line in f:
#             file, label = line.strip().split(':')
#             file_to_label[os.path.basename(file).strip()] = [float(i) for i in label.strip().split(' ')]
#     return file_to_label


# def get_misaligned(raw_dir):
#     misaligned_file = os.path.join(raw_dir, MISALIGNED_FILES)
#
#     if not os.path.exists(misaligned_file):
#         return {}  # all good
#     else:
#         files = {}
#         for line in open(misaligned_file, 'r').readlines():
#             wav_name, onset, duration = [val.strip() for val in line.split(':')]
#             files[os.path.basename(wav_name)] = (float(onset), float(duration))
#
#         # files = [os.path.abspath(line.strip()) for line in open(misaligned_file, 'r').readlines()]
#         return files


def process_raw_data(raw_dir, output_dir, pre_window, window_size, neg_word, pref, features_exec_file,
                     sample_rate=16000, is_test=False):
    """
    raw dir contains wav and TextGrids.
    - convert WAV TO Sample_rate(default=16Khz)
    convert each file to sample rate and then extract features. its done file by file because of memory-space limits.
     the script doesnt save the entire dataset in new sample rate, but save only one at a time.
    """

    init_processing_lists(pref, output_dir)
    voice_starts_dict = get_voice_starts_dict(raw_dir)
    # misaligned_files = get_misaligned(raw_dir)
    # file_to_label_dict = None
    # if not is_test:
    #     file_to_label_dict = get_file_to_label_dict(raw_dir)

    wav_files = [glob.glob(os.path.join(raw_dir, '*{}'.format(ext))) for ext in ['.WAV', '.wav']]
    wav_files = [os.path.basename(item) for sublist in wav_files for item in sublist]  # flatten list

    for wav_file in tqdm.tqdm(wav_files):
        # if not wav_file.lower().endswith("wav"):
        #     continue

        if wav_file.endswith(".WAV"):
            fix_extention(os.path.join(raw_dir, wav_file))  # convert "WAV" to "wav" if nessecary
            wav_file = wav_file.replace('WAV', 'wav')
        temp_wav_path = os.path.join(output_dir, "temp_" + wav_file)

        try:
            conv_file_to_Khz(os.path.join(raw_dir, wav_file), sample_rate, temp_wav_path)
            renamed_wav_file = wav_file if is_test else add_prefix(pref, wav_file)  # add prevoiced/voiceless/voiced
            #
            # if misaligned_files.__contains__(wav_file) == True:  # voice_start missed the VOT window.
            #     voice_starts, wav_duration = misaligned_files[wav_file]
            # else:  # the voice_start succeed so use its onset
            voice_starts, wav_duration = voice_starts_dict[wav_file]

            start_point, finish_point, ft_filename_line = update_lists_for_extraction(output_dir, temp_wav_path,
                                                                                      renamed_wav_file,
                                                                                      voice_starts, pre_window,
                                                                                      window_size, wav_duration)
            extract_ft(os.path.join(output_dir, temp_input_list), os.path.join(output_dir, temp_ft_out_filename_list),
                       features_exec_filename=features_exec_file)
            remove_first_row(ft_filename_line)  # the 1st row in the features file is the size, remove it

            # if not is_test:  # create  labels file
            #     vot_onset, vot_offset, _ = file_to_label_dict[os.path.splitext(wav_file)[0]]
            #     create_labels(os.path.join(output_dir, renamed_wav_file.replace('.wav', '.labels')), start_point,
            #                   vot_onset, vot_offset, neg_word, window_size)
            # else:  # doesnt have labels, so create dummy labels, with 0 ,0 as onset/offset
            create_labels(os.path.join(output_dir, renamed_wav_file.replace('.wav', '.test.labels')), start_point,
                          start_point, start_point, neg_word, window_size)

            os.remove(temp_wav_path)
            os.remove(os.path.join(output_dir, temp_input_list))
            os.remove(os.path.join(output_dir, temp_ft_out_filename_list))

        except Exception as e:
            try:
                os.remove(temp_wav_path)
                os.remove(os.path.join(output_dir, temp_input_list))
                os.remove(os.path.join(output_dir, temp_ft_out_filename_list))
            except:
                pass
            print("[error] Failed to process file:'{}'\n {} ".format(wav_file, e))


def remove_first_row(file):
    with open(file, 'r') as fin:
        data = fin.read().splitlines(True)
    # print("len :{}".format(data[0]))
    with open(file, 'w') as fout:
        fout.writelines(data[1:])


def add_prefix(pref, name):
    return pref + '_' + name


def create_labels(labels_filename, window_starts, onset, offset, neg_word, window_size):
    """
    The feature extractor(linux_VotFrontEnd2) ignores 0.002sec at the beginning of the wav, so:
     if the window starts at 0.000sec then -2 frames to onset and offsets indexes.
     if the window starts at 0.001sec then -1 frames to onset and offsets indexes.
     else, the window starts at 0.002+ then frames aren't should be shifted
    """
    # calc timing->frames
    # TODO NOTE: when predicting add the shift to the prediction frame to get the right frame in the whole wav
    shift = 0.002 - window_starts if window_starts < 0.002 else window_starts
    onset_frame = int(1000 * (float(onset) - shift))
    offset_frame = int(1000 * (float(offset) - shift))
    is_positive_vot = int(not labels_filename.__contains__(neg_word))  # 0=negative, 1=positive

    with open(labels_filename, 'w')as f:
        f.write('1 2\n')
        f.write('{} {}\n'.format(onset_frame, offset_frame))
        f.write('{}\n'.format(is_positive_vot))
        f.write('offset_from_start {}\n'.format(round(shift, 3)))
        f.write('window_size_sec {}\n'.format(round(window_size, 4)))


def update_lists_for_extraction(output_dir, wav_path, original_wav_name, voice_starts, pre_window, window_size,
                                total_len):
    """
    process 1 file at a time to save space
    start at random time before according to pre and finish at random time after the offset according to post
    """
    global all_files_input_info_list, all_files_out_filenames_list

    start_point = round(max(0, float(voice_starts) - pre_window), 3)
    finish_point = round(min(float(total_len), float(start_point + window_size)), 3)

    ft_extract_line = '"{}" {:.3f} {:.3f} 0 0'.format(os.path.abspath(wav_path), start_point, finish_point)
    output_filename_line = os.path.abspath(os.path.join(output_dir, original_wav_name.replace('wav', 'features')))

    # file specific
    with open(os.path.join(output_dir, temp_input_list), 'w') as f:
        f.write(ft_extract_line)
    with open(os.path.join(output_dir, temp_ft_out_filename_list), 'w') as f:
        f.write(output_filename_line)

    # all files lists- saving those lists for future needs -> if someone needs to know the extraction window
    with open(os.path.join(output_dir, all_files_input_info_list), 'a') as f:
        f.write(ft_extract_line + '\n')
    with open(os.path.join(output_dir, all_files_out_filenames_list), 'a') as f:
        f.write(output_filename_line + '\n')
    return start_point, finish_point, output_filename_line


def conv_file_to_Khz(filename, sample_rate, temp_wav_name):
    # using volume decrease to 90% to reduce clipping
    cmd = "sox -v 0.90 {} -r {} -b 16 {}".format(filename, sample_rate, temp_wav_name)
    utilities.easy_call(cmd)


def read_timing(file):
    if not os.path.exists(file):
        assert False, "[ERROR] missing .timing file for : '{},\n <><> PLEASE RUN 'textgrid_to_timing_old.py' FIRST<><>".format(
            file)

    with open(file, 'r') as f:
        temp = f.readline().split()
        assert len(temp) == 3, "Wrong timing file : {}, ".format(file)

    onset = temp[0].strip()
    offset = temp[1].strip()
    total_len = temp[2].strip()
    return onset, offset, total_len


def main(args):
    parser = argparse.ArgumentParser(description='Get timing form TextGrid')
    parser.add_argument('input_dir', type=str, help='Path to TextGrid dir')
    parser.add_argument('output_dir', default=None, help='Path to output dir')
    parser.add_argument('--window_size', type=float, default=0.25, help='window size in sec')
    parser.add_argument('--pre', type=float, default=0.05, help='window begins --pre sec before voice starts')
    parser.add_argument('--neg_word', default="prevoiced", help='negative vot files are named with this word')
    parser.add_argument('--prefix', type=str, required=True,
                        help='prefix for new files, labels depends on it, prevoiced if negative,voiced otherwise')
    parser.add_argument('--test', action='store_true', help='no labels, get only features')
    parser.add_argument('--features_file', type=str, default=None, help='path to VotFrontEnd2')

    args = parser.parse_args(args)

    if not os.path.isdir(args.input_dir) or not os.path.isdir(args.output_dir):
        print("[Error] Wrong input or output Path ")
        exit()

    print("Input path : '{}' \nOutput path : '{}'".format(args.input_dir, args.output_dir))
    print(args)

    process_raw_data(args.input_dir, args.output_dir, args.pre, args.window_size, args.neg_word, args.prefix,
                     args.features_file,
                     sample_rate=16000, is_test=args.test)

    # remove_timing_files(args.output_dir)
    print("Finished.\n The features and labels files are at : '{}'".format(args.output_dir))


if __name__ == '__main__':
    ####################################################################
    ##  Extract features from Wav files according to pre-set window.
    ##
    ##  requirement: must be run after 'extract_voice_starts.py'(produces a file with an extraction window for each wav)
    ##
    ##  This script: convert files to 16Khz and then for every file extract features
    ##  from a pre-set window which the extract_voice_starts.py produced
    ##
    ##  The script inputs are the 1)src dir which has the .wavs and .Textgrids, 2)output dir
    ##  the script also get as input the duration needed before(--pre) and after(--post) the vot
    ##
    ##  NOTE when labeling : the feature extractor always outputs 4 frames less then original.
    ##                  0.002sec at the beggining and 0.002sec at the end are dropped
    ##  $python features_extractor_old.py <src> <out> --pre <limits for the random> --post <limits for the random>
    ##
    ##  for example :
    ##      $python features_extractor_old.py <src> <out> --pre 0.06 0.08 --post 0.04 0.05
    ####################################################################

    main(sys.argv[1:])
