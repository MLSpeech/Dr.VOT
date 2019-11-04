import sys
import os
import numpy as np
from praatio import pitch_and_intensity
import threading
import glob
from tqdm import tqdm
import argparse

__author__ = 'YosiShrem'
SAMPLE_STEP = 0.001
PITCH_DURATION_THRESHOLD = 0.05
INTENS_DURATION_THRESHOLD = 0.3
THREAD_COUNT=16

def run_PI_for_dir(raw_dir, praat_path,gender=None):
    print(f"run PI_extractor for dir {raw_dir}, using {THREAD_COUNT} threads\n")
    wav_files = [glob.glob(os.path.join(raw_dir,'*{}'.format(ext))) for ext in ['.WAV','.wav']]
    wav_files = [item for sublist in wav_files for item in sublist] #flatten list
    threads=[None]*(THREAD_COUNT) #2 genders
    i=0
    for wav_file in tqdm(wav_files):

        male_out_file = wav_file.replace('.wav', '_mPI.txt').replace('.WAV', '_mPI.txt')
        threads[i] = threading.Thread(target=pitch_and_intensity.extractPI,
                                           kwargs=dict(inputFN=wav_file, outputFN=male_out_file, praatEXE=praat_path,
                                                       minPitch=50, maxPitch=350, sampleStep=SAMPLE_STEP,
                                                       silenceThreshold=0))
        female_out_file = wav_file.replace('.wav', '_fPI.txt').replace('.WAV', '_fPI.txt')
        threads[i +1] = threading.Thread(target=pitch_and_intensity.extractPI,
                                           kwargs=dict(inputFN=wav_file, outputFN=female_out_file, praatEXE=praat_path,
                                                       minPitch=75, maxPitch=450, sampleStep=SAMPLE_STEP,
                                                       silenceThreshold=0))
        if os.path.exists(male_out_file): os.remove(male_out_file)  # just making sure to use the new files
        if os.path.exists(female_out_file): os.remove(female_out_file)
        i+=2
        if i+1 >= THREAD_COUNT:
            for thread in threads:
                if thread!= None:thread.start()
            for thread in threads:
                if thread != None:thread.join()
            threads = [None] * THREAD_COUNT  # 2 genders
            i=0
    #leftovers
    for thread in threads:
        if thread != None: thread.start()
    for thread in threads:
        if thread != None: thread.join()
    print("extracted PI for dir :{}\n".format(raw_dir))






def get_voice_start(input, out_wav_path, praat_path, debug=False, gender=None,skip_extract_PI=False):
    # male_pitch

    male_out_file = out_wav_path.replace('.wav', '_mPI.txt').replace('.WAV', '_mPI.txt')
    female_out_file = input.replace('.wav', '_fPI.txt').replace('.WAV', '_fPI.txt')


    if not skip_extract_PI:
        male_min_pitch = 50
        male_max_pitch = 350
        female_min_pitch = 75
        female_max_pitch = 450
        male_thread = threading.Thread(target=pitch_and_intensity.extractPI,
                                       kwargs=dict(inputFN=input, outputFN=male_out_file, praatEXE=praat_path,
                                                   minPitch=male_min_pitch, maxPitch=male_max_pitch, sampleStep=SAMPLE_STEP,
                                                   silenceThreshold=0))

        female_thread = threading.Thread(target=pitch_and_intensity.extractPI,
                                         kwargs=dict(inputFN=input, outputFN=female_out_file, praatEXE=praat_path,
                                                     minPitch=female_min_pitch, maxPitch=female_max_pitch, sampleStep=SAMPLE_STEP,
                                                     silenceThreshold=0))
        # run both with threads for speedup of X1.5
        female_thread.start()
        male_thread.start()
        female_thread.join()
        male_thread.join()

    pitch_exists_m, intens_threshold = get_intens_threshold(male_out_file, debug=debug)
    male = parse_PI_file(male_out_file, intens_threshold)
    if gender == 'male': return male

    pitch_exists_f, intens_threshold = get_intens_threshold(female_out_file, debug=debug)
    female = parse_PI_file(female_out_file, intens_threshold)
    if gender == 'female': return female

    # female_pitch
    # min_pitch = 75
    # max_pitch = 750
    # female_out_file = input.replace('.wav', '_fPI.txt').replace('.WAV', '_fPI.txt')
    # pitch_and_intensity.extractPI(input, female_out_file, praat_path, min_pitch, max_pitch, sampleStep=SAMPLE_STEP,
    #                               silenceThreshold=0)
    # pitch_exists_f,intens_threshold = get_intens_threshold(female_out_file, debug=debug)
    # female = parse_PI_file(female_out_file, intens_threshold)
    # if gender == 'female': return female

    # check is pitch exists, if in both return the MAX(empirically better), otherwise the ones with the pitch
    if (pitch_exists_f and pitch_exists_m) or (
            pitch_exists_f == False and pitch_exists_m == False):  # exists/missing in both
        if male > female:
            os.remove(female_out_file)
        else:
            os.remove(male_out_file)
        # unspecified gender
        return max(male, female)
    else:  # exists only in one of them
        if pitch_exists_m:
            os.remove(female_out_file)
            return male
        else:
            os.remove(male_out_file)
            return female


def get_intens_threshold(filename, ratio=0.7, debug=False, threshold=30):
    intens_list = []
    is_pitch = True
    with open(filename, 'r') as f:  # only where there is pitch
        f.readline()  # description line
        for line in f:
            time, pit, intens = line.split(',')

            if pit.strip().replace('.', '', 1).isdigit():  # pitch exist
                if intens.strip().replace('.', '', 1).isdigit() and float(intens) > threshold:  # intens is defines
                    intens_list += [float(intens)]

    if len(intens_list) < PITCH_DURATION_THRESHOLD * 1000:  # couldnt fint pitch so use the avg intense
        is_pitch = False
        with open(filename, 'r') as f:
            f.readline()  # description line
            for line in f:
                time, pit, intens = line.split(',')
                if intens.strip().replace('.', '', 1).isdigit():  # intens is defines
                    intens_list += [float(intens)]

    intens_threshold = np.array(intens_list).mean() * ratio
    if debug: print("normalized intens is : {:.2f}\n".format(intens_threshold))
    return is_pitch, intens_threshold


def parse_PI_file(filename, intens_threshold, debug=False):
    """
    go over the pitch and look for the first segment of pitch where the intensity also increases,
     and return the first point where one of them started to ascend.
    return the timing where the intes/pitch starts to increase
    """

    intens_start = 0
    last_pitch_start = 0
    first_intens = 0
    pitch_start = 0
    pitch_len = 0
    intens_len = 0
    with open(filename, 'r') as f:
        f.readline()  # description line
        for line in f:
            time, pit, intens = line.split(',')

            if pit.strip().replace('.', '', 1).isdigit():  # pitch exist
                if pitch_start == 0:  # pitch starts
                    pitch_start = round(float(time), 3)
                    last_pitch_start = pitch_start
                pitch_len += SAMPLE_STEP
            else:
                # len=0
                if pitch_len > 0: pitch_len -= SAMPLE_STEP
                pitch_start = 0

            if intens.strip().replace('.', '', 1).isdigit() and float(intens) > intens_threshold:
                # update start
                if intens_start == 0:
                    intens_start = round(float(time), 3)
                intens_len += SAMPLE_STEP
                # update first
                if first_intens == 0:
                    first_intens = round(float(time), 3)
            else:

                intens_start = 0
                intens_len = 0

            # break condition
            if (pitch_len > PITCH_DURATION_THRESHOLD or intens_len > INTENS_DURATION_THRESHOLD) \
                    and pitch_start != 0 and intens_start != 0:
                break

    if pitch_start == 0 and last_pitch_start == 0:  # no pitch at all
        intens_start = max(intens_start, first_intens)
        if debug: print("No pitch at : [{}] , use only intens, start at : [{}]".format(filename, intens_start))
        return intens_start

    if pitch_start == 0 or intens_start == 0:
        if debug: print("PITCH and intensity manipulation didnt work on [{}],"
                        " so using basic extraction\n".format(filename))
        pitch_start = max(0, last_pitch_start - 0.1)
        intens_start = max(intens_start, first_intens)
        # print("start at : {}\n".format(min(pitch_start, intens_start)))

    if intens_start > pitch_start:
        if debug: print("[DEBUG] INTENS after pitch , file :{}\n".format(filename))
    # if intens_start + 0.15 < pitch_start:
    #     print("Intens too early, file :{}\n".format(filename))
    #     return min(intens_start + 0.15, pitch_start)

    if pitch_len < PITCH_DURATION_THRESHOLD:
        if debug: print("using INTENS threshold, not pitch, file :{}\n".format(filename))

    return min(pitch_start, intens_start)




def main(args):
    parser = argparse.ArgumentParser(description='Get timing form TextGrid')
    parser.add_argument('input_dir', type=str, help='Path to TextGrid dir')
    parser.add_argument('--praat_path', default='/home/yosi/custom_commands/praat', type=str, help='path to praat executable')


    args = parser.parse_args(args)
    assert os.path.isdir(args.input_dir), "Couldn't find input dir [{}]".format(args.input_dir)
    assert os.path.exists(args.praat_path), "Couldn't find praat path : {}".format(args.praat_path)

    run_PI_for_dir(args.input_dir, args.praat_path)

if __name__ == '__main__':
    main(sys.argv[1:])
