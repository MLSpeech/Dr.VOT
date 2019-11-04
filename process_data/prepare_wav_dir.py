from boltons import fileutils
import os
from shutil import copyfile,SameFileError
import sys
import argparse


__author__ = 'YosiShrem'

files_dict_fname="files.txt"

def main(args):
    try:
        parser = argparse.ArgumentParser(description='copy all wav files from all sub dirs to out_dir')
        parser.add_argument('--input_dir', type=str, help='Path to TextGrid dir',required=True)
        parser.add_argument('--output_dir', type=str, help='Path to output dir',required=True)
        args = parser.parse_args(args)


        assert os.path.exists(args.input_dir),f"Invalid Path, couldn't find [{args.input_dir}]"
        assert os.path.exists(args.output_dir),f"Invalid Path, couldn't find [{args.output_dir}]"

        wav_files = list(fileutils.iter_find_files(args.input_dir, "*.wav"))+list(fileutils.iter_find_files(args.input_dir, "*.WAV"))

        counter=0
        files_dict={}
        for file in wav_files:
            files_dict[counter] = file
            if os.path.exists(os.path.join(args.output_dir,f"{counter}.wav")):
                os.remove(os.path.join(args.output_dir,f"{counter}.wav"))
            copyfile(file,os.path.join(args.output_dir,f"{counter}.wav"))


            counter+=1

        print(f"Finished to copy '*.wav' files to {args.output_dir}")
        with open(os.path.join(args.output_dir,files_dict_fname),'w') as f:
            f.write(f"input_dir : {args.input_dir}\n")
            f.write(f"output_dir : {args.output_dir}\n")
            for k,v in files_dict.items():
                f.write(f"{k}:{v}\n")
        print(f"Finished to write the files dictionary to {os.path.join(args.output_dir,files_dict_fname)}")


    except Exception as e:
        print(f"Failed to process the data, error {e}")
        exit(1) #FAIL




if __name__ == '__main__':
    main(sys.argv[1:])
