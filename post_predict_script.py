import os
import argparse
from shutil import copyfile
from tqdm import tqdm

__author__ = 'YosiShrem'

"""
replace the filenames in the summary.csv and then rename the textgrids at the same way the wav files are organized ;
 hierarchically.
"""



parser = argparse.ArgumentParser(description='VOT segmentor Predictor')

parser.add_argument('--summary', type=str, default=None,required=True, help='summary file with only base filenames')
parser.add_argument('--filenames',type=str, default=None,required=True, help='path for files.txt the dictionary which maps each wav to origin path')
parser.add_argument('--tg_dir',type=str, default=None,required=True, help='path for textgrids_dir')
args = parser.parse_args()
assert os.path.exists(args.summary) ,f"Couldn't find {args.summary}"
assert os.path.exists(args.filenames) ,f"Couldn't find {args.filenames}"
assert os.path.exists(args.tg_dir) ,f"Couldn't find {args.tg_dir}"

try:
    print(f"reading <{args.filenames}> to get the mapping...")
    full_paths = {}

    with open(args.filenames, 'r') as f:
        origin_dir=f.readline().split(':')[1].strip()
        f.readline() # skip features_path line
        for line in f:
            k,v = line.strip().split(':')
            full_paths[k] = v[len(origin_dir):].strip('/')

    # .replace(".wav",".TextGrid").replace(".WAV",".TextGrid")
    print(f"reading <{args.summary}> for re-write...")
    filenames= {}
    with open(args.summary, 'r') as f:
        title = f.readline()
        for line in f:
            base_name, vot_type, duration = line.strip().split(',')
            filenames[base_name] = [vot_type, duration]

    print(f"writing new_summary.csv at  {os.path.join(os.path.dirname(args.summary),'new_summary.csv')}...")
    with open(os.path.join(os.path.dirname(args.summary),"new_summary.csv"),'w') as f:
        f.write(title)
        for k,full_path in full_paths.items():
            try:
                vot_type, duration = filenames[k]
                f.write("{},{},{}\n".format(full_path, vot_type, duration ))
            except:
                print (k)


    """ create hierarchy textgrids ,same as input """

    new_hirerchical_dir = os.path.join(args.tg_dir, 'hierarchical_tg')

    print(f"copying all .TextGrids hierarchically to <{new_hirerchical_dir}> ...")
    for file in tqdm(os.listdir(args.tg_dir)):
        if not file.lower().endswith(".textgrid"):
            continue

        base_name,pred_suffix = file.split("_") #get only the integer from <idx_predPOS/NEG.TextGrid>
        #get location to copy each idx.TextGrid
        dst = full_paths[base_name][:-len('.wav')] + "_" + pred_suffix
        os.makedirs(os.path.join(new_hirerchical_dir,os.path.dirname(dst)), exist_ok=True)
        copyfile(os.path.join(args.tg_dir,file), os.path.join(new_hirerchical_dir,dst))
except Exception as e:
    print(f"failed to post-process the data, Error:{e}")
    exit(1)
#
# for _,path in full_paths.items():
#
#
#
#     tg_src_path = os.path.join(os.path.dirname(args.summary),os.path.basename(path)[:-4]) # remove ".WAV/.wav"
#     if os.path.exists(tg_src_path + "_predPOS.TextGrid"):
#         tg_src_path += "_predPOS.TextGrid"
#     else:
#         tg_src_path += "_predNEG.TextGrid"
#     tg_dst_path = os.path.join(os.path.dirname(args.summary),os.path.dirname(path),os.path.basename(tg_src_path))
#     tg_dst_dir = os.path.dirname(tg_dst_path)
#     if not os.path.exists(tg_dst_dir): os.makedirs(tg_dst_dir)
#     copyfile(tg_src_path, tg_dst_path)
#
