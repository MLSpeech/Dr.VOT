from __future__ import print_function
import torch
import torch.nn as nn
import sys
import os

proj_path = os.path.abspath(os.path.join(__file__, '..', '..'))
sys.path.append(proj_path)
import argparse
import torch
import numpy as np

import data_utils
from data_utils import POS,NEG
from model.model import VOT_Seg,VOT_tagger
from model.structured_layer import structured_layer


from tqdm import tqdm
import helpers.textgrid as tg

__author__ = 'YosiShrem'
POS_STRING = "POS_VOT"
NEG_STRING = "NEG_VOT"


MARGIN = 1
DEBUG_MAX_SAMPLES = 20


def create_tg_simple(path, window_length, window_offset, pred,length):
    textgrid = tg.TextGrid()
    window_tier = tg.IntervalTier("Window")
    window_tier.add(window_offset, window_offset + window_length * 0.001, "window")
    window_tier.add(window_offset + window_length * 0.001, length-0.001, " ")
    # window_tier.add(window_offset + window_length * 0.001, window_offset + window_length * 0.001 + MARGIN, " ")
    textgrid.append(window_tier)

    vot_onset, vot_offset, pred_type = pred
    vot_tier = tg.IntervalTier("VOT")
    vot_tier.add(window_offset + vot_onset * 0.001, window_offset + vot_offset * 0.001, pred_type)
    vot_tier.add(window_offset + vot_offset * 0.001, length-0.001, "")
    # vot_tier.add(window_offset + vot_offset * 0.001, window_offset + vot_offset * 0.001 + MARGIN, "")
    textgrid.append(vot_tier)

    predicted_type = "POS" if POS_STRING == pred_type else "NEG"
    path = path.replace(".TextGrid", predicted_type + ".TextGrid")
    textgrid.write(path)


def create_tg(path, window_length, window_offset, preds, basic_preds,labeled_vot=None,labeled_type=None):
    textgrid = tg.TextGrid()
    window_tier = tg.IntervalTier("Window")
    window_tier.add(window_offset, window_offset + window_length * 0.001, "window")
    window_tier.add(window_offset + window_length * 0.001, window_offset + window_length * 0.001 + MARGIN, " ")
    textgrid.append(window_tier)

    for (vot_onset, vot_offset, pred_type), tier_name in zip(preds, ["if pos", "if neg", "prediction"]):
        vot_tier = tg.IntervalTier(tier_name)
        # word frame is larger so create some margin- 10% . the model isnt effected by this tier
        vot_tier.add(window_offset + vot_onset * 0.001, window_offset + vot_offset * 0.001, pred_type)
        vot_tier.add(window_offset + vot_offset * 0.001, window_offset + vot_offset * 0.001 + MARGIN, "")
        textgrid.append(vot_tier)

    vot_onset, vot_offset = basic_preds
    vot_tier = tg.IntervalTier("Basic_segmentor")
    # word frame is larger so create some margin- 10% . the model isnt effected by this tier
    vot_tier.add(window_offset + vot_onset * 0.001, window_offset + vot_offset * 0.001, "VOT")
    vot_tier.add(window_offset + vot_offset * 0.001, window_offset + vot_offset * 0.001 + MARGIN, "")
    textgrid.append(vot_tier)

    if labeled_type is not None and labeled_vot[0]>0 and labeled_vot[1]>0: #unlabeled is 0,0 as onset and offset
        vot_onset, vot_offset = labeled_vot[0],labeled_vot[1]
        vot_tier = tg.IntervalTier("Labeled VOT")
        # word frame is larger so create some margin- 10% . the model isnt effected by this tier
        vot_tier.add(window_offset + vot_onset * 0.001, window_offset + vot_offset * 0.001, POS_STRING if labeled_type==1 else NEG_STRING)
        vot_tier.add(window_offset + vot_offset * 0.001, window_offset + vot_offset * 0.001 + MARGIN, "")
        textgrid.append(vot_tier)


    predicted_type = "POS" if POS_STRING == preds[-1][-1] else "NEG"
    path = path.replace(".TextGrid", predicted_type + ".TextGrid")
    textgrid.write(path)


def predict(model, structured, tagger, device, data_loader, debug=False):
    """
    for every file, return the vot if positive, vot if negative, vot according to predicted type
    add the offset to all(the beginning  of the search window)

    :return: [ (filename, (pos_vot), (neg_vot), (pred_vot)),
                ....]
    """
    model.eval()
    structured.eval()
    tagger.eval()
    filenames = []
    pos_vots = []
    neg_vots = []
    pred_vots = []
    for idx, (data, (filename, window_offset,labeled_vot,labeled_vot_type)) in enumerate(tqdm(data_loader)):

        filenames += [
            (filename[0], window_offset.item(), data.shape[1],labeled_vot[0],labeled_vot_type)]  # first dim is the batch size,2nd is len,3rd n_features

        data = data.to(device).float()
        phi = model(data)  # rnn
        vot_scores = tagger(phi[-1]).view(-1)  # tagger

        # POS case
        scores = torch.Tensor([0, 0])
        scores[POS] = 1
        w_phi = structured(phi, scores).cpu()  # structured
        _, _, pos_onset, pos_offset = structured.predict(w_phi)
        pos_vots += [(pos_onset + 1, pos_offset + 1, POS_STRING)]  # see Note at the bottom for the '+1'

        # NEG case
        scores = torch.Tensor([0, 0])
        scores[NEG] = 1
        w_phi = structured(phi, scores).cpu()  # structured
        _, _, neg_onset, neg_offset = structured.predict(w_phi)
        neg_vots += [(neg_onset + 1, neg_offset + 1, NEG_STRING)]  # see Note at the bottom for the '+1'

        pred_vots += [pos_vots[-1] if vot_scores[POS] > vot_scores[NEG] else neg_vots[-1]]

        if debug and idx > DEBUG_MAX_SAMPLES:
            break
    return filenames, pos_vots, neg_vots, pred_vots


def basic_predict(model, structured, device, data_loader, debug=False):
    model.eval()
    structured.eval()

    filenames = []
    pred_vots = []

    for idx, (data, (filename, window_offset,_,_)) in enumerate(tqdm(data_loader)):

        filenames += [
            (filename[0], window_offset.item(), data.shape[1])]  # first dim is the batch size,2nd is len,3rd n_features

        data = data.to(device).float()
        phi = model(data)  # rnn

        scores = torch.Tensor([0, 0])
        w_phi = structured(phi, scores).cpu()  # structured
        _, _, onset, offset = structured.predict(w_phi)
        pred_vots += [(onset + 1, offset + 1)]  # see Note at the bottom for the '+1'

        if debug and idx > DEBUG_MAX_SAMPLES:
            break
    return filenames, pred_vots


def write_predictions(filenames, pos_vots, neg_vots, pred_vots, basic_preds, out_dir,durations):
    """
    output TextGrids and CSV with all the data
    :param filenames:
    :param pos_vots:
    :param neg_vots:
    :param pred_vots:
    :param out_dir:
    :return:
    """
    durations=  get_durations(durations)
    predictions = {}

    base_out_tg = "{}_pred.TextGrid"
    for idx in range(len(filenames)):
        filename, window_offset, window_length,labeled_vot,labeled_type = filenames[idx]
        out_tg = base_out_tg.format(os.path.join(out_dir, filename))
        # create_tg(out_tg, window_length, window_offset, [pos_vots[idx], neg_vots[idx], pred_vots[idx]],
        #           basic_preds[idx],labeled_vot,labeled_type)#TODO- basic
        create_tg_simple(out_tg, window_length, window_offset,pred_vots[idx],length= durations[filename])
    print("Predictions can be found at : [{}]".format(out_dir))

def write_csv(summary_path,filenames, preds_vots):
    with open(summary_path,'w') as f:
        f.write("filename, type, duration(msec)\n")
        for idx in range(len(filenames)):
            filename, _, _, _, _= filenames[idx]
            pred_onset,pred_offset,vot_type = preds_vots[idx]
            f.write("{},{},{}\n".format(filename,vot_type,pred_offset-pred_onset))






def get_model(path, basic=False):
    # creation
    model = VOT_Seg()
    structured = structured_layer()
    tagger = None
    if not basic:
        tagger = VOT_tagger()

    name = path.strip('.')
    if not os.path.exists(name + ".model"):
        assert "Couldnt find {} to load, path doesnt exist".format(name + ".model")
    if not os.path.exists(name + '.structured'):
        assert "Couldnt find {} to load, path doesnt exist".format(name + ".structured")
    if not basic and not os.path.exists(name + '.tagger'):
        assert "Couldnt find {} to load, path doesnt exist".format(name + ".tagger")

    print("Loading Models...")
    model.load_state_dict(torch.load(name + ".model", map_location=lambda storage, loc: storage))
    structured.load_state_dict(torch.load(name + ".structured", map_location=lambda storage, loc: storage))
    if not basic: tagger.load_state_dict(torch.load(name + ".tagger", map_location=lambda storage, loc: storage))
    print("Done Loading")

    return model, structured, tagger

def get_durations(durations_fname):
    #create dict of {file:duration} for matching textgrids
    print(f"reading <{durations_fname}> to get the duration of each file to match the generated textgrid...")
    durations = {}

    with open(durations_fname, 'r') as f:
        for line in f:
            k,_,v = line.strip().split(':') #fullpath,windows_starts,duration
            durations[os.path.basename(k.strip()).split('.')[0]] = float(v)
    return durations

parser = argparse.ArgumentParser(description='VOT segmentor Predictor')

parser.add_argument('--load', default="final_models/adv_model.", help='laod model')
# parser.add_argument('--load_basic', default="final_models/basic_model.", help='laod model')TODO- basic
parser.add_argument('--inference', default="features/", type=str, help='path to inference dir')
parser.add_argument('--out_dir', default="out_textgrids/", type=str, help='path to output dir for textgrids')
parser.add_argument('--debug', action='store_true', help='predict only 20 examples for short run-time')
parser.add_argument('--durations',type=str, default=None,required=True, help='path for voice_starts.txt that contains the duration of each file')

args = parser.parse_args()
try:
    assert os.path.exists(args.durations), f"Couldn't find {args.durations}"

    if not args.load:
        assert False, "must load a model"

    if args.out_dir=="./out_textgrids" and not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)
    if args.out_dir != None:
        assert os.path.isdir(args.out_dir), "Couldn't Find --out_dir path :< {} >\n".format(os.path.join(os.getcwd(),args.out_dir))
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = data_utils.get_inference_dataset(args.inference,args.debug)
    model, structured, tagger = get_model(args.load)
    # basic_model, basic_structured, _ = get_model(args.load_basic, basic=True)#TODO- basic

    filenames, pos_vots, neg_vots, pred_vots = predict(model, structured, tagger, device, test_loader, args.debug)
    # basic_filenames, basic_pred_vots = basic_predict(basic_model, basic_structured, device, test_loader, args.debug)# TODO- basic
    # write_predictions(filenames, pos_vots, neg_vots, pred_vots, basic_pred_vots, args.out_dir or args.inference)#TODO- basic
    write_predictions(filenames, pos_vots, neg_vots, pred_vots, None, args.out_dir or args.inference,durations= args.durations)
    write_csv(os.path.join(args.out_dir,"summary.csv"),filenames,pred_vots)

except Exception as e:
    print(f"Failed to run dr.VOT :{e}")
    exit(1)


