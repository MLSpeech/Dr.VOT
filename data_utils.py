from select import select

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import os
import numpy as np
import torch
import torch.optim as optim
import pickle
import argparse
from tqdm import tqdm
POS=1
NEG=0


__author__ = 'YosiShrem'

dataset_pickle = "data.pickle"
# index_dict = {'dmit': 0, 'natalia': 1, 'murphy': 2 ,'shultz':3}
index_dict = {}
def get_dict_len():
    return len(index_dict)
def init_index_dict(list_of_path):
    global index_dict
    i=0
    for path in list_of_path:
        if path.__contains__('dmit'):
            index_dict['dmit'] =i
            i+=1
        if path.__contains__('natalia'):
            index_dict['natalia'] =i
            i+=1
        if path.__contains__('murphy'):
            index_dict['murphy'] =i
            i+=1
        if path.__contains__('shultz'):
            index_dict['shultz'] =i
            i+=1

def get_dataset_index(path):
    """
    :return: dataset index
    """
    for k, v in index_dict.items():
        if path.__contains__(k):
            print("indexed [{}] as [{}]".format(k, v))
            return v
    assert False, "unknown data when indexing"


def extract_vot_and_tag(path):
    # vot - 2nd line and first 2 numbers
    # tag - 3rd row
    vot = np.fromstring(open(path).readlines()[1].strip(), sep=' ')[:2]
    tag = np.float(open(path).readlines()[2])
    return np.append(vot, tag)



def get_inference_dataset(dataset_path,debug=False):
    """
    for every file, read the features and the "offset from start" in the .labels,
    and return array of features and array of (filename,offset)
    [(features,(filename,offset)),..,]
    """

    if not os.path.exists(dataset_path):
        assert False, "Couldn't find path : '{}'".format(dataset_path)
    print("\nprocessing data :'{}'\n".format(dataset_path))

    path = os.getcwd()
    os.chdir(dataset_path)

    dataset = []
    for file in tqdm(os.listdir('.')):
        if not file.endswith('features'):
            continue
        name = file.replace(".features", "")  # removing "features"
        x = np.loadtxt(name + '.features')
        np.nan_to_num(x, copy=False)
        #get labels file
        if os.path.exists(name + '.test.labels'):
            labels_file = open(name + '.test.labels').readlines()
        elif os.path.exists(name + '.labels'):
            labels_file = open(name + '.labels').readlines()
        else:
            continue
        file_info = (name , float(labels_file[-2].split(' ')[-1]),
                     np.fromstring(labels_file[1].strip(), sep=' ')[:2],
                     float(labels_file[2]))#(file name,window_offset,(onset,offset),vot_type)

        dataset.append([torch.from_numpy(x).float(), file_info])
        if debug and len(dataset)>100:
            break
    os.chdir(path)

    return DataLoader(dataset,shuffle=False)

def read_txt_labels(dataset_path, re_read):
    # check if already exists pickle
    if not os.path.exists(dataset_path):
        assert False, "Couldn't find path : '{}'".format(dataset_path)
    if os.path.exists(os.path.join(dataset_path, dataset_pickle)) and not re_read:
        with open(os.path.join(dataset_path, dataset_pickle), 'rb') as f:
            return pickle.load(f)

    # create pickle
    dataset = []
    path = os.getcwd()
    print("\nprocessing data :'{}'\n".format(dataset_path))
    os.chdir(dataset_path)
    for file in tqdm(os.listdir('.')):
        if not file.endswith('features'):
            continue
        name = file.replace(".features", "")  # removing "features"
        x = np.loadtxt(name + '.features')
        y = extract_vot_and_tag(name + '.labels')

        if y[1] > len(x):#check if label can be reached,i.e. the vot is in the search window x
            continue

        np.nan_to_num(x, copy=False)
        dataset.append((torch.from_numpy(x).float(), torch.from_numpy(y)))
    with open(dataset_pickle, 'wb') as f:
        print("Saving data...."),
        pickle.dump(dataset, f)
    os.chdir(path)
    return dataset
#np.sort(np.array([(y[1]-y[0]).item() if y[2]>0 else 30 for x,y in dataset])) to get sorted vot length



class ImbalancedDatasetSampler(Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset)))

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        #return dataset.data[idx][1][-1].item()
        return dataset.get_dataset_index(idx), dataset.data[idx][1][-1].item()

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class VOT_dataset(Dataset):
    def __init__(self, list_of_paths, re_read=False, index_datasets=False):
        """
        :param add_dataset_number: used for adversarial training.
                assign id_number for every dataset in the list of paths
        """
        self.data = []
        self.boundries = {}
        self.index_datasets=index_datasets
        # handles 1 element and not list
        list_of_paths = list_of_paths if isinstance(list_of_paths, list) else [list_of_paths]
        init_index_dict(list_of_paths)
        for path in list_of_paths:
            dataset = read_txt_labels(path, re_read)
            if index_datasets:
                index = get_dataset_index(path)
                self.boundries[index] = (len(self.data), len(dataset) + len(self.data) - 1)
            self.data += dataset

        self.len = len(self.data)
        if not self.index_datasets:
            self.boundries[0] = (0,self.len)

    def get_dataset_index(self, idx):
        for k, v in self.boundries.items():
            if idx >= v[0] and idx <= v[1]:
                return k
        assert False, "Failed to get dataset index,\n boundries: {}\n asked for idx: [{}]".format(self.boundries, idx)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        if self.index_datasets:
            return self.data[item][0], torch.cat((self.data[item][1],
                                               torch.DoubleTensor([self.get_dataset_index(item)])), 0)
        return self.data[item]
    # def add_dataset_id(self):


def get_dataloader(list_of_paths, train=True, re_read=False, index_datasets=False):
    data = VOT_dataset(list_of_paths, re_read, index_datasets)
    if train:
        dataloader = DataLoader(data, batch_size=1, sampler=ImbalancedDatasetSampler(data), num_workers=4)
        num_of_ft = dataloader.dataset[0][0].shape[1]
    else:
        dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=4)
        num_of_ft = dataloader.dataset[0][0].shape[1]

    return dataloader, num_of_ft



def run_check(dataloaders):
    for dataloader in dataloaders:
        for x, y in dataloader:
            if len(y.view(-1)) < 2:
                assert ("label too short < 2")

            if y.view(-1)[-1] > len(x.view(-1, x.shape[-1])):
                assert ("target frame is bigger than len(x)")




if __name__ == '__main__':
    print("utils")
    # for data in [bb_train, bb_val, bb_test]:
    #     print("re-save [{}]".format(data))
    #     dataset = VOT_dataset(data, True)
    #     dataloader = DataLoader(dataset, batch_size=1,
    #                             shuffle=True, num_workers=4)

    # testing adding pos/neg to labels
    # path = "val/"
    parser = argparse.ArgumentParser(description='data process')
    parser.add_argument('--path', type=str, required=True, nargs='+', help='path to the data folder')
    parser.add_argument('--index', action='store_true', help='index data sets')
    parser.add_argument('--no_re_read', action='store_true', help='dont re-read')
    args = parser.parse_args()
    # path = "/data/shremjo/vot/new_processed/dmitrieva/neg/"
    dataset = VOT_dataset(args.path, re_read=not args.no_re_read, index_datasets=args.index)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=4, sampler=ImbalancedDatasetSampler(dataset))

    s = 0
    k = 0
    for i, j in dataloader:
        s += j.view(-1)[-1].item()
        k += 1
    print(s)
    print("total :", k)