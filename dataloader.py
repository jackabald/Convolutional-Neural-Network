# python imports
import os
import pickle
import hashlib
import urllib
import tarfile
import shutil
import time
from PIL import Image
from tqdm import tqdm

# torch imports
import torch
from torch.utils import data


data_urls = {"data": "http://miniplaces.csail.mit.edu/data/data.tar.gz",
             "train": "http://raw.githubusercontent.com/CSAILVision/miniplaces/master/data/train.txt",
             "val": "http://raw.githubusercontent.com/CSAILVision/miniplaces/master/data/val.txt"}
data_md5 = "265825ec94f79390e4f1e38045a69059"


def calculate_md5(fpath, chunk_size=1024*1024):
    md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()

def gen_bar_updater():
    pbar = tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update

def download_url(url, folder):
    """Download a file from a url and place it in folder.
    Args:
        url (str): URL to download file from
        folder (str): Directory to place downloaded file in
    """
    fpath = os.path.join(os.path.expanduser(folder),
                         os.path.basename(url))

    os.makedirs(os.path.expanduser(folder), exist_ok=True)

    if os.path.exists(fpath):
        return

    try:
        print('Downloading ' + url + ' to ' + fpath)
        urllib.request.urlretrieve(
            url, fpath,
            reporthook=gen_bar_updater()
        )
    except (urllib.error.URLError, IOError) as err:
        print('Failed download.')
        raise err
    return

def extract_targz(src_file, dst_path):
    # create dst folder / extract all files
    print('Extracting ' + src_file + ' to' + dst_path)
    os.makedirs(os.path.expanduser(dst_path), exist_ok=True)
    with tarfile.open(src_file, 'r:gz') as tar:
        tar.extractall(path=dst_path)

class MiniPlaces(data.Dataset):
    """
    A simple dataloader for mini places
    """
    def __init__(self,
                 root,
                 label_file=None,
                 num_classes=100,
                 download=False,
                 split="train",
                 transform=None):
        assert split in ["train", "val", "test"]
        # root folder, split
        self.root_folder = os.path.join(root, "miniplaces")
        self.split = split
        self.transform = transform
        self.n_classes = num_classes

        # download dataset
        if download:
            self._download_dataset(root)

        # load all labels
        if label_file is None:
            label_file = os.path.join(self.root_folder, split + '.txt')
        if not os.path.exists(label_file):
            raise ValueError(
                'Label file {:s} does not exist!'.format(label_file))
        with open(label_file) as f:
            lines = f.readlines()

        # store the file list
        file_label_list = []
        for line in lines:
            filename, label_id = line.rstrip('\n').split(' ')
            label_id = int(label_id)
            filename = os.path.join(self.root_folder, filename)
            file_label_list.append((filename, label_id))

        self.img_label_list = self._load_dataset(file_label_list)

    def _download_dataset(self, data_folder):
        # data folder and data file
        data_folder = os.path.expanduser(data_folder)
        data_file = os.path.join(data_folder,
                                 os.path.basename(data_urls['data']))

        # if we need to download the full dataset
        require_download = True
        if os.path.exists(data_file):
            file_md5 = calculate_md5(data_file)
        else:
            file_md5 = None
        if file_md5 == data_md5:
            require_download = False

        if (not require_download) and \
           os.path.exists(os.path.join(data_folder, 'miniplaces')):
            # only download the annotations
            download_url(data_urls[self.split],
                         os.path.join(data_folder, 'miniplaces'))
        else:
            # corner case: a corrupted file
            if os.path.exists(data_file) and (file_md5 != data_md5):
                print("File corrupted. Remove and re-download ...")
                os.remove(data_file)
            # corner case: the subfolder already exists
            if os.path.exists(os.path.join(data_folder, 'miniplaces')):
                shutil.rmtree(os.path.join(data_folder, 'miniplaces'))
            # download and extract the tar.gz file
            download_url(data_urls['data'], data_folder)
            extract_targz(data_file, data_folder)
            # setup the folders
            print("Setting up data folders ...")
            shutil.move(os.path.join(data_folder, 'images'),
                        os.path.join(data_folder, 'miniplaces'))
            shutil.rmtree(os.path.join(data_folder, 'objects'))
            # download the annotations
            download_url(data_urls[self.split],
                         os.path.join(data_folder, 'miniplaces'))
        return

    def _load_dataset(self, file_label_list):
        cached_filename = os.path.join(self.root_folder,
                                       'cached_{:s}.pkl'.format(self.split))
        if os.path.exists(cached_filename):
            # load dataset into memory
            print("=> Loading from cached file {:s} ...".format(cached_filename))
            try:
                img_label_list = pickle.load(open(cached_filename, "rb"))
            except (RuntimeError, TypeError, NameError):
                print("Can't load cached file. Please remove the file and rebuild the cache!")
        else:
            # load dataset into memory
            print("Loading {:s} set into memory. This might take a while ...".format(self.split))
            img_label_list = tuple()
            for filename, label_id in tqdm(file_label_list):
                img = Image.open(filename).convert('RGB')
                img = img.resize((32, 32), Image.BILINEAR)
                label = label_id
                img_label_list += ((img, label), )
            pickle.dump(img_label_list, open(cached_filename, "wb"))
        return img_label_list

    def __len__(self):
        return len(self.img_label_list)

    def __getitem__(self, index):
        # load img and label
        img, label = self.img_label_list[index]

        # apply data augmentation
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def get_index_mapping(self):
        # load the train label file
        train_label_file = os.path.join(self.root_folder, self.split + '.txt')
        if not os.path.exists(train_label_file):
            raise ValueError(
                'Label file {:s} does not exist!'.format(train_label_file))
        with open(train_label_file) as f:
            lines = f.readlines()

        # get the category names
        id_index_map = {}
        for line in lines:
            filename, label_id = line.rstrip('\n').split(' ')
            cat_name = filename.split('/')[-2]
            id_index_map[label_id] = cat_name

        # return a dictionary that maps an ID to its category name
        return id_index_map
