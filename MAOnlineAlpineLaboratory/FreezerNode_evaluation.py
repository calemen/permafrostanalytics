"""MIT License

Copyright (c) 2019, Swiss Federal Institute of Technology (ETH Zurich), Matthias Meyer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

import stuett
import torch
import numpy as np
import scipy
import argparse
import datetime as dt
import os
import pandas as pd
import xarray as xr

from datasets import DatasetFreezer, DatasetMerger
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import traceback
import sys
import shutil

from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go

import stuett
from stuett.global_config import get_setting, setting_exists, set_setting
from stuett.data import Freezer, Spectrogram, Rescale, To_db, To_Tensor

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from ignite.metrics import Accuracy

from pathlib import Path

from PIL import Image

import numpy as np
import json
import pandas as pd
import os
from skimage import io as imio
import io, codecs
import time

from models import SimpleCNN

parser = argparse.ArgumentParser(description="Pytorch Neural Network Classification")
parser.add_argument(
    "--classifier",
    type=str,
    default="seismic",
    help="Classification type either 'seismic', 'image' or 'wind'",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=16,
    help="input batch size for training (default: 16)",
)
parser.add_argument(
    "--epochs", type=int, default=2, help="number of epochs to train (default: 500)"
)
parser.add_argument(
    "--lr", type=float, default=0.001, help="learning rate for optimizer"
)
parser.add_argument(
    "--linear_decrease_start_epoch",
    type=int,
    default=1,
    help="At which epoch to start the linear decrease",
)
parser.add_argument(
    "--use_frozen",
    action="store_true",
    default=False,
    help="Using cached/preprocessed dataset",
)
parser.add_argument(
    "--reload_frozen",
    action="store_true",
    default=False,
    help="Reloads the cached/preprocessed dataset",
)
parser.add_argument(
    "--reload_all",
    action="store_true",
    default=False,
    help="Reloads the cached/preprocessed dataset, the labels",
)
parser.add_argument(
    "--resume", type=str, default=None, help="Resume from given model checkpoint"
)
parser.add_argument(
    "--augment", action="store_true", default=False, help="augment data at runtime"
)
parser.add_argument(
    "--tmp_dir",
    default=str(
        Path(__file__).absolute().parent.joinpath("..", "data", "user_dir", "tmp")
    ),
    help="folder to store logs and model checkpoints",
)
parser.add_argument(
    "--run_id",
    default=dt.datetime.now().strftime("%Y%m%d-%H%M%S"),
    help="id for this run. If not provided it will be the current timestamp",
)
parser.add_argument(
    "-p",
    "--path",
    type=str,
    default=str(Path(__file__).absolute().parent.joinpath("..", "data")),
    help="The path to the folder containing the permafrost hackathon data",
)
parser.add_argument(
    "-l",
    "--local",
    action="store_true",
    default=True,
    help="Only use local files and not data from Azure",
)
parser.add_argument(
    "--annotations_path",
    type=str,
    default=str(Path(__file__).absolute().parent.joinpath("..", "data", "annotations")),
    help="The path to the folder containing the annotation data",
)
args = parser.parse_args()

################## PARAMETERS ###################
#################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)
data_path = Path(args.path)
annotations_path = Path(args.annotations_path)
label_filename = "automatic_labels_mountaineers.csv" #"annotations.csv"
tmp_dir = Path(args.tmp_dir)
os.makedirs(tmp_dir, exist_ok=True)


if args.classifier == "wind" or args.classifier == "seismic":
    prefix = "seismic_data/4D/" #"geophones/binaries/4D/"
elif args.classifier == "image":
    prefix="timelapse_images_fast"
else:
    raise RuntimeError("Please specify either 'seismic', 'image' or 'wind' classifier")

if args.reload_all:
    args.reload_frozen = True

############ SETTING UP DATA LOADERS ############
#################################################
if not args.local:
    from stuett.global_config import get_setting, setting_exists

    account_name = (
        get_setting("azure")["account_name"]
        if setting_exists("azure")
        else "storageaccountperma8980"
    )
    account_key = (
        get_setting("azure")["account_key"] if setting_exists("azure") else None
    )
    store = stuett.ABSStore(
        container="hackathon-on-permafrost",
        prefix=prefix,
        account_name=account_name,
        account_key=account_key,
    )
    annotation_store = stuett.ABSStore(
        container="hackathon-on-permafrost",
        prefix="annotations",
        account_name=account_name,
        account_key=account_key,
    )
else:
    store = stuett.DirectoryStore(Path(data_path).joinpath(prefix))

    print(Path(data_path).joinpath(prefix))
    
    if ("MH36/2017/EHE.D/4D.MH36.A.EHE.D.20170101_000000.miniseed" not in store):
        raise RuntimeError(f"Please provide a valid path to the permafrost {prefix} data or see README how to download it")
    annotation_store = stuett.DirectoryStore(annotations_path)
    if label_filename not in annotation_store:
        print(
            "WARNING: Please provide a valid path to the permafrost annotation data or see README how to download it"
        )

################## Transform ################
#################################################
def get_seismic_transform():
    #obspy.detrend?
    def to_db(x, min_value=1e-10, reference=1.0):
        value_db = 10.0 * xr.ufuncs.log10(xr.ufuncs.maximum(min_value, x))
        value_db -= 10.0 * xr.ufuncs.log10(xr.ufuncs.maximum(min_value, reference))
        return value_db

    spectrogram = stuett.data.Spectrogram(
        nfft=512, stride=512, dim="time", sampling_rate=250
    )

    transform = transforms.Compose(
        [
            lambda x: x / x.max(),  # rescale to -1 to 1
            spectrogram,  # spectrogram
            lambda x: to_db(x).values.squeeze(),
            lambda x: Tensor(x),
        ]
    )

    return transform

def get_image_transform():
    # TODO: add image transformations
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    transform = transforms.Compose([transforms.ToTensor(), normalize])

    return transform

########## Annotation Balancing #################
#################################################
# Load the labels
label = stuett.data.BoundingBoxAnnotation(
    filename=label_filename, store=annotation_store
)()
#print(label)
# we are not interest in any x or y position (since there are none in the labels)
label = label.drop_vars(["start_x", "end_x", "start_y", "end_y"])

# Currently, the dataset contains of only one label 'mountaineer'
# The labelled section without mountaineer outnumber the sections with one (approx 10:1)
# To train succesfully we need a balanced dataset of positive and negative examples
# Here, we balance it by choosing number of random non-mountaineer sections which
# is approximatly the same number as the mountaineer sections.
# NOTE: Adjust this section if you want to train with different label classes!!
#no_label_mask = label.isnull()
#label_mask = label.notnull()
#ratio = (no_label_mask.sum() / label_mask.sum()).values.astype(int)
#no_label_indices = np.argwhere(no_label_mask.values)[::ratio].squeeze()
#label_mask[no_label_indices] = True
#label = label[label_mask]
print("Number of labels which are checked against the data: ", len(label))

# here we load a predefined list from our server
# If you want to regenerate your list add reload_all as an argument to the script
label_list_file = tmp_dir.joinpath(f"{args.classifier}_list.csv").resolve()
if not label_list_file.exists() and not args.reload_all:
    # save in store
    with open(label_list_file, "wb") as f:
        f.write(annotation_store[f"{args.classifier}_list.csv"])


###### SELECTING A CLASSIFIER TYPE ##############
#################################################
# Load the data source
def load_seismic_source():
    seismic_channels = ["EHE", "EHN", "EHZ"]
    seismic_node = stuett.data.SeismicSource(
        store=store, station="MH36", channel=seismic_channels, start_time="2017-01-01", end_time="2017-01-31"
    )
    return seismic_node, len(seismic_channels)

def load_image_source():
    image_node = stuett.data.MHDSLRFilenames(
        store=store, force_write_to_remote=True, as_pandas=False,
    )
    return image_node, 3


if args.classifier == "image":
    from datasets import ImageDataset as Dataset

    transform = None
    data_node, num_channels = load_image_source()
elif args.classifier == "wind" or args.classifier == "seismic":
    from datasets import SeismicDataset as Dataset

    transform = get_seismic_transform()
    data_node, num_channels = load_seismic_source()


############# LOADING DATASET ###################
#################################################
bypass_freeze = not args.use_frozen

if args.classifier == "image" or args.classifier == "seismic":
    dataset_slice = {"time": slice("2017-01-01", "2017-01-31")}
    batch_dims = {"time": stuett.to_timedelta(10, "minutes")}
elif args.classifier == "wind":
    dataset_slice={"time": slice("2017-01-01", "2017-12-31")}
    batch_dims={"time": stuett.to_timedelta(60, "minutes")}

# seismic_channels = ["EHE", "EHN", "EHZ"]
# seismic_data_node = stuett.data.SeismicSource(store=store, station="MH36", channel=seismic_channels, start_time="2017-08-01", end_time="2017-09-01")
# seismic_data = seismic_data_node()
# print(seismic_data)
freeze_store_path = tmp_dir.joinpath("frozen", "FreezerNodeEvaluation")
freeze_store = stuett.DirectoryStore(freeze_store_path)
results_path = tmp_dir.joinpath("FreezerNode_evaluation")
results = {}
os.makedirs(results_path, exist_ok=True)
max_chunk = 6

evaluation_starttime = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
try:
    shutil.rmtree(freeze_store_path)
except:
    pass

for chunk_size in range(1,max_chunk+1):
    for use_frozen in [True, False]:
        bypass_freeze = not use_frozen

        data_node, num_channels = load_seismic_source()
        freezer_node = Freezer(store=freeze_store, groupname="classifier", dim="time", offset=pd.to_timedelta("10 minutes"), store_chunk_size=chunk_size, bypass_freeze=bypass_freeze)
        rescale_node = Rescale()
        spectogram_node = Spectrogram(nfft=512, stride=512, dim="time", sampling_rate=250)
        to_db_node = To_db()
        to_tensor_node = To_Tensor()
        data_node = to_tensor_node(freezer_node(to_db_node(spectogram_node(rescale_node(data_node(delayed=True), delayed=True), delayed=True), delayed=True), delayed=True), delayed=True)

        train_dataset = Dataset(
            label_list_file=label_list_file,
            transform=None,
            store=store,
            mode="train",
            label=label,
            data=data_node,
            dataset_slice=dataset_slice,
            batch_dims=batch_dims,
        )
        test_dataset = Dataset(
            label_list_file=label_list_file,
            transform=None,
            store=store,
            mode="test",
            label=label,
            data=data_node,
            dataset_slice=dataset_slice,
            batch_dims=batch_dims,
        )

        # Set up pytorch data loaders
        shuffle = True
        train_sampler = None
        num_workers = 0
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            sampler=train_sampler,
            # drop_last=True,
            num_workers=num_workers,
        )

        validation_sampler = None
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            sampler=validation_sampler,
            # drop_last=True,
            num_workers=num_workers,
        )

        for j in [1, 2]:
            starttime = time.time()
            for i, data in enumerate(tqdm(train_loader), 0):
                pass
            for i, data in enumerate(tqdm(test_loader), 0):
                pass
            endtime = time.time()
            time_tmp = endtime - starttime
            time_tmp = time.strftime('%H:%M:%S', time.gmtime(time_tmp))
            total_size = 0
            for root, dirs, files in os.walk(freeze_store_path):
                #print(f"root: {root}, dirs: {dirs}, files: {files}")
                for f in files:
                    total_size += os.path.getsize(os.path.join(root, f))
            total_size = total_size / 1024
            print(f"j: {j}, time: {time_tmp}, total_size: {total_size}kb")
            results[f"chunk_size:{chunk_size}, use_frozen:{use_frozen}, iteration:{j}"] = [time_tmp, total_size]
        if chunk_size < max_chunk:
            try:
                shutil.rmtree(freeze_store_path)
            except:
                pass
f = open(results_path.joinpath(f"{evaluation_starttime}_results.txt"),"w")
f.write( str(results) )
f.close()
