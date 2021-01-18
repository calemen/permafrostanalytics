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
import zarr

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
import dask

from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go

import stuett
from stuett.global_config import get_setting, setting_exists, set_setting


import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset

from pathlib import Path

from PIL import Image

import numpy as np
import json
import pandas as pd
import os
from skimage import io as imio
import io, codecs
import time

import csv

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
parser.add_argument(
    "--initial_env",
    action="store_true",
    default=False,
    help="Inital hackathon virtual environment is used",
)
parser.add_argument(
    "--copy_seismic_data",
    action="store_true",
    default=True,
    help="Copy seismic raw data to local scratch before running and use this data as source. If folder seismic_data already exists it will not copy again",
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
    prefix_seismic = "seismic_data/4D/"
elif args.classifier == "image":
    prefix="timelapse_images_fast"
else:
    raise RuntimeError("Please specify either 'seismic', 'image' or 'wind' classifier")

if args.reload_all:
    args.reload_frozen = True

if args.initial_env:
    from datasets import DatasetFreezer, DatasetMerger
    from stuett.data import (
    SeismicSource,
    DataCollector,
    MinMaxDownsampling,
    GSNDataSource,
    LTTBDownsampling,
    MHDSLRImages,
    MHDSLRFilenames,
    Spectrogram,
    Freezer
    )
    from stuett.convenience import (
        DirectoryStore
    )
else:
    from stuett.data import Freezer, Spectrogram, Rescale, To_db, To_Tensor
    from stuett.data import DatasetFreezer, DatasetMerger

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

if args.initial_env:
    class Rescale(stuett.core.StuettNode):
        def __init__(
            self,
            dim=None,
        ):
            super().__init__(
                dim=dim
            )

        def forward(self, data=None, request=None):
            data = data/data.max()
            return data

    class To_db(stuett.core.StuettNode):
        def __init__(
            self,
            dim=None,
        ):
            super().__init__(
                dim=dim
            )

        def forward(self, data=None, request=None, min_value=1e-10, reference=1.0):
            data = 10.0 * xr.ufuncs.log10(xr.ufuncs.maximum(min_value, data))
            data -= 10.0 * xr.ufuncs.log10(xr.ufuncs.maximum(min_value, reference))
            return data

    class To_Tensor(stuett.core.StuettNode):
        def __init__(
            self,
            dim=None,
        ):
            super().__init__(
                dim=dim
            )

        def forward(self, data=None, request=None):
            data = data.values.squeeze()
            return Tensor(data)

###### SELECTING A CLASSIFIER TYPE ##############
#################################################
# Load the data source
def load_seismic_source():
    seismic_channels = ["EHE", "EHN", "EHZ"]
    seismic_node = stuett.data.SeismicSource(
        store=store, station="MH36", channel=seismic_channels,
    )
    return seismic_node, len(seismic_channels)

def load_image_source():
    image_node = stuett.data.MHDSLRFilenames(
        store=store, force_write_to_remote=True, as_pandas=False,
    )
    return image_node, 3

if args.classifier == "image":
    if args.initial_env:
        from datasets import ImageDataset as Dataset
    else:
        from stuett.data import ImageDataset as Dataset

    transform = None
    data_node, num_channels = load_image_source()
elif args.classifier == "wind" or args.classifier == "seismic":
    if args.initial_env:
        from datasets import SeismicDataset as Dataset
    else:
        from stuett.data import SeismicDataset as Dataset

    transform = get_seismic_transform()
    data_node, num_channels = load_seismic_source()


############# LOADING DATASET ###################
#################################################
bypass_freeze = not args.use_frozen

if args.classifier == "image" or args.classifier == "seismic":
    dataset_slice = {"time": slice("2017-03-01", "2017-03-14")}
    batch_dims = {"time": stuett.to_timedelta(60, "minutes")}
elif args.classifier == "wind":
    dataset_slice={"time": slice("2017-01-01", "2017-12-31")}
    batch_dims={"time": stuett.to_timedelta(60, "minutes")}

# seismic_channels = ["EHE", "EHN", "EHZ"]
# seismic_data_node = stuett.data.SeismicSource(store=store, station="MH36", channel=seismic_channels, start_time="2017-08-01", end_time="2017-09-01")
# seismic_data = seismic_data_node()
# print(seismic_data)

evaluation_starttime = dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

local_scratch_path = Path("/scratch/barthc/")
local_scratch_path_tmp = local_scratch_path.joinpath(f"{evaluation_starttime}")
local_scratch_data_path = local_scratch_path.joinpath("data")
os.makedirs(local_scratch_data_path, exist_ok=True)
if args.copy_seismic_data:
    seismic_zip_name = "seismic_data.zip"
    seismic_data_path = local_scratch_data_path.joinpath("seismic_data")
    if not os.path.isfile(local_scratch_data_path.joinpath(seismic_zip_name)):
        print("Copying zip file")
        shutil.copy2(Path(data_path).joinpath(seismic_zip_name), local_scratch_data_path)
    if not os.path.isdir(seismic_data_path):
        print("Extracting zip file")
        with zipfile.ZipFile(local_scratch_data_path.joinpath(seismic_zip_name), "r") as zip_file:
            zip_file.extractall(seismic_data_path)
    store_seismic = stuett.DirectoryStore(seismic_data_path.joinpath(prefix_seismic))
    if ("MH36/2017/EHE.D/4D.MH36.A.EHE.D.20170101_000000.miniseed" not in store_seismic):
            raise RuntimeError(f"Please provide a valid path to the permafrost {prefix_seismic} data or see README how to download it")
freeze_store_path = local_scratch_path_tmp.joinpath("frozen", "FreezerNodeEvaluation")
freeze_store = stuett.DirectoryStore(freeze_store_path)
results_path = tmp_dir.joinpath("FreezerNode_evaluation")
os.makedirs(results_path, exist_ok=True)
chunk_size = 1
#synchronizer=zarr.ThreadSynchronizer()
synchronizer=zarr.ProcessSynchronizer(path=freeze_store_path)

label_list_file = None

max_workers = 4
min_workers = 4
num_iterations = 10
results = []
results.append(["iteration","Data node","num_workers", "use_frozen", "storing/loading","Calculation time formatted", "Calculation time raw"])

dask.config.set(scheduler='single-threaded')

try:
    for num_workers in range(min_workers,max_workers+1):
        for use_frozen in [True]:
            bypass_freeze = not use_frozen

            data_node, num_channels = load_seismic_source()
            if args.initial_env:
                freezer_node = Freezer(store=freeze_store, groupname="classifier", dim="time", offset=pd.to_timedelta("60 minutes"))
            else:
                freezer_node = Freezer(store=freeze_store, groupname="classifier", dim="time", offset=pd.to_timedelta("60 minutes"), synchronizer=synchronizer, bypass_freeze=bypass_freeze)
            rescale_node = Rescale()
            spectogram_node = Spectrogram(nfft=512, stride=512, dim="time", sampling_rate=250)
            to_db_node = To_db()
            to_tensor_node = To_Tensor()
            data_node_freezer = lambda x: to_tensor_node(freezer_node(to_db_node(spectogram_node(rescale_node(x, delayed=True), delayed=True), delayed=True), delayed=True), delayed=True)
                #data_node_freezer = to_tensor_node(freezer_node(to_db_node(spectogram_node(rescale_node(data_node, delayed=True), delayed=True), delayed=True), delayed=True), delayed=True)

            train_dataset_freezernode = Dataset(
                label_list_file=label_list_file,
                transform=data_node_freezer,
                store=store,
                mode="train",
                label=label,
                data=data_node,
                dataset_slice=dataset_slice,
                batch_dims=batch_dims,
            )

            # Set up pytorch data loaders
            shuffle = False
            train_sampler = None
            train_loader_freezernode = DataLoader(
                train_dataset_freezernode,
                batch_size=args.batch_size,
                shuffle=shuffle,
                sampler=train_sampler,
                # drop_last=True,
                num_workers=num_workers,
            )

            validation_sampler = None

            train_dataset_raw = Dataset(
                label_list_file=label_list_file,
                transform=transform,
                store=store,
                mode="train",
                label=label,
                data=data_node,
                dataset_slice=dataset_slice,
                batch_dims=batch_dims,
            )

            for iteration in range(num_iterations): #zeit pro anzahl samples (wenn zu klein dann pro 1000 samples)
                try:
                    shutil.rmtree(freeze_store_path)
                except:
                    pass
                for j in ["storing", "loading"]:
                    starttime = time.time()
                    for i, data in enumerate(tqdm(train_loader_freezernode), 0):
                        pass
                    endtime = time.time()
                    time_tmp = endtime - starttime
                    time_tmp_formatted = time.strftime('%H:%M:%S', time.gmtime(time_tmp))
                    # total_size = 0
                    # for root, dirs, files in os.walk(freeze_store_path):
                    #     #print(f"root: {root}, dirs: {dirs}, files: {files}")
                    #     for f in files:
                    #         total_size += os.path.getsize(os.path.join(root, f))
                    # total_size = total_size / 1024
                    print(f"j: {j}, time: {time_tmp_formatted}")
                    results.append([iteration, "FreezerNode", num_workers, use_frozen, j, time_tmp_formatted, time_tmp])
            
            # try:

            # for iteration in range(num_iterations):
            #     try:
            #         shutil.rmtree(dataset_freezer_path)
            #     except:
            #         pass
            #     for j in ["storing", "loading"]:
            #         starttime = time.time()
            #         train_frozen = DatasetFreezer(
            #         train_dataset_raw, path=dataset_freezer_path, bypass=bypass_freeze
            #         )
            #         train_frozen.freeze(reload=False)

            #         train_loader_datasetfreezer = DataLoader(
            #             train_frozen,
            #             batch_size=args.batch_size,
            #             shuffle=shuffle,
            #             sampler=train_sampler,
            #             # drop_last=True,
            #             num_workers=num_workers,
            #         )
            #         for i, data in enumerate(tqdm(train_loader_datasetfreezer), 0):
            #             pass
            #         endtime = time.time()
            #         time_tmp = endtime - starttime
            #         time_tmp_formatted = time.strftime('%H:%M:%S', time.gmtime(time_tmp))
            #         # total_size = 0
            #         # for root, dirs, files in os.walk(dataset_freezer_path):
            #         #     #print(f"root: {root}, dirs: {dirs}, files: {files}")
            #         #     for f in files:
            #         #         total_size += os.path.getsize(os.path.join(root, f))
            #         # total_size = total_size / 1024
            #         print(f"j: {j}, time: {time_tmp_formatted}")
            #         results.append([iteration, "DatasetFreezer", num_workers, use_frozen, j, time_tmp_formatted, time_tmp])
            train_loader_raw = DataLoader(
                        train_dataset_raw,
                        batch_size=args.batch_size,
                        shuffle=shuffle,
                        sampler=train_sampler,
                        # drop_last=True,
                        num_workers=num_workers,
                    )
            for iteration in range(num_iterations):
                for j in ["loading"]:
                    starttime = time.time()

                    for i, data in enumerate(tqdm(train_loader_raw), 0):
                        pass
                    endtime = time.time()
                    time_tmp = endtime - starttime
                    time_tmp_formatted = time.strftime('%H:%M:%S', time.gmtime(time_tmp))
                    # total_size = 0
                    # for root, dirs, files in os.walk(dataset_freezer_path):
                    #     #print(f"root: {root}, dirs: {dirs}, files: {files}")
                    #     for f in files:
                    #         total_size += os.path.getsize(os.path.join(root, f))
                    # total_size = total_size / 1024
                    print(f"j: {j}, time: {time_tmp_formatted}")
                    results.append([iteration, "Baseline", num_workers, use_frozen, j, time_tmp_formatted, time_tmp])
except Exception as e:
    import traceback
    print(traceback.format_exc())

with open(results_path.joinpath(f"{evaluation_starttime}_results.csv"),"w") as f:
    wr = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC,delimiter=",", lineterminator='\n')
    wr.writerow(results)
print(f"Resultsfile written: {evaluation_starttime}_results.csv")
try:
    shutil.rmtree(local_scratch_path_tmp)
except:
    pass
