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
import dask

from tqdm import tqdm

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

parser = argparse.ArgumentParser(description="Pytorch Neural Network Classification")
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
args = parser.parse_args()

################## PARAMETERS ###################
#################################################
data_path = Path(args.path)
tmp_dir = Path(args.tmp_dir)
os.makedirs(tmp_dir, exist_ok=True)

prefix_seismic = "seismic_data/4D/" #"geophones/binaries/4D/"
prefix_image = "timelapse_images_fast"

############ SETTING UP DATA LOADERS ############
#################################################
if not args.local:
    account_name = (
        get_setting("azure")["account_name"]
        if setting_exists("azure")
        else "storageaccountperma8980"
    )
    account_key = (
        get_setting("azure")["account_key"] if setting_exists("azure") else None
    )
    store_seismic = stuett.ABSStore(
        container="hackathon-on-permafrost",
        prefix=prefix_seismic,
        account_name=account_name,
        account_key=account_key,
    )
    store_image = stuett.ABSStore(
        container="hackathon-on-permafrost",
        prefix=prefix_image,
        account_name=account_name,
        account_key=account_key,
    )
else:
    store_seismic = stuett.DirectoryStore(Path(data_path).joinpath(prefix_seismic))
    store_image = stuett.DirectoryStore(Path(data_path).joinpath(prefix_image))

    print(Path(data_path).joinpath(prefix_seismic))
    
    if ("MH36/2017/EHE.D/4D.MH36.A.EHE.D.20170101_000000.miniseed" not in store_seismic):
        raise RuntimeError(f"Please provide a valid path to the permafrost {prefix_seismic} data or see README how to download it")

###### Load the data source ##############
#################################################
def load_seismic_source():
    seismic_channels = ["EHE", "EHN", "EHZ"]
    seismic_node = stuett.data.SeismicSource(
        store=store_seismic, station="MH36", channel=seismic_channels, start_time="2017-01-01", end_time="2017-01-31"
    )
    return seismic_node

def load_image_source():
    image_node = stuett.data.MHDSLRFilenames(
        store=store_image, force_write_to_remote=True, as_pandas=False,
    )
    return image_node


evaluation_starttime = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

local_scratch_path = Path("/scratch/barthc/")
local_scratch_path_tmp = local_scratch_path.joinpath(f"{evaluation_starttime}")
freeze_store_path = local_scratch_path_tmp.joinpath("frozen", "FreezerNodeEvaluation")
freeze_store = stuett.DirectoryStore(freeze_store_path)
results_path = tmp_dir.joinpath("FreezerNode_evaluation")
results = []
results.append(["iteration","Data node","storing/loading","Calculation time formatted", "Calculation time raw"])
os.makedirs(results_path, exist_ok=True)
max_workers = 8
num_iterations = 10
synchronizer=zarr.ThreadSynchronizer()
#synchronizer=zarr.ProcessSynchronizer(path=freeze_store_path)

data_node_seismic = load_seismic_source()
data_node_image = load_image_source()


start_time = f"2017-03-01 00:00"
end_time = f"2017-03-01 00:59"
start_time = pd.to_datetime(start_time)
end_time = pd.to_datetime(end_time)

request = {"start_time":start_time, "end_time":end_time}

try:
    for offset in [pd.to_timedelta("10 seconds"), pd.to_timedelta("1 minutes"), pd.to_timedelta("10 minutes"), pd.to_timedelta("30 minutes"), pd.to_timedelta("60 minutes")]:
        freezer_node_seismic = Freezer(store=freeze_store, groupname="seismic", dim="time", offset=offset, synchronizer=synchronizer)
        freezer_node_spectogram = Freezer(store=freeze_store, groupname="spectogram", dim="time", offset=offset, synchronizer=synchronizer)
        spectogram_node = Spectrogram(nfft=512, stride=512, dim="time", sampling_rate=250)

        data_node_freezed_seismic = freezer_node_seismic(data_node_seismic(delayed=True), delayed=True)
        data_node_freezed_spectogram = freezer_node_seismic(spectogram_node(data_node_seismic(delayed=True), delayed=True), delayed=True)
        data_node_spectogram = spectogram_node(data_node_seismic(delayed=True), delayed=True)
        for iteration in range(num_iterations):
            #baseline raw
            try:
                shutil.rmtree(freeze_store_path)
            except:
                pass
            for j in ["loading"]:
                starttime = time.time()
                data = data_node_seismic(request)
                endtime = time.time()
                time_tmp = endtime - starttime
                time_tmp_formatted = time.strftime('%H:%M:%S', time.gmtime(time_tmp))
                print(f"j: {j}, time: {time_tmp}")
                results.append([iteration, "baseline raw", j, time_tmp_formatted, time_tmp])
            #Freezer raw
            try:
                shutil.rmtree(freeze_store_path)
            except:
                pass
            for j in ["storing", "loading"]:
                starttime = time.time()
                data = stuett.core.configuration(data_node_freezed_seismic, request)
                data = dask.compute(data)
                endtime = time.time()
                time_tmp = endtime - starttime
                time_tmp_formatted = time.strftime('%H:%M:%S', time.gmtime(time_tmp))
                print(f"j: {j}, time: {time_tmp}")
                results.append([iteration, "Freezer raw", j, time_tmp_formatted, time_tmp])
            #baseline spectogram
            try:
                shutil.rmtree(freeze_store_path)
            except:
                pass
            for j in ["loading"]:
                starttime = time.time()
                data = stuett.core.configuration(data_node_spectogram, request)
                data = dask.compute(data)
                endtime = time.time()
                time_tmp = endtime - starttime
                time_tmp_formatted = time.strftime('%H:%M:%S', time.gmtime(time_tmp))
                print(f"j: {j}, time: {time_tmp}")
                results.append([iteration, "baseline spectogram", j, time_tmp_formatted, time_tmp])
            #Freezer spectogram
            try:
                shutil.rmtree(freeze_store_path)
            except:
                pass
            for j in ["storing", "loading"]:
                starttime = time.time()
                data = stuett.core.configuration(data_node_freezed_spectogram, request)
                data = dask.compute(data)
                endtime = time.time()
                time_tmp = endtime - starttime
                time_tmp_formatted = time.strftime('%H:%M:%S', time.gmtime(time_tmp))
                print(f"j: {j}, time: {time_tmp}")
                results.append([iteration, "Freezer spectogram", j, time_tmp_formatted, time_tmp])
except Exception as e:
    import traceback
    print(traceback.format_exc())

with open(results_path.joinpath(f"{evaluation_starttime}_results_data_loading.csv"),"w") as f:
    wr = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC,delimiter=",")
    wr.writerow(results)

try:
    shutil.rmtree(local_scratch_path_tmp)
except:
    pass
