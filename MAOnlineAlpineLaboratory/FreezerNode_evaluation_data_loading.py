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
from stuett.data import (
    SeismicSource,
    DataCollector,
    MinMaxDownsampling,
    GSNDataSource,
    LTTBDownsampling,
    MHDSLRImages,
    MHDSLRFilenames,
    DirectoryStore,
    Spectrogram,
    Freezer,
    Spectrogram,
    Rescale,
    To_db,
    To_Tensor
)
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
import zipfile

import csv

import concurrent.futures
import threading

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
parser.add_argument(
    "--copy_seismic_data",
    action="store_true",
    default=True,
    help="Copy seismic raw data to local scratch before running and use this data as source. If folder seismic_data already exists it will not copy again",
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
synchronizer=zarr.ThreadSynchronizer()
#synchronizer=zarr.ProcessSynchronizer(path=freeze_store_path)


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

data_node_seismic = load_seismic_source()
data_node_image = load_image_source()
stride = 512
spectogram_node = Spectrogram(nfft=512, stride=stride, dim="time", sampling_rate=250)

def seismic_collector(): 
    
    return DataCollector(
    data_paths = [
        data_node_seismic(delayed=True)] + [
                MinMaxDownsampling(rate=rate, dim="time")(
                data=data_node_seismic(delayed=True), delayed=True
        )
        for rate in [4, 12, 450, 5000, 360000, 3600000]
    ],
    granularities=[
        dt.timedelta(minutes=20),
        dt.timedelta(minutes=40),
        dt.timedelta(hours=2),
        dt.timedelta(hours=16),
        dt.timedelta(days=40),
        dt.timedelta(days=332),
        dt.timedelta(days=36500),
    ],
    )

def seismic_collector_freezer(freezer_offset): 

    synchronizer = zarr.ThreadSynchronizer()
    
    return DataCollector(
    data_paths = [
        data_node_seismic(delayed=True)] + [
        Freezer(freeze_store,"seismic_sampling_rate_" + str(rate) ,"time",freezer_offset,synchronizer)(
                MinMaxDownsampling(rate=rate, dim="time")(
                data=data_node_seismic(delayed=True), delayed=True
        ),delayed = True)
        for rate in [4, 12, 450, 5000, 360000, 3600000]
    ],
    granularities=[
        dt.timedelta(minutes=20),
        dt.timedelta(minutes=40),
        dt.timedelta(hours=2),
        dt.timedelta(hours=16),
        dt.timedelta(days=40),
        dt.timedelta(days=332),
        dt.timedelta(days=36500),
    ],
    )

def spectogram_collector(): 
    
        return DataCollector(
    data_paths = [
        spectogram_node(data_node_seismic(delayed=True), delayed=True)] + [
                MinMaxDownsampling(rate=rate, dim="time")(
                data=spectogram_node(data_node_seismic(delayed=True), delayed=True), delayed=True
        )
        for rate in [int(15000/stride), int(150000/stride), int(9000000/stride), int(21600000/stride)]
    ],
    granularities=[
        dt.timedelta(hours=3),
        dt.timedelta(days=3),
        dt.timedelta(days=30),
        dt.timedelta(days=300),
        dt.timedelta(days=36500),
    ],
    )

def spectogram_collector_freezer(freezer_offset): 

    synchronizer = zarr.ThreadSynchronizer()
    
    return DataCollector(
    data_paths = [
        spectogram_node(data_node_seismic(delayed=True), delayed=True)] + [
        Freezer(freeze_store,"spectogram_sampling_rate_" + str(rate) ,"time",freezer_offset,synchronizer)(
                MinMaxDownsampling(rate=rate, dim="time")(
                data=spectogram_node(data_node_seismic(delayed=True), delayed=True), delayed=True
        ),delayed = True)
        for rate in [int(15000/stride), int(150000/stride), int(9000000/stride), int(21600000/stride)]
    ],
    granularities=[
        dt.timedelta(hours=3),
        dt.timedelta(days=3),
        dt.timedelta(days=30),
        dt.timedelta(days=300),
        dt.timedelta(days=36500),
    ],
    )


def processing(data_node, start_time, end_time, step):
    print(f"node: {data_node}, start_time: {start_time}, end_time: {end_time}, step: {step}")
    current_time = start_time
    offset = stuett.to_timedelta(step)
    off = offset - stuett.to_timedelta(f"4 ms")
    while current_time <= end_time:
        request = {"start_time": current_time, "end_time": current_time + off}
        data = stuett.core.configuration(data_node, request)
        data = dask.compute(data)
        current_time = current_time + offset
    return None


if __name__ == "__main__":

    start_time = f"2017-03-01 00:00"
    end_time = f"2017-07-01 00:00"
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)

    max_workers = 0
    min_workers = 0
    num_iterations = 1
    max_hours = 10000
    hours_log_smallstep = range(1,10)
    hours_log_bigstep = [10, 100]#1,10,100,
    results = []
    results.append(["iteration","Data node","storing/loading","num_workers","Freezer offset","Calculation time formatted", "Calculation time raw"])

    request = {"start_time":start_time, "end_time":end_time}
    #dask.config.set(scheduler='single-threaded')

    try: 
        for bigstep in hours_log_bigstep:
            for smallstep in hours_log_smallstep:
                hours = bigstep * smallstep
                if bigstep == 1:
                    step = 1
                elif hours >= 800:
                    step = 50
                else:
                    step = 10
                for offset in [pd.to_timedelta("60 minutes")]:
                    data_collector_seismic = seismic_collector()
                    data_collector_seismic_freezer = seismic_collector_freezer(offset)
                    data_spectogram_collector = spectogram_collector()
                    data_spectogram_collector_freezer = spectogram_collector_freezer(offset)
                    start_time_tmp = stuett.to_datetime(start_time)
                    end_time_tmp = stuett.to_datetime(start_time) + stuett.to_timedelta(f"{hours} hours")
                    request = {"start_time":start_time_tmp, "end_time":end_time_tmp}
                    data_node_raw = data_collector_seismic(request)
                    data_node_raw_frozen = data_collector_seismic_freezer(request)
                    data_node_spectogram = data_spectogram_collector(request)
                    data_node_freezed_spectogram = data_spectogram_collector_freezer(request)

                    for iteration in range(num_iterations):
                        for num_workers in range(min_workers, max_workers+1):
                            #baseline raw
                            for j in ["loading"]:
                                starttime = time.time()
                                if num_workers == 0:
                                    for i in range(0, hours, step):
                                        start_time_tmp = stuett.to_datetime(start_time) + stuett.to_timedelta(f"{i} hours")
                                        end_time_tmp = stuett.to_datetime(start_time) + stuett.to_timedelta(f"{i + step} hours") - stuett.to_timedelta(f"4 ms")
                                        request = {"start_time":start_time_tmp, "end_time":end_time_tmp}
                                        data_raw = stuett.core.configuration(data_node_raw, request)
                                        data_raw = dask.compute(data_raw)
                                else:
                                    offsets_count = int(np.ceil(hours/step))
                                    workers_slice = int(np.floor(offsets_count/num_workers))
                                    for x in range(num_workers):
                                        start_time_tmp = stuett.to_datetime(start_time) + stuett.to_timedelta(f"{x * workers_slice * step} hours")
                                        end_time_tmp = stuett.to_datetime(start_time) + stuett.to_timedelta(f"{max((x + 1) * workers_slice * step, hours)} hours") - stuett.to_timedelta(f"4 ms")
                                        request = {"start_time":start_time_tmp, "end_time":end_time_tmp}
                                        threads = [threading.Thread(target=processing, args=(data_node_raw, start_time_tmp, end_time_tmp, step))]

                                    for t in threads:
                                        t.start()
                                    for t in threads:
                                        t.join()
                                endtime = time.time()
                                time_tmp = endtime - starttime
                                time_tmp_formatted = time.strftime('%H:%M:%S', time.gmtime(time_tmp))
                                print(f"baseline raw {j}: hours: {hours}, time: {time_tmp}")#, shape: {data_raw[0].shape}")
                                results.append([hours, "baseline raw", j, num_workers, offset, time_tmp_formatted, time_tmp])
                            #Freezer raw
                            try:
                                shutil.rmtree(freeze_store_path)
                            except:
                                pass
                            for j in ["storing", "loading"]:
                                starttime = time.time()
                                if j == "storing":
                                    for i in range(0, hours, step):
                                        start_time_tmp = stuett.to_datetime(start_time) + stuett.to_timedelta(f"{i} hours")
                                        end_time_tmp = stuett.to_datetime(start_time) + stuett.to_timedelta(f"{i + step} hours") - stuett.to_timedelta(f"4 ms")
                                        request = {"start_time":start_time_tmp, "end_time":end_time_tmp}
                                        data_raw_frozen = stuett.core.configuration(data_node_raw_frozen, request)
                                        data_raw_frozen = dask.compute(data_raw_frozen)
                                else:
                                    start_time_tmp = stuett.to_datetime(start_time)
                                    end_time_tmp = stuett.to_datetime(start_time) + stuett.to_timedelta(f"{hours} hours") - stuett.to_timedelta(f"4 ms")
                                    request = {"start_time":start_time_tmp, "end_time":end_time_tmp}
                                    data_raw_frozen = stuett.core.configuration(data_node_raw_frozen, request)
                                    data_raw_frozen = dask.compute(data_raw_frozen)
                                endtime = time.time()
                                time_tmp = endtime - starttime
                                time_tmp_formatted = time.strftime('%H:%M:%S', time.gmtime(time_tmp))
                                print(f"Freezer raw {j}: hours: {hours}, time: {time_tmp}, shape: {data_raw_frozen[0].shape}")
                                results.append([hours, "Freezer raw", j, offset, time_tmp_formatted, time_tmp])
                            #baseline spectogram
                            for j in ["loading"]:
                                starttime = time.time()
                                for i in range(0, hours, step):
                                    start_time_tmp = stuett.to_datetime(start_time) + stuett.to_timedelta(f"{i} hours")
                                    end_time_tmp = stuett.to_datetime(start_time) + stuett.to_timedelta(f"{i + step} hours") - stuett.to_timedelta(f"4 ms")
                                    request = {"start_time":start_time_tmp, "end_time":end_time_tmp}
                                    data_spectogram = stuett.core.configuration(data_node_spectogram, request)
                                    data_spectogram = dask.compute(data_spectogram)
                                endtime = time.time()
                                time_tmp = endtime - starttime
                                time_tmp_formatted = time.strftime('%H:%M:%S', time.gmtime(time_tmp))
                                print(f"Baseline spectogram {j}: hours: {hours}, time: {time_tmp}, shape: {data_spectogram[0].shape}")
                                results.append([hours, "Baseline spectogram", j, offset, time_tmp_formatted, time_tmp])
                            #Freezer spectogram
                            try:
                                shutil.rmtree(freeze_store_path)
                            except:
                                pass
                            for j in ["storing", "loading"]:
                                starttime = time.time()
                                if j == "storing":
                                    for i in range(0, hours, step):
                                        start_time_tmp = stuett.to_datetime(start_time) + stuett.to_timedelta(f"{i} hours")
                                        end_time_tmp = stuett.to_datetime(start_time) + stuett.to_timedelta(f"{i + step} hours") - stuett.to_timedelta(f"4 ms")
                                        request = {"start_time":start_time_tmp, "end_time":end_time_tmp}
                                        data_spectogram_frozen = stuett.core.configuration(data_node_freezed_spectogram, request)
                                        data_spectogram_frozen = dask.compute(data_spectogram_frozen)
                                else:
                                    start_time_tmp = stuett.to_datetime(start_time)
                                    end_time_tmp = stuett.to_datetime(start_time) + stuett.to_timedelta(f"{hours} hours") - stuett.to_timedelta(f"4 ms")
                                    request = {"start_time":start_time_tmp, "end_time":end_time_tmp}
                                    data_spectogram_frozen = stuett.core.configuration(data_node_freezed_spectogram, request)
                                    data_spectogram_frozen = dask.compute(data_spectogram_frozen)
                                endtime = time.time()
                                time_tmp = endtime - starttime
                                time_tmp_formatted = time.strftime('%H:%M:%S', time.gmtime(time_tmp))
                                print(f"Freezer spectogram {j}: hours: {hours}, time: {time_tmp}, shape: {data_spectogram_frozen[0].shape}")
                                results.append([hours, "Freezer spectogram", j, offset, time_tmp_formatted, time_tmp])
    except Exception as e:
        import traceback
        print(traceback.format_exc())

    with open(results_path.joinpath(f"{evaluation_starttime}_results_data_loading.csv"),"w") as f:
        wr = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC,delimiter=",", lineterminator='\n')
        wr.writerow(results)
    print(f"Resultsfile written: {evaluation_starttime}_results_data_loading.csv")
    try:
        shutil.rmtree(local_scratch_path_tmp)
    except:
        pass
