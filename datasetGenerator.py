"""Dataset generators for the Anagnor landslide model.

Three NetCDF-backed sources are stacked along the channel dimension to form a
single 32-channel input tensor per sample:

    SurfaceTemp      ->  15 channels  (GISTEMP tempanomaly, prior 15 time steps)
    Elevation        ->   1 channel   (NLDAS elevation patch)
    IRprecipitation  ->  16 channels  (TRMM IR precipitation, prior 16 frames)
"""

import datetime
import random
from datetime import timedelta
from random import randrange

import cv2
import numpy as np
import pandas as pd
import torch
from netCDF4 import Dataset, date2index
from torch.utils.data import Dataset as TorchDataset


IMG_SIZE = 1024


def random_date(start: datetime.datetime, end: datetime.datetime) -> datetime.datetime:
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    return start + timedelta(seconds=randrange(int_delta))


def _to_hwc(matrix: np.ndarray) -> np.ndarray:
    return np.transpose(matrix, (1, 2, 0))


def _to_chw(matrix: np.ndarray) -> np.ndarray:
    return np.transpose(matrix, (2, 0, 1))


def splice2Dnc(nc, iter_, x_i, x_j, y_i, y_j, cx="latitude", cy="longitude", swap=False):
    lat = nc.variables[cx][:]
    lon = nc.variables[cy][:]

    where_j = np.where((lon >= x_i) & (lon <= x_j))[0]
    where_i = np.where((lat >= y_i) & (lat <= y_j))[0]

    if len(where_i) == 0:
        i0, i1 = 0, 10
    else:
        i0, i1 = where_i[0], where_i[-1] + 1

    j0, j1 = where_j[0], where_j[-1] + 1
    if swap:
        return iter_[j0:j1, i0:i1]
    return iter_[i0:i1, j0:j1]


def splice3Dnc(nc, iter_, x_i, x_j, y_i, y_j, z_i, z_j):
    lat = nc.variables["lat"][:]
    lon = nc.variables["lon"][:]

    where_j = np.where((lon >= x_i) & (lon <= x_j))[0]
    where_i = np.where((lat >= y_i) & (lat <= y_j))[0]

    i0, i1 = where_i[0], where_i[-1] + 1
    j0, j1 = where_j[0], where_j[-1] + 1
    return iter_[z_i:z_j, i0:i1, j0:j1]


class DatasetGen:
    def __init__(self, path: str):
        self.img_size = IMG_SIZE
        self.path = path

    def getData(self, day, month, year, long, lat):  # noqa: N802
        raise NotImplementedError


class SurfaceTemp(DatasetGen):
    def __init__(self, path: str = "gistemp1200_GHCNv4_ERSSTv5.nc"):
        super().__init__(path=path)
        self.data = Dataset(path)
        self.temp_anomaly = self.data.variables["tempanomaly"][:]

    def getData(self, day, month, year, long, lat):
        timeindex = date2index(datetime.datetime(year, month, day), self.data.variables["time"])
        matrix = splice3Dnc(self.data, self.temp_anomaly, -3, 3, -3, 3, timeindex - 15, timeindex)
        img = cv2.resize(_to_hwc(np.array(matrix)), (self.img_size, self.img_size))
        return _to_chw(img)


class Elevation(DatasetGen):
    def __init__(self, path: str = "NLDAS.nc"):
        super().__init__(path=path)
        self.data = Dataset(path)
        self.elevation = self.data.variables["elevation"][:]

    def getData(self, day, month, year, long, lat):
        matrix = splice2Dnc(self.data, self.elevation, long - 3, long + 3, lat - 3, lat + 3)
        img = np.array([matrix])
        img = cv2.resize(_to_hwc(img), (self.img_size, self.img_size))
        img = np.expand_dims(img, 2)
        return _to_chw(img)


class IRprecipitation(DatasetGen):
    def __init__(self, path: str = "TRMMDataset/3B42_Daily.{:04d}{:02d}{:02}.7.nc4"):
        super().__init__(path=path)
        self.dt = datetime.timedelta(days=3)

    def getData(self, day, month, year, long, lat):
        last = None
        for _ in range(-15, 1):
            p_new = datetime.datetime(year, month, day) - self.dt
            year, month, day = p_new.year, p_new.month, p_new.day
            nc = Dataset(self.path.format(year, month, day))
            vars_ = nc.variables["IRprecipitation_cnt"][:]
            matrix = splice2Dnc(nc, vars_, long - 3, long + 3, lat - 3, lat + 3, "lat", "lon", swap=True)
            matrix = np.array([matrix])
            last = matrix if last is None else np.vstack((matrix, last))

        last = last.astype(np.float32)
        img = cv2.resize(_to_hwc(last), (self.img_size, self.img_size))
        return _to_chw(img)


class CustomDataset(TorchDataset):
    """Pairs each landslide event with a random negative sample (label=0)."""

    def __init__(self, csv_file: str = "datasets/nasa_global_landslide_catalog_point.csv"):
        ds = pd.read_csv(csv_file)
        self.event_date = ds["event_date"].reset_index(drop=True)
        self.lat = ds["latitude"].reset_index(drop=True)
        self.long = ds["longitude"].reset_index(drop=True)

        self.st = SurfaceTemp()
        self.el = Elevation()
        self.pres = IRprecipitation()

        self.neg_start = datetime.datetime.strptime("1/1/2001 1:30 PM", "%m/%d/%Y %I:%M %p")
        self.neg_end = datetime.datetime.strptime("1/1/2019 4:50 AM", "%m/%d/%Y %I:%M %p")

    def __len__(self) -> int:
        return len(self.event_date)

    def __getitem__(self, idx: int):
        if idx % 2 == 0:
            date = datetime.datetime.fromisoformat(self.event_date[idx])
            lat = self.lat[idx]
            long = self.long[idx]
            label = 1.0
        else:
            date = random_date(self.neg_start, self.neg_end)
            lat = random.randint(-90, 90)
            long = random.randint(0, 180)
            label = 0.0

        if long < 0:
            long += 180

        d1 = self.st.getData(15, date.month, date.year, int(lat), int(long))
        d2 = self.el.getData(date.day, date.month, date.year, long, lat)
        d3 = self.pres.getData(date.day, date.month, 2000, long, lat - 50)

        sample = np.vstack((d1, d2, d3)).astype(np.float32)
        return torch.from_numpy(sample), torch.tensor([label], dtype=torch.float32)


if __name__ == "__main__":
    ds = CustomDataset()
    print(f"dataset size: {len(ds)}")
    x, y = ds[0]
    print(f"sample shape: {tuple(x.shape)}  label: {y.item()}")
