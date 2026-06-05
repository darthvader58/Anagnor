# Anagnor

Anagnor is a machine-learning system for **global landslide detection** that
fuses NASA's GISTEMP 4.0 surface-temperature anomaly grids with NLDAS elevation
and TRMM precipitation rasters. A 3D-convolutional binary classifier learns,
from the NASA Global Landslide Catalog, whether the climatic and physiographic
conditions around a given lat/long on a given date are consistent with a
landslide event.

The repository contains the full pipeline: dataset construction from NetCDF
sources, a PyTorch model, a training loop with train/val tracking, and a
visualization module producing geographic and training-curve plots.

**Authors:** Yash Jha, Shashwat Raj, Garv Jain, Bhavya Verma, Rishit Yadav, Vir Malhotra.

---

## Data sources

| Layer | Variable | Source | File |
|---|---|---|---|
| Surface temperature anomaly | `tempanomaly` (15 prior monthly steps) | **NASA GISTEMP v4** (GHCN v4 + ERSST v5) | `gistemp1200_GHCNv4_ERSSTv5.nc` |
| Elevation | `elevation` (static raster patch) | NASA NLDAS | `NLDAS.nc` |
| IR precipitation | `IRprecipitation_cnt` (16 prior 3-day frames) | NASA TRMM 3B42 Daily v7 | `TRMMDataset/3B42_Daily.YYYYMMDD.7.nc4` |
| Event labels | landslide events with lat/long/date | NASA Global Landslide Catalog | `datasets/nasa_global_landslide_catalog_point.csv` |

GISTEMP 4.0 is the load-bearing climatic signal: monthly temperature anomalies
on a 2°×2° grid that capture the regional warming and seasonal heating patterns
known to destabilise slopes. Each training sample slices a 6°×6° window around
the candidate location and stacks the prior 15 monthly anomaly maps as input
channels.

The three rasters are co-registered to a 1024×1024 patch and concatenated along
the channel dimension:

```
SurfaceTemp (15)  +  Elevation (1)  +  IRprecipitation (16)  =  32 channels
```

## Model

`AnagnorModel` (see [model.py](model.py)) is a small convolutional binary
classifier operating on the 32-channel raster:

```
Conv2d(32 → 12, k=3) → ReLU → MaxPool(10) → Dropout
Conv2d(12 →  3, k=3) → ReLU → MaxPool(10) → Dropout
Flatten → FC(300→120) → FC(120→84) → FC(84→1)
```

The final FC produces a logit; pair with `BCEWithLogitsLoss` during training and
apply `torch.sigmoid` at inference for a landslide-probability score in [0, 1].

## Training pipeline

[`datasetGenerator.py`](datasetGenerator.py) builds positive and negative
samples on the fly:

* **Positive (`label = 1`)** — a real catalog event: read the date and lat/long
  from the CSV, then extract the GISTEMP / NLDAS / TRMM windows around that
  point and time.
* **Negative (`label = 0`)** — a random date/location pair, sampled so the
  model also learns what *non-events* look like.

[`train.py`](train.py) does an 80/20 train/val split (seeded), trains with SGD +
momentum and `BCEWithLogitsLoss`, tracks per-epoch loss and accuracy on both
splits, and writes them to `checkpoints/metrics.json`. The best validation
checkpoint is saved to `checkpoints/best.pth`.

## Visualizations

[`visualize.py`](visualize.py) renders both data plots and training plots
(outputs land in `plots/`):

| Plot | Source | Library |
|---|---|---|
| `world_map.png` — events on a Robinson world projection, coloured by trigger | landslide CSV | cartopy + seaborn |
| `density.png` — KDE heatmap of event locations | landslide CSV | seaborn + cartopy |
| `country_choropleth.png` — events per country on a Natural Earth choropleth | landslide CSV | geopandas + cartopy |
| `categories.png` — top triggers and event sizes | landslide CSV | seaborn |
| `training_curves.png` — train vs validation loss & accuracy per epoch | `checkpoints/metrics.json` | matplotlib |

## Quickstart

```bash
pip install -r requirements.txt
```

Download the NetCDF inputs (the catalog CSV is already bundled):

```bash
wget https://data.giss.nasa.gov/pub/gistemp/gistemp1200_GHCNv4_ERSSTv5.nc.gz
gunzip gistemp1200_GHCNv4_ERSSTv5.nc.gz
# Place NLDAS.nc in the repo root, and TRMM daily files under TRMMDataset/.
```

Train:

```bash
python train.py --epochs 10 --batch-size 8
```

Plot:

```bash
python visualize.py all          # geo plots + training curves
python visualize.py geo          # geo plots only
python visualize.py training     # train vs validation curves only
python main.py --date 2008-01-15 # quick-look GISTEMP anomaly map for one day
```

## Repository layout

```
.
├── datasets/                          # NASA Global Landslide Catalog (CSV)
├── checkpoints/                       # model weights + metrics.json (gitignored)
├── plots/                             # generated PNGs (gitignored)
├── tests/                             # ad-hoc inspection scripts for each NetCDF source
├── datasetGenerator.py                # SurfaceTemp / Elevation / IRprecipitation + CustomDataset
├── model.py                           # AnagnorModel
├── train.py                           # training loop with train/val tracking
├── visualize.py                       # geo + training visualizations
├── main.py                            # CLI viewer for a single GISTEMP day
└── requirements.txt
```

## Detailed background

Watch the project walkthrough:
https://www.youtube.com/watch?v=u5XusYBq1h0

Conventional landslide-warning systems focus on rainfall and surface-water
erosion. Anagnor adds **uneven heating of land** (via GISTEMP anomalies),
**precipitation history** (TRMM), and **terrain** (NLDAS elevation) into a
single multi-channel input, on the hypothesis that the joint signal is more
predictive than any single channel.

The original project also proposed a low-cost underground radon-detection
device — measuring radon and its daughter-product concentrations through a
degassing chamber feeding a Lucas scintillation cell — as a means of inferring
tectonic activity that can precede landslides. That hardware sits outside this
repository; this codebase is the data + ML portion of the system.
