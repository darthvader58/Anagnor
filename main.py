"""Quick-look plot of the GISTEMP surface temperature anomaly for a chosen date.

For the full visualization suite (geo maps + training curves) use:

    python visualize.py all
"""

import argparse
import datetime

import matplotlib.pyplot as plt
from netCDF4 import Dataset, date2index


def plot_temperature_anomaly(nc_path: str, date: datetime.date, out: str | None) -> None:
    data = Dataset(nc_path)
    idx = date2index(datetime.datetime(date.year, date.month, date.day), data.variables["time"])
    anomaly = data.variables["tempanomaly"][idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(anomaly, cmap="RdBu_r", origin="lower", vmin=-8, vmax=8,
                   extent=[-180, 180, -90, 90])
    ax.set_title(f"GISTEMP surface temperature anomaly — {date.isoformat()}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.colorbar(im, ax=ax, label="Temperature anomaly (°C)", shrink=0.8)
    fig.tight_layout()

    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"wrote {out}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nc", default="gistemp1200_GHCNv4_ERSSTv5.nc", help="Path to the GISTEMP NetCDF file.")
    parser.add_argument("--date", default="2008-01-15", help="YYYY-MM-DD; nearest available time step is used.")
    parser.add_argument("--out", default=None, help="If set, save to this path instead of showing interactively.")
    args = parser.parse_args()

    plot_temperature_anomaly(args.nc, datetime.date.fromisoformat(args.date), args.out)


if __name__ == "__main__":
    main()
