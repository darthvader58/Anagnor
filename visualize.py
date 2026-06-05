"""Visualizations for the Anagnor landslide dataset and trained models.

Two families of plots:

  Geo plots (from the NASA landslide catalog CSV):
    * world map of landslide event points, coloured by trigger      (cartopy)
    * 2D kernel-density heatmap of event locations                  (seaborn)
    * choropleth of event counts per country                        (geopandas)
    * categorical breakdowns by trigger and landslide size          (seaborn)

  Training plots (from checkpoints/metrics.json written by train.py):
    * loss curve   : train vs validation
    * accuracy curve : train vs validation

Usage:
    python visualize.py all
    python visualize.py geo
    python visualize.py training
    python visualize.py map --out plots/world_map.png
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


CSV_DEFAULT = "datasets/nasa_global_landslide_catalog_point.csv"
METRICS_DEFAULT = "checkpoints/metrics.json"
PLOTS_DIR_DEFAULT = "plots"


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def load_landslides(csv_path: str = CSV_DEFAULT) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[(df["latitude"].between(-90, 90)) & (df["longitude"].between(-180, 180))]
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Geo plots
# ---------------------------------------------------------------------------

def plot_world_map(df: pd.DataFrame, out: str) -> None:
    """Scatter landslide events on a world map, coloured by trigger."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    top_triggers = df["landslide_trigger"].value_counts().head(6).index.tolist()
    plot_df = df[df["landslide_trigger"].isin(top_triggers)]

    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#f0f0f0")
    ax.add_feature(cfeature.OCEAN, facecolor="#dceaf3")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="#555")
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="#888")

    palette = sns.color_palette("tab10", n_colors=len(top_triggers))
    for trigger, color in zip(top_triggers, palette):
        sub = plot_df[plot_df["landslide_trigger"] == trigger]
        ax.scatter(
            sub["longitude"], sub["latitude"],
            s=8, alpha=0.55, color=color, label=trigger,
            transform=ccrs.PlateCarree(),
        )

    ax.legend(loc="lower left", frameon=True, title="Trigger", fontsize=9)
    ax.set_title("NASA Global Landslide Catalog — events by trigger", fontsize=14)

    _ensure_dir(out)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_density_heatmap(df: pd.DataFrame, out: str) -> None:
    """KDE density of landslide events on a PlateCarree world map."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#f7f7f7")
    ax.add_feature(cfeature.OCEAN, facecolor="#e8f0f5")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="#555")

    sns.kdeplot(
        x=df["longitude"], y=df["latitude"],
        ax=ax, fill=True, cmap="rocket_r",
        levels=20, thresh=0.02, alpha=0.7,
    )
    ax.set_title("Landslide event density (KDE)", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    _ensure_dir(out)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_country_choropleth(df: pd.DataFrame, out: str) -> None:
    """Choropleth of event counts per country using cartopy's Natural Earth shapefile."""
    import geopandas as gpd
    import cartopy.io.shapereader as shpreader

    shp = shpreader.natural_earth(resolution="110m", category="cultural", name="admin_0_countries")
    world = gpd.read_file(shp)

    # Hand-pick aliases for the obvious mismatches between the catalog and Natural Earth.
    aliases = {
        "United States": "United States of America",
        "Tanzania": "United Republic of Tanzania",
        "Bahamas": "The Bahamas",
        "Congo (Brazzaville)": "Republic of the Congo",
        "Congo (Kinshasa)": "Democratic Republic of the Congo",
        "Czech Republic": "Czechia",
        "Serbia": "Republic of Serbia",
        "Cote d'Ivoire": "Ivory Coast",
    }
    counts = df["country_name"].replace(aliases).value_counts().rename_axis("country").reset_index(name="events")

    merged = world.merge(counts, left_on="NAME", right_on="country", how="left")
    merged["events"] = merged["events"].fillna(0)

    fig, ax = plt.subplots(figsize=(16, 8))
    merged.plot(
        column="events",
        ax=ax,
        cmap="OrRd",
        legend=True,
        edgecolor="#888",
        linewidth=0.3,
        legend_kwds={"label": "Landslide events", "shrink": 0.6},
        missing_kwds={"color": "#eeeeee", "label": "No data"},
    )
    ax.set_title("Landslide events per country", fontsize=14)
    ax.set_axis_off()

    _ensure_dir(out)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_category_breakdown(df: pd.DataFrame, out: str) -> None:
    """Two-panel seaborn bar chart: top triggers and landslide sizes."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    trigger_counts = df["landslide_trigger"].value_counts().head(10)
    sns.barplot(
        x=trigger_counts.values, y=trigger_counts.index,
        ax=axes[0], hue=trigger_counts.index, palette="rocket", legend=False,
    )
    axes[0].set_title("Top 10 landslide triggers")
    axes[0].set_xlabel("Event count")
    axes[0].set_ylabel("")

    size_order = ["small", "medium", "large", "very_large", "catastrophic"]
    size_counts = df["landslide_size"].value_counts()
    size_counts = size_counts.reindex([s for s in size_order if s in size_counts.index])
    sns.barplot(
        x=size_counts.index, y=size_counts.values,
        ax=axes[1], hue=size_counts.index, palette="mako", legend=False,
    )
    axes[1].set_title("Events by landslide size")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Event count")

    fig.tight_layout()
    _ensure_dir(out)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(metrics_path: str, out: str) -> None:
    """Plot train vs validation loss + accuracy from train.py's metrics.json."""
    with open(metrics_path) as f:
        h = json.load(f)

    epochs = range(1, len(h["train_loss"]) + 1)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(epochs, h["train_loss"], marker="o", label="train")
    axes[0].plot(epochs, h["val_loss"], marker="o", label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE loss")
    axes[0].legend()

    axes[1].plot(epochs, h["train_acc"], marker="o", label="train")
    axes[1].plot(epochs, h["val_acc"], marker="o", label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    fig.suptitle("Anagnor training — train vs validation", fontsize=14)
    fig.tight_layout()

    _ensure_dir(out)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _run(targets: set[str], csv: str, metrics: str, plots_dir: str, single_out: Optional[str]) -> None:
    if any(t in targets for t in ("geo", "map", "density", "choropleth", "categories", "all")):
        df = load_landslides(csv)
    else:
        df = None

    if "map" in targets or "geo" in targets or "all" in targets:
        plot_world_map(df, single_out or os.path.join(plots_dir, "world_map.png"))
    if "density" in targets or "geo" in targets or "all" in targets:
        plot_density_heatmap(df, single_out or os.path.join(plots_dir, "density.png"))
    if "choropleth" in targets or "geo" in targets or "all" in targets:
        plot_country_choropleth(df, single_out or os.path.join(plots_dir, "country_choropleth.png"))
    if "categories" in targets or "geo" in targets or "all" in targets:
        plot_category_breakdown(df, single_out or os.path.join(plots_dir, "categories.png"))
    if "training" in targets or "all" in targets:
        if os.path.exists(metrics):
            plot_training_curves(metrics, single_out or os.path.join(plots_dir, "training_curves.png"))
        else:
            print(f"no metrics file at {metrics} — run train.py first to populate it")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "targets", nargs="+",
        choices=["all", "geo", "training", "map", "density", "choropleth", "categories"],
        help="Which plots to generate.",
    )
    parser.add_argument("--csv", default=CSV_DEFAULT)
    parser.add_argument("--metrics", default=METRICS_DEFAULT)
    parser.add_argument("--plots-dir", default=PLOTS_DIR_DEFAULT)
    parser.add_argument("--out", default=None, help="Override output path (only meaningful for a single target).")
    args = parser.parse_args()

    _run(set(args.targets), args.csv, args.metrics, args.plots_dir, args.out)


if __name__ == "__main__":
    main()
