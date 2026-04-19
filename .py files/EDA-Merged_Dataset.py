"""
EDA (Exploratory Data Analysis) for Delivery Delay Prediction Dataset
Target Variable: is_delayed (1 = Delayed, 0 = On Time)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import warnings
from pathlib import Path
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  USER CONFIG — set these two paths before running
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH  = Path(r"final_dataset.csv")   # path to your CSV file
OUTPUT_DIR = Path(r"eda_outputs")         # folder where plots will be saved
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)   # auto-creates the folder

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
PALETTE        = "gnuplot"
TARGET         = "is_delayed"
NUM_FEATURES   = [
    "approval_delay", "estimated_delivery_time", "purchase_day_of_week",
    "purchase_hour", "total_items", "total_price", "total_freight_value",
    "distance_km", "product_volume_cm3", "product_weight_grams",
]
BINARY_FEATURES = ["is_same_city", "is_same_state"]
CAT_GROUPS     = [c for c in [
    "category_group_books", "category_group_construction",
    "category_group_electronics", "category_group_entertainment",
    "category_group_fashion", "category_group_food",
    "category_group_garden", "category_group_health_beauty",
    "category_group_home", "category_group_industry",
    "category_group_lifestyle", "category_group_office",
    "category_group_other", "category_group_pet",
    "category_group_sports_toys",
] if True]


def _gnuplot_colors(n: int) -> list:
    """Sample n evenly spaced colours from the gnuplot colormap."""
    cmap = plt.cm.get_cmap(PALETTE, n)
    return [cmap(i) for i in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
class DeliveryDelayEDA:
    """
    Exploratory Data Analysis class for the delivery-delay dataset.

    Usage
    -----
    eda = DeliveryDelayEDA("final_dataset.csv")
    eda.run_full_eda()          # generates every plot
    """

    # ── Constructor ──────────────────────────────────────────────────────────
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df       = pd.read_csv(filepath)
        self._preprocess()
        print(f"✅ Loaded dataset  →  {self.df.shape[0]:,} rows  ×  {self.df.shape[1]} columns")

    # ── Preprocessing ────────────────────────────────────────────────────────
    def _preprocess(self):
        """Derive a few helper columns used across plots."""
        df = self.df
        df["delay_label"]    = df[TARGET].map({0: "On Time", 1: "Delayed"})
        df["distance_band"]  = pd.cut(
            df["distance_km"],
            bins=[0, 200, 500, 1000, 2000, df["distance_km"].max() + 1],
            labels=["<200 km", "200–500 km", "500–1000 km", "1000–2000 km", ">2000 km"],
        )
        df["price_band"] = pd.cut(
            df["total_price"],
            bins=[0, 50, 150, 300, 600, df["total_price"].max() + 1],
            labels=["<50", "50–150", "150–300", "300–600", ">600"],
        )
        self.df = df

    # ═════════════════════════════════════════════════════════════════════════
    # 1. DATASET OVERVIEW
    # ═════════════════════════════════════════════════════════════════════════
    def plot_dataset_overview(self, save: bool = True):
        """
        Summary dashboard: shape, data types, missing values, target split.
        """
        df   = self.df
        fig  = plt.figure(figsize=(18, 10), facecolor="#0d0d0d")
        gs   = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)
        colors = _gnuplot_colors(6)

        # ── (a) Target distribution – donut ──────────────────────────────
        ax0 = fig.add_subplot(gs[0, 0])
        counts  = df[TARGET].value_counts()
        labels  = ["On Time", "Delayed"]
        clrs    = [colors[1], colors[4]]
        wedges, texts, autotexts = ax0.pie(
            counts, labels=labels, colors=clrs, autopct="%1.1f%%",
            startangle=140, pctdistance=0.75,
            wedgeprops=dict(width=0.5, edgecolor="#0d0d0d", linewidth=2),
        )
        for t in texts + autotexts:
            t.set_color("white")
            t.set_fontsize(11)
        ax0.set_title("Target Distribution\n(is_delayed)", color="white",
                       fontsize=13, fontweight="bold", pad=10)

        # ── (b) Data-type breakdown ───────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 1])
        dtype_map = {
            "float64": "Float",
            "int64":   "Integer",
            "bool":    "Boolean",
            "object":  "Object",
        }
        dtype_counts = df.dtypes.astype(str).map(dtype_map).fillna("Other").value_counts()
        bars = ax1.bar(dtype_counts.index, dtype_counts.values,
                       color=_gnuplot_colors(len(dtype_counts)),
                       edgecolor="#0d0d0d", linewidth=1.5)
        for bar, val in zip(bars, dtype_counts.values):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     str(val), ha="center", va="bottom", color="white", fontsize=10)
        ax1.set_facecolor("#1a1a1a")
        ax1.set_title("Feature Data Types", color="white", fontsize=13, fontweight="bold")
        ax1.tick_params(colors="white")
        ax1.spines[["top","right"]].set_visible(False)
        ax1.spines[["left","bottom"]].set_color("#444")
        ax1.set_ylabel("Count", color="white")

        # ── (c) Missing values ────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 2])
        missing = df.isnull().sum()
        if missing.sum() == 0:
            ax2.text(0.5, 0.5, " No Missing\nValues!", transform=ax2.transAxes,
                     ha="center", va="center", fontsize=18, color="#00ff88",
                     fontweight="bold")
        else:
            missing[missing > 0].plot(kind="barh", ax=ax2,
                                       color=colors[3], edgecolor="#0d0d0d")
        ax2.set_facecolor("#1a1a1a")
        ax2.set_title("Missing Values", color="white", fontsize=13, fontweight="bold")
        ax2.tick_params(colors="white")
        ax2.spines[["top","right"]].set_visible(False)
        ax2.spines[["left","bottom"]].set_color("#444")

        # ── (d) Key statistics table ──────────────────────────────────────
        ax3 = fig.add_subplot(gs[1, :])
        ax3.set_facecolor("#1a1a1a")
        stats = df[NUM_FEATURES].describe().T[["mean","std","min","50%","max"]].round(2)
        stats.columns = ["Mean","Std","Min","Median","Max"]
        stats = stats.reset_index().rename(columns={"index": "Feature"})
        table = ax3.table(
            cellText=stats.values,
            colLabels=stats.columns,
            cellLoc="center", loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        for (row, col), cell in table.get_celld().items():
            cell.set_facecolor("#2a2a2a" if row % 2 == 0 else "#1a1a1a")
            cell.set_text_props(color="white")
            cell.set_edgecolor("#444")
            if row == 0:
                cell.set_facecolor("#3a1a5a")
                cell.set_text_props(color="white", fontweight="bold")
        ax3.axis("off")
        ax3.set_title("Numerical Feature Statistics", color="white",
                       fontsize=13, fontweight="bold", pad=8)

        fig.suptitle("📦 Delivery Delay EDA – Dataset Overview",
                     color="white", fontsize=17, fontweight="bold", y=1.01)
        plt.savefig(OUTPUT_DIR / "01_dataset_overview.png",
                    dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
        plt.close()
        print("  ✔ Saved: 01_dataset_overview.png")

    # ═════════════════════════════════════════════════════════════════════════
    # 2. NUMERICAL FEATURES vs TARGET (violin + strip)
    # ═════════════════════════════════════════════════════════════════════════
    def plot_numeric_vs_target(self, save: bool = True):
        """Violin plots of key numeric features split by delay status."""
        df     = self.df
        feats  = ["approval_delay","estimated_delivery_time","distance_km",
                  "total_freight_value","product_weight_grams","total_price"]
        titles = ["Approval Delay (days)","Estimated Delivery Time",
                  "Distance (km)","Freight Value","Product Weight (g)","Total Price"]

        fig, axes = plt.subplots(2, 3, figsize=(19, 11), facecolor="#0d0d0d")
        axes = axes.flatten()
        clrs = _gnuplot_colors(2)

        for i, (feat, title) in enumerate(zip(feats, titles)):
            ax = axes[i]
            ax.set_facecolor("#1a1a1a")
            subset = df[[feat, "delay_label"]].copy()
            subset[feat] = np.log1p(subset[feat])   # log-scale for skewed data
            sns.violinplot(data=subset, x="delay_label", y=feat,
                           palette={"On Time": clrs[0], "Delayed": clrs[1]},
                           inner="quartile", ax=ax, linewidth=1.2,
                           order=["On Time","Delayed"])
            ax.set_title(f"log(1+{title})\nvs Delay Status",
                          color="white", fontsize=12, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel(f"log(1+{feat})", color="white", fontsize=9)
            ax.tick_params(colors="white")
            ax.spines[["top","right"]].set_visible(False)
            ax.spines[["left","bottom"]].set_color("#444")
            # annotate medians
            for j, label in enumerate(["On Time","Delayed"]):
                med = np.log1p(df[df["delay_label"]==label][feat].median())
                ax.text(j, med, f"med={np.expm1(med):.1f}", ha="center",
                         va="bottom", color="yellow", fontsize=8, fontweight="bold")

        fig.suptitle("🎻 Numerical Features vs Delivery Delay  (log-scaled)",
                     color="white", fontsize=16, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "02_numeric_vs_target.png",
                    dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
        plt.close()
        print("  ✔ Saved: 02_numeric_vs_target.png")

    # ═════════════════════════════════════════════════════════════════════════
    # 3. DISTANCE BAND vs DELAY RATE
    # ═════════════════════════════════════════════════════════════════════════
    def plot_distance_delay(self, save: bool = True):
        """Delay rate increases steeply with shipping distance."""
        df   = self.df
        agg  = (df.groupby("distance_band", observed=True)[TARGET]
                  .agg(["mean","count"])
                  .reset_index())
        agg.columns = ["Distance Band","Delay Rate","Orders"]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="#0d0d0d")
        colors = _gnuplot_colors(len(agg))

        # Left: delay rate bar
        ax0 = axes[0]
        ax0.set_facecolor("#1a1a1a")
        bars = ax0.bar(agg["Distance Band"], agg["Delay Rate"] * 100,
                       color=colors, edgecolor="#0d0d0d", linewidth=1.5)
        ax0.yaxis.set_major_formatter(mtick.PercentFormatter())
        for bar, val in zip(bars, agg["Delay Rate"]):
            ax0.text(bar.get_x() + bar.get_width() / 2,
                      bar.get_height() + 0.2, f"{val*100:.1f}%",
                      ha="center", va="bottom", color="white", fontsize=10)
        ax0.set_title("Delay Rate by Shipping Distance", color="white",
                       fontsize=13, fontweight="bold")
        ax0.set_xlabel("Distance Band", color="white")
        ax0.set_ylabel("Delay Rate (%)", color="white")
        ax0.tick_params(colors="white", rotation=15)
        ax0.spines[["top","right"]].set_visible(False)
        ax0.spines[["left","bottom"]].set_color("#444")

        # Right: scatter of orders vs delay rate
        ax1 = axes[1]
        ax1.set_facecolor("#1a1a1a")
        sc = ax1.scatter(agg["Orders"], agg["Delay Rate"] * 100,
                         c=range(len(agg)), cmap=PALETTE,
                         s=200, edgecolors="white", linewidths=1.5, zorder=5)
        for _, row in agg.iterrows():
            ax1.annotate(row["Distance Band"],
                          (row["Orders"], row["Delay Rate"] * 100),
                          textcoords="offset points", xytext=(6, 4),
                          color="white", fontsize=9)
        ax1.set_title("Order Volume vs Delay Rate (by Distance)", color="white",
                       fontsize=13, fontweight="bold")
        ax1.set_xlabel("Order Volume", color="white")
        ax1.set_ylabel("Delay Rate (%)", color="white")
        ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax1.tick_params(colors="white")
        ax1.spines[["top","right"]].set_visible(False)
        ax1.spines[["left","bottom"]].set_color("#444")

        fig.suptitle("📍 Distance vs Delivery Delay", color="white",
                     fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "03_distance_vs_delay.png",
                    dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
        plt.close()
        print("  ✔ Saved: 03_distance_vs_delay.png")

    # ═════════════════════════════════════════════════════════════════════════
    # 4. TIME-BASED PATTERNS
    # ═════════════════════════════════════════════════════════════════════════
    def plot_time_patterns(self, save: bool = True):
        """Delay rates by hour-of-day and day-of-week (heatmap + line)."""
        df   = self.df
        days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

        fig, axes = plt.subplots(1, 2, figsize=(18, 6), facecolor="#0d0d0d")

        # ── Hour-of-day line plot ─────────────────────────────────────────
        ax0 = axes[0]
        ax0.set_facecolor("#1a1a1a")
        hour_delay = df.groupby("purchase_hour")[TARGET].mean() * 100
        cmap       = plt.cm.get_cmap(PALETTE, len(hour_delay))
        for h in range(len(hour_delay)):
            ax0.bar(h, hour_delay.iloc[h], color=cmap(h),
                    edgecolor="#0d0d0d", linewidth=0.8)
        ax0.plot(hour_delay.index, hour_delay.values, color="yellow",
                  linewidth=2, marker="o", markersize=4, zorder=5)
        ax0.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax0.set_title("Delay Rate by Purchase Hour", color="white",
                       fontsize=13, fontweight="bold")
        ax0.set_xlabel("Hour of Day (0–23)", color="white")
        ax0.set_ylabel("Delay Rate (%)", color="white")
        ax0.tick_params(colors="white")
        ax0.spines[["top","right"]].set_visible(False)
        ax0.spines[["left","bottom"]].set_color("#444")
        ax0.set_xticks(range(0, 24, 2))

        # ── Day-of-week bar ───────────────────────────────────────────────
        ax1 = axes[1]
        ax1.set_facecolor("#1a1a1a")
        dow_delay = df.groupby("purchase_day_of_week")[TARGET].mean() * 100
        colors    = _gnuplot_colors(7)
        bars = ax1.bar(days, dow_delay.values, color=colors,
                        edgecolor="#0d0d0d", linewidth=1.5)
        for bar, val in zip(bars, dow_delay.values):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                      bar.get_height() + 0.05, f"{val:.2f}%",
                      ha="center", va="bottom", color="white", fontsize=10)
        ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax1.set_title("Delay Rate by Day of Week", color="white",
                       fontsize=13, fontweight="bold")
        ax1.set_xlabel("Day of Week (0=Mon)", color="white")
        ax1.set_ylabel("Delay Rate (%)", color="white")
        ax1.tick_params(colors="white")
        ax1.spines[["top","right"]].set_visible(False)
        ax1.spines[["left","bottom"]].set_color("#444")

        fig.suptitle("⏰ Time-Based Purchase Patterns vs Delivery Delay",
                     color="white", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "04_time_patterns.png",
                    dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
        plt.close()
        print("  ✔ Saved: 04_time_patterns.png")

    # ═════════════════════════════════════════════════════════════════════════
    # 5. APPROVAL DELAY vs DELIVERY DELAY
    # ═════════════════════════════════════════════════════════════════════════
    def plot_approval_delay(self, save: bool = True):
        """KDE + boxplot of approval_delay capped at 10 days."""
        df = self.df
        df_cap = df[df["approval_delay"] <= 10].copy()

        fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="#0d0d0d")
        clrs = _gnuplot_colors(2)

        # KDE
        ax0 = axes[0]
        ax0.set_facecolor("#1a1a1a")
        for label, color in zip(["On Time","Delayed"], clrs):
            subset = df_cap[df_cap["delay_label"] == label]["approval_delay"]
            subset.plot.kde(ax=ax0, color=color, linewidth=2.5, label=label)
            ax0.fill_between(
                np.linspace(subset.min(), subset.max(), 200),
                0,
                [0] * 200,  # placeholder – kde fill via collection
                alpha=0.1, color=color,
            )
        ax0.set_title("Approval Delay Density by Delay Status\n(capped at 10 days)",
                       color="white", fontsize=12, fontweight="bold")
        ax0.set_xlabel("Approval Delay (days)", color="white")
        ax0.set_ylabel("Density", color="white")
        ax0.legend(facecolor="#2a2a2a", labelcolor="white")
        ax0.tick_params(colors="white")
        ax0.spines[["top","right"]].set_visible(False)
        ax0.spines[["left","bottom"]].set_color("#444")

        # Delay rate bucketed by approval_delay bins
        ax1 = axes[1]
        ax1.set_facecolor("#1a1a1a")
        df_cap["approval_bin"] = pd.cut(
            df_cap["approval_delay"], bins=[-0.1,0,1,2,3,5,10],
            labels=["0","1","2","3","4-5","6-10"],
        )
        agg = df_cap.groupby("approval_bin", observed=True)[TARGET].mean() * 100
        colors = _gnuplot_colors(len(agg))
        bars = ax1.bar(agg.index.astype(str), agg.values, color=colors,
                        edgecolor="#0d0d0d", linewidth=1.5)
        for bar, val in zip(bars, agg.values):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                      bar.get_height() + 0.1, f"{val:.1f}%",
                      ha="center", va="bottom", color="white", fontsize=10)
        ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax1.set_title("Delay Rate by Approval Delay (days)",
                       color="white", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Approval Delay Bucket (days)", color="white")
        ax1.set_ylabel("Delay Rate (%)", color="white")
        ax1.tick_params(colors="white")
        ax1.spines[["top","right"]].set_visible(False)
        ax1.spines[["left","bottom"]].set_color("#444")

        fig.suptitle("⏳ Approval Delay vs Delivery Delay",
                     color="white", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "05_approval_delay.png",
                    dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
        plt.close()
        print("  ✔ Saved: 05_approval_delay.png")

    # ═════════════════════════════════════════════════════════════════════════
    # 6. SAME CITY / SAME STATE  +  PRICE BAND
    # ═════════════════════════════════════════════════════════════════════════
    def plot_binary_features(self, save: bool = True):
        """Delay rates for same-city, same-state, and price bands."""
        df   = self.df
        fig, axes = plt.subplots(1, 3, figsize=(19, 6), facecolor="#0d0d0d")

        for ax, feat, title in zip(
            axes[:2],
            ["is_same_city","is_same_state"],
            ["Same City","Same State"],
        ):
            ax.set_facecolor("#1a1a1a")
            agg   = df.groupby(feat)[TARGET].mean() * 100
            lbls  = ["Different","Same"]
            clrs  = _gnuplot_colors(2)
            bars  = ax.bar(lbls, agg.values, color=clrs, edgecolor="#0d0d0d",
                            linewidth=1.5, width=0.45)
            for bar, val in zip(bars, agg.values):
                ax.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.1, f"{val:.2f}%",
                         ha="center", va="bottom", color="white",
                         fontsize=12, fontweight="bold")
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())
            ax.set_title(f"Delay Rate: {title}", color="white",
                          fontsize=13, fontweight="bold")
            ax.set_ylabel("Delay Rate (%)", color="white")
            ax.tick_params(colors="white")
            ax.spines[["top","right"]].set_visible(False)
            ax.spines[["left","bottom"]].set_color("#444")
            diff = abs(agg.iloc[0] - agg.iloc[1])
            ax.text(0.5, 0.92, f"Δ = {diff:.2f}%", transform=ax.transAxes,
                     ha="center", color="yellow", fontsize=11)

        # Price band delay rate
        ax2 = axes[2]
        ax2.set_facecolor("#1a1a1a")
        agg2   = df.groupby("price_band", observed=True)[TARGET].mean() * 100
        colors = _gnuplot_colors(len(agg2))
        bars   = ax2.bar(agg2.index.astype(str), agg2.values,
                          color=colors, edgecolor="#0d0d0d", linewidth=1.5)
        for bar, val in zip(bars, agg2.values):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                      bar.get_height() + 0.05, f"{val:.1f}%",
                      ha="center", va="bottom", color="white", fontsize=10)
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax2.set_title("Delay Rate by Total Price Band (BRL)",
                       color="white", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Price Band (BRL)", color="white")
        ax2.set_ylabel("Delay Rate (%)", color="white")
        ax2.tick_params(colors="white")
        ax2.spines[["top","right"]].set_visible(False)
        ax2.spines[["left","bottom"]].set_color("#444")

        fig.suptitle("🏠 Location & Price Features vs Delivery Delay",
                     color="white", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "06_binary_price_features.png",
                    dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
        plt.close()
        print("  ✔ Saved: 06_binary_price_features.png")

    # ═════════════════════════════════════════════════════════════════════════
    # 7. CATEGORY GROUP vs DELAY RATE
    # ═════════════════════════════════════════════════════════════════════════
    def plot_category_delay(self, save: bool = True):
        """Horizontal bar chart of delay rate per product category group."""
        df    = self.df
        rates = {}
        for col in CAT_GROUPS:
            name  = col.replace("category_group_","").replace("_"," ").title()
            rates[name] = df[df[col] == True][TARGET].mean() * 100
        rates = pd.Series(rates).sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(14, 7), facecolor="#0d0d0d")
        ax.set_facecolor("#1a1a1a")
        colors = _gnuplot_colors(len(rates))[::-1]
        bars   = ax.barh(rates.index, rates.values, color=colors,
                          edgecolor="#0d0d0d", linewidth=1.2)
        for bar, val in zip(bars, rates.values):
            ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                     f"{val:.2f}%", va="center", color="white", fontsize=10)
        overall = df[TARGET].mean() * 100
        ax.axvline(overall, color="yellow", linewidth=2, linestyle="--",
                    label=f"Overall avg: {overall:.1f}%")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_title("Delay Rate by Product Category Group",
                      color="white", fontsize=15, fontweight="bold")
        ax.set_xlabel("Delay Rate (%)", color="white")
        ax.tick_params(colors="white")
        ax.spines[["top","right"]].set_visible(False)
        ax.spines[["left","bottom"]].set_color("#444")
        ax.legend(facecolor="#2a2a2a", labelcolor="white", fontsize=11)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "07_category_delay.png",
                    dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
        plt.close()
        print("  ✔ Saved: 07_category_delay.png")

    # ═════════════════════════════════════════════════════════════════════════
    # 8. CORRELATION HEATMAP
    # ═════════════════════════════════════════════════════════════════════════
    def plot_correlation_heatmap(self, save: bool = True):
        """Correlation matrix of all numerical features including target."""
        df       = self.df
        num_cols = NUM_FEATURES + [TARGET]
        corr     = df[num_cols].corr()

        fig, ax = plt.subplots(figsize=(14, 11), facecolor="#0d0d0d")
        ax.set_facecolor("#0d0d0d")
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f",
            cmap=PALETTE, center=0, linewidths=0.5,
            linecolor="#0d0d0d", ax=ax,
            annot_kws={"size": 9, "color": "white"},
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title("Correlation Heatmap – Numerical Features + Target",
                      color="white", fontsize=15, fontweight="bold", pad=12)
        ax.tick_params(colors="white", labelsize=9)
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right")

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "08_correlation_heatmap.png",
                    dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
        plt.close()
        print("  ✔ Saved: 08_correlation_heatmap.png")

    # ═════════════════════════════════════════════════════════════════════════
    # 9. FEATURE IMPORTANCE (point-biserial correlation with target)
    # ═════════════════════════════════════════════════════════════════════════
    def plot_feature_importance(self, save: bool = True):
        """
        Horizontal bar of |correlation| with is_delayed for numeric features.
        """
        df    = self.df
        corrs = (df[NUM_FEATURES + BINARY_FEATURES + [TARGET]]
                   .corr()[TARGET]
                   .drop(TARGET)
                   .abs()
                   .sort_values(ascending=True))

        fig, ax = plt.subplots(figsize=(13, 7), facecolor="#0d0d0d")
        ax.set_facecolor("#1a1a1a")
        colors = _gnuplot_colors(len(corrs))
        bars   = ax.barh(corrs.index, corrs.values, color=colors,
                          edgecolor="#0d0d0d", linewidth=1.2)
        for bar, val in zip(bars, corrs.values):
            ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                     f"{val:.4f}", va="center", color="white", fontsize=10)
        ax.set_title("|Pearson Correlation| with is_delayed",
                      color="white", fontsize=15, fontweight="bold")
        ax.set_xlabel("|Correlation Coefficient|", color="white")
        ax.tick_params(colors="white")
        ax.spines[["top","right"]].set_visible(False)
        ax.spines[["left","bottom"]].set_color("#444")

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "09_feature_importance.png",
                    dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
        plt.close()
        print("  ✔ Saved: 09_feature_importance.png")

    # ═════════════════════════════════════════════════════════════════════════
    # 10. MULTIVARIATE – Distance × Approval Delay coloured by target
    # ═════════════════════════════════════════════════════════════════════════
    def plot_multivariate(self, save: bool = True):
        """Scatter: distance_km vs approval_delay, hue = is_delayed."""
        df     = self.df
        sample = df.sample(min(8000, len(df)), random_state=42)

        fig, ax = plt.subplots(figsize=(14, 7), facecolor="#0d0d0d")
        ax.set_facecolor("#1a1a1a")
        clrs = {0: _gnuplot_colors(4)[0], 1: _gnuplot_colors(4)[3]}
        for label, grp in sample.groupby(TARGET):
            ax.scatter(
                grp["distance_km"], np.log1p(grp["approval_delay"]),
                c=[clrs[label]] * len(grp), alpha=0.35, s=15,
                label="Delayed" if label else "On Time",
                edgecolors="none",
            )
        ax.set_title("Distance vs log(Approval Delay) — coloured by Delay Status",
                      color="white", fontsize=14, fontweight="bold")
        ax.set_xlabel("Distance (km)", color="white")
        ax.set_ylabel("log(1 + Approval Delay)", color="white")
        ax.tick_params(colors="white")
        ax.spines[["top","right"]].set_visible(False)
        ax.spines[["left","bottom"]].set_color("#444")
        legend = ax.legend(facecolor="#2a2a2a", labelcolor="white",
                            fontsize=11, markerscale=2)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "10_multivariate_scatter.png",
                    dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
        plt.close()
        print("  ✔ Saved: 10_multivariate_scatter.png")

    # ═════════════════════════════════════════════════════════════════════════
    # 11. CUSTOMER STATE DELAY RATES (top 10)
    # ═════════════════════════════════════════════════════════════════════════
    def plot_state_delay(self, save: bool = True):
        """Delay rate by top customer states."""
        df     = self.df
        states = [c for c in df.columns if c.startswith("customer_state_")]
        rates  = {}
        for col in states:
            name = col.replace("customer_state_", "")
            sub  = df[df[col] == True]
            if len(sub) > 200:          # only states with enough data
                rates[name] = sub[TARGET].mean() * 100
        rates = pd.Series(rates).sort_values(ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(14, 6), facecolor="#0d0d0d")
        ax.set_facecolor("#1a1a1a")
        colors = _gnuplot_colors(len(rates))
        bars   = ax.bar(rates.index, rates.values, color=colors,
                         edgecolor="#0d0d0d", linewidth=1.5)
        for bar, val in zip(bars, rates.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.05, f"{val:.1f}%",
                     ha="center", va="bottom", color="white", fontsize=10)
        overall = df[TARGET].mean() * 100
        ax.axhline(overall, color="yellow", linewidth=2, linestyle="--",
                    label=f"Overall: {overall:.1f}%")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_title("Top 10 Customer States by Delay Rate",
                      color="white", fontsize=15, fontweight="bold")
        ax.set_xlabel("State", color="white")
        ax.set_ylabel("Delay Rate (%)", color="white")
        ax.tick_params(colors="white")
        ax.spines[["top","right"]].set_visible(False)
        ax.spines[["left","bottom"]].set_color("#444")
        ax.legend(facecolor="#2a2a2a", labelcolor="white")

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "11_state_delay.png",
                    dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
        plt.close()
        print("  ✔ Saved: 11_state_delay.png")

    # ═════════════════════════════════════════════════════════════════════════
    # MASTER RUNNER
    # ═════════════════════════════════════════════════════════════════════════
    def run_full_eda(self):
        """Execute all EDA plots in sequence."""
        print("\n" + "=" * 60)
        print("  DELIVERY DELAY EDA  –  Generating 11 plots")
        print("=" * 60)
        self.plot_dataset_overview()
        self.plot_numeric_vs_target()
        self.plot_distance_delay()
        self.plot_time_patterns()
        self.plot_approval_delay()
        self.plot_binary_features()
        self.plot_category_delay()
        self.plot_correlation_heatmap()
        self.plot_feature_importance()
        self.plot_multivariate()
        self.plot_state_delay()
        print("\n✅ All plots saved to ")

    # ═════════════════════════════════════════════════════════════════════════
    # UTILITY: print key textual insights
    # ═════════════════════════════════════════════════════════════════════════
    def print_insights(self):
        """Print a concise summary of major data insights."""
        df = self.df
        sep = "─" * 55

        print(f"\n{sep}")
        print("  📊  KEY DATA INSIGHTS")
        print(sep)

        # Overall delay rate
        delay_rate = df[TARGET].mean() * 100
        print(f"\n1. Overall Delay Rate : {delay_rate:.2f}%  "
              f"({df[TARGET].sum():,} / {len(df):,} orders)")

        # Distance
        grp = df.groupby("distance_band", observed=True)[TARGET].mean() * 100
        print(f"\n2. Distance Impact :")
        for band, rate in grp.items():
            print(f"     {str(band):15s}  →  {rate:.2f}%")

        # Same state
        ss = df.groupby("is_same_state")[TARGET].mean() * 100
        print(f"\n3. Same-State Shipments  : {ss[1]:.2f}%  delay")
        print(f"   Cross-State Shipments  : {ss[0]:.2f}%  delay")

        # Approval delay
        zero_appr = df[df["approval_delay"] == 0][TARGET].mean() * 100
        hi_appr   = df[df["approval_delay"] > 3][TARGET].mean() * 100
        print(f"\n4. Approval Delay = 0d  : {zero_appr:.2f}%  delayed")
        print(f"   Approval Delay > 3d  : {hi_appr:.2f}%  delayed")

        # Day of week
        dow = df.groupby("purchase_day_of_week")[TARGET].mean() * 100
        print(f"\n5. Highest delay day  : {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][dow.idxmax()]}  ({dow.max():.2f}%)")
        print(f"   Lowest  delay day  : {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][dow.idxmin()]}  ({dow.min():.2f}%)")

        # Categories
        cat_rates = {}
        for col in CAT_GROUPS:
            name = col.replace("category_group_","").replace("_"," ").title()
            cat_rates[name] = df[df[col] == True][TARGET].mean() * 100
        best  = max(cat_rates, key=cat_rates.get)
        worst = min(cat_rates, key=cat_rates.get)
        print(f"\n6. Highest delay category : {best}  ({cat_rates[best]:.2f}%)")
        print(f"   Lowest  delay category : {worst}  ({cat_rates[worst]:.2f}%)")

        print(f"\n{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    eda = DeliveryDelayEDA(DATA_PATH)
    eda.print_insights()
    eda.run_full_eda()
