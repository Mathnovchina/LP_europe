import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import geopandas as gpd
from matplotlib.colors import Normalize

print("Starting environmental spending analysis with maps...")

# ---- file/location settings ----
DATA_PATH = Path("data/eurostat_green_panel.csv")
COUNTRY_COL = "geo"
YEAR_COL = "time"

print(f"Loading data from {DATA_PATH}...")
# Load data
raw_df = pd.read_csv(DATA_PATH)

# Filter for 1995-2022, exclude UK
df = raw_df.copy()
df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
df = df[(df[YEAR_COL] >= 1995) & (df[YEAR_COL] <= 2022)]
df = df[df[COUNTRY_COL] != "UK"]

print(f"Loaded data for {df[COUNTRY_COL].nunique()} countries (excluding UK)")

# ==================== Calculate Environmental Spending Metrics ====================

# 1. Environmental spending share of GDP
share_df = df[[COUNTRY_COL, "env_total_const_2015", "gdp_const_2015"]].dropna()
share_df["env_total_const_2015"] = pd.to_numeric(share_df["env_total_const_2015"], errors="coerce")
share_df["gdp_const_2015"] = pd.to_numeric(share_df["gdp_const_2015"], errors="coerce")
share_df = share_df.dropna()
share_df = share_df[share_df["gdp_const_2015"] != 0]
share_df["green_share"] = (share_df["env_total_const_2015"] / share_df["gdp_const_2015"]) * 100.0

# Average by country
share_stats = share_df.groupby(COUNTRY_COL).agg(
    mean_share=("green_share", "mean"),
    std_share=("green_share", "std"),
    n_obs=("green_share", "count")
).reset_index()
share_stats["err_68"] = share_stats["std_share"]

# 2. Environmental spending per capita
per_capita_df = df[[COUNTRY_COL, "env_total_const_2015", "population"]].dropna()
per_capita_df["env_total_const_2015"] = pd.to_numeric(per_capita_df["env_total_const_2015"], errors="coerce")
per_capita_df["population"] = pd.to_numeric(per_capita_df["population"], errors="coerce")
per_capita_df = per_capita_df.dropna()
per_capita_df = per_capita_df[per_capita_df["population"] != 0]
# Convert millions of euros to euros per capita
per_capita_df["env_per_capita"] = (per_capita_df["env_total_const_2015"] * 1_000_000) / per_capita_df["population"]

# Average by country
per_capita_stats = per_capita_df.groupby(COUNTRY_COL).agg(
    mean_per_capita=("env_per_capita", "mean"),
    std_per_capita=("env_per_capita", "std"),
    n_obs=("env_per_capita", "count")
).reset_index()
per_capita_stats["err_68"] = per_capita_stats["std_per_capita"]

# ==================== Load European Geometries ====================
print("Loading European country geometries...")
# Download Natural Earth data directly
import urllib.request
import zipfile
import os

# Check if we already have the data
ne_path = Path("ne_110m_admin_0_countries")
if not ne_path.exists():
    print("Downloading Natural Earth data...")
    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    zip_path = "ne_countries.zip"
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("ne_110m_admin_0_countries")
    os.remove(zip_path)
    print("Download complete.")

world = gpd.read_file("ne_110m_admin_0_countries")

# Use ADM0_A3 as fallback for countries with -99 in ISO_A3
world['ISO_CODE'] = world['ISO_A3'].where(world['ISO_A3'] != '-99', world['ADM0_A3'])

# Map ISO2 codes to ISO3 codes used in geopandas
iso2_to_iso3 = {
    'AT': 'AUT', 'BE': 'BEL', 'BG': 'BGR', 'CH': 'CHE', 'CY': 'CYP',
    'CZ': 'CZE', 'DE': 'DEU', 'DK': 'DNK', 'EE': 'EST', 'EL': 'GRC',
    'ES': 'ESP', 'FI': 'FIN', 'FR': 'FRA', 'HR': 'HRV', 'HU': 'HUN',
    'IE': 'IRL', 'IS': 'ISL', 'IT': 'ITA', 'LT': 'LTU', 'LU': 'LUX',
    'LV': 'LVA', 'MT': 'MLT', 'NL': 'NLD', 'NO': 'NOR', 'PL': 'POL',
    'PT': 'PRT', 'RO': 'ROU', 'SE': 'SWE', 'SI': 'SVN', 'SK': 'SVK'
}

# Prepare data for mapping
share_stats['iso3'] = share_stats[COUNTRY_COL].map(iso2_to_iso3)
per_capita_stats['iso3'] = per_capita_stats[COUNTRY_COL].map(iso2_to_iso3)

# Merge with geometries
europe_share = world[world['ISO_CODE'].isin(share_stats['iso3'])].merge(
    share_stats[['iso3', 'mean_share']], left_on='ISO_CODE', right_on='iso3', how='left'
)
europe_per_capita = world[world['ISO_CODE'].isin(per_capita_stats['iso3'])].merge(
    per_capita_stats[['iso3', 'mean_per_capita']], left_on='ISO_CODE', right_on='iso3', how='left'
)

print(f"Countries plotted on map: {len(europe_share[europe_share['mean_share'].notna()])}")

# Note: Malta (MT) is too small to appear in low-resolution Natural Earth data
# We'll add it manually as a marker
malta_coords = (14.5, 35.9)  # Approximate center of Malta

# ==================== Create 2x2 Plot ====================
print("Creating 2x2 plot with choropleth maps and histograms...")

fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(2, 2, hspace=0.15, wspace=0.15, left=0.05, right=0.95, top=0.97, bottom=0.05,
                      width_ratios=[1, 1], height_ratios=[1, 1])

# Define color schemes
cmap_share = plt.cm.YlGn  # Yellow-Green for share
cmap_per_capita = plt.cm.YlOrRd  # Yellow-Orange-Red for per capita

# Create normalization functions for Malta manual markers
from matplotlib.colors import Normalize
norm_share = Normalize(vmin=share_stats['mean_share'].min(), vmax=share_stats['mean_share'].max())
norm_pc = Normalize(vmin=per_capita_stats['mean_per_capita'].min(), vmax=per_capita_stats['mean_per_capita'].max())

# ==================== TOP LEFT: Choropleth Map - Environmental Spending Share ====================
ax1 = fig.add_subplot(gs[0, 0])

# Plot all European context (light gray)
world_europe = world[world['CONTINENT'] == 'Europe']
world_europe.plot(ax=ax1, color='lightgray', edgecolor='white', linewidth=0.5)

# Plot data countries with color scale
europe_share.plot(
    column='mean_share',
    ax=ax1,
    legend=False,
    cmap=cmap_share,
    edgecolor='black',
    linewidth=0.8,
    missing_kwds={'color': 'lightgray'}
)

# Add country labels
for idx, row in europe_share.iterrows():
    if pd.notna(row['mean_share']):
        # Find the ISO2 code
        iso2 = [k for k, v in iso2_to_iso3.items() if v == row['ISO_CODE']]
        if iso2:
            centroid = row['geometry'].centroid
            # Manual position for France to be on its territory
            if iso2[0] == 'FR':
                ax1.text(2.5, 47, iso2[0], 
                        fontsize=7, ha='center', va='center', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
            else:
                ax1.text(centroid.x, centroid.y, iso2[0], 
                        fontsize=7, ha='center', va='center', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))

ax1.set_xlim(-25, 35)
ax1.set_ylim(35, 72)
ax1.set_xlabel('Longitude', fontsize=11)
ax1.set_ylabel('Latitude', fontsize=11)
ax1.set_title('(A) Average Environmental Spending Share of GDP by Country\n(%, 1995-2022)', 
              fontsize=12, fontweight='bold', pad=20, y=1.0)
ax1.axis('off')

# Add dashed box around Iceland to show artificial positioning
from matplotlib.patches import Rectangle
iceland_box = Rectangle((-24.5, 63.3), width=11, height=3.5, 
                        fill=False, edgecolor='gray', linewidth=1.5, 
                        linestyle='--', zorder=99)
ax1.add_patch(iceland_box)

# Add Malta manually if not in map
if 'MT' in share_stats[COUNTRY_COL].values:
    mt_value = share_stats[share_stats[COUNTRY_COL] == 'MT']['mean_share'].values[0]
    mt_color = cmap_share(norm_share(mt_value))
    ax1.scatter(malta_coords[0], malta_coords[1], s=300, c=[mt_color], 
               edgecolors='black', linewidth=2, alpha=0.9, marker='o', zorder=100)
    ax1.text(malta_coords[0], malta_coords[1], 'MT', 
            fontsize=7, ha='center', va='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'), zorder=101)

# Add colorbar
norm_share = Normalize(vmin=share_stats['mean_share'].min(), vmax=share_stats['mean_share'].max())
sm_share = plt.cm.ScalarMappable(cmap=cmap_share, norm=norm_share)
sm_share.set_array([])
cbar1 = plt.colorbar(sm_share, ax=ax1, orientation='horizontal', pad=0.08, aspect=40, shrink=0.9)
cbar1.set_label('Environmental Spending Share (%)', fontsize=10)

# Adjust map position to make it slightly larger
pos1 = ax1.get_position()
ax1.set_position([pos1.x0, pos1.y0 - pos1.height * 0.1, pos1.width * 1.15, pos1.height * 1.15])

# ==================== TOP RIGHT: Histogram - Environmental Spending Share ====================
ax2 = fig.add_subplot(gs[0, 1])
share_stats_sorted = share_stats.sort_values('mean_share', ascending=False)
colors_share = [cmap_share(norm_share(val)) for val in share_stats_sorted['mean_share']]

bars2 = ax2.bar(
    range(len(share_stats_sorted)),
    share_stats_sorted['mean_share'],
    yerr=share_stats_sorted['err_68'],
    capsize=5,
    color=colors_share,
    edgecolor='black',
    alpha=0.8,
    linewidth=1
)

ax2.set_ylabel('Environmental Spending Share (%)', fontsize=11)
ax2.set_title('(B) Average Environmental Spending Share of GDP by Country\n(68% interval, %, 1995-2022)', 
              fontsize=12, fontweight='bold', pad=20, y=1.0)
ax2.set_xticks(range(len(share_stats_sorted)))
ax2.set_xticklabels(share_stats_sorted[COUNTRY_COL], rotation=45, ha='right', fontsize=9)
ax2.axhline(0, color='black', linewidth=0.8)
ax2.margins(x=0.01)
ax2.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.5)

# Adjust histogram position to align bottom with map
pos2 = ax2.get_position()
ax2.set_position([pos2.x0, pos2.y0 + pos2.height * 0.08, pos2.width, pos2.height * 0.95])

# ==================== BOTTOM LEFT: Choropleth Map - Environmental Spending Per Capita ====================
ax3 = fig.add_subplot(gs[1, 0])

# Plot all European context (light gray)
world_europe.plot(ax=ax3, color='lightgray', edgecolor='white', linewidth=0.5)

# Plot data countries with color scale
europe_per_capita.plot(
    column='mean_per_capita',
    ax=ax3,
    legend=False,
    cmap=cmap_per_capita,
    edgecolor='black',
    linewidth=0.8,
    missing_kwds={'color': 'lightgray'}
)

# Add country labels
for idx, row in europe_per_capita.iterrows():
    if pd.notna(row['mean_per_capita']):
        # Find the ISO2 code
        iso2 = [k for k, v in iso2_to_iso3.items() if v == row['ISO_CODE']]
        if iso2:
            centroid = row['geometry'].centroid
            # Manual position for France to be on its territory
            if iso2[0] == 'FR':
                ax3.text(2.5, 47, iso2[0], 
                        fontsize=7, ha='center', va='center', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
            else:
                ax3.text(centroid.x, centroid.y, iso2[0], 
                        fontsize=7, ha='center', va='center', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))

ax3.set_xlim(-25, 35)
ax3.set_ylim(35, 72)
ax3.set_xlabel('Longitude', fontsize=11)
ax3.set_ylabel('Latitude', fontsize=11)
ax3.set_title('(C) Average Environmental Spending Per Capita by Country\n(2015 prices, 1995-2022)', 
              fontsize=12, fontweight='bold', pad=20, y=1.0)
ax3.axis('off')

# Add dashed box around Iceland to show artificial positioning
iceland_box2 = Rectangle((-24.5, 63.3), width=11, height=3.5, 
                         fill=False, edgecolor='gray', linewidth=1.5, 
                         linestyle='--', zorder=99)
ax3.add_patch(iceland_box2)
# Add Malta manually if not in map
if 'MT' in per_capita_stats[COUNTRY_COL].values:
    mt_value = per_capita_stats[per_capita_stats[COUNTRY_COL] == 'MT']['mean_per_capita'].values[0]
    mt_color = cmap_per_capita(norm_pc(mt_value))
    ax3.scatter(malta_coords[0], malta_coords[1], s=300, c=[mt_color], 
               edgecolors='black', linewidth=2, alpha=0.9, marker='o', zorder=100)
    ax3.text(malta_coords[0], malta_coords[1], 'MT', 
            fontsize=7, ha='center', va='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'), zorder=101)
# Add colorbar
norm_per_capita = Normalize(vmin=per_capita_stats['mean_per_capita'].min(), 
                             vmax=per_capita_stats['mean_per_capita'].max())
sm_per_capita = plt.cm.ScalarMappable(cmap=cmap_per_capita, norm=norm_per_capita)
sm_per_capita.set_array([])
cbar3 = plt.colorbar(sm_per_capita, ax=ax3, orientation='horizontal', pad=0.08, aspect=40, shrink=0.9)
cbar3.set_label('Environmental Spending Per Capita (€)', fontsize=10)

# Adjust map position to make it slightly larger
pos3 = ax3.get_position()
ax3.set_position([pos3.x0, pos3.y0 - pos3.height * 0.1, pos3.width * 1.15, pos3.height * 1.15])

# ==================== BOTTOM RIGHT: Histogram - Environmental Spending Per Capita ====================
ax4 = fig.add_subplot(gs[1, 1])
per_capita_stats_sorted = per_capita_stats.sort_values('mean_per_capita', ascending=False)
colors_per_capita = [cmap_per_capita(norm_per_capita(val)) for val in per_capita_stats_sorted['mean_per_capita']]

bars4 = ax4.bar(
    range(len(per_capita_stats_sorted)),
    per_capita_stats_sorted['mean_per_capita'],
    yerr=per_capita_stats_sorted['err_68'],
    capsize=5,
    color=colors_per_capita,
    edgecolor='black',
    alpha=0.8,
    linewidth=1
)

ax4.set_ylabel('Environmental Spending Per Capita (€, 2015 prices)', fontsize=11)
ax4.set_title('(D) Average Environmental Spending Per Capita by Country\n(68% interval, 2015 prices, 1995-2022)', 
              fontsize=12, fontweight='bold', pad=20, y=1.0)
ax4.set_xticks(range(len(per_capita_stats_sorted)))
ax4.set_xticklabels(per_capita_stats_sorted[COUNTRY_COL], rotation=45, ha='right', fontsize=9)
ax4.axhline(0, color='black', linewidth=0.8)
ax4.margins(x=0.01)
ax4.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.5)

# Adjust histogram position to align bottom with map
pos4 = ax4.get_position()
ax4.set_position([pos4.x0, pos4.y0 + pos4.height * 0.08, pos4.width, pos4.height * 0.95])

plt.savefig("Figure_environmental_spending_2x2.png", dpi=300, bbox_inches="tight")
plt.close()

print("  Saved: Figure_environmental_spending_2x2.png")

# ==================== Print Statistics ====================
print("\n" + "="*80)
print("ENVIRONMENTAL SPENDING SUMMARY STATISTICS (1995-2022, excluding UK)")
print("="*80)

print("\nENVIRONMENTAL SPENDING SHARE OF GDP (%):")
print("-" * 80)
print(f"Minimum: {share_stats['mean_share'].min():.3f}% ({share_stats.loc[share_stats['mean_share'].idxmin(), COUNTRY_COL]})")
print(f"Maximum: {share_stats['mean_share'].max():.3f}% ({share_stats.loc[share_stats['mean_share'].idxmax(), COUNTRY_COL]})")
print(f"Mean: {share_stats['mean_share'].mean():.3f}%")
print(f"Median: {share_stats['mean_share'].median():.3f}%")

print("\nENVIRONMENTAL SPENDING PER CAPITA (€, 2015 prices):")
print("-" * 80)
print(f"Minimum: {per_capita_stats['mean_per_capita'].min():.2f}€ ({per_capita_stats.loc[per_capita_stats['mean_per_capita'].idxmin(), COUNTRY_COL]})")
print(f"Maximum: {per_capita_stats['mean_per_capita'].max():.2f}€ ({per_capita_stats.loc[per_capita_stats['mean_per_capita'].idxmax(), COUNTRY_COL]})")
print(f"Mean: {per_capita_stats['mean_per_capita'].mean():.2f}€")
print(f"Median: {per_capita_stats['mean_per_capita'].median():.2f}€")

print("\n" + "="*80)
print("Figure saved successfully!")
print("="*80)
