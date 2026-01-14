import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("Creating 1x2 plot: GDP per capita evolution and classification heatmap...")

# ---- file/location settings ----
DATA_PATH = Path("data/eurostat_green_panel.csv")
COUNTRY_COL = "geo"
YEAR_COL = "time"
GDP_PC_COL = "real_gdp_per_capita"

# Load data
raw_df = pd.read_csv(DATA_PATH)
df = raw_df[[COUNTRY_COL, YEAR_COL, GDP_PC_COL]].copy()
df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
df[GDP_PC_COL] = pd.to_numeric(df[GDP_PC_COL], errors="coerce")
df = df.dropna(subset=[YEAR_COL, GDP_PC_COL])
df = df[(df[YEAR_COL] >= 1995) & (df[YEAR_COL] <= 2022)]

# Exclude UK from analysis
df = df[df[COUNTRY_COL] != "UK"]

# Calculate yearly median (without UK)
yearly_median = df.groupby(YEAR_COL)[GDP_PC_COL].median().reset_index()
yearly_median.columns = [YEAR_COL, "median_gdp_pc"]

# Merge and classify
df = df.merge(yearly_median, on=YEAR_COL, how="left")
df["income_group"] = np.where(df[GDP_PC_COL] > df["median_gdp_pc"], "High", "Low")

# Get initial classification (1995) for each country
initial_class = df[df[YEAR_COL] == 1995][[COUNTRY_COL, "income_group"]].drop_duplicates()
initial_class_dict = dict(zip(initial_class[COUNTRY_COL], initial_class["income_group"]))

# Generate distinct colors for each country
countries = sorted(df[COUNTRY_COL].unique())
n_countries = len(countries)
if n_countries > 20:
    cmap1 = plt.colormaps.get_cmap('tab20b')
    cmap2 = plt.colormaps.get_cmap('tab20c')
    colors_list = [cmap1(i/20) for i in range(20)] + [cmap2(i/(n_countries-20)) for i in range(n_countries-20)]
else:
    cmap = plt.colormaps.get_cmap('tab20c')
    colors_list = [cmap(i/n_countries) for i in range(n_countries)]

country_colors = dict(zip(countries, colors_list))

# Create 1x2 figure
fig = plt.figure(figsize=(24, 10))
gs = fig.add_gridspec(1, 2, hspace=0.15, wspace=0.15, left=0.05, right=0.98, top=0.93, bottom=0.08)

# ==================== LEFT: Spaghetti plot - all countries over time ====================
ax1 = fig.add_subplot(gs[0, 0])

# Store final positions for labeling
final_positions = []

for country in countries:
    country_data = df[df[COUNTRY_COL] == country].sort_values(YEAR_COL)
    if len(country_data) > 0:
        color = country_colors[country]
        initial_group = initial_class_dict.get(country, "Low")
        linewidth = 2.0 if initial_group == "High" else 1.5
        ax1.plot(country_data[YEAR_COL], country_data[GDP_PC_COL], 
                 color=color, alpha=0.85, linewidth=linewidth)
        
        # Store final position for label
        final_year = country_data[YEAR_COL].iloc[-1]
        final_value = country_data[GDP_PC_COL].iloc[-1]
        final_positions.append((country, final_year, final_value, color))

# Sort by final value to handle overlaps
final_positions.sort(key=lambda x: x[2])

# Adjust positions to avoid overlaps
MIN_SPACING = 2000
adjusted_positions = []
for i, (country, year, value, color) in enumerate(final_positions):
    adjusted_value = value
    for prev_item in adjusted_positions:
        prev_adj_value = prev_item[2]
        if abs(adjusted_value - prev_adj_value) < MIN_SPACING:
            adjusted_value = prev_adj_value + MIN_SPACING
    adjusted_positions.append((country, year, adjusted_value, color, value))

# Add country labels with adjusted positions
for country, year, adj_value, color, orig_value in adjusted_positions:
    ax1.text(year + 0.4, adj_value, country, fontsize=8, va='center', 
             color=color, fontweight='bold', alpha=1.0)
    if abs(adj_value - orig_value) > MIN_SPACING * 0.3:
        ax1.plot([year, year + 0.3], [orig_value, adj_value], 
                color=color, linewidth=0.5, alpha=0.4, linestyle=':')

# Plot yearly median on top
ax1.plot(yearly_median[YEAR_COL], yearly_median["median_gdp_pc"], 
         color='red', linewidth=3, linestyle='--', label='Yearly Median', zorder=100, alpha=0.9)

ax1.set_xlabel("Year", fontsize=12)
ax1.set_ylabel("Real GDP per Capita (2015 €)", fontsize=12)
ax1.set_title("(A) Real GDP per Capita Evolution by Country (1995-2022)", 
              fontsize=14, fontweight='bold', pad=20)
ax1.grid(True, linestyle='--', linewidth=0.6, alpha=0.4)
ax1.legend(loc='upper left', fontsize=11)
ax1.set_xlim(1995, 2023)

# ==================== RIGHT: Classification heatmap ====================
ax2 = fig.add_subplot(gs[0, 1])

# Prepare classification data
classification_pivot = df.pivot_table(
    index=COUNTRY_COL, 
    columns=YEAR_COL, 
    values="income_group", 
    aggfunc="first"
)

# Convert to binary (1 = High, 0 = Low)
classification_binary = classification_pivot.map(lambda x: 1 if x == "High" else 0)

# Sort by average classification (countries with more years in high income on top)
row_avg = classification_binary.mean(axis=1).sort_values(ascending=False)
classification_binary = classification_binary.loc[row_avg.index]

# Use blue colormap instead of green
sns.heatmap(
    classification_binary, 
    cmap=["lightblue", "#1f77b4"],  # Light blue for low, dark blue for high
    cbar_kws={"label": "Income Group", "ticks": [0.25, 0.75]},
    linewidths=0.5,
    linecolor="white",
    ax=ax2
)
colorbar = ax2.collections[0].colorbar
colorbar.set_ticklabels(["Low", "High"])
ax2.set_xlabel("Year", fontsize=12)
ax2.set_ylabel("Country", fontsize=12)
ax2.set_title("(B) Income Classification Stability by Country and Year (1995-2022)\nDark Blue = High Income | Light Blue = Low Income", 
              fontsize=14, fontweight='bold', pad=20)

plt.savefig("Figure_gdp_classification_1x2.png", dpi=300, bbox_inches="tight")
plt.close()

print("  Saved: Figure_gdp_classification_1x2.png")
print("\nFigure created successfully!")
