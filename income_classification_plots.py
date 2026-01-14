import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("Starting income classification analysis...")

# ---- file/location settings ----
DATA_PATH = Path("data/eurostat_green_panel.csv")
COUNTRY_COL = "geo"
YEAR_COL = "time"
GDP_PC_COL = "real_gdp_per_capita"

print(f"Loading data from {DATA_PATH}...")
# Load data
raw_df = pd.read_csv(DATA_PATH)
df = raw_df[[COUNTRY_COL, YEAR_COL, GDP_PC_COL]].copy()
df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
df[GDP_PC_COL] = pd.to_numeric(df[GDP_PC_COL], errors="coerce")
df = df.dropna(subset=[YEAR_COL, GDP_PC_COL])
df = df[(df[YEAR_COL] >= 1995) & (df[YEAR_COL] <= 2022)]

# Exclude UK from analysis
df = df[df[COUNTRY_COL] != "UK"]

print(f"Loaded {len(df)} observations (excluding UK)")

# Calculate yearly median (without UK)
yearly_median = df.groupby(YEAR_COL)[GDP_PC_COL].median().reset_index()
yearly_median.columns = [YEAR_COL, "median_gdp_pc"]

# Merge and classify
df = df.merge(yearly_median, on=YEAR_COL, how="left")
df["income_group"] = np.where(df[GDP_PC_COL] > df["median_gdp_pc"], "High", "Low")

# Calculate average GDP per capita by country over entire period
avg_gdp_by_country = df.groupby(COUNTRY_COL)[GDP_PC_COL].mean().reset_index()
avg_gdp_by_country.columns = [COUNTRY_COL, "avg_gdp_pc"]
avg_gdp_by_country = avg_gdp_by_country.sort_values("avg_gdp_pc", ascending=False)

# Overall median (across all years)
overall_median = df[GDP_PC_COL].median()

print("Creating Figure 1: Bar chart with median line...")
# ==================== FIGURE 1: Bar chart with median line ====================
fig1, ax1 = plt.subplots(figsize=(14, 7))
colors = ["#2d8659" if val > overall_median else "#66b266" for val in avg_gdp_by_country["avg_gdp_pc"]]
bars = ax1.bar(
    range(len(avg_gdp_by_country)),
    avg_gdp_by_country["avg_gdp_pc"],
    color=colors,
    edgecolor="#1a4d33",
    alpha=0.85,
)
ax1.axhline(overall_median, color="red", linestyle="--", linewidth=2, label=f"Overall Median: {overall_median:,.0f} €")
ax1.set_ylabel("Average Real GDP per Capita (2015 €)")
ax1.set_title("Average Real GDP per Capita by Country (1995-2022) with Overall Median Threshold")
ax1.set_xticks(range(len(avg_gdp_by_country)))
ax1.set_xticklabels(avg_gdp_by_country[COUNTRY_COL], rotation=45, ha="right")
ax1.legend(loc="upper right")
ax1.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
plt.tight_layout()
plt.savefig("Figure_income_classification_bar.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: Figure_income_classification_bar.png")

print("Creating Figure 2: Evolution of median over time...")
# ==================== FIGURE 2: Evolution of median over time ====================
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(yearly_median[YEAR_COL], yearly_median["median_gdp_pc"], 
         marker="o", linewidth=2.5, markersize=6, color="#2d8659", label="Yearly Median GDP per Capita")
ax2.fill_between(yearly_median[YEAR_COL], yearly_median["median_gdp_pc"], 
                 alpha=0.3, color="#66b266")
ax2.set_xlabel("Year")
ax2.set_ylabel("Real GDP per Capita (2015 €)")
ax2.set_title("Evolution of Median Real GDP per Capita Across European Countries (1995-2022)")
ax2.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
ax2.legend()
plt.tight_layout()
plt.savefig("Figure_median_evolution.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: Figure_median_evolution.png")

print("Creating Figure 3: Spaghetti plot - all countries over time...")
# ==================== FIGURE 3: Spaghetti plot - all countries over time ====================
fig3, ax3 = plt.subplots(figsize=(18, 10))

# Get initial classification (1995) for each country
initial_class = df[df[YEAR_COL] == 1995][[COUNTRY_COL, "income_group"]].drop_duplicates()
initial_class_dict = dict(zip(initial_class[COUNTRY_COL], initial_class["income_group"]))

# Generate distinct colors for each country
countries = sorted(df[COUNTRY_COL].unique())
n_countries = len(countries)
# Use a colormap with distinct colors
if n_countries > 20:
    # Combine two colormaps for more countries
    cmap1 = plt.colormaps.get_cmap('tab20b')
    cmap2 = plt.colormaps.get_cmap('tab20c')
    colors_list = [cmap1(i/20) for i in range(20)] + [cmap2(i/(n_countries-20)) for i in range(n_countries-20)]
else:
    cmap = plt.colormaps.get_cmap('tab20c')
    colors_list = [cmap(i/n_countries) for i in range(n_countries)]

country_colors = dict(zip(countries, colors_list))

# Store final positions for labeling
final_positions = []

for country in countries:
    country_data = df[df[COUNTRY_COL] == country].sort_values(YEAR_COL)
    if len(country_data) > 0:
        color = country_colors[country]
        initial_group = initial_class_dict.get(country, "Low")
        # Use thicker lines for high-income countries
        linewidth = 2.0 if initial_group == "High" else 1.5
        linestyle = '-' if initial_group == "High" else '-'
        ax3.plot(country_data[YEAR_COL], country_data[GDP_PC_COL], 
                 color=color, alpha=0.85, linewidth=linewidth, linestyle=linestyle)
        
        # Store final position for label
        final_year = country_data[YEAR_COL].iloc[-1]
        final_value = country_data[GDP_PC_COL].iloc[-1]
        final_positions.append((country, final_year, final_value, color))

# Sort by final value to handle overlaps
final_positions.sort(key=lambda x: x[2])

# Adjust positions to avoid overlaps
MIN_SPACING = 2000  # Minimum spacing between labels in GDP per capita units
adjusted_positions = []
for i, (country, year, value, color) in enumerate(final_positions):
    adjusted_value = value
    # Check against all previously placed labels
    for prev_item in adjusted_positions:
        prev_adj_value = prev_item[2]
        if abs(adjusted_value - prev_adj_value) < MIN_SPACING:
            # Adjust upward if too close
            adjusted_value = prev_adj_value + MIN_SPACING
    adjusted_positions.append((country, year, adjusted_value, color, value))

# Add country labels with adjusted positions
for country, year, adj_value, color, orig_value in adjusted_positions:
    ax3.text(year + 0.4, adj_value, country, fontsize=8, va='center', 
             color=color, fontweight='bold', alpha=1.0)
    # Draw a small connecting line if position was adjusted significantly
    if abs(adj_value - orig_value) > MIN_SPACING * 0.3:
        ax3.plot([year, year + 0.35], [orig_value, adj_value], 
                color=color, linewidth=0.5, alpha=0.6, linestyle='-')

# Plot median line
ax3.plot(yearly_median[YEAR_COL], yearly_median["median_gdp_pc"], 
         color="black", linestyle="--", linewidth=2.5, label="Median GDP per Capita", zorder=100)

ax3.set_xlabel("Year", fontsize=12)
ax3.set_ylabel("Real GDP per Capita (2015 €)", fontsize=12)
ax3.set_title("Evolution of Real GDP per Capita by Country (1995-2022)", fontsize=13)
ax3.set_xlim(1995, 2024.5)  # Extend x-axis to make room for labels
ax3.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
ax3.legend(fontsize=11)
plt.tight_layout()
plt.savefig("Figure_gdp_evolution_all_countries.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: Figure_gdp_evolution_all_countries.png")

print("Creating Figure 4: Heatmap - Income classification stability...")
# ==================== FIGURE 4: Heatmap - Income classification stability ====================
# Create pivot table: countries x years
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

fig4, ax4 = plt.subplots(figsize=(16, 10))
sns.heatmap(
    classification_binary, 
    cmap=["#66b266", "#2d8659"],
    cbar_kws={"label": "Income Group", "ticks": [0.25, 0.75]},
    linewidths=0.5,
    linecolor="white",
    ax=ax4
)
colorbar = ax4.collections[0].colorbar
colorbar.set_ticklabels(["Low", "High"])
ax4.set_xlabel("Year")
ax4.set_ylabel("Country")
ax4.set_title("Income Classification Stability by Country and Year (1995-2022)\nDark Green = High Income | Light Green = Low Income")
plt.tight_layout()
plt.savefig("Figure_income_classification_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: Figure_income_classification_heatmap.png")

print("Creating Figure 5: Count of high vs low income countries over time...")
# ==================== FIGURE 5: Count of high vs low income countries over time ====================
income_counts = df.groupby([YEAR_COL, "income_group"]).size().reset_index(name="count")
income_counts_pivot = income_counts.pivot(index=YEAR_COL, columns="income_group", values="count").fillna(0)

fig5, ax5 = plt.subplots(figsize=(12, 6))
ax5.bar(income_counts_pivot.index, income_counts_pivot["High"], 
        label="High-Income Countries", color="#2d8659", alpha=0.85, edgecolor="#1a4d33")
ax5.bar(income_counts_pivot.index, income_counts_pivot["Low"], 
        bottom=income_counts_pivot["High"], label="Low-Income Countries", 
        color="#66b266", alpha=0.85, edgecolor="#1a4d33")
ax5.set_xlabel("Year")
ax5.set_ylabel("Number of Countries")
ax5.set_title("Distribution of High-Income vs Low-Income Countries Over Time (1995-2022)")
ax5.legend()
ax5.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
plt.tight_layout()
plt.savefig("Figure_income_distribution_over_time.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: Figure_income_distribution_over_time.png")

print("Creating Figure 6: Classification transitions...")
# ==================== FIGURE 6: Classification transitions ====================
# Calculate how many times each country changed classification
transitions = []
for country in df[COUNTRY_COL].unique():
    country_data = df[df[COUNTRY_COL] == country].sort_values(YEAR_COL)
    if len(country_data) > 1:
        changes = (country_data["income_group"] != country_data["income_group"].shift()).sum() - 1
        transitions.append({COUNTRY_COL: country, "transitions": changes})

transitions_df = pd.DataFrame(transitions).sort_values("transitions", ascending=False)

fig6, ax6 = plt.subplots(figsize=(14, 7))
colors_trans = ["#d9534f" if t > 0 else "#2d8659" for t in transitions_df["transitions"]]
ax6.bar(range(len(transitions_df)), transitions_df["transitions"], 
        color=colors_trans, edgecolor="#1a4d33", alpha=0.85)
ax6.set_ylabel("Number of Classification Changes")
ax6.set_title("Income Classification Stability: Number of Times Each Country Changed Group (1995-2022)")
ax6.set_xticks(range(len(transitions_df)))
ax6.set_xticklabels(transitions_df[COUNTRY_COL], rotation=45, ha="right")
ax6.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
plt.tight_layout()
plt.savefig("Figure_classification_transitions.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: Figure_classification_transitions.png")

print("\n" + "="*80)
print("INCOME CLASSIFICATION SUMMARY STATISTICS")
print("="*80)
print(f"\nOverall Median Real GDP per Capita (1995-2022): {overall_median:,.2f} €")
print(f"Median in 1995: {yearly_median[yearly_median[YEAR_COL]==1995]['median_gdp_pc'].values[0]:,.2f} €")
print(f"Median in 2020: {yearly_median[yearly_median[YEAR_COL]==2020]['median_gdp_pc'].values[0]:,.2f} €")
print(f"Growth in Median: {((yearly_median[yearly_median[YEAR_COL]==2020]['median_gdp_pc'].values[0] / yearly_median[yearly_median[YEAR_COL]==1995]['median_gdp_pc'].values[0]) - 1) * 100:.1f}%")

print("\n" + "-"*80)
print("COUNTRIES BY AVERAGE INCOME LEVEL (1995-2022)")
print("-"*80)
print("\nHIGH-INCOME COUNTRIES (Above Overall Median):")
high_income = avg_gdp_by_country[avg_gdp_by_country["avg_gdp_pc"] > overall_median]
for idx, row in high_income.iterrows():
    print(f"  {row[COUNTRY_COL]}: {row['avg_gdp_pc']:>10,.0f} €")

print("\nLOW-INCOME COUNTRIES (Below Overall Median):")
low_income = avg_gdp_by_country[avg_gdp_by_country["avg_gdp_pc"] <= overall_median]
for idx, row in low_income.iterrows():
    print(f"  {row[COUNTRY_COL]}: {row['avg_gdp_pc']:>10,.0f} €")

print("\n" + "-"*80)
print("CLASSIFICATION STABILITY")
print("-"*80)
print(f"\nCountries with NO classification changes: {len(transitions_df[transitions_df['transitions']==0])}")
print(f"Countries with classification changes: {len(transitions_df[transitions_df['transitions']>0])}")
if len(transitions_df[transitions_df['transitions']>0]) > 0:
    print("\nMost volatile countries:")
    for idx, row in transitions_df[transitions_df['transitions']>0].head(10).iterrows():
        print(f"  {row[COUNTRY_COL]}: {int(row['transitions'])} changes")

print("\n" + "="*80)
print("All figures have been saved!")
print("="*80)
