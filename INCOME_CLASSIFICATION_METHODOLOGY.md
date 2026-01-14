# Income-Based Country Classification Methodology

## Overview
This document details the methodology used to classify European countries into income groups based on real GDP per capita for the period 1995-2020.

## Data Source

### File Used
- **Primary Data File**: `data/eurostat_green_panel.csv`
- **Source**: Eurostat database, compiled via the `dataset_builder.py` script
- **Coverage**: 31 European countries from 1995 to 2022

### Variables Used
1. **geo**: Country code (ISO 2-letter code)
2. **time**: Year of observation
3. **real_gdp_per_capita**: Real GDP per capita in 2015 constant prices (€)
   - Unit: EUR per inhabitant
   - Base year: 2015
   - Source variable in dataset: `real_gdp_per_capita`

### Countries Included (N=31)
Austria (AT), Belgium (BE), Bulgaria (BG), Cyprus (CY), Czech Republic (CZ), Germany (DE), Denmark (DK), Estonia (EE), Greece (EL), Spain (ES), Finland (FI), France (FR), Croatia (HR), Hungary (HU), Ireland (IE), Iceland (IS), Italy (IT), Lithuania (LT), Luxembourg (LU), Latvia (LV), Malta (MT), Netherlands (NL), Norway (NO), Poland (PL), Portugal (PT), Romania (RO), Sweden (SE), Slovenia (SI), Slovakia (SK), Switzerland (CH), United Kingdom (UK)

## Time Period
- **Analysis Period**: 1995-2020 (26 years)
- **Rationale**: 
  - Ensures data completeness and comparability
  - Excludes COVID-19 pandemic effects (2021-2022)
  - Provides sufficient historical depth for trend analysis

## Classification Methodology

### Hypothesis and Rationale
The classification is based on the following premises:
1. **No widely accepted geographic partition**: The literature lacks a standardized geographic classification for European countries that captures economic heterogeneity
2. **Income-based grouping is highly stable**: Real GDP per capita changes gradually over time, providing a stable classification criterion
3. **Median-based threshold**: Using the median rather than mean reduces sensitivity to outliers (e.g., Luxembourg's exceptionally high GDP per capita)

### Classification Rule
Countries are classified into two groups for each year:

**High-Income Countries**: 
- Real GDP per capita > Yearly Median GDP per capita

**Low-Income Countries**: 
- Real GDP per capita ≤ Yearly Median GDP per capita

### Key Features
1. **Yearly median calculation**: The classification threshold is recalculated for each year to account for overall economic growth
2. **Dynamic classification**: Countries can theoretically move between groups if their relative position changes
3. **Balanced groups**: By definition, the median ensures approximately equal numbers of countries in each group

## Results

### Summary Statistics (1995-2020)
- **Overall Median Real GDP per Capita**: €24,260
- **Median in 1995**: €18,220
- **Median in 2020**: €24,680
- **Median Growth**: 35.5% over the period

### High-Income Countries (N=15)
Countries with average real GDP per capita above the overall median (€24,260):
1. Luxembourg (LU): €89,315
2. Switzerland (CH): €71,613
3. Norway (NO): €64,352
4. Denmark (DK): €46,205
5. Ireland (IE): €45,749
6. Iceland (IS): €44,532
7. Sweden (SE): €41,437
8. Netherlands (NL): €38,789
9. United Kingdom (UK): €37,396
10. Austria (AT): €37,263
11. Finland (FI): €36,978
12. Germany (DE): €34,666
13. Belgium (BE): €34,553
14. France (FR): €31,445
15. Italy (IT): €28,653

### Low-Income Countries (N=16)
Countries with average real GDP per capita below the overall median (€24,260):
1. Spain (ES): €22,657
2. Cyprus (CY): €21,744
3. Malta (MT): €18,164
4. Greece (EL): €17,386
5. Slovenia (SI): €17,201
6. Portugal (PT): €17,131
7. Czech Republic (CZ): €14,123
8. Estonia (EE): €13,023
9. Slovakia (SK): €11,668
10. Hungary (HU): €10,045
11. Croatia (HR): €10,038
12. Lithuania (LT): €9,683
13. Latvia (LV): €9,263
14. Poland (PL): €9,037
15. Romania (RO): €6,508
16. Bulgaria (BG): €5,246

### Classification Stability
- **Countries with NO classification changes**: 30 out of 31 (96.8%)
- **Countries with classification changes**: 1 out of 31 (3.2%)
  - **Malta (MT)**: Transitioned from Low-Income (1995-2019) to High-Income (2020)

This exceptional stability validates the hypothesis that income-based grouping using real GDP per capita is highly stable over the 1995-2020 timespan.

## Visualization Outputs

The following figures were generated to illustrate the classification:

1. **Figure_income_classification_bar.png**: Average real GDP per capita by country with overall median threshold
2. **Figure_median_evolution.png**: Evolution of the median GDP per capita over time
3. **Figure_gdp_evolution_all_countries.png**: Spaghetti plot showing all countries' trajectories with unique colors
4. **Figure_income_classification_heatmap.png**: Heatmap showing classification by country and year
5. **Figure_income_distribution_over_time.png**: Count of high vs low-income countries over time
6. **Figure_classification_transitions.png**: Number of classification changes by country

## Implementation

### Code
- **Script**: `income_classification_plots.py`
- **Language**: Python 3.14
- **Dependencies**: pandas, numpy, matplotlib, seaborn

### Processing Steps
1. Load data from `eurostat_green_panel.csv`
2. Filter for years 1995-2020
3. Calculate yearly median GDP per capita
4. Classify each country-year observation as High or Low income
5. Generate summary statistics and visualizations
6. Export figures as PNG files (300 DPI)

## Validation

The classification methodology was validated through:
1. **Consistency check**: Verifying that the median produces balanced groups each year
2. **Stability analysis**: Confirming minimal classification changes over time
3. **Face validity**: Comparing results with economic intuition (e.g., Scandinavian countries, Luxembourg, Switzerland in high-income; Eastern European countries in low-income)

## Limitations and Considerations

1. **Binary classification**: The two-group classification is a simplification; some countries near the median may be misclassified in specific years
2. **Relative measure**: Classification is relative to the sample; a "low-income" European country would be high-income globally
3. **Single indicator**: GDP per capita does not capture all dimensions of economic development (inequality, quality of life, etc.)
4. **Exchange rate effects**: Real GDP per capita in constant 2015 euros may be affected by base year exchange rate choices

## References

- **Eurostat**: European Commission Eurostat Database
- **Dataset Builder**: `dataset_builder.py` - Custom script for data extraction and harmonization
- **Classification Script**: `income_classification_plots.py` - Analysis and visualization

## Date of Analysis
January 13, 2026

---

*For questions or clarifications regarding this methodology, please refer to the source code in `income_classification_plots.py` or the data compilation process in `dataset_builder.py`.*
