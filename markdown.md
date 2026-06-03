% Machine Learning for County-Level Corn and Cotton Yield Prediction (Texas)

Authors: Syed, Ali, Wajeeh Moonwar, Gustavo Hernandez (Texas State University)

## Abstract
This report summarizes a walk-forward XGBoost analysis predicting county-level Corn and Cotton yields in Texas. The pipeline ingests USDA NASS yield records, NASA POWER meteorological data, and USDA soil variables, constructs biologically informed features, and evaluates model performance using rolling-origin (walk-forward) validation. The analysis below reports the actual model metrics produced by executing `code.py` in this workspace.

## Key Results (from this run)
- **Corn (target: bu/acre)**: Cumulative walk-forward R² = 0.6710; RMSE = 24.69 bu/acre; MAE = 19.54 bu/acre.
- **Cotton (target: lbs/acre)**: Cumulative walk-forward R² = 0.4948; RMSE = 248.62 lbs/acre; MAE = 209.17 lbs/acre.

Per-year Corn R² (selected): 2018 = 0.7658, 2019 = 0.2679, 2020 = 0.7346, 2021 = 0.8035, 2022 = 0.6681, 2023 = 0.2814, 2024 = 0.6673, 2025 = 0.5208.

## Methods (concise)
- Data sources: `USDA_MASTER_CORN.csv`, `USDA_COTTON_DRYLAND_COMPLETE.csv`, `USDA_COTTON_IRRIGATED_COMPLETE.csv`, county soil (`TEXAS_COUNTY_SOIL_TRUE.csv`), and NASA POWER weather pulls.
- Geospatial processing: county coordinates were mapped to NASA POWER endpoints first, then the REST API was queried so each county-year record could receive aligned weather observations.
- Feature engineering: seasonal aggregations aligned to crop phenology windows and engineered bio-climatic proxies (e.g., drought stress, vapor pressure deficit, moisture balance).
- Modeling: XGBoost regressors trained under a rolling-origin validation scheme with an out-of-time start year of 2018.

## Comparison with group write-up
The original group draft reported Corn performance near an R² ≈ 0.6774, RMSE ≈ 24.45 bu/ac and Cotton R² ≈ 0.5479, RMSE ≈ 235.20 lbs/ac. Our execution of `code.py` produced slightly different numeric results (Corn R² = 0.6710 vs. 0.6774 reported; Cotton R² = 0.4948 vs. 0.5479 reported). These differences are modest for Corn and larger for Cotton, and can be explained by any of the following:

- Non-deterministic training effects (XGBoost randomness unless fully seeded across all operations).
- Differences in data versions or preprocessing (e.g., dropped rows due to missing soil mapping or slight CSV variations).
- Single-year validation exposure for Cotton (the Cotton pipeline had fewer biologically complete years), increasing sensitivity to dataset splits.

## Updated Findings and Recommendations
- The Corn model demonstrates robust predictive skill (R² ≈ 0.67) and stable year-to-year performance in several test years (notably 2018, 2020, 2021). The RMSE (~24.7 bu/acre) and MAE (~19.5 bu/acre) are consistent with an operationally useful forecasting tool at the county scale.
- The Cotton model shows lower aggregate predictive power (R² ≈ 0.49) and substantially higher error magnitudes (RMSE ≈ 249 lbs/acre). Given the Cotton crop's indeterminate development and sparser merged records, additional data (extended years, improved irrigation tags, or farm-level observations) is recommended before deploying a production forecasting system.
- To reduce numeric variance between runs and improve reproducibility, set deterministic seeds for XGBoost (and any numpy randomness), fix training thread counts, and record the exact CSV versions used in the run.

## Actions performed in this workspace
- Executed `code.py` and saved console output to `results.txt` (generated dashboards and PDF/PNG figures in the workspace).
- Generated a polished PDF version of this report as `polished_report.pdf`.
- Updated this markdown to reflect the precise metrics observed when running the pipeline locally.

## Conclusion
The implemented XGBoost workflow is an effective, interpretable approach for county-level Corn yield prediction in Texas, while Cotton prediction requires more data and careful validation. The updated markdown reflects the measured performance from the current run; if you would like, I can (a) run the pipeline multiple times with fixed random seeds to confirm stability, (b) further polish figures and embed them into this report, or (c) produce a short summary slide deck.

## References
See the original references in the draft for background on XGBoost and agronomic sources.
