# MLProject

Texas county-level crop yield forecasting with XGBoost.

## Purpose
This project builds an interpretable machine learning pipeline to predict Corn and Cotton yield using weather, soil, and crop history. The goal is to test whether a biologically informed model can capture meaningful yield variation at the county scale while remaining practical to run and explain.

## Summary
The workflow combines USDA yield records, USDA soil data, and NASA POWER weather data. A geospatial preprocessing step maps each county coordinate to the NASA POWER REST API so that monthly weather observations can be collected consistently for every county-year record. Those inputs are merged into a single modeling table, engineered into crop-specific seasonal features, and evaluated with rolling-origin validation.

## Process
1. Load Corn and Cotton yield records from USDA files.
2. Load USDA soil attributes and county-level soil mappings.
3. Use county coordinates to query NASA POWER through its REST API and retrieve weather data.
4. Merge weather, soil, and crop data into biologically complete county-year rows.
5. Engineer weather proxies such as drought stress, moisture balance, and vapor pressure deficit.
6. Train XGBoost regressors with walk-forward validation to simulate real forecasting conditions.

## Data Sources
- NASA POWER API for weather observations.
- USDA NASS for crop yield data.
- USDA soil data for county-level edaphic variables.

## Results
The current local run produced the following cumulative walk-forward metrics:

- **Corn**: R² = 0.6710, RMSE = 24.69 bu/acre, MAE = 19.54 bu/acre.
- **Cotton**: R² = 0.4948, RMSE = 248.62 lbs/acre, MAE = 209.17 lbs/acre.

Corn showed the stronger and more stable fit across multiple validation years. Cotton remained predictive, but the smaller merged dataset and the crop's indeterminate growth habit made the model less stable.

## Report Files
- [polished_report.pdf](polished_report.pdf)
- [markdown.md](markdown.md)
- [results.txt](results.txt)

## Notes
The repository also includes the scripts used to run the analysis and generate the PDF report. If you rerun the pipeline, the reported metrics may shift slightly depending on the exact data files and model randomness.


