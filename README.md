# AgriTech

**Datasets Overview**
Dataset contain historical data from 2000 to 2024. During the training phase, we withheld 20% of the tail part of the historical data. The model was evaluated by predicting the planting progress for these withheld "past weeks" and comparing its predictions against the actual historical records.

**Model Overview:**
We utilized an **XGBoost Regressor** (Extreme Gradient Boosting) to predict crop yield using 3 different datasets.


