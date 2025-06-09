# House Price Movement Predictor

Predict whether Dallas home prices will rise over the next quarter, using macroeconomic indicators and Zillow housing data.

## Overview

This project demonstrates a proof-of-concept machine-learning pipeline for forecasting short-term changes in home values. By combining Federal Reserve economic series with Zillow’s home value and sale price indices, we train a Random Forest classifier to predict if median prices will increase in the following quarter.

## Data Sources

- **Federal Reserve (FRED)**
  - `CPIAUCSL.csv`: Consumer Price Index for All Urban Consumers  
  - `RRVRUSQ156N.csv`: Rental Vacancy Rate in the U.S.  
  - `MORTGAGE30US.csv`: 30-Year Fixed Mortgage Rate  

- **Zillow**
  - `Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_month.csv`: Zillow Home Value Index (35th–65th percentile tier)  
  - `Metro_median_sale_price_uc_sfrcondo_sm_week.csv`: Median sale price (raw weekly data)  

## Methodology

1. **Load and preprocess** each series into pandas DataFrames with a DateTime index.  
2. **Concatenate** Fed series into a single “wide” table indexed by date.  
3. **Align** and merge with Zillow indices on matching dates.  
4. **Feature engineering**  
   - Adjust prices for seasonal variation.  
   - Compute lagged features (current vs. next quarter).  
   - Derive year-over-year ratios.  
5. **Define target**  
   ```python
   price_data["change"] = (price_data["next_quarter"] > price_data["Adj_price"]).astype(int)
