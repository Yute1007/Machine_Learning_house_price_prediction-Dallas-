# House Price Movement Predictor

Predict whether Dallas home prices will rise over the next quarter, using macroeconomic indicators and Zillow housing data, with stepwise model improvements and a visual performance chart.

## Overview

This project is a proof-of-concept machine-learning pipeline for forecasting short-term home value changes in the Dallas metro area. It demonstrates how to:

1. Ingest and merge Federal Reserve economic series with Zillow home price indices  
2. Engineer lagged and ratio-based features  
3. Train a Random Forest classifier  
4. Evaluate, improve, and visualize predictive performance

## Data Sources

- **Federal Reserve (FRED) Series**  
  - `CPIAUCSL.csv` — Consumer Price Index (All Urban Consumers)  
  - `RRVRUSQ156N.csv` — Rental Vacancy Rate  
  - `MORTGAGE30US.csv` — 30-Year Fixed Mortgage Rate  

- **Zillow Indices**  
  - `Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_month.csv` — 35th–65th percentile Home Value Index  
  - `Metro_median_sale_price_uc_sfrcondo_sm_week.csv` — Median sale price (weekly)

## Methodology

1. **Load & Preprocess**  
   - Read each CSV into a pandas DataFrame, parse dates, and set a quarterly/weekly index.  
   - Forward-fill or interpolate missing points to align frequencies.

2. **Merge Data**  
   - Combine Fed series into one “wide” DataFrame.  
   - Join Zillow home value and sale-price columns on matching dates.

3. **Feature Engineering**  
   - **Lagged features:** current vs. next‐quarter adjusted prices  
   - **Year-over-year ratios:** 52-week rolling mean ratios to capture trends  
   - **Target definition:**  
     ```python
     price_data["change"] = (price_data["next_quarter"] > price_data["Adj_price"]).astype(int)
     ```

4. **Modeling & Backtesting**  
   - **Baseline model:**  
     ```python
     rf = RandomForestClassifier(min_samples_split=10, random_state=1)
     preds, accuracy = backtest(price_data, predictors, target)
     # → baseline accuracy: ~59.3%
     ```
   - **Improved model:** add `yearly_ratios` to predictors  
     ```python
     preds2, accuracy2 = backtest(price_data, predictors + yearly_ratios, target)
     # → improved accuracy: ~64.7%
     ```

5. **Visualization**  
   - Plot actual adjusted price against time, coloring points **green** where the model correctly predicted a rise and **red** where it did not.
   - Example:

     ```python
     import matplotlib.pyplot as plt

     plot_data = price_data.iloc[START:].copy()
     plot_data["pred_match"] = (preds2 == (plot_data[target] == 1))
     color_map = plot_data["pred_match"].map({True: "green", False: "red"})

     plot_data.reset_index().plot.scatter(
         x="index", y="Adj_price", color=color_map
     )
     plt.title("Adjusted Home Prices: Green = Correct Rise Prediction")
     plt.xlabel("Date")
     plt.ylabel("Adjusted Price")
     plt.show()
     ```

## Results

- **Baseline Random Forest accuracy:** 59.3%  
- **After adding year-over-year ratio features:** 64.7%  
- **Final scatter plot** highlights correct vs. incorrect rise predictions over time.

![Prediction Performance](assets/prediction_scatter.png)

> *Green dots* = correctly predicted price increases  
> *Red dots* = missed or false alarms

## Installation

```bash
git clone https://github.com/<your-username>/house-price-predictor.git
cd house-price-predictor
python3 -m venv venv
source venv/bin/activate      # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
