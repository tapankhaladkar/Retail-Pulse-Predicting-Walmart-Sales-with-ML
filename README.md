# 📊 Retail Pulse: Predicting Walmart Sales with Data Science 🚀

![Retail Forecasting](https://img.shields.io/badge/Retail-Forecasting-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-orange)
![Python](https://img.shields.io/badge/Python-Data%20Science-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📌 Project Overview

Retail businesses often struggle with **inventory management** and **sales forecasting** due to
fluctuating economic conditions. This project **analyzes and forecasts Walmart's weekly sales**
across 45 stores using **Exploratory Data Analysis, statistical analysis, and a machine learning
regression model**.

🚀 **Objectives of this project:**

- ✔️ Identify key factors affecting sales (unemployment, CPI, temperature, fuel price).
- ✔️ Detect seasonal sales trends and rank store performance.
- ✔️ Forecast the **next 12 weeks of sales** for each store.

---

## 📂 Dataset Description

The dataset is **Walmart Sales Data** (`Walmart.csv`), containing **6,435 rows** and **8 columns** —
45 stores observed weekly (every Friday) from **2010-02-05 to 2012-10-26**, 143 weeks per store,
with no missing values, no duplicate rows and no gaps in the weekly series.

| **Feature**    | **Description**                                        |
|----------------|--------------------------------------------------------|
| `Store`        | Store number (unique identifier)                        |
| `Date`         | Week of sales (`DD-MM-YYYY`)                            |
| `Weekly_Sales` | Sales for the given store in that week                  |
| `Holiday_Flag` | Whether it is a holiday week (1 = Yes, 0 = No)          |
| `Temperature`  | Temperature in the region that week (°F)                |
| `Fuel_Price`   | Cost of fuel in the region                              |
| `CPI`          | Consumer Price Index                                    |
| `Unemployment` | Unemployment rate                                       |

> The dataset is the widely circulated public "Walmart Store Sales" extract, included here for
> reproducibility. It is redistributed for educational use; the MIT license in this repository
> covers the code and documentation, not the underlying data.

---

## 🗂️ Repository Structure

```
.
├── Walmart.csv                      # Source dataset (6,435 × 8)
├── Walmart.ipynb                    # EDA + forecasting notebook
├── Capstone Report - Walmart.pdf    # Written project report
├── requirements.txt                 # Pinned runtime dependencies
├── LICENSE                          # MIT
└── README.md
```

---

## ⚡ Getting Started

```bash
git clone https://github.com/tapankhaladkar/Retail-Pulse-Predicting-Walmart-Sales-with-ML.git
cd Retail-Pulse-Predicting-Walmart-Sales-with-ML

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter lab Walmart.ipynb
```

Then run the notebook top to bottom. `Walmart.csv` is read from the repository root, so no path
configuration is needed.

---

## 🔎 Data Analysis & Insights

The exploratory analysis addresses five questions:

- ✔️ **Does unemployment affect sales?** — Pearson correlation computed *per store* (with p-values)
  rather than pooled across stores, since mean weekly sales differ ~8x between the largest and smallest store.
- ✔️ **Do sales follow a seasonal trend?** — Monthly and weekly aggregation, plus holiday-week
  comparison.
- ✔️ **How does temperature impact sales?** — Overall correlation, quartile buckets, and store-level
  correlations.
- ✔️ **How does CPI affect sales?** — Store-level correlation and linear-regression slope.
- ✔️ **Best and worst performing stores** — Ranked by mean weekly sales, with coefficient of
  variation as a consistency measure and a separate holiday-period ranking.

---

## 📊 Predictive Modeling

To forecast the **next 12 weeks** of sales:

- **Model:** `RandomForestRegressor` (scikit-learn), trained **separately for each of the 45 stores**.
- **Features:** calendar (`Month`, `Week`), external factors (`Temperature`, `Fuel_Price`, `CPI`,
  `Unemployment`, `Holiday_Flag`) and autoregressive features (`Sales_Lag1`, `Sales_Lag2`,
  `Sales_Rolling_Mean`, `Sales_Lag52`).
- **Forecasting strategy:** recursive multi-step — each predicted week is fed back as the next
  week's lag, with calendar features advancing and the holiday flag taken from the real calendar.
- **Validation:** chronological holdout (never a random split), scored with MAE / RMSE / MAPE
  against naive and seasonal-naive baselines on two windows.

### Results

| Window | Naive | Seasonal naive | **Random Forest** |
|---|---|---|---|
| **A** — Aug–Oct 2012 (no major holiday) | 6.01% | 5.46% | **3.88%** ✅ |
| **B** — Nov 2011–Jan 2012 (Thanksgiving + Christmas) | 13.79% | **6.25%** ✅ | 11.06% |

*MAPE, lower is better. Winner in bold.*

**The model wins on ordinary weeks and loses on the seasonal peak.** The dataset spans 143 weeks
and contains only two November–December periods, so a holdout starting in November leaves no prior
holiday season in training. `Sales_Lag52` carries 63% of feature importance and lag features 78% in
total — this is a smoothed-persistence forecaster with a year-over-year anchor, and it cannot
manufacture a +68% spike it has never observed.

Because the requested horizon (Nov 2012–Jan 2013) falls in window B, the notebook reports both the
Random Forest forecast and a seasonal-naive reference, and recommends the latter for the holiday
weeks.

## 💻 Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-008080?style=for-the-badge&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

---

## 🚧 Status & Roadmap

This repository is undergoing a correctness and reproducibility pass:

- [x] **Repository hygiene** — license, dependency manifest, `.gitignore`, accurate documentation,
      removal of stray `nbconvert` export artifacts.
- [x] **Notebook correctness** — reads the CSV from the repository root, removed the target leakage
      in the rolling-mean feature, replaced the random train/test split with a chronological
      holdout, fixed the recursive forecast loop so calendar features advance across the horizon,
      carried exogenous values forward per store rather than globally, applied the real holiday
      calendar, and restored a working Matplotlib style call. The notebook now executes end to end
      from a clean clone.
- [x] **Evaluation** — MAE / RMSE / MAPE on two chronological holdout windows, benchmarked against
      naive and seasonal-naive baselines, plus feature importance.
- [ ] **Findings** — re-derive the written conclusions and store rankings in
      `Capstone Report - Walmart.pdf` from the corrected notebook output.

## 📄 License

Released under the [MIT License](LICENSE).

## 👤 Author

**Tapan Khaladkar** — [GitHub](https://github.com/tapankhaladkar)
