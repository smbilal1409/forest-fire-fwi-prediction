# 🌲 Forest Fire Weather Index (FWI) Prediction

An end-to-end machine learning web application that predicts the **Fire Weather Index (FWI)** from live weather and fuel-moisture readings, and classifies the result into an actionable fire-risk level — **Low, Moderate, High, or Extreme**.

Built on the **Algerian Forest Fires dataset**, trained with **Ridge Regression**, served through a **Flask** backend, and deployable to both **AWS Elastic Beanstalk** and **Vercel**.

---

## 🔍 Overview

Forest fires are one of the most destructive natural hazards, and predicting fire risk *before* conditions turn dangerous is critical for early warning systems. The **Fire Weather Index (FWI)** is a numeric rating used by meteorologists to estimate fire intensity potential based on weather conditions and fuel moisture codes.

This project builds a regression model that predicts FWI directly from raw weather station inputs, then wraps it in a simple web interface so anyone — not just someone with a Python environment — can enter conditions and get an instant risk read.

**What this project demonstrates:**
- End-to-end ML workflow: raw data → cleaning → EDA → feature engineering → model selection → deployment
- Comparing multiple regression algorithms and justifying the final choice with metrics, not guesswork
- Packaging a trained model into a production-style Flask API
- Cloud deployment experience across two different platforms (AWS EB and Vercel)

---

## 📊 Dataset

**Source:** Algerian Forest Fires Dataset (UCI Machine Learning Repository)

The dataset covers two regions of Algeria — **Bejaia** and **Sidi Bel-Abbes** — over the summer fire season, combining daily weather observations with the Canadian Forest Fire Weather Index System components.

| Feature | Description |
|---|---|
| Temperature | Noon temperature (°C) |
| RH | Relative Humidity (%) |
| Ws | Wind speed (km/h) |
| Rain | Total rainfall (mm) |
| FFMC | Fine Fuel Moisture Code |
| DMC | Duff Moisture Code |
| ISI | Initial Spread Index |
| Classes | Fire / Not Fire (binary label from original observations) |
| Region | Bejaia (0) or Sidi Bel-Abbes (1) |
| **FWI** | **Target** — Fire Weather Index (continuous) |

---

## 🧹 Exploratory Data Analysis

Before modeling, the raw data was inspected and cleaned:

- **Missing & malformed rows** — the raw CSV contains a stray header row splitting the two regions; these were removed and the two regional blocks were merged with a `Region` indicator column added back in.
- **Data type correction** — several numeric columns were read in as strings due to embedded whitespace/formatting and were cast to `float`/`int`.
- **Outlier review** — boxplots were used per feature to check for sensor/entry errors (e.g. implausible rainfall or temperature spikes).
- **Correlation analysis** — a heatmap was used to check multicollinearity between the fire-behavior indices (FFMC, DMC, ISI) themselves, and against the FWI target, since these codes are cumulative and expected to correlate.
- **Class balance check** — the `Classes` (fire / not fire) column was checked for imbalance, since it's used as an input feature rather than the prediction target.
- **Distribution plots** — histograms/KDE plots per feature to understand skew (rainfall in particular is heavily right-skewed, as expected for a semi-arid climate with occasional storms).

> 📈 *Add your actual EDA notebook screenshots here — correlation heatmap, distribution plots, and boxplots make this section far more convincing to anyone reviewing the repo.*

---

## 🛠 Feature Engineering

- Dropped the `day`, `month`, `year` columns after confirming seasonality wasn't the modeling focus (the project targets instantaneous risk from current conditions, not time-series forecasting)
- Encoded `Region` as a binary indicator (0 = Bejaia, 1 = Sidi Bel-Abbes)
- Applied **StandardScaler** to normalize all numeric features before training, since Ridge Regression is sensitive to feature scale
- Train/test split performed before scaling to avoid data leakage — the scaler is fit only on training data, then reused (via `scaler.pkl`) for both test evaluation and live inference

---

## 🤖 Model Experimentation

Multiple regression algorithms were trained and compared on the same train/test split to select the best-performing model for this task:

| Model | Notes |
|---|---|
| Linear Regression | Baseline model |
| **Ridge Regression** | ✅ Selected — L2 regularization handles multicollinearity between FFMC/DMC/ISI well |
| Lasso Regression | Tested for feature sparsity; underperformed vs. Ridge on this feature set |
| Elastic Net | Combined L1/L2 penalty, tested as a middle ground |

Ridge Regression was selected as the final model because the fire-behavior index features are naturally correlated with each other (each builds cumulatively on the last in the Canadian FWI system), and Ridge's L2 penalty handles that multicollinearity more gracefully than an unregularized linear model — without zeroing out features the way Lasso can.

> 📊 *Add your actual R², MAE, and RMSE comparison table here for each model — this is the single most convincing section for anyone evaluating your ML skills, so don't leave it as prose only.*

---

## 🏆 Final Model

- **Algorithm:** Ridge Regression
- **Preprocessing:** StandardScaler (fit on training data, saved as `scaler.pkl`)
- **Artifacts:** `models/ridge.pkl`, `models/scaler.pkl`
- **Inference:** raw form inputs → scaled → passed to Ridge model → continuous FWI score → mapped to a risk band

**Risk classification thresholds used in the app:**

| FWI Range | Risk Level |
|---|---|
| < 5 | 🟢 Low |
| 5 – 15 | 🟡 Moderate |
| 15 – 30 | 🟠 High |
| 30+ | 🔴 Extreme |

---

## 🧰 Tech Stack

**Language & ML:** Python, scikit-learn, NumPy, Pandas
**Backend:** Flask
**Frontend:** HTML, CSS (custom-themed, responsive)
**Deployment:** AWS Elastic Beanstalk, Vercel
**Version Control:** Git & GitHub

---

## 📁 Project Structure

```
FWI-prediction-system/
├── myapplication.py          # Flask app entry point
├── requirements.txt
├── vercel.json                # Vercel deployment config
├── .ebextensions/
│   └── 01_flask.config        # AWS Elastic Beanstalk config
├── models/
│   ├── ridge.pkl               # Trained Ridge Regression model
│   └── scaler.pkl              # Fitted StandardScaler
├── templates/
│   └── Home.html
├── static/
│   └── css/
│       └── style.css
└── notebooks/
    └── EDA_and_model_training.ipynb
```

---

## 🚀 Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/forest-fire-fwi-prediction.git
cd forest-fire-fwi-prediction
```

**2. Create a virtual environment & install dependencies**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

**3. Run locally**
```bash
python myapplication.py
```
Visit `http://127.0.0.1:5000` in your browser.

---

## 🔮 Future Improvements

- Add a confidence interval or prediction range alongside the point estimate
- Log predictions to a database for monitoring model drift over time
- Add unit tests for the Flask routes and prediction pipeline
- Experiment with ensemble models (Random Forest, XGBoost) as a comparison baseline
- Add CI/CD via GitHub Actions for automatic deployment on push

---

## 👤 Author

Muhammad Bilal Sheikh
Built as an end-to-end ML deployment project — data cleaning, EDA, model selection, Flask API, and cloud deployment.

📫 [LinkedIn: linkedin.com/in/bilalsheikhmuhammad ](#) · [GitHub: https://github.com/smbilal1409](#)