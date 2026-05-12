# 🏨 Hotel Booking Cancellation Prediction

> A machine learning system to predict hotel booking cancellations using the CRISP-DM framework, built with LightGBM and deployed as an interactive Streamlit dashboard.

---

## 👥 Team

| Name             |
| ---------------- |
| Salma Abdelhamid |
| Malak Mehana     |

**Course:** Data Mining  
**Institution:** SUT

---

## 🌐 Domain

**Hospitality & Revenue Management**  
Hotel booking cancellations cost the industry an estimated $2.1 billion annually. This project builds a binary classification model to predict whether a booking will be canceled — using only information available at the time of booking — enabling hotels to act proactively and protect revenue.

---

## 📦 Dataset

- **Name:** Hotel Booking Demand
- **Source:** Kaggle — https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
- **Original Paper:** Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. _Data in Brief, 22_, 41–49. https://doi.org/10.1016/j.dib.2018.11.126
- **Size:** 119,390 bookings | 32 features | 2015–2017
- **Target:** `is_canceled` (0 = Not Canceled, 1 = Canceled)
- **Class split:** 73% not canceled / 27% canceled

---

## 📁 Project Structure

## 📁 Project Structure

````text
├── data/
│   └── data_link_kaggle.txt          # Link to download the Kaggle dataset
├── deployment/
│   ├── app.py                        # Streamlit dashboard
│   └── requirements.txt              # Python dependencies for deployment
├── notebooks/
│   └── Hotel booking demand-Final Attempt.ipynb # Main Jupyter Notebook for exploratory analysis/modeling
├── phase1/
│   ├── PHASE1_PRESENTATION.pptx      # Phase 1 presentation slides
│   └── PHASE1_REPORT.docx            # Phase 1 project report
├── phase2/
│   ├── Predicting Hotel Booking Cancellations.pptx # Phase 2 presentation slides
│   └── PHASE2_REPORT.docx            # Phase 2 project report
├── .gitignore                        # Git ignore file
└── README.md                         # This file

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/the-salma-ahmad/dm-final-hospitality-team15.git
cd [repo-folder]
````

### 2. Create and activate a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Download `hotel_bookings.csv` from [Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) and place it in the **root folder** (same level as `app.py`).

```
├── app.py
├── hotel_bookings.csv   ← here
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`

---

## 📊 Dashboard Tabs

| Tab                    | Description                                                                   |
| ---------------------- | ----------------------------------------------------------------------------- |
| Overview               | Hotel type breakdown, seasonality, cancellation distribution                  |
| Revenue & ADR          | Revenue trends, ADR by month, lead time vs ADR distributions                  |
| Cancellation Analysis  | Cancellation rates by month, lead time, customer type, country                |
| Guests & Segments      | Guest composition, special requests, market segment summary                   |
| Correlation & Features | Heatmap, engineered binary feature analysis                                   |
| Data Explorer          | Filtered raw data viewer + CSV download                                       |
| 🤖 ML Results          | Model comparison, confusion matrix, ROC/PR curves, feature importances, SMOTE |

> **Note:** The ML Results tab requires clicking **"▶ Train Models & Show Results"** to trigger training. Results are cached for the session after the first run (~30 seconds).

---

## 🤖 ML Pipeline Summary

- **Models trained:** Logistic Regression, LightGBM, LightGBM (Tuned)
- **Best model:** LightGBM (`scale_pos_weight=2.67`)
- **Split:** 70% train / 15% validation / 15% test (stratified)
- **Threshold tuning:** Tuned on validation set to guarantee Recall ≥ 70%
- **Imbalance handling:** `scale_pos_weight` + SMOTE comparison
- **Test ROC-AUC:** 0.894 | **F1 (Canceled):** 0.69 | **Recall:** 0.76

---

## 📚 Key References

- Antonio et al. (2019). Hotel booking demand datasets. _Data in Brief_. https://doi.org/10.1016/j.dib.2018.11.126
- Ke et al. (2017). LightGBM. _NeurIPS_. https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html
- Chawla et al. (2002). SMOTE. _JAIR_. https://doi.org/10.1613/jair.953
- Chapman et al. (2000). CRISP-DM 1.0. SPSS Inc.
