# Tsunami Risk Prediction

Machine learning models for predicting tsunami occurrence using USGS seismic data.

## Features

- **Data Collection**: Using USGS API for collecting historical earthquake data
- **Model Implementation**: 
  - Baseline models: Decision Tree, Random Forest, XGBoost, KNN, Logistic Regression
  - Hyperparameter optimization using Optuna (Random Forest, XGBoost)
  - Best model: XGBoost (F1: 0.868, Accuracy: 0.912, ROC-AUC: 0.945)

## Tech Stack

- **Language**: Python 3.13
- **ML Libraries**: scikit-learn, XGBoost, Optuna
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, folium
- **Prediction App**: joblib, streamlit

## Project Structure
```
earthquake-tsunami-ml-project/
├── App/
│   ├── app.py                  # The streamlit app for prediction
│
├── Data/
│   ├── collect_earthquake_data.py                      # USGS API data collection script
│   ├── earthquake_tsunami_raw_data.csv                 # Raw USGS earthquake data (2013.1.1 - 2025.10.31) for modeling 
│   ├── earthquake_tsunami_raw_data_pred.csv            # Raw USGS earthquake data (2025.11.1 - 2025.11.20) for prediction
│   ├── X_test.csv                                      # Cleaned data for testing models - X
│   ├── y_test.csv                                      # Cleaned data for testing models - X
│
├── Notebooks/
│   ├── 01_Explanatory Data Analysis.ipynb              # EDA
│   ├── 02_Data Processing and Modeling.ipynb           # Data cleaning, processing and modeling
│   ├── 03_Prediction.ipynb                             # Prediction demo notebook 
│
├── Model/                             # Saved models and preprocessing artifacts
│   ├── best_model_xgboost.pkl         # Final trained XGBoost model
│   ├── best_model_xgboost_info.json   # Model hyperparameters and metrics
│   ├── scaler.pkl                     # StandardScaler for feature scaling
│   ├── scaler_config.json             # Scaler configuration
│   ├── onehot_encoder.pkl             # OneHotEncoder for categorical features
│   ├── onehot_encoder_config.json     # Encoder configuration
│   └── column_medians.json            # Median values for imputation
│
├── Slides/
│   ├──                # PPT for presentation
│   └──                # pdf version of slides
│
├── Summary/
│   ├──                # Summary report
│   
├── requirements.txt           # Python dependencies
└── README.md
```

## Installation
```bash
git clone https://github.com/RubyXZZZ/earthquake-tsunami-ml-project.git
cd earthquake-tsunami-ml-project
pip install -r requirements.txt
```

## Usage
#### 1. **Collect Data Script**:
**Note**: Data is already stored in the `Data/earthquake_tsunami_raw_data.csv`. If you want to re-run the script, you can run `collect_earthquake_data.py`. You can also modify the data range parameters and output file path in the script.
```bash
cd Data
python collect_earthquake_data.py
```
#### 2. **Notebook**:
**Note**: Make sure that mark notebook as trusted in your IDE.
```bash
cd Notebooks
```



## Data Source

Earthquake data sourced from [USGS Earthquake Catalog API](https://earthquake.usgs.gov/fdsnws/event/1/)

