# Earthquake - Tsunami Prediction System

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
│   ├── earthquake_tsunami_raw_data.csv                 # Raw USGS earthquake data (1/1/2013-10/31/2025) for modeling 
│   ├── earthquake_tsunami_raw_data_pred.csv            # Raw USGS earthquake data (11/1/2025-11/20/2025) for prediction
│   ├── X_test.csv                                      # Cleaned data for testing models - X
│   ├── y_test.csv                                      # Cleaned data for testing models - y
│
├── Notebooks/
│   ├── 01_Exploratory Data Analysis.ipynb              # EDA
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
│   └──FinalProject_Slides.pdf         # pdf version of slides
│
├── Summary/
│   ├── Project Summary.pdf            # Summary report
│   
├── requirements.txt           # Python dependencies
└── README.md
```

## Set up Environment
#### 1. **Clone Repository**:
```bash
git clone https://github.com/RubyXZZZ/earthquake-tsunami-ml-project.git
cd earthquake-tsunami-ml-project
```
#### 2. **Create and Activate Virtual Environment**:
```bash
# Using conda (Recommended)
conda create --name your_env_name python=3.13
conda activate your_env_name

# Or using venv (Requires Python 3.13)
python -m venv your_venv_name
# MacOs/Linux
source your_venv_name/bin/activate
# Windows
.\your_venv_name\Scripts\activate
```
#### 3. **Install Dependencies**:
**Note**: Check environment information in `requirements.txt`.
```bash
pip install -r requirements.txt
```

## How to Run
#### 1. **Collect Data Script**:
**Note**: Data is already stored in the `Data/earthquake_tsunami_raw_data.csv`. 
If you want to re-run the script, you can run `collect_earthquake_data.py`. 
You can also modify the data range parameters and output file path in the script. 
Please rename the output file path to avoid overwriting existing files.
```bash
cd Data
python collect_earthquake_data.py
```
#### 2. **Jupyter Notebooks**:
**Note**: If you do not see any pictures or texts, please close and reopen the notebooks. This may be due to the large amount of pictures in the notebooks.<br>
There is an interactive heatmap at the end of `01_Exploratory Data Analysis.ipynb`, make sure to run this notebook locally to view the map.
Also please make sure to mark notebooks as trusted in your IDE. If you are using Jupyter Notebook/Jupyter Lab, navigate to the file menu at the top of the interface then select the "trust notebook" option.
If you are using Pycharm, the "trust notebook" option is at the top right of the interface. <br>
Open and run the notebooks in order:
1. 01_Exploratory Data Analysis.ipynb
2. 02_Data Processing and Modeling.ipynb
3. 03_Prediction.ipynb
```bash
cd Notebooks
```
#### 3. **Streamlit App**:
**Note**: Please use terminal to run the app. Please make sure to navigate into the App folder before running the Streamlit App. If the terminal asks for your email address, please ignore it and press Enter/Return.
When using the app, please enter legit numbers for manual input section.
```bash
cd App
streamlit run app.py
```
The app will open at http://localhost:8501


## Data Source

Earthquake data sourced from [USGS Earthquake Catalog API](https://earthquake.usgs.gov/fdsnws/event/1/)

