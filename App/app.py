import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib

st.set_page_config(
    page_title="Tsunami Prediction System",
    page_icon="🌊",
    layout="wide"
)

st.markdown("<h1 style='text-align: center;'>Earthquake Tsunami Prediction System</h1>", unsafe_allow_html=True)
st.markdown("---")

@st.cache_resource
def load_models():
    try:
        with open("../Model/best_model_xgboost.pkl", "rb") as f:
            model = joblib.load(f)
        with open("../Model/best_model_xgboost_info.json", "r") as f:
            model_info = json.load(f)
        with open("../Model/scaler.pkl", "rb") as f:
            scaler = joblib.load(f)
        with open("../Model/scaler_config.json", "r") as f:
            scaler_config = json.load(f)
        with open("../Model/onehot_encoder.pkl", "rb") as f:
            onehot_encoder = joblib.load(f)
        with open("../Model/onehot_encoder_config.json", "r") as f:
            onehot_encoder_config = json.load(f)
        with open("../Model/column_medians.json", "r") as f:
            column_medians = json.load(f)
        return model, model_info, scaler, scaler_config, onehot_encoder, onehot_encoder_config, column_medians
    except Exception as e:
        st.error(f"Error loading {e}")
        return None, None, None, None, None, None, None

model, model_info, scaler, scaler_config, onehot_encoder, onehot_encoder_config, column_medians = load_models()

drop_cols = ["title", "place", "alert", "datetime", "code", "status", "net", "type", "cdi", "mmi", "sig", "nst"]
drop_engineered_cols = ["latitude", "longitude", "month"]

@st.cache_data
def load_default_data():
    df_demo = pd.read_csv("../Data/earthquake_tsunami_raw_data_pred.csv")
    df_demo_X = df_demo.drop(["tsunami"], axis=1)
    df_demo_y = df_demo["tsunami"]
    return df_demo_X, df_demo_y

def impute_nulls(df, medians):
    for col, median in medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(median)
    return df

def convert_to_cartesian(df):
    lat_rad = np.radians(df["latitude"])
    lon_rad = np.radians(df["longitude"])

    df["X"] = np.cos(lat_rad) * np.cos(lon_rad)
    df["Y"] = np.cos(lat_rad) * np.sin(lon_rad)
    df["Z"] = np.sin(lat_rad)
    return df

def convert_month_to_cyclical(df):
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    return df

def process_data(df, drop_columns, medians, ohe, ohe_config, standard_scaler, standard_scaler_config, is_default=True):
    df = df.copy()
    if is_default:
        # Filter out non-earthquake events like nuclear explosion
        df = df[df["type"] == "earthquake"]
        original_df = df.copy()
        # Drop unwanted columns
        df = df.drop(columns=drop_columns, axis=1)
    else:
        original_df = df.copy()
    # Impute null values with medians
    df = impute_nulls(df, medians)
    # Convert latitude and longitude to Cartesian coordinates
    df = convert_to_cartesian(df)
    # Convert month
    df = convert_month_to_cyclical(df)
    # Encode magType
    encoded_array = ohe.transform(df[ohe_config["target_cols"]])
    encoded_feature_names = ohe.get_feature_names_out(ohe_config["target_cols"])
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoded_feature_names,
        index=df.index
    )
    # Drop original magType column and add encoded columns
    df = df.drop(columns=ohe_config["target_cols"], axis=1)
    df = pd.concat([df, encoded_df], axis=1)

    # Drop features that were engineered (latitude, longitude, month)
    df = df.drop(columns=drop_engineered_cols, axis=1)
    # Scale data
    df[standard_scaler_config["scaling_cols"]] = standard_scaler.transform(df[standard_scaler_config["scaling_cols"]])
    return df, original_df


st.subheader("Input Method")
input_method = st.radio(
    "You can input your own data or use our preloaded data",
    ["Manual Input", "Load Demo Data (7 samples)"],
    horizontal=True
)
st.markdown("---")

if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False
    st.session_state.results = None

if input_method == "Manual Input":
    st.subheader("Enter Earthquake Parameters")
    col1, col2 = st.columns(2)
    with col1:
        magnitude = st.text_input("Magnitude", placeholder="e.g., 6.5", help="Earthquake magnitude")
        dmin = st.text_input("Dmin (distance from epicenter to nearest station in degrees)", placeholder="e.g., 2.5", help="Distance to nearest station in degrees")
        gap = st.text_input("Gap (largest azimuthal gap between adjacent stations in degrees)", placeholder="e.g., 15", help="Azimuthal gap in degrees")
        depth = st.text_input("Depth (km)", placeholder="e.g., 25", help="Depth of earthquake in kilometers")
        latitude = st.text_input("Latitude", placeholder="e.g., 35.5", help="Latitude in degrees (-90 to 90)")
        longitude = st.text_input("Longitude", placeholder="e.g., 140.2", help="Longitude in degrees (-180 to 180)")
    with col2:
        year = st.text_input("Year", placeholder="e.g., 2023", help="Year of earthquake")
        month = st.text_input("Month", placeholder="e.g., 3", help="Month (1-12)")
        magType = st.selectbox(
            "Magnitude Type",
            options=["mb", "ml", "mw", "mwb", "mwc", "mwr", "mww"],
            help="Type of magnitude calculation"
        )
        rms = st.text_input("RMS (root mean square travel time residual)", placeholder="e.g., 0.95", help="Root mean square travel time residual")
    st.markdown("---")
    if st.button("PREDICT TSUNAMI RISK", key="manual_predict", type="primary", width='stretch'):
        try:
            input_data = pd.DataFrame({
                "magnitude": [float(magnitude)],
                "dmin": [float(dmin)],
                "gap": [float(gap)],
                "depth": [float(depth)],
                "latitude": [float(latitude)],
                "longitude": [float(longitude)],
                "year": [int(year)],
                "month": [int(month)],
                "magType": [magType],
                "rms": [float(rms)]
            })
            if model is not None and scaler is not None:
                input_processed, input_original = process_data(
                    input_data,
                    drop_cols,
                    column_medians,
                    onehot_encoder,
                    onehot_encoder_config,
                    scaler,
                    scaler_config,
                    is_default=False
                )

                prediction = model.predict(input_processed)[0]
                try:
                    proba = model.predict_proba(input_processed)[0]
                    confidence = max(proba) * 100
                except:
                    confidence = None

                st.session_state.prediction_made = True
                st.session_state.results = {
                    "type": "single",
                    "prediction": prediction,
                    "confidence": confidence,
                    "input_data": input_original
                }
        except ValueError as ve:
            st.error("Please enter valid numeric values for all fields!")
            st.error(f"Details: {str(ve)}")
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
else:
    st.subheader("Using 7 Demo Earthquake Data")

    try:
        df_X, df_y = load_default_data()
        st.write("Preview of demo data:")
        preview_cols = ["magnitude", "dmin", "gap", "depth", "latitude", "longitude", "year", "month", "magType", "rms"]
        display_cols = [col for col in preview_cols if col in df_X.columns]
        st.dataframe(df_X[display_cols], width='stretch')
        st.markdown("---")

        if st.button("PREDICT TSUNAMI RISK", key="demo_predict", type="primary", width='stretch'):
            if model is not None and scaler is not None:
                try:
                    demo_processed, demo_original = process_data(
                        df_X,
                        drop_cols,
                        column_medians,
                        onehot_encoder,
                        onehot_encoder_config,
                        scaler,
                        scaler_config,
                        is_default=True
                    )

                    predictions = model.predict(demo_processed)
                    try:
                        probas = model.predict_proba(demo_processed)
                        confidences = [max(proba) * 100 for proba in probas]
                    except:
                        confidences = [None] * len(predictions)

                    actual_labels = df_y.values

                    st.session_state.prediction_made = True
                    st.session_state.results = {
                        "type": "batch",
                        "predictions": predictions,
                        "confidences": confidences,
                        "actual": actual_labels,
                        "data": demo_original
                    }
                except Exception as e:
                    st.error(f"Error making predictions: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())

    except Exception as e:
        st.error(f"Error loading demo data: {str(e)}")

if st.session_state.prediction_made and st.session_state.results is not None:
    st.markdown("---")
    st.subheader("Prediction Results")

    results = st.session_state.results

    if results["type"] == "single":
        prediction = results["prediction"]
        confidence = results["confidence"]

        if prediction == 1:
            st.markdown("<h2 style='color: #cc0000; text-align: center;'>TSUNAMI PREDICTED</h2>",
                        unsafe_allow_html=True)
            st.markdown(
                f"<p style='text-align: center;'><strong>Confidence:</strong> {confidence:.1f}%</p>"
                if confidence else "<p style='text-align: center;'><strong>Confidence:</strong> N/A</p>",
                unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='color: #006600; text-align: center;'>NO TSUNAMI PREDICTED</h2>",
                        unsafe_allow_html=True)
            st.markdown(
                f"<p style='text-align: center;'><strong>Confidence:</strong> {confidence:.1f}%</p>"
                if confidence else "<p style='text-align: center;'><strong>Confidence:</strong> N/A</p>",
                unsafe_allow_html=True)

        st.markdown("#### Input Parameters Used:")
        st.dataframe(results['input_data'], width='stretch')

    else:
        predictions = results["predictions"]
        confidences = results["confidences"]
        actual = results.get("actual", None)
        data = results["data"]

        tsunami_count = sum(predictions)
        no_tsunami_count = len(predictions) - tsunami_count
        tsunami_pct = (tsunami_count / len(predictions)) * 100
        no_tsunami_pct = (no_tsunami_count / len(predictions)) * 100

        st.write(
            f"**Summary:** {tsunami_count} Tsunamis predicted ({tsunami_pct:.0f}%), {no_tsunami_count} No Tsunami ({no_tsunami_pct:.0f}%)")

        st.markdown("#### Results Table:")

        results_df = data[["magnitude", "depth", "latitude", "longitude"]].copy()
        results_df["Prediction"] = ["Tsunami" if p == 1 else "No Tsunami" for p in predictions]
        results_df["Confidence"] = [f"{c:.1f}%" if c is not None else "N/A" for c in confidences]

        if actual is not None:
            results_df["Actual"] = ["Tsunami" if a == 1 else "No Tsunami" for a in actual]
            results_df["Correct"] = ["✅" if p == a else "❌" for p, a in zip(predictions, actual)]

        st.dataframe(results_df, width='stretch')



st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>INFO6105 | Earthquake & Tsunami Prediction System | Zhan Tang & Xinru Zhang</p>
    </div>
""", unsafe_allow_html=True)






