from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import os
import logging
from xgboost import XGBRegressor

app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
models_path = "models"
preprocessors_path = "preprocessors"
data_path = "data"

# Load models with error handling
def load_model(model_filename):
    model_path = os.path.join(models_path, model_filename)
    try:
        model = XGBRegressor()
        model.load_model(model_path)  # Ensure this is the correct method for loading
        logger.info(f"Loaded model '{model_filename}' successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model '{model_filename}': {e}")
        return None

# Load XGBoost models
dep_delay_model = load_model("opt_dep_delay_model.booster")
arr_delay_model = load_model("opt_arr_delay_model.booster")

# Load reason model from pickle
try:
    with open("models/opt_delay_reason_model.pkl", 'rb') as file:
        delay_reason_model = pickle.load(file)
    logger.info("Loaded delay reason model successfully.")
except Exception as e:
    logger.error(f"Failed to load delay reason model: {e}")
    delay_reason_model = None

# Load preprocessors
def load_preprocessor(preprocessor_filename):
    preprocessor_path = os.path.join(preprocessors_path, preprocessor_filename)
    try:
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        logger.info(f"Loaded preprocessor '{preprocessor_filename}' successfully.")
        return preprocessor
    except Exception as e:
        logger.error(f"Failed to load preprocessor '{preprocessor_filename}': {e}")
        return None

dep_preprocessor = load_preprocessor("preprocessor_dep.pkl")
arr_preprocessor = load_preprocessor("preprocessor_arr.pkl")
reasons_preprocessor = load_preprocessor("preprocessor_delay_reason.pkl")

# Load route statistics
try:
    route_stats = pd.read_csv(os.path.join(data_path, "mean_airtime_elapsedtime_by_route.csv"))
    logger.info("Loaded route statistics successfully.")
except Exception as e:
    logger.error(f"Failed to load route statistics: {e}")
    route_stats = None

# Define input data model
class InputData(BaseModel):
    ORIGIN: str
    DEST: str
    AIRLINE_CODE: str
    DEP_HOUR: int
    ARR_HOUR: int
    PEAK_HOUR: int
    BUSY_ORIGIN: int
    BUSY_DEST: int
    HOLIDAY_SEASON: int
    DAY_OF_WEEK: int
    WEEK_OF_YEAR: int
    MONTH: int
    CRS_ELAPSED_TIME: float

# Helper functions
def get_mean_air_time(origin, dest):
    match = route_stats[(route_stats['ORIGIN'] == origin) & (route_stats['DEST'] == dest)]
    if not match.empty:
        return match['AIR_TIME'].values[0]
    else:
        raise HTTPException(status_code=400, detail=f"No mean AIR_TIME data found for route {origin} to {dest}.")

def get_mean_elapsed_time(origin, dest):
    match = route_stats[(route_stats['ORIGIN'] == origin) & (route_stats['DEST'] == dest)]
    if not match.empty:
        return match['ELAPSED_TIME'].values[0]
    else:
        raise HTTPException(status_code=400, detail=f"No mean ELAPSED_TIME data found for route {origin} to {dest}.")

# Prediction Endpoint
@app.post("/predict_all")
async def predict_all(input_data: InputData):
    if dep_preprocessor is None or dep_delay_model is None or arr_delay_model is None or reasons_preprocessor is None or delay_reason_model is None:
        raise HTTPException(status_code=500, detail="Model or preprocessor not loaded.")

    try:
        input_df = pd.DataFrame([input_data.dict()])

        # Get AIR_TIME and ELAPSED_TIME
        input_df['AIR_TIME'] = get_mean_air_time(input_df['ORIGIN'][0], input_df['DEST'][0])
        input_df['ELAPSED_TIME'] = get_mean_elapsed_time(input_df['ORIGIN'][0], input_df['DEST'][0])
        input_df['SCHEDULED_VS_ACTUAL_TIME'] = input_df['ELAPSED_TIME'] - input_df['CRS_ELAPSED_TIME']

        # Predict Departure Delay
        dep_features = dep_preprocessor.transform(input_df)
        predicted_dep_delay = dep_delay_model.predict(dep_features)

        # Ensure the prediction result is a single value
        if len(predicted_dep_delay) != 1:
            raise ValueError("Unexpected output size for departure delay prediction.")
        predicted_dep_delay = predicted_dep_delay[0]

        input_df['DEP_DELAY'] = predicted_dep_delay  # Add predicted departure delay to input DataFrame

        # Predict Arrival Delay
        arr_features = arr_preprocessor.transform(input_df)
        predicted_arr_delay = arr_delay_model.predict(arr_features)

        # Ensure the prediction result is a single value
        if len(predicted_arr_delay) != 1:
            raise ValueError("Unexpected output size for arrival delay prediction.")
        predicted_arr_delay = predicted_arr_delay[0]

        input_df['ARR_DELAY'] = predicted_arr_delay  # Add predicted arrival delay to input DataFrame

        # Predict Delay Reasons
        reasons_features = reasons_preprocessor.transform(input_df)
        predicted_reasons = delay_reason_model.predict(reasons_features)

        # Convert numpy array predictions to a list
        predicted_reasons = [int(reason) for reason in predicted_reasons[0]]

        # Prepare response
        total_delay = predicted_dep_delay + predicted_arr_delay
        return {
            "predicted_dep_delay": float(predicted_dep_delay),
            "predicted_arr_delay": float(predicted_arr_delay),
            "total_delay": float(total_delay),
            "predicted_reasons": dict(
                zip(['CARRIER', 'LATE_AIRCRAFT', 'NAS', 'SECURITY', 'WEATHER'], predicted_reasons))
        }
    except Exception as e:
        logger.error(f"Error in predicting all: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while predicting.")


