import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from fastapi import FastAPI
from pydantic import BaseModel
from preprocessing.preprocessing import preprocess_data

app = FastAPI()

# Load the model
model = joblib.load('artifacts/model.joblib')

class Item(BaseModel):
    body_type: str
    sex: str
    diet: str
    how_often_shower: str
    heating_energy_source: str
    transport: str
    vehicle_type: str
    social_activity: str
    monthly_grocery_bill: float
    frequency_of_traveling_by_air: str
    vehicle_monthly_distance_km: float
    waste_bag_size: str
    waste_bag_weekly_count: int
    how_long_tv_pc_daily_hour: float
    how_many_new_clothes_monthly: int
    how_long_internet_daily_hour: float
    energy_efficiency: str
    recycling: str
    cooking_with: str

@app.post("/predict")
def predict(item: Item):
    df = pd.read_csv('utils/example.csv')
    data = [[item.body_type, item.sex, item.diet, item.how_often_shower, item.heating_energy_source,
           item.transport, item.vehicle_type, item.social_activity, item.monthly_grocery_bill,
           item.frequency_of_traveling_by_air, item.vehicle_monthly_distance_km,
           item.waste_bag_size, item.waste_bag_weekly_count, item.how_long_tv_pc_daily_hour,
           item.how_many_new_clothes_monthly, item.how_long_internet_daily_hour,
           item.energy_efficiency, item.recycling, item.cooking_with]]

    new_data_df = pd.DataFrame(data, columns = df.columns)
    df = pd.concat([df, new_data_df], ignore_index = True)

    df_DMatrix = preprocess_data(df)
    prediction = model.predict(df_DMatrix)[1]
    pred = np.round(prediction, 2)

    # Return the result
    return {"Predicted Monthly Emission": str(pred)}