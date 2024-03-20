import ast
import pandas as pd
import xgboost as xgb
from joblib import load
from utils.myDictionaries import *
pd.set_option('future.no_silent_downcasting', True)

scaler = load('artifacts/scaler.joblib')

def process_column(df, column_name):
        df[column_name] = df[column_name].apply(ast.literal_eval)

        df_exploded = df.explode(column_name)
        df_one_hot = pd.get_dummies(df_exploded[column_name]).groupby(df_exploded.index).sum()
        df_one_hot.columns = column_name + "_" + df_one_hot.columns
        del df[column_name]

        df = df.join(df_one_hot)
        return df

def preprocess_data(df):

    # Replace categorical values with numerical values
    df['body_type'] = df['body_type'].replace(bodyTypeDict)
    df['sex'] = df['sex'].replace(sexDict)
    df['diet'] = df['diet'].replace(dietDict)
    df['how_often_shower'] = df['how_often_shower'].replace(showerDict)
    df['heating_energy_source'] = df['heating_energy_source'].replace(energySourceDict)
    df['transport'] = df['transport'].replace(transportDict)
    df['vehicle_type'] = df['vehicle_type'].replace(fuelsDict)
    df['social_activity'] = df['social_activity'].replace(socialDict)
    df['frequency_of_traveling_by_air'] = df['frequency_of_traveling_by_air'].replace(airTravelDict)
    df['waste_bag_size'] = df['waste_bag_size'].replace(wasteBagSizeDict)
    df['energy_efficiency'] = df['energy_efficiency'].replace(energyEfficiencyDict)
    df['recycling'] = df['recycling'].replace("[]", "['no recycling']")

    df['wasteBagType'] = df['waste_bag_size'] + df['waste_bag_weekly_count']

    df['recycling'] = df['recycling'].astype(str)
    df['cooking_with'] = df['cooking_with'].astype(str)

    df = process_column(df, 'recycling')
    df = process_column(df, 'cooking_with')
    df = df.astype(int)

    df_scaled = scaler.transform(df)
    testing = xgb.DMatrix(df_scaled, label=None)

    return testing