import joblib
import requests
import numpy as np
import pandas as pd
import streamlit as st
from preprocessing.preprocessing import preprocess_data

model = joblib.load('artifacts/model.joblib')
st.set_page_config(page_title="Calculate Carbon", page_icon="🌍")

st.markdown("<h1 style='text-align: center; color: #34343c;'>Individual Carbon Footprint Prediction</h1>", unsafe_allow_html=True)
st.write(' ')
st.write(' ')

with st.container(border=True, height = 310):
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["About you", "Vehicle Information", "Energy Usage", "Habits", "Waste Management"])

    with tab1:
        tab1_col1, tab1_col2 = st.columns(2)
        with tab1_col1:
            with st.container(border = True):
                sex = st.selectbox('Your gender', ['Male', 'Female'])
                sex = sex.lower()

        with tab1_col2:
            with st.container(border = True):
                body_type = st.selectbox('Your body type', ['Underweight', 'Normal', 'Overweight', 'Obese'])
                body_type = body_type.lower()

        with tab1_col1:
            with st.container(border = True):
                how_often_shower = st.selectbox('How often do you shower', ['Daily', 'Twice a day', 'Less frequently', 'More frequently'])
                how_often_shower = how_often_shower.lower()

        with tab1_col2:
            with st.container(border = True):
                diet = st.selectbox('Diet', ['Vegetarian', 'Omnivore', 'Vegan', 'Pescatarian'])
                diet = diet.lower()

    with tab2:
        tab2_col1, tab2_col2 = st.columns(2)
        with tab2_col1:
            with st.container(border = True):
                transport = st.selectbox('Transport', ['Walk/Bicycle', 'Public', 'Private'])
                transport = transport.lower()

        with tab2_col2:
            with st.container(border = True):
                vehicle_type = st.selectbox('Vehicle type', ['Electric', 'Petrol', 'Diesel', 'Hybrid', 'LPG', 'No Fuel'])
                vehicle_type = vehicle_type.lower()

        with tab2_col1:
            with st.container(border = True):
                vehicle_monthly_distance_km = st.number_input('Monthly Distance Travelled (In KMs)', 0, 1000, 200)

        with tab2_col2:
            with st.container(border = True):
                frequency_of_traveling_by_air = st.selectbox('How often do you travel by Plane?', ['Never', 'Very frequently', 'Frequently', 'Rarely'])
                frequency_of_traveling_by_air = frequency_of_traveling_by_air.lower()

    with tab3:
        tab3_col1, tab3_col2 = st.columns(2)
        with tab3_col1:
            with st.container(border = True):
                heating_energy_source = st.selectbox('Heating Energy Source', ['Electricity', 'Coal', 'Wood', 'Natural Gas'])
                heating_energy_source = heating_energy_source.lower()
                
        with tab3_col2:
            with st.container(border = True):
                energy_efficiency = st.selectbox('Are you energy efficient', ['Sometimes', 'Yes', 'No'])

        with tab3_col1:
            with st.container(border = True):
                cooking_with = st.multiselect('Cooking Appliances', ['Stove', 'Oven', 'Microwave', 'Grill'])

        with tab3_col2:
            with st.container(border = True):
                recycling = st.multiselect('What do you recycle?', ['Paper', 'Plastic', 'Glass', 'Metal', 'No Recycling'])

    with tab4:
        tab4_col1, tab4_col2 = st.columns(2)
        with tab4_col1:
            with st.container(border = True):
                social_activity = st.selectbox('Social Activity', ['Never', 'Often', 'Sometimes'])
                social_activity = social_activity.lower()

        with tab4_col2:
            with st.container(border = True):
                how_many_new_clothes_monthly = st.number_input('Monthly clothes purchased', 0, 100, 1)

        with tab4_col1:
            with st.container(border = True):
                how_long_tv_pc_daily_hour = st.number_input('Daily TV / PC watch time (in hours)', 0, 24, 4)

        with tab4_col2:
            with st.container(border = True):
                how_long_internet_daily_hour = st.number_input('Daily internet usage (in hours)', 0, 24, 4)

    with tab5:
        tab5_col1, tab5_col2 = st.columns(2)
        with tab5_col1:
            with st.container(border = True):
                waste_bag_size = st.selectbox('Waste Bag Size', ['Small', 'Medium', 'Large', 'Extra Large'])
                waste_bag_size = waste_bag_size.lower()

        with tab5_col2:
            with st.container(border = True):
                waste_bag_weekly_count = st.number_input('Weekly waste bag count', 1, 7, 1)

        with tab5_col1:
            with st.container(border = True):
                monthly_grocery_bill = st.number_input('Monthly Grocery Bill (in dollars)', 50, 300, 70)

pred_col1, pred_col2, pred_col3 = st.columns(3)

m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #b1f3b1;
}
</style>""", unsafe_allow_html=True)

button = pred_col2.button("Predict Carbon Footprint")

if button:
    data = {
        "body_type": body_type,
        "sex": sex,
        "diet": diet,
        "how_often_shower": how_often_shower,
        "heating_energy_source": heating_energy_source,
        "transport": transport,
        "vehicle_type": vehicle_type,
        "social_activity": social_activity,
        "monthly_grocery_bill": monthly_grocery_bill,
        "frequency_of_traveling_by_air": frequency_of_traveling_by_air,
        "vehicle_monthly_distance_km": vehicle_monthly_distance_km,
        "waste_bag_size": waste_bag_size,
        "waste_bag_weekly_count": waste_bag_weekly_count,
        "how_long_tv_pc_daily_hour": how_long_tv_pc_daily_hour,
        "how_many_new_clothes_monthly": how_many_new_clothes_monthly,
        "how_long_internet_daily_hour": how_long_internet_daily_hour,
        "energy_efficiency": energy_efficiency,
        "recycling": str(recycling),
        "cooking_with": str(cooking_with)
    }

    response = requests.post("http://20.235.218.72/predict", json=data)
    pred = response.json()["Predicted Monthly Emission"]

    st.markdown(f"<h5 style='text-align: center; color: #34343c;'>Predicted Monthly Emission: {pred} kgCO2e</h5>", unsafe_allow_html=True)