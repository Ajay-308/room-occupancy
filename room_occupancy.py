import streamlit as st
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open(r"occupancy.pkl", 'rb'))

st.title('Room Occupancy Predictor')

cols = {}

for column in ['Date', 'Time', 'S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp', 'S1_Light',
               'S2_Light', 'S3_Light', 'S4_Light', 'S1_Sound', 'S2_Sound', 'S3_Sound',
               'S4_Sound', 'S5_CO2', 'S5_CO2_Slope', 'S6_PIR', 'S7_PIR']:
    if column == 'Date' or column == 'Time':
        col_value = st.text_input(f"Enter {column}", value='')
    else:
        col_value = st.number_input(f"Enter {column}", value=0.0, step=0.1)
    cols[column] = col_value

if st.button('Predict Occupancy'):
    input_data = []
    for column in ['Date', 'Time', 'S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp', 'S1_Light',
                   'S2_Light', 'S3_Light', 'S4_Light', 'S1_Sound', 'S2_Sound', 'S3_Sound',
                   'S4_Sound', 'S5_CO2', 'S5_CO2_Slope', 'S6_PIR', 'S7_PIR']:
        if column == 'Date' or column == 'Time':
            # Convert date and time strings to datetime objects
            input_data.append(pd.to_datetime(cols[column]))
        else:
            input_data.append(cols[column])

    # Convert date and time to float
    input_data[0] = input_data[0].view('int64').astype('float64')
    input_data[1] = input_data[1].to_pydatetime().time()
    input_data[1] = input_data[1].hour * 3600 + input_data[1].minute * 60 + input_data[1].second
    
    input_data = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_data)

    if prediction == 1:
        st.write('Room is occupied.')
    else:
        st.write('Room is not occupied.')
