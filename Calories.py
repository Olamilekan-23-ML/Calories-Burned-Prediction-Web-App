#___IMPORTING DEPENDENCIES___#
import pickle 
import streamlit as st 
import numpy as np
#___LOADING MODEL___#
load_model = pickle.load(('mymodel.pkl', 'rb'))
#___ENCODING___#
encoder = {'Gender':{'Male':0, 'Female':1}}
#___CREATING FORM___#
st.title("🏃‍♂️CALORIES BURNED PREDICTOR")
gender = st.selectbox('**Gender**', [ 'Male', 'Female'], index=None)
age = st.number_input("**Age**", min_value=10, max_value=80, value=None)
height = st.slider("**Height (cm)**",120.0,230.0, value=None, step=0.1)
weight = st.slider("**Weight (kg)**",30.0,150.0, value=None, step=0.1)
time = st.slider("**Duration (mins)**",1,60, value=None)
heart_rate =st.slider("**Heart Rate bpm**",60,130, value=None)
temp =st.slider("**Body Temp (°C)**",36.0,42.0, value=None, step=0.1)
def encode_inputs():
    return[
        encoder['Gender'][gender],
        float(age),
        float(height),
        float(weight),
        float(time),
        float(heart_rate),
        float(temp)]
st.markdown('---')
#____CREATING THE BUTTON___#
if st.button("🔥Predict Calories", use_container_width=True):
    try:
      #___PREDICTIVE SYSTEM___#
      input_encoder = encode_inputs()
      input_data_as_numpy_array = np.asarray(input_encoder)
      input_data_reshape = input_data_as_numpy_array.reshape(1,-1)
      prediction = load_model.predict(input_data_reshape)
      st.markdown('---')
      #___PREDICTION OUTPUT___#
      st.success(f'✅Estimated Calories Burned: {prediction[0]:.1f}kcal')
    #___FILLING ALL THE FIELDS REQUIREMENT___#
    except:

      st.error('❌Please fill all the fields before predicting!')




