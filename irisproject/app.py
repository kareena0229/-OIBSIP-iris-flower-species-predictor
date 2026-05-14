import streamlit as st
import joblib
import pandas as pd

# 1. Load the Iris model
model = joblib.load('iris_logic_model.pkl')

st.title("🌸 Iris Flower Species Predictor")
st.write("Adjust the sliders to see which species the model identifies.")

# 2. Create sliders for the 4 flower measurements
sepal_l = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_w = st.slider("Sepal Width (cm)", 2.0, 5.0, 3.5)
petal_l = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_w = st.slider("Petal Width (cm)", 0.1, 3.0, 0.2)

# 3. Predict button
if st.button("Identify Species"):
    # Create a dataframe for the input
    input_df = pd.DataFrame([[sepal_l, sepal_w, petal_l, petal_w]], 
                            columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
    
    # Get prediction
    prediction = model.predict(input_df)[0]
    
    # Display result with styling
    st.success(f"The predicted species is: **{prediction}**")