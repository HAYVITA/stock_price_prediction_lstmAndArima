import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMAResults

# Load data
@st.cache
def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=True)
    return data

# Load ARIMA model
@st.cache(allow_output_mutation=True)
def load_arima_model(model_path):
    model = pickle.load(open(model_path, 'rb'))
    return model

# Function to predict using ARIMA model
def predict_arima(model, data, days):
    pred = model.predict(start=len(data), end=len(data)+days, typ='levels')
    return pred

def main():
    st.title('Stock Price Prediction with ARIMA')

    # Upload file and select model
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.write('Uploaded data:')
        st.write(data.head())

        model_path = 'model_pickle_apple_arima.sav'  # Update with your model path
        model = load_arima_model(model_path)

        days = st.number_input('Enter number of days to predict:', min_value=1, value=7)

        if st.button('Predict'):
            pred = predict_arima(model, data[' Close/Last'], days)
            
            # Plotting historical data
            fig1 = go.Figure([go.Scatter(x=data['Date'], y=data[' Close/Last'])])
            st.plotly_chart(fig1, use_container_width=True)

            # Plotting predicted values
            x = [i for i in range(1, days+2)]
            fig2 = go.Figure([go.Scatter(x=x, y=pred)])
            st.plotly_chart(fig2, use_container_width=True)

            st.write('Predicted Values:')
            st.write(pred)

if __name__ == '__main__':
    main()
