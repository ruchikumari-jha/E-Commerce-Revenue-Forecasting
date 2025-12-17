import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Sales Forecasting AI", layout="wide")

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://www.marketingaiinstitute.com/hs-fs/hubfs/2022%20Website%20Redesign/Background%20Graphics/MAII_Backgrounds_hexagonal-wave.jpg?width=1600&height=900&name=MAII_Backgrounds_hexagonal-wave.jpg");
             background-attachment: fixed;
             background-size: cover;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()
st.title("E-Commerce Revenue Forecasting")
# st.markdown("### Advanced LSTM Prediction System")

st.sidebar.header("📂 Upload Data")
uploaded_orders = st.sidebar.file_uploader("Upload 'List of Orders.csv'", type="csv")
uploaded_details = st.sidebar.file_uploader("Upload 'Order Details.csv'", type="csv")

if uploaded_orders and uploaded_details:
    orders = pd.read_csv(uploaded_orders)
    details = pd.read_csv(uploaded_details)
    
    df = pd.merge(orders, details, on='Order ID', how='inner')
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True, errors='coerce')
    df['Revenue'] = df['Amount']
    
    st.success("Files Loaded & Merged Successfully! 🔥")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Orders", df['Order ID'].nunique())
    col2.metric("Total Revenue", f"₹{df['Revenue'].sum():,.0f}")
    col3.metric("Top Category", df['Category'].mode()[0])

    categories = df['Category'].unique()
    selected_cat = st.selectbox("Select Category to Forecast", np.append(['All Categories'], categories))

    if selected_cat != 'All Categories':
        df_filtered = df[df['Category'] == selected_cat]
    else:
        df_filtered = df

    df_weekly = df_filtered.set_index('Order Date').resample('W')['Revenue'].sum().reset_index()
    
    st.subheader(f"📈 Historical Revenue ({selected_cat})")
    st.line_chart(df_weekly.set_index('Order Date')['Revenue'])

    if st.button(f"Forecast of {selected_cat}"):
        with st.spinner('Training Neural Network... (This helps accuracy!)'):
            
            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(df_weekly['Revenue'].values.reshape(-1, 1))
            
            time_step = 4  
            X, y = [], []
            for i in range(time_step, len(data_scaled)):
                X.append(data_scaled[i-time_step:i, 0])
                y.append(data_scaled[i, 0])
            
            X, y = np.array(X), np.array(y)
            X = X.reshape(X.shape[0], X.shape[1], 1)
            
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
            model.add(LSTM(50))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')
            
            model.fit(X, y, epochs=50, batch_size=8, verbose=0)
            
            future_inputs = data_scaled[-time_step:].reshape(1, time_step, 1)
            future_preds = []
            
            for _ in range(12):
                pred = model.predict(future_inputs, verbose=0)
                future_preds.append(pred[0,0])
                future_inputs = np.append(future_inputs[:,1:,:], pred.reshape(1,1,1), axis=1)
            
            future_rev = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
            
            last_date = df_weekly['Order Date'].iloc[-1]
            future_dates = pd.date_range(start=last_date, periods=13, freq='W')[1:]
            
            forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Revenue': future_rev.flatten()})
            
            st.success("Forecasting Complete!")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df_weekly['Order Date'], df_weekly['Revenue'], label='Historical', marker='o')
            ax.plot(forecast_df['Date'], forecast_df['Predicted Revenue'], label='Forecast', linestyle='--', color='red', marker='x')
            ax.set_title(f"12-Week Revenue Forecast for {selected_cat}")
            ax.legend()
            st.pyplot(fig)
            
            st.write("### Predicted Values")
            st.dataframe(forecast_df.style.format({"Predicted Revenue": "₹{:.2f}"}))

else:
    st.info("Please upload the CSV files to begin the magic! 👈")
