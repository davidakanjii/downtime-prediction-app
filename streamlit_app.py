import streamlit as st
import pandas as pd
import joblib
import datetime

# Load the trained models
clf_pipeline = joblib.load('breakdown_classifier_pipeline.pkl')
reg_pipeline = joblib.load('downtime_regression_pipeline.pkl')

# Set up the title and layout
st.set_page_config(page_title="Breakdown and Downtime Prediction", layout="wide")

# Project description using WERKS and WERKS CODE
werks_code = st.text_input("WERKS CODE (Location of Plant)", "A112")  
werks = st.text_input("Project Description (WERKS)", "Kaduna Noodles")  

st.title(f"Breakdown and Downtime Prediction for {werks} ({werks_code})")

# Sidebar for prediction mode selection
st.sidebar.title("Choose Prediction Mode")
prediction_mode = st.sidebar.radio("Prediction Type", ("Single Prediction", "Batch Prediction (CSV/Excel)"))

# Add a download button for sample CSV in the sidebar
st.sidebar.header("Download Sample Data")
with open("sample1.csv", "rb") as file:
    st.sidebar.download_button(
        label="Download Sample CSV",
        data=file,
        file_name="sample1.csv",
        mime="text/csv"
    )

if prediction_mode == "Single Prediction":
    st.header("Single Prediction Mode")

    # Collect input for single prediction
    order_number = st.text_input("Order Number")
    start_date = st.date_input("Start Date")
    start_time = st.time_input("Start Time", datetime.time(0, 0))
    end_date = st.date_input("End Date")
    end_time = st.time_input("End Time", datetime.time(0, 0))
    line = st.text_input("Line")
    shift = st.number_input("Shift", min_value=0, max_value=3, step=1)
    equipment = st.text_input("Equipment")
    group = st.text_input("Group")
    notification_number = st.text_input("Notification Number (optional)")
    pot_min = st.number_input("POT Min (optional)", min_value=0.0, step=0.1)
    pot_hour = st.number_input("POT Hour (optional)", min_value=0.0, step=0.1)

    if st.button("Predict"):
        start_datetime = datetime.datetime.combine(start_date, start_time)
        end_datetime = datetime.datetime.combine(end_date, end_time)
        operation_duration = (end_datetime - start_datetime).total_seconds()

        input_data = {
            'Order Number': [order_number],
            'WERKS CODE': [werks_code],
            'Group': [group],
            'Notification Number': [notification_number],
            'Shift': [shift],
            'POT Min': [pot_min],
            'POT Hour': [pot_hour],
            'Equiment': [equipment],
            'Operation Duration': [operation_duration],
            'Start Hour': [start_datetime.hour],
            'End Hour': [end_datetime.hour],
            'Day of Week': [start_datetime.weekday()],
            'Line': [line],
        }

        input_df = pd.DataFrame(input_data)

        breakdown_pred = clf_pipeline.predict(input_df)[0]
        if breakdown_pred == 1:
            downtime_pred = reg_pipeline.predict(input_df)[0]
            st.success(f"Breakdown Prediction: Yes, Downtime: {downtime_pred} seconds")
        else:
            st.success("No Breakdown, so no downtime.")

else:
    st.header("Batch Prediction Mode")

    uploaded_file = st.file_uploader("Upload a CSV or Excel file for batch predictions", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                data = pd.read_excel(uploaded_file)

            st.write("Data Preview:")
            st.write(data.head())

            required_columns = ['Order Number', 'WERKS CODE', 'Group', 'Notification Number',
                                'Start Date', 'End Date', 'Line', 'Shift', 'Equiment']
            if all(col in data.columns for col in required_columns):
                data['Start Date'] = pd.to_datetime(data['Start Date'])
                data['End Date'] = pd.to_datetime(data['End Date'])

                data['Operation Duration'] = (data['End Date'] - data['Start Date']).dt.total_seconds()
                data['Start Hour'] = data['Start Date'].dt.hour
                data['End Hour'] = data['End Date'].dt.hour
                data['Day of Week'] = data['Start Date'].dt.dayofweek

                data.drop(columns=['Start Date', 'End Date'], inplace=True)

                breakdown_preds = clf_pipeline.predict(data)
                downtime_preds = [
                    reg_pipeline.predict(data.iloc[[i]])[0] if breakdown_preds[i] == 1 else None
                    for i in range(len(breakdown_preds))
                ]

                data['Breakdown Prediction'] = breakdown_preds
                data['Downtime Prediction'] = downtime_preds

                st.write("Prediction Results:")
                st.write(data[['Breakdown Prediction', 'Downtime Prediction']])

                csv_data = data.to_csv(index=False)
                st.download_button(label="Download Predictions as CSV", data=csv_data, mime="text/csv")

            else:
                st.error("The uploaded file is missing some required columns.")
        except Exception as e:
            st.error(f"Error processing the file: {e}")
