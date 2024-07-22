import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import plotly.express as px

# Load and preprocess the dataset
data = pd.read_csv("water_potability.csv")
data = data.dropna()
X = data.drop('Potability', axis=1)
y = data['Potability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=["Home", "Dataset", "Data Visualization", "Model Accuracy", "About"],
        icons=["house-fill", "database-fill", "graph-up-arrow", "boxes", "info-circle-fill"],
        menu_icon="menu-button-wide-fill",
        default_index=0
    )

# Home Page: Input water quality parameters and predict potability
if selected == "Home":
    # st.logo("logo1.png", icon_image="logo1.png")
    st.image("HydroSense logo-1.png")
    st.caption("### Enter water quality values to predict potability:")

    # Input fields for water quality parameters
    ph = st.number_input('pH Level', min_value=0.0, max_value=14.0, value=7.0)
    hardness = st.number_input('Hardness', min_value=0.0, value=150.0)
    solids = st.number_input('Solids (mg/L)', min_value=0.0, value=500.0)
    chloramines = st.number_input('Chloramines (ppm)', min_value=0.0, value=5.0)
    sulfate = st.number_input('Sulfate (mg/L)', min_value=0.0, value=250.0)
    conductivity = st.number_input('Conductivity (μS/cm)', min_value=0.0, value=500.0)
    organic_carbon = st.number_input('Organic Carbon (ppm)', min_value=0.0, value=10.0)
    trihalomethanes = st.number_input('Trihalomethanes (μg/L)', min_value=0.0, value=50.0)
    turbidity = st.number_input('Turbidity (NTU)', min_value=0.0, value=3.0)

    # Button to make the prediction
    if st.button("Predict Potability"):
        input_data = np.array([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]])
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.success("The water is safe to drink.")
        else:
            st.error("The water is not safe to drink.")
    
    # Display model accuracy
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

# Dataset Page: Display the dataset
if selected == "Dataset":
    st.title("Dataset")
    # st.write("### Dataset")
    st.dataframe(data)

# Data Visualization Page: Visualize data distributions and correlations
if selected == "Data Visualization":
    st.title("Data Visualization")
    # st.write("### Distribution of Unsafe and Safe Water")
    # counts = data['Potability'].value_counts()
    # fig, ax = plt.subplots()
    # ax.bar(counts.index, counts.values, color=['red', 'green'])
    # ax.set_xlabel('Potability')
    # ax.set_ylabel('Count')
    # ax.set_title('Distribution of Unsafe and Safe Water')
    # ax.set_xticks([0, 1])
    # ax.set_xticklabels(['Unsafe', 'Safe'])
    # st.pyplot(fig)

    # st.write("### Factors Affecting Water Quality")
    columns = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
    for column in columns:
        figure = px.histogram(data, x=column, color='Potability', title=f'Factors Affecting Water Quality: {column}')
        st.plotly_chart(figure)

# Model Accuracy Page: Display model accuracy
if selected == "Model Accuracy":
    st.title("Model Accuracy")

    # Introduction
    st.markdown("""
    

    Welcome to the Model Accuracy page!
    Here, we present the performance metrics of our machine learning
    model designed to predict whether water is potable or not.

    ### Overview

    Our model has been trained using a comprehensive dataset containing
    various water quality parameters. To ensure the highest level of 
    accuracy, we have utilized state-of-the-art machine learning
    techniques. Below, you will find detailed metrics that highlight
    the effectiveness of our model.
    """)

    # Key Metrics Section
    st.markdown("""
    ### Key Metrics

    - **Accuracy**  : The percentage of correct predictions out of all 
                    predictions made. This is a primary indicator of
                    the model's overall performance.
    - **Precision** : This metric shows the ratio of true positive
                    predictions to the total predicted positives.
                    It indicates how many of the predicted potable 
                    water instances were actually potable.
    - **Recall**    : This represents the ratio of true positive predictions
                    to the total actual positives. It measures the ability
                    of the model to identify all potable water instances.
    - **F1 Score**  : The harmonic mean of precision and recall. It provides
                    a balance between precision and recall, offering a
                    single metric to evaluate the model.
    """)

    # Add performance metrics (replace these with actual values from your model)
    # accuracy = 0.92
    # precision = 0.88
    # recall = 0.85
    # f1_score = 0.86

    st.markdown("""
    We continually refine our model to ensure the highest accuracy and reliability
    in predicting water potability. Below, you can see the current performance 
    metrics of our model.
    """)

    # Displaying the metrics
    st.metric(label="Accuracy", value=f"{accuracy * 100:.2f}%")
    st.metric(label="Precision", value=f"{precision * 100:.2f}%")
    st.metric(label="Recall", value=f"{recall * 100:.2f}%")
    st.metric(label="F1 Score", value=f"{f1 * 100:.2f}%")

# About Page: Provide information about the application
if selected == "About":
    st.title("About")

    # Introduction
    st.markdown("""
    
    Welcome to HydroSense, your reliable companion for ensuring safe drinking water. HydroSense is a cutting-edge web application designed to predict the potability of water using advanced machine learning algorithms.
    """)
    # Mission Section
    st.markdown("""
    ### Our Mission

    Our mission is to provide everyone with access to accurate water quality assessments, helping to ensure safe and clean drinking water for all. We believe that everyone deserves access to potable water, and we're dedicated to making that a reality through technology.
    """)

    # Vision Section
    st.markdown("""
    ### Our Vision

    We envision a world where water quality is easily accessible and understandable, empowering individuals and communities to make informed decisions about their water consumption.
    """)

    # Key Features Section
    st.markdown("""
    ### Key Features

    - **User-Friendly Interface**: An intuitive and easy-to-navigate interface.
    - **Accurate Predictions**: State-of-the-art machine learning algorithms for high accuracy.
    - **Real-Time Analysis**: Get immediate results for your water quality assessment.
    - **Comprehensive Data**: Analysis based on a wide range of water quality parameters.
    """)

    st.markdown("""
    ### How It Works

    1. **Data Input**: Enter the required water quality parameters into the app.
    2. **Model Analysis**: Our machine learning model processes the data and predicts water potability.
    3. **Results**: View the prediction results and additional information about water safety.
    """)

    # Why Water Potability Matters Section
    st.markdown("""
    ### Why Water Potability Matters

    Access to potable water is essential for health and well-being. Contaminated water can lead to serious health issues, including gastrointestinal illnesses, reproductive problems, and neurological disorders. HydroSense aims to provide an easy and efficient way to monitor water quality and ensure the safety of your drinking water.
    """)

    # Contact Us Section
    st.markdown("""
    ### Contact Us

    If you have any questions or feedback, please feel free to contact us at [support@hydrosense.com](mailto:support@hydrosense.com).
    """)

