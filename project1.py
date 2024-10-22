import pandas as pd
import numpy as np
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv(r"E:\Sem -5\Machine learning\spirulina_dataset_final.csv")

# Select necessary features for prediction
selected_features = [
    'Initial pH', 'Illumination Intensity', 'NaNO3', 'Inoculum Level', 
    'Culture Time', 'Seawater Medium', 'NaHCO3', 'Temperature', 
    'K2HPO4', 'Salinity', 'Aeration', 'Light Time', 'Dark Time', 
    'BG11 Medium', 'Na2CO3', 'Pond System', 'Urea', 'Trace Elements', 
    'Biomass Yield'
]

# Encode 'Protein Content' column
def categorize_protein_content(value):
    if value == 0.5:
        return 1  # Low
    elif value == 0.75:
        return 2  # Medium
    else:
        return 3  # High

df['Protein Content'] = df['Protein Content'].apply(categorize_protein_content)

# Prepare data for model training
X = df[selected_features]
y = df['Protein Content']

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Streamlit UI Elements
st.title("Spirulina Protein Content Predictor")

# Create input fields for all features
user_input = {}
for feature in selected_features:
    user_input[feature] = st.number_input(f"{feature}:", value=0.0)

# Prediction logic
if st.button("Predict"):
    try:
        # Prepare input data
        input_data = np.array([list(user_input.values())])
        
        # Predict protein content class
        prediction = clf.predict(input_data)
        
        # Display the result
        st.success(f"Predicted Protein Content Class: {prediction[0]}")

    except ValueError:
        st.error("Please enter valid numeric values!")
