# For deployment 

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Page config
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")
st.title("🚢 Titanic Survival Prediction System")
st.write("This app trains a model to predict if a passenger would survive the Titanic disaster.")

# Load dataset
df = pd.read_csv('train.csv')
st.subheader("📊 Raw Data")
st.dataframe(df.head())

# Data preprocessing
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])  # male:1, female:0
df['Embarked'] = label_encoder.fit_transform(df['Embarked'])  # C=0, Q=1, S=2

X = df.drop('Survived', axis=1)
y = df['Survived']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Results
st.subheader("📈 Model Performance")
st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
st.text("Classification Report:")
st.code(classification_report(y_test, y_pred), language='text')

# Feature importance
st.subheader("🔍 Feature Importance")

feature_importance = model.feature_importances_
feature_names = X.columns
sorted_indices = np.argsort(feature_importance)[::-1]

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(X.shape[1]), feature_importance[sorted_indices], color='skyblue')
ax.set_xticks(range(X.shape[1]))
ax.set_xticklabels([feature_names[i] for i in sorted_indices], rotation=45)
ax.set_title("Most Important Features for Survival Prediction")
ax.set_xlabel("Features")
ax.set_ylabel("Importance Score")
st.pyplot(fig)

st.success("✅ Model training and evaluation completed.")