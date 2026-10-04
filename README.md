# 🎯 AI-Powered Skill Demand Prediction System

## 📌 Project Overview

The **AI-Powered Skill Demand Prediction System** is a machine learning-based web application that predicts a suitable career path based on a user's technical skills.

The system also provides career prediction probabilities, skill demand analysis, career recommendations, and learning recommendations.

## 🎯 Objectives

- Predict a suitable career path using Machine Learning.
- Analyze the demand for different technical skills.
- Compare multiple Machine Learning algorithms.
- Display career prediction probabilities.
- Recommend alternative career paths.
- Provide learning recommendations based on the predicted career.
- Provide an interactive Streamlit web interface.

## 🤖 Machine Learning Models

The project compares three classification algorithms:

1. Logistic Regression
2. Decision Tree
3. Random Forest

### Model Performance

| Model | Accuracy |
|---|---:|
| Logistic Regression | 88.89% |
| Decision Tree | 77.78% |
| Random Forest | 77.78% |

**Best Model:** Logistic Regression  
**Test Accuracy:** 88.89%

> Note: The current dataset is a small prototype dataset created for this project. The accuracy should not be interpreted as real-world hiring-market performance.

## 🧠 Skills Used for Prediction

- Python
- Machine Learning
- SQL
- Data Analysis
- Deep Learning
- NLP
- Cloud Computing
- Power BI
- Web Development
- Java

## ✨ Main Features

### 1. Career Prediction
The trained Machine Learning model predicts the most suitable career based on selected skills.

### 2. Career Prediction Probability
The application displays the probability associated with each predicted career.

### 3. Skill Demand Analysis
The system analyzes the project dataset and displays the relative demand of the available skills.

### 4. Career Recommendations
The application recommends additional career options based on the model's prediction probabilities.

### 5. Learning Recommendations
The system provides a learning path based on the predicted career.

## 💻 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit

## 📂 Project Structure

```text
AI_Powered_Skill_Prediction/
│
├── dataset/
│   └── career_prediction.csv
│
├── models/
│   └── career_model.pkl
│
├── app.py
├── train_model.py
└── README.md