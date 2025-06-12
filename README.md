# Personalized Metabolism Tracker

This project forecasts body weight based on daily nutrition, sleep, and activity patterns. It uses XGBoost for modeling, SQLite for data storage, and Streamlit for an interactive web interface.

## Pipeline Overview

### 1. Exploratory Data Analysis (EDA)
- Visualization of weight over time
- Calories vs weight trends
- Correlation matrix of all health metrics
- Sleep vs weight analysis
- Activity vs weight analysis

### 2. Modeling
- Features (X): 7-day rolling averages of calories, protein, carbs, fat, steps, activity minutes, and sleep hours
- Target (Y): 7-day rolling average of weight
- Model: XGBoost Regressor
- Performance: R² = 0.85, MSE = 0.05

### 3. SQLite Integration
- Data is stored in a local SQLite database (`metabolism_tracker.db`)
- The dashboard queries this database to fetch and filter user records dynamically

### 4. Streamlit Dashboard
- Users can visualize actual vs predicted weight
- Sliders allow for simulation of calorie, sleep, and step habit changes
- Forecasts next 7 days of predicted weight using the trained model
- Provides basic health recommendations based on user trends

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/ashwin-ternflat/metabolism-tracker.git
cd metabolism-tracker

pip install -r requirements.txt

streamlit run metabolism_tracker/app.py
