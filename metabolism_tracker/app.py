import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import datetime
import xgboost
import joblib

reg = joblib.load("notebooks/xgb_weight_model.pkl") # load pretrained model

conn = sqlite3.connect("metabolism_tracker.db") # load data from sqlite
query = "SELECT * FROM metabolism_data"
df = pd.read_sql(query, conn)
conn.close()

features = ['calories_roll_7', 'protein_roll_7', 'carbs_roll_7', 'fat_roll_7', 'activity_minutes_roll_7', 'steps_roll_7', 'sleep_hours_roll_7']

df["date"] = pd.to_datetime(df["date"])


min_date = df["date"].min().to_pydatetime()
max_date = df["date"].max().to_pydatetime()

# Streamlit app config
st.set_page_config(page_title="Metabolism Tracker", layout="wide")

st.title("Personalized Metabolism Tracker")
st.markdown("Monitor your weight, nutrition, and activity trends.")


date_range = st.slider(
    "Select Date Range:",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY-MM-DD"
)

# Filter DataFrame by selected date range

filtered_df = df[(df["date"] >= date_range[0]) & (df["date"] <= date_range[1])]


st.subheader("Weight vs Predicted Weight")
fig, ax = plt.subplots()
ax.plot(filtered_df["date"], filtered_df["weight_roll_7"], label="Actual Weight (7d Avg)")
ax.plot(filtered_df["date"], filtered_df["predicted_weight_roll_7"], label="Predicted Weight")
ax.set_xlabel("Date")
ax.set_ylabel("Weight (kg)")
ax.legend()
st.pyplot(fig)


st.subheader("Nutrition Overview (7-day Rolling Averages)")
st.write(
    filtered_df[["calories_roll_7", "protein_roll_7", "carbs_roll_7", "fat_roll_7"]].describe()
)



st.subheader(" Activity & Sleep Trends")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Steps (7d Avg)**")
    st.line_chart(filtered_df.set_index("date")[["steps_roll_7"]])

with col2:
    st.markdown("**Sleep Hours (7d Avg)**")
    st.line_chart(filtered_df.set_index("date")[["sleep_hours_roll_7"]])

 # Forecast Weight

st.subheader("Forecasted Weight (Next 7 Days)")

st.markdown("Simulate future trends by adjusting your habits")
cal_adjust = st.slider("Daily Calorie Change (forecast)", -500, 500, 0)
step_adjust = st.slider("Daily Step Change (forecast)", 0, 2000, 100)
sleep_adjust = st.slider("Daily Sleep Change (forecast)", -2.0, 2.0, 0.0, step=0.25)

# Initiate from the last known rolling value

latest_input = filtered_df[features].iloc[-1:].copy()
forecast_weights = []

# 7 days

for i in range(7):
    pred = reg.predict(latest_input)[0]
    forecast_weights.append(pred)

    # Apply changes directly to rolling values 
    latest_input["calories_roll_7"] += cal_adjust / 7
    latest_input["steps_roll_7"] += step_adjust / 7
    latest_input["sleep_hours_roll_7"] += sleep_adjust / 7



future_dates = pd.date_range(start=filtered_df["date"].max() + pd.Timedelta(days=1), periods=7)
forecast_df = pd.DataFrame({
    "date": future_dates,
    "predicted_weight": forecast_weights
})

st.line_chart(forecast_df.set_index("date"))

with st.expander(" Show Forecast Table"):
    st.dataframe(forecast_df)


# Recommendations section
st.subheader(" Personalized Recommendations")

latest = filtered_df.iloc[-1]

recs = []



if latest["calories_roll_7"] < 1800:
    recs.append(" You might need to increase your calorie intake for maintenance.")
elif latest["calories_roll_7"] > 2500:
    recs.append("Your average calorie intake is high—consider moderating it if weight loss is your goal.")


if latest["sleep_hours_roll_7"] < 6:
    recs.append("Try to get at least 7–8 hours of sleep for better metabolism.")


if latest["steps_roll_7"] < 5000:
    recs.append(" Increase daily movement—consider walking more to improve fat burn.")


if latest["protein_roll_7"] < 50:
    recs.append(" You may want to increase your protein intake to support muscle maintenance.")


if recs:
    for r in recs:
        st.markdown(f"- {r}")
else:
    st.markdown("Everything looks on track! Keep it up.")

