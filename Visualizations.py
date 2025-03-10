import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the merged CSV files
df_final = pd.read_csv("merged_final_exam_data.csv")
df_midterm1 = pd.read_csv("merged_midterm1_data.csv")
df_midterm2 = pd.read_csv("merged_midterm2_data.csv")

# Concatenate all exam data into one DataFrame
df_all = pd.concat([df_final, df_midterm1, df_midterm2], ignore_index=True)
df_all["Exam"] = df_all["Exam"].astype("category")

# List of sensor columns
sensors = ["EDA", "BVP", "TEMP", "HR"]

# ---------------- Visualization 1 ----------------
# Time Series Plots for Final Exam: Average sensor values over time (per minute)
df_final_grouped = df_final.groupby("Minute")[sensors].mean().reset_index()
plt.figure(figsize=(12, 8))
for i, sensor in enumerate(sensors, 1):
    plt.subplot(2, 2, i)
    plt.plot(df_final_grouped["Minute"], df_final_grouped[sensor], marker="o", linestyle='-')
    plt.title(f"Final Exam - Average {sensor} over Time")
    plt.xlabel("Minute")
    plt.ylabel(sensor)
    plt.grid(True)
plt.tight_layout()
plt.savefig("viz_final_timeseries.png")
plt.show()

# ---------------- Visualization 2 ----------------
# Boxplots comparing sensor distributions across exam types (all exams)
plt.figure(figsize=(12, 8))
for i, sensor in enumerate(sensors, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x="Exam", y=sensor, data=df_all)
    plt.title(f"{sensor} Distribution by Exam")
    plt.xlabel("Exam")
    plt.ylabel(sensor)
plt.tight_layout()
plt.savefig("viz_sensor_boxplots.png")
plt.show()

# ---------------- Visualization 3 ----------------
# Scatter Plot: HR vs EDA colored by Exam
plt.figure(figsize=(10, 6))
sns.scatterplot(x="HR", y="EDA", hue="Exam", style="Exam", data=df_all, palette="Set1", alpha=0.7)
plt.title("Scatter Plot: HR vs EDA by Exam")
plt.xlabel("Heart Rate (HR)")
plt.ylabel("Electrodermal Activity (EDA)")
plt.grid(True)
plt.savefig("viz_hr_vs_eda.png")
plt.show()

# ---------------- Visualization 4 ----------------
# Correlation Heatmap among sensor readings (all exams)
df_sensors = df_all[sensors].dropna()
corr = df_sensors.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Heatmap Among Sensors")
plt.savefig("viz_correlation_heatmap.png")
plt.show()

# ---------------- Visualization 5 ----------------
# Pairplot (scatter matrix) for sensor readings for Final Exam only
sns.pairplot(df_final[sensors].dropna())
plt.suptitle("Pairplot of Sensor Readings for Final Exam", y=1.02)
plt.savefig("viz_pairplot_final.png")
plt.show()
