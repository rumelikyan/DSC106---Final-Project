import os
import glob
import pandas as pd
import numpy as np

# Define exam duration in minutes (3 hours = 180 minutes)
FINAL_EXAM_DURATION = 180

def read_sensor_data(file_path, column_name):
    """
    Reads and processes a physiological data file (EDA, HR, TEMP, BVP).

    Parameters:
    - file_path (str): Path to the CSV file.
    - column_name (str): Name of the sensor data column.

    Returns:
    - pd.DataFrame: DataFrame with sensor data and corresponding timestamps.
    """
    with open(file_path, "r") as file:
        session_start = float(file.readline().strip())  # Unix timestamp
        sample_rate = float(file.readline().strip())      # Sample rate in Hz
    # Read actual data, skipping the first two rows
    df = pd.read_csv(file_path, skiprows=2, header=None, names=[column_name])
    # Compute timestamps based on sample rate
    timestamps = np.arange(len(df)) / sample_rate
    df["Timestamp"] = session_start + timestamps
    return df

def aggregate_to_minutes(df, exam_duration):
    """
    Aggregates sensor data to 1 measurement per minute using the average.
    Only data within the exam duration (in minutes) is retained.

    Parameters:
    - df (pd.DataFrame): DataFrame with sensor data and a "Timestamp" column.
    - exam_duration (int): Duration of the exam in minutes.

    Returns:
    - pd.DataFrame: DataFrame aggregated by minute with exactly exam_duration rows.
    """
    df["Minute"] = (df["Timestamp"] - df["Timestamp"].min()) // 60
    df = df[df["Minute"] < exam_duration]
    df_agg = df.groupby("Minute").mean().reset_index()
    # Ensure there is one row for every minute 0 to exam_duration-1
    all_minutes = pd.DataFrame({"Minute": np.arange(exam_duration)})
    df_agg = all_minutes.merge(df_agg, on="Minute", how="left")
    return df_agg

# Base directory containing student folders (S1, S2, ..., S10)
BASE_DIR = r"C:\Users\ruben\Documents\GitHub\DSC106---Final-Project\Data"
student_dirs = sorted(glob.glob(os.path.join(BASE_DIR, "S*")))

# Sensors to process
SENSORS = {
    "EDA": "EDA.csv",
    "HR": "HR.csv",
    "TEMP": "TEMP.csv",
    "BVP": "BVP.csv",
}

# List to hold merged final exam data for all students
final_exam_data_list = []

for student_dir in student_dirs:
    student_id = os.path.basename(student_dir)
    final_dir = os.path.join(student_dir, "Final")
    if not os.path.isdir(final_dir):
        print(f"Final exam folder not found for {student_id}. Skipping...")
        continue

    sensor_dfs = {}
    for sensor_name, filename in SENSORS.items():
        file_path = os.path.join(final_dir, filename)
        if os.path.exists(file_path):
            try:
                df_sensor = read_sensor_data(file_path, sensor_name)
                df_sensor_agg = aggregate_to_minutes(df_sensor, FINAL_EXAM_DURATION)
                sensor_dfs[sensor_name] = df_sensor_agg
            except Exception as e:
                print(f"Error processing {sensor_name} for {student_id}: {e}")
        else:
            print(f"{filename} not found for {student_id} in Final exam.")
    # Merge sensor data on 'Minute' for this student
    if "EDA" in sensor_dfs:
        df_student = sensor_dfs["EDA"]
        for sensor_name, df in sensor_dfs.items():
            if sensor_name != "EDA":
                df_student = df_student.merge(df, on="Minute", how="outer", suffixes=("", f"_{sensor_name}"))
        # Add metadata
        df_student["Student"] = student_id
        df_student["Exam"] = "Final"
        final_exam_data_list.append(df_student)
    else:
        print(f"No EDA data for {student_id}; skipping merging for this student.")

# Concatenate all students' final exam data
if final_exam_data_list:
    df_final_exam = pd.concat(final_exam_data_list, ignore_index=True)
    # Reorder columns: Student, Exam, Minute, then sensor columns (prioritize original sensor names)
    desired_cols = ["Student", "Exam", "Minute", "EDA", "BVP", "TEMP", "HR"]
    existing_cols = [col for col in desired_cols if col in df_final_exam.columns]
    df_final_exam = df_final_exam[existing_cols]
    # Save to CSV
    df_final_exam.to_csv("merged_final_exam_data.csv", index=False)
    print("Merged final exam data for all students has been saved to merged_final_exam_data.csv.")
else:
    print("No final exam data was merged.")
