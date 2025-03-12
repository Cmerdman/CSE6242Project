# ==========================
# IMPORT REQUIRED LIBRARIES
# ==========================
import heapq
import networkx as nx
import pandas as pd
import os

# ==========================
# DEFINE FILE PATHS
# ==========================
data_dir = "/content"

# ==========================
# FUNCTION: Read Speed Limits Data
# ==========================
def read_SpeedLimit_data():
    """Reads NYC Speed Limits and returns a DataFrame."""
    file_path = os.path.join(data_dir, "NYC_Speed_Limits.csv")

    if not os.path.exists(file_path):
        print("Error: NYC_Speed_Limits.csv not found!")
        return None

    df = pd.read_csv(file_path, dtype=str)  # Read as string to avoid dtype warnings

    # Ensure correct column names
    df.columns = df.columns.str.strip()  # Remove any accidental spaces
    if 'street' not in df.columns or 'postvz_sl' not in df.columns:
        print("Error: Required columns missing in NYC_Speed_Limits.csv")
        print("Available columns:", df.columns)  # Debugging output
        return None

    # Rename columns
    df = df.rename(columns={'street': 'RoadName', 'postvz_sl': 'SpeedLimit'})

    # Convert speed limit to numeric and fill missing values
    df['SpeedLimit'] = pd.to_numeric(df['SpeedLimit'], errors='coerce').fillna(25)

    # Drop duplicates
    df = df.drop_duplicates(subset=['RoadName'])

    return df

# ==========================
# FUNCTION: Read Traffic Data
# ==========================
def read_Traffic_data():
    """Reads Traffic Data and computes a Traffic Modifier."""
    file_path = os.path.join(data_dir, "Traffic_Volume_Counts_20250304.csv")

    if not os.path.exists(file_path):
        print("Error: Traffic data file not found!")
        return None

    df = pd.read_csv(file_path, dtype=str, low_memory=False)  # Handle mixed types

    # Ensure correct column names
    df.columns = df.columns.str.strip()  # Remove accidental spaces
    if 'Roadway Name' not in df.columns:
        print("Error: Column 'Roadway Name' missing in Traffic Data!")
        print("Available columns:", df.columns)  # Debugging output
        return None

    # Convert to uppercase
    df['Roadway Name'] = df['Roadway Name'].astype(str).str.upper()

    # Sum all hourly traffic columns to get total volume per street
    traffic_columns = df.columns[7:]  # Assuming first 7 columns are metadata
    df['Total_Traffic'] = df[traffic_columns].apply(pd.to_numeric, errors='coerce').sum(axis=1)

    # Group by street and compute Traffic Modifier
    traffic_counts = df.groupby('Roadway Name')['Total_Traffic'].sum().reset_index()
    avg_traffic = traffic_counts['Total_Traffic'].mean()
    traffic_counts['Traffic_Modifier'] = traffic_counts['Total_Traffic'] / avg_traffic

    return traffic_counts.rename(columns={'Roadway Name': 'RoadName'})

# ==========================
# FUNCTION: Read Accident Data (FIXED)
# ==========================
def read_Accident_data():
    """Reads Accident Data and computes an Accident Modifier."""
    file_path = os.path.join(data_dir, "Motor_Vehicle_Collisions_-_Crashes_20250304.csv")

    if not os.path.exists(file_path):
        print("Error: Accident data file not found!")
        return None

    df = pd.read_csv(file_path, dtype=str, low_memory=False)  # Handle mixed types

    # Auto-detect correct column for street name
    possible_names = ['ON STREET NAME', 'CROSS STREET NAME', 'OFF STREET NAME']
    road_name_col = next((col for col in df.columns if col.strip() in possible_names), None)

    if not road_name_col:
        print("Error: No valid road name column found in Accident Data!")
        print("Available columns:", df.columns)  # Debugging output
        return None

    # Convert to uppercase
    df[road_name_col] = df[road_name_col].astype(str).str.upper()

    # Remove rows with missing or blank street names
    df = df[df[road_name_col].str.strip() != ""]  # Remove empty street names

    # Count accidents per street
    accident_counts = df.groupby(road_name_col).size().reset_index(name='Accident_Count')

    # Compute accident modifier
    avg_accidents = accident_counts['Accident_Count'].mean()
    accident_counts['Accident_Modifier'] = accident_counts['Accident_Count'] / avg_accidents

    return accident_counts.rename(columns={road_name_col: 'RoadName'})

# ==========================
# MAIN FUNCTION
# ==========================
def main():
    # Load speed limits, traffic, and accident data
    Speed_Limit_df = read_SpeedLimit_data()
    Traffic_df = read_Traffic_data()
    Accidents_df = read_Accident_data()

    # Check if any dataset is missing
    if Speed_Limit_df is None or Traffic_df is None or Accidents_df is None:
        print("Error: One or more datasets could not be loaded!")
        return

    # Print data previews to confirm correctness
    print("\nSpeed Limits Data Preview:")
    print(Speed_Limit_df.head())

    print("\nTraffic Data Preview:")
    print(Traffic_df.head())

    print("\nAccident Data Preview:")
    print(Accidents_df.head())

    # Save processed data
    Speed_Limit_df.to_csv('/content/Processed_Speed_Limits.csv', index=False)
    Traffic_df.to_csv('/content/Processed_Traffic.csv', index=False)
    Accidents_df.to_csv('/content/Processed_Accidents.csv', index=False)

if __name__ == "__main__":
    main()
