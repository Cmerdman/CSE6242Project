# ==========================
# IMPORT REQUIRED LIBRARIES
# ==========================
import heapq
import geopy.distance
import networkx as nx
import pandas as pd
import os
import geopy
#import googlemaps
#import requests
#import json
from math import radians, sin, cos, sqrt, atan2

# ==========================
# DEFINE FILE PATHS
# ==========================
data_dir = "Data"
taxi_dir = "Raw"
output_dir = "Cleaned"

#Coordinate lookup table for pickup and dropoff ids
coordinate_lookup = pd.read_csv("Debug/taxitripdict.csv")
distance_lookup = pd.read_csv("Debug/taxitripdictdist.csv")
distance_lookup['Estimated_Margin_of_Error'] = distance_lookup['Estimated_Margin_of_Error'].astype(float)
distance_lookup['Estimated_Dist'] = distance_lookup['Estimated_Dist'].astype(float)
distance_lookup['startend'] = distance_lookup['startend'].astype(str)

#Preprocessing of Data
#Data Files (Stored in Data folder):
#   newyork.graphml: Graph node representation of New York City streets.  From https://www.kaggle.com/datasets/crailtap/street-network-of-new-york-in-graphml
#   NYC_Speed_Limits.csv: Speed Limits for each street in New York City.  From https://data.cityofnewyork.us/Transportation/VZV_Speed-Limits/7n5j-865y   NOTE: Unless posted otherwise, default speed limit is 25mph in New York City
#   Traffic_Volume_Counts_20250304: File of Traffic Data.
#   Motor_Vehicle_Collisions_-_Crashes_20250304.csv: File of Accident Data.
#   Raw/yellow_tripdata_20XX_01.parquet: Files of the actual taxi trips from our dataset, all from the yellow zone, in January, for years 2009-2024


#Functions to read the various data files.
def read_SpeedLimit_data():
    """
    Reads the NYC_Speed_Limits file and returns the speed limit for each street as a Pandas DataFrame.

    Returns:
            - Speed_Limits_df: DataFrame containing speed limits for each street.
               - Columns: RoadName, SpeedLimit (in MPH)
    """

    """Reads NYC Speed Limits and returns a DataFrame."""
    file_path = os.path.join(data_dir, "NYC_Speed_Limits.csv")

    if not os.path.exists(file_path):
        print("Error: NYC_Speed_Limits.csv not found!")
        #return None

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
    Speed_Limits_df = df.drop_duplicates(subset=['RoadName'])

    
    '''
    #Read File
    Speed_Limits_df = pd.read_csv("Data/NYC_Speed_Limits.csv")
    #Remove unnecessary columns
    Speed_Limits_df = Speed_Limits_df.drop(['the_geom', 'postvz_sg', 'Shape_Leng'], axis=1)
    #Rename remaining columns for readability
    Speed_Limits_df = Speed_Limits_df.rename(columns={'street':'RoadName', 'postvz_sl': 'SpeedLimit'})
    #Set Default Speed limit to 25 where it is not defined (Where Speedlimit == 0 in data)
    Speed_Limits_df = Speed_Limits_df.replace(0,25)
    #Drop duplicates
    Speed_Limits_df = Speed_Limits_df.drop_duplicates(subset=['RoadName'])
    '''

    
    return Speed_Limits_df

def read_Accident_data():
    """
    Reads the Accident Data file and returns the Accident Modifier for each street as a Pandas DataFrame.

    Returns:
            - Accidents_df: DataFrame containing  for each street.
               - Columns: RoadName, Accident Modifier 

    NOTES:
            - Keep RoadName in ALL CAPS.  Convert road names to ALL CAPS if they are not already.  There should be no underscores
            - Accident Modifier: Defined as how far the number of accidents on any given street deviates from the mean for any one street.
                -I.e. 
                    Average_accidents = Total_Accidents/Number_of_Streets 
                        - (NOTE: Number of streets is the number of streets in the file, not the number of streets in New York.  WHY: We are only observing a small timeframe for accidents)
                    Accident_Modifier['Street'] = Accidents_on_street/Average_accidents
                - An Accident Modifier less than 1 indicates that the street has less accidents than average.
                - An Accident Modifier of 1 indicates that the street has the same ammount of accidents as any other street.  We will assume the modifier is 1 for any streets that aren't in the data file
                - An Accident Modifier greater than 1 indicates that the street has more accidents than average
    """
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

def read_Traffic_data():
    """
    Reads the Traffic Data file and returns the Traffic Modifier for each street as a Pandas DataFrame.
    NOTE: This is the same as calculating the Accident Modifier

    Returns:
            - Traffic_df: DataFrame containing  for each street.
               - Columns: RoadName, Traffic Modifier 

    NOTES:
            - Keep RoadName in ALL CAPS.  Convert road names to ALL CAPS if they are not already.  There should be no underscores
            - Traffic Modifier: Defined as how much the traffic on any given street deviates from the mean for any one street.
                -I.e. 
                    Average_Traffic = Total_Traffic/Number_of_Streets 
                        - (NOTE: Number of streets is the number of streets in the file, not the number of streets in New York.  WHY: We are only observing a small timeframe for accidents)
                    Traffic_Modifier['Street'] = Traffic_on_street/Average_Traffic
                - A Traffic Modifier less than 1 indicates that the street has less Traffic than average.
                - A Traffic Modifier of 1 indicates that the street has the same ammount of Traffic as any other street.  We will assume the modifier is 1 for any streets that aren't in the data file
                - A Traffic Modifier greater than 1 indicates that the street has more Traffic than average
    """
    """Reads Traffic Data and computes a Traffic Modifier."""
    file_path = os.path.join(data_dir,"Traffic_Volume_Counts_20250304.csv")

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

def read_Trip_data(year):
    """
    Reads a the trip data file for the current year and returns data as Pandas DataFrames.
    
    Args: 
        - Year: The Current Year selected by the user
    Returns:

    NOTES:
        - We already have a function to remove NAN values and out-of-bounds data called in main once the data is read.
        - Note that the function to remove NAN values will remove rows with ANY NA values (keep this in mind for which rows to drop.  i.e. Do we want to drop a row if the company name is NA here shouldn't cause the row to be omitted)

    """
    """Reads Traffic Data and computes a Traffic Modifier."""
    file_path = os.path.join(data_dir,taxi_dir,f'yellow_tripdata_{year}-01.parquet')

    if not os.path.exists(file_path):
        print(f"Error: Taxi Trip data file not found for year {year}!")
        return None

    df = pd.read_parquet(file_path)
    #Distribute data between the three time groups, keeping the original distribution:
    #NOTE: Distribution: Rush Hour = 42.0700%, Mid Day = 29.8441%, Night Shift= 28.0859% 
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    df['start_time'] = df['tpep_pickup_datetime'].dt.time
    df['end_time'] = df['tpep_dropoff_datetime'].dt.time
    #Assign Groups
    start = '7:00'
    end = '9:00'
    start =pd.to_datetime(start).time()
    end =pd.to_datetime(end).time()
    mask = (df['start_time'].between(start, end, inclusive='both')) | (df['end_time'].between(start, end, inclusive='both')) #Where the trip starts or ends in morning rush hour
    df_rush_hour_1 = df[mask]
    df_rush_hour_1['Time Zone'] = "Rush Hour"
    df = df[~mask]
    #Evening Rush Hour
    start = '15:00'
    end = '20:00'
    start =pd.to_datetime(start).time()
    end =pd.to_datetime(end).time()
    mask = (df['start_time'].between(start, end, inclusive='both')) | (df['end_time'].between(start, end, inclusive='both')) #Where the trip starts or ends in evening rush hour
    df_rush_hour_2 = df[mask]
    df_rush_hour_2['Time Zone'] = "Rush Hour"
    df = df[~mask]
    #Mid Day:
    start = '9:00'
    end = '15:00'
    start =pd.to_datetime(start).time()
    end =pd.to_datetime(end).time()
    mask = (df['start_time'].between(start, end, inclusive='both')) | (df['end_time'].between(start, end, inclusive='both')) #Where the trip starts or ends in evening rush hour
    df_mid_day = df[mask]
    df_mid_day["Time Zone"] = "Mid Day"
    df = df[~mask]


    #Night Shift (the remaining data):
    df_night_shift = df[~mask]
    df_night_shift["Time Zone"] = "Night Shift"


    #Keep only the first approx two million records distributed so this doesn't take forever
    df_rush_hour =  pd.concat([df_rush_hour_1, df_rush_hour_1], ignore_index=True)
    df_rush_hour = df_rush_hour.head(420700*2)
    df_mid_day = df_mid_day.head(298441*2)
    df_night_shift = df_night_shift.head(280859*2)
    df = pd.concat([df_rush_hour, df_mid_day, df_night_shift], ignore_index=True)
    #print(len(df))

    #NOTE: Data scheme changed after 2010, so we need to process 2010 and 2009's data differently
    #The following IF statement converts the data to the same scheme as the data after 2010
    #Commented our as we chose to omit 2009 and 2010
    '''
    if year<2011:
        #Rename the columns to match all years
        df = df.rename(columns={"Start_Lon": "pickup_longitude", "Start_Lat": "pickup_latitude", "End_Lon": "dropoff_longitude", "End_Lat": "dropoff_latitude"})
        df = df.rename(columns={"vendor_name": "VendorID", "vendor_id": "VendorID", "Trip_Pickup_DateTime": "tpep_pickup_datetime", "Trip_Dropoff_DateTime": "tpep_dropoff_datetime", "pickup_datetime": "tpep_pickup_datetime", "dropoff_datetime":"tpep_dropoff_datetime",
                                "Passenger_Count": "passenger_count", "Trip_Distance": "trip_distance", "Rate_Code": "RatecodeID", "rate_code": "RatecodeID", "store_and_forward": "store_and_fwd_flag",
                                "Payment_Type": "payment_type", "Fare_Amt": "fare_amount", "surcharge": "extra", "Tip_Amt": "tip_amount", "Tolls_Amt": "tolls_amount", "Total_Amt": "total_amount"})
        
        #Add new columns from years after 2010 to match the naming scheme, setting default values to 0
        df['improvement_surcharge'] = 0
        df['congestion_surcharge'] = 0
        df['airport_fee'] = 0

        #convert payment type using the dictionary 1= Credit card, 2= Cash, 3= No charge, 4 = Dispute, 5 = Unknown, 6= Voided trip
        map = {'credit card':1, 'cre': 1, 'cash':2, 'cas':2, 'no charge':3, 'dispute':4, 'unknown':5, 'voided trip':6}
        df['payment_type'] = df['payment_type'].str.lower()
        df['payment_type'] = df['payment_type'].map(lambda x: map.get(x,5))

        #convert vendor id using the dictionary 1= Creative Mobile Technologies, LLC; 2= VeriFone Inc.
        #Note that this will always be 3 for 2009 and 2010 as the vendor ids are not present in the data
        map = {'creative mobile technologies, llc':1, 'verifone inc':2}
        df['VendorID'] = df['VendorID'].str.lower()
        df['VendorID'] = df['VendorID'].map(lambda x: map.get(x,3))

        #Convert store_and_fwd_flag to Y/N
        df['store_and_fwd_flag'] = df['store_and_fwd_flag'].map({0:'N', 1:'Y'})
    '''
    #For 2011 onward, we must add the latitude and longitude of pickup and dropoff locations
    #else:
    #They made a typo naming the airport fee column for 2024 only
    if year == 2024:
        df = df.rename(columns={"Airport_fee":"airport_fee"})

    #Add the latitude and longitude of the pickup and dropoff locations
    #NOTE: This function works because the coordinate lookup data is stored in order of the LocationID.  However, because python zero indexes, we have to subtract 1 from the ids.
    #Drop rows where the pickup or dropoff id is greater than 263 (as this is outside of our range)
    df = df[df['PULocationID'] <= 263]
    df = df[df['DOLocationID'] <= 263]
    #print(len(df))
    df['startend'] = df['PULocationID'].astype(str) + df['DOLocationID'].astype(str)
    
    #ensure both are strings
    df['startend'] = df['startend'].astype(str)
    #add distance
    #NOTE: Removed due to inaccuracies in estimating distance due to use of zones
    #df = df.merge(distance_lookup, on='startend', how='inner')
    #print(len(df))
    '''
    df = df.merge(coordinate_lookup, left_on='PULocationID', right_on='zone_id', how='inner')
    df = df.rename(columns={"lat":"pickup_latitude", "long":"pickup_longitude"})
    df = df.merge(coordinate_lookup, left_on='DOLocationID', right_on='zone_id', how='inner')
    df = df.rename(columns={"lat":"dropoff_latitude", "long":"dropoff_longitude"})
    df = df.drop("zone_id_x", axis=1)
    df = df.drop("zone_id_y", axis=1)
    '''
    #print(df.head())
    #print(df.columns)
        

    #Change null values to 0 for RatecodeID, store_and_fwd_flag, mta_tax, extra, improvement_surcharge, congestion_surcharge, and airport_fee
    #NOTE: These default to null for any trip that does not have a one of these values, but these are not invalid trips
    #NOTE: This also defaults to null when store_and_fwd_flag is false in 2009 and 2010
    df['RatecodeID'] = df['RatecodeID'].fillna(0)
    df['store_and_fwd_flag'] = df['store_and_fwd_flag'].fillna('N')
    df['mta_tax'] = df['mta_tax'].fillna(0.0)
    df['extra'] = df['extra'].fillna(0.0)
    df['improvement_surcharge'] = df['improvement_surcharge'].fillna(0.0)
    df['congestion_surcharge'] = df['congestion_surcharge'].fillna(0.0)
    df['airport_fee'] = df['airport_fee'].fillna(0.0)
    
    #Convert pickup and dropoff times to datetime
    #df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    #df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    #Calculate the trip time in minutes
    df['Trip_Time'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds()/60
    #print(len(df))
    #NOTE: Originally changed to feet, but decided to keep miles
    #df['trip_distance'] = df['trip_distance']*5280

    df = df.reset_index()

    #Normalize Data
    df = normalize_data(df)

    return df

#Function to read the grpahml file to a dataframe
def read_graphml_to_dataframe():
    """
    Reads a GraphML file and returns node and edge data as Pandas DataFrames.

    Returns:
        tuple: A tuple containing two Pandas DataFrames:
            - nodes_df: DataFrame containing node attributes.
               - Columns: Lat, Long, RoadID
            - edges_df: DataFrame containing edge attributes.
                -Columns: SourceID, TargetID, EdgeID, RoadName, Length (in Yards), OneWay
    """
    #Obtain Nodes
    graph = nx.read_graphml('Data/newyork.graphml')
    nodes_df = pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient='index')
    #Removes the highway and ref columns of the dataframe, as they are always NAN and renames the others for ease of understanding
    nodes_df = nodes_df.drop(['highway', 'ref'], axis=1)
    #Rename remaining columns for readability
    nodes_df = nodes_df.rename(columns={'y': 'Lat', 'x': 'Long', 'osmid':'RoadID'})

    #Obtain Edges
    edge_data = graph.edges(data=True)
    # Create a list to hold edge data
    edge_list = []
    for u, v, data in edge_data:
        edge_dict = {'source': u, 'target': v}
        edge_dict.update(data)
        edge_list.append(edge_dict)

    # Create a DataFrame from edge data
    edges_df = pd.DataFrame(edge_list)
    
    #Remove unnecessary columns: 
    edges_df = edges_df.drop(['highway', 'geometry', 'bridge', 'tunnel', 'access', 'lanes', 'key', 'ref', 'access', 'width', 'service', 'maxspeed'], axis=1)
    #Rename remaining columns for readability
    edges_df = edges_df.rename(columns={'source':'SourceID', 'target':'TargetID', 'osmid':'EdgeID', 'name':'RoadName', 'length':'Length', 'oneway':'OneWay'})
    #Convert Roadnames to all caps to match the NYC_Speed_Limits.csv file
    edges_df['RoadName'] = edges_df['RoadName'].str.upper()
    #Convert numerical values to floats
    edges_df['Length'] = edges_df['Length'].astype(float)
    edges_df['TargetID'] = edges_df['TargetID'].astype(float)
    edges_df['SourceID'] = edges_df['SourceID'].astype(float)
    
    return nodes_df, edges_df

#Function that removes all points in the data outside of the latitude and longitude bounds of New York City 
def preprocess_remove_out_of_bound(df_input):
    longitude_bounds = [-75, -72]
    latitude_bounds = [40, 42]
    
    pickup_in_bound = ((df_input.pickup_longitude>longitude_bounds[0]) &
                       (df_input.pickup_longitude<longitude_bounds[1]) & 
                       (df_input.pickup_latitude>latitude_bounds[0]) &
                       (df_input.pickup_latitude<latitude_bounds[1]))
    dropoff_in_bound = ((df_input.dropoff_longitude>longitude_bounds[0]) &
                        (df_input.dropoff_longitude<longitude_bounds[1]) &
                        (df_input.dropoff_latitude>latitude_bounds[0]) &
                        (df_input.dropoff_latitude<latitude_bounds[1]))
    #df_in_pickup = df_input[pickup_in_bound]
    #df_in_dropoff = df_input[dropoff_in_bound]
    #df_out = df_in_pickup.join(df_in_dropoff)
    
    #return df_out
    return df_input[pickup_in_bound & dropoff_in_bound]

#Function to remove all rows and columns with NaN from the data.
def preprocess_remove_na(df_input):
    df_input = df_input.dropna(how = 'any', axis = 'rows')
    return df_input

#Function to append speed limit data to edges dataframe, used for Dijkstra's Algorithm
#Additionally adds time estimate (in minutes) for how long it takes to travel each edge (length/speed limit)
def preprocess_add_speed_limits_to_edges(edges_df, speed_limits_df):
    edges_df = pd.merge(edges_df, speed_limits_df, on='RoadName', how='left')
    #Sets any missed streets to default speed limit of 25 MPH
    edges_df = edges_df.fillna(25.0)
    #Calculates time estimate to travel each edge in minutes
    #NOTE: 1MPH = 88 Feet per minute
    #NOTE: Length is in yards, so we multiply by 3 to get feet
    edges_df['TravelTime'] = (edges_df['Length']*3)/(edges_df['SpeedLimit']*88) 
    
    return edges_df

#Function to append accident data to edges dataframe, used for Dijkstra's Algorithm
def preprocess_add_accidents_to_edges(edges_df, accidents_df):
    edges_df = pd.merge(edges_df, accidents_df, on='RoadName', how='left')
    #Sets any missed streets to default modifier of 1.0
    edges_df = edges_df.fillna(1.0)
    return edges_df

#Function to append traffic data to edges dataframe, used for Dijkstra's Algorithm
def preprocess_add_traffic_to_edges(edges_df, traffic_df):
    edges_df = pd.merge(edges_df, traffic_df, on='RoadName', how='left')
    #Sets any missed streets to default modifier of 1.0
    edges_df = edges_df.fillna(1.0)
    return edges_df

#Normalizes each of the following columns of the dataset: passenger_count, trip_distance, fare_amount, extra, tip_amount, tolls_amount,	total_amount, congestion_surcharge, airport_fee, Trip_Time
#Grouped by start and end zones (column startend)
#NOTE: This uses mean normalization.  We could try min max normalization.
def normalize_data(df):
    df_means = df.groupby('startend').agg({'passenger_count': 'mean', 'trip_distance': 'mean', 'fare_amount': 'mean', 'extra': 'mean', 'tip_amount': 'mean', 'tolls_amount': 'mean', 'total_amount':'mean', 'congestion_surcharge': 'mean', 'airport_fee': 'mean', 'Trip_Time': 'mean'})
    df_means = df_means.fillna(0)
    #print(df_means)
    df_stds = df.groupby('startend').agg({'passenger_count': 'std', 'trip_distance': 'std', 'fare_amount': 'std', 'extra': 'std', 'tip_amount': 'std', 'tolls_amount': 'std', 'total_amount':'std', 'congestion_surcharge': 'std', 'airport_fee': 'std', 'Trip_Time': 'std'})
    df_stds = df_stds.fillna(0)
    #print(df_stds)

    #NOTE: Column_x is means, Column_y is standard deviations
    df_grouped = df_means.merge(df_stds, on='startend', how="inner")
    #print(df_grouped)

    df = df.merge(df_grouped, on='startend', how="inner")
    #Calculate normalized variables:
    df['norm_passenger_count'] = (df['passenger_count']-df['passenger_count_x'])/df['passenger_count_y']
    df['norm_trip_distance'] = (df['trip_distance']-df['trip_distance_x'])/df['trip_distance_y']
    df['norm_fare_amount'] = (df['fare_amount']-df['fare_amount_x'])/df['fare_amount_y']
    df['norm_extra'] = (df['extra']-df['extra_x'])/df['extra_y']
    df['norm_tip_amount'] = (df['tip_amount']-df['tip_amount_x'])/df['tip_amount_y']
    df['norm_tolls_amount'] = (df['tolls_amount']-df['tolls_amount_x'])/df['tolls_amount_y']
    df['norm_total_amount'] = (df['total_amount']-df['total_amount_x'])/df['total_amount_y']
    df['norm_congestion_surcharge'] = (df['congestion_surcharge']-df['congestion_surcharge_x'])/df['congestion_surcharge_y']
    df['norm_airport_fee'] = (df['airport_fee']-df['airport_fee_x'])/df['airport_fee_y']
    df['norm_Trip_Time'] = (df['Trip_Time']-df['Trip_Time_x'])/df['Trip_Time_y']

    #print(df.keys())

    #Drop extra columns
    df = df.drop("passenger_count_x", axis=1)
    df = df.drop("passenger_count_y", axis=1)
    df = df.drop("trip_distance_x", axis=1)
    df = df.drop("trip_distance_y", axis=1)
    df = df.drop("fare_amount_x", axis=1)
    df = df.drop("fare_amount_y", axis=1)
    df = df.drop("extra_x", axis=1)
    df = df.drop("extra_y", axis=1)
    df = df.drop("tip_amount_x", axis=1)
    df = df.drop("tip_amount_y", axis=1)
    df = df.drop("tolls_amount_x", axis=1)
    df = df.drop("tolls_amount_y", axis=1)
    df = df.drop("total_amount_x", axis=1)
    df = df.drop("total_amount_y", axis=1)
    df = df.drop("congestion_surcharge_x", axis=1)
    df = df.drop("congestion_surcharge_y", axis=1)
    df = df.drop("airport_fee_x", axis=1)
    df = df.drop("airport_fee_y", axis=1)
    df = df.drop("Trip_Time_x", axis=1)
    df = df.drop("Trip_Time_y", axis=1)

    return df

def debug_process_taxizone_data():
    """
    Reads the Taxi Zone Data file and returns the Taxi Zone Data as a Pandas DataFrame."
    """
    #Read File
    Taxi_Zone_df = pd.read_csv("Debug/taxizonelookup_clean.csv", dtype=str)
    #drop na
    Taxi_Zone_df = Taxi_Zone_df.dropna()
    Taxi_Zone_df['lat'] = 0.0
    Taxi_Zone_df['long'] = 0.0
    #get the first lat long point for each zone
    for index, row in Taxi_Zone_df.iterrows():
        stringlatlong = str(row['text'])
        #print(row['zone_id'])
        if stringlatlong is not None:
            if stringlatlong[0] == "P":
                stringlatlong = stringlatlong[9:]
            else: 
                stringlatlong = stringlatlong[15:]
            
            stringlatlong = stringlatlong.replace('POINT (', '')
            stringlatlong = stringlatlong.replace(')', '')
            stringlatlong = stringlatlong.replace(',', '')
            stringlatlong = stringlatlong.split(' ')
            Taxi_Zone_df.loc[index, 'lat'] = float(stringlatlong[0])
            Taxi_Zone_df.loc[index, 'long'] = float(stringlatlong[1])

    #drop text column
    Taxi_Zone_df = Taxi_Zone_df.drop(['text'], axis=1)
    #print to new csv
    Taxi_Zone_df.to_csv('Debug/taxizonelookup.csv', index=False)

#Create a dictionary of all possible trips
def debug_process_taxizone_alltrips():
    data_df = pd.read_csv("Debug/taxizonelookup.csv", dtype=str)
    trip_df = pd.DataFrame(columns=["Start_ID", "Start_Lat", "Start_Lon", "End_ID", "End_Lat", "End_Lon"])
    #Create each possible route
    itter = 0
    for i in range(len(data_df)):
        for j in range(i, len(data_df)):
            trip_df.loc[itter] = [data_df.loc[i,"zone_id"], data_df.loc[i,"lat"], data_df.loc[i,"long"],data_df.loc[j,"zone_id"], data_df.loc[j,"lat"], data_df.loc[j,"long"]]
            itter = itter + 1
            
    #print to new csv
    #print(itter) #Should be 34716
    trip_df.to_csv("Debug/taxitripdict.csv", index=False)

#Get estimated distance for each trip.
def debug_dist_calulation():
    trip_df = pd.read_csv("Debug/taxitripdict.csv")
    trip_df["Estimated_Dist"] = 0.0
    trip_df["Estimated_Margin_of_Error"] = 0.0
    trip_df["startend"] = "PLACEHOLDER"
    for i in range(len(trip_df)):
        lat1=trip_df.loc[i,"Start_Lat"] 
        lon1=trip_df.loc[i,"Start_Lon"] 
        lat2=trip_df.loc[i,"End_Lat"]  
        lon2=trip_df.loc[i,"End_Lon"] 
        point1 = (lat1, lon1)
        point2 = (lat2, lon2)
        #calculate straight line distance between the lat long start and end.
        trip_df.loc[i,'Estimated_Dist'] = geopy.distance.geodesic(point1,point2).miles
        #Calculate a margin of error.  This is caclulated based on the change in 1 degree longitude as latitude changes.  For the latitude of new york, this is about 5 miles per degree latitude.
        #NOTE: Explination of Calculation: This calculation is only based on the change in latitude because the distance between one degree longitude changes at each latitude while the distance between one degree at a constant latitude is constant.
        #Distance between one degree latitude is always constant.
        #NOTE: This will always be positive.  It's impossible to have a negative margin of error.
        trip_df.loc[i,'Estimated_Margin_of_Error'] = abs(5*(abs(lat1)-abs(lat2)))
        #For joining on other table: Make a column of start and end id
        trip_df.loc[i,'startend'] = str(trip_df.loc[i,'Start_ID']) + str(trip_df.loc[i,'End_ID'])

    trip_df.to_csv("Debug/taxitripdictdist.csv", index=False)

#Gets the distrubtion of data between our three timzones.
def debug_data_distribution(year):
    file_path = os.path.join(data_dir,taxi_dir,f'yellow_tripdata_{year}-01.parquet')
    df = pd.read_parquet(file_path)
    df = df[['tpep_pickup_datetime','tpep_dropoff_datetime']]
    length = len(df)
    #start = '7:00'
    #start =pd.to_datetime(start).time()
    #print(start)
    
    #get hour of day for each trip
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    df['start_time'] = df['tpep_pickup_datetime'].dt.time
    df['end_time'] = df['tpep_dropoff_datetime'].dt.time
    #print(df.head(2))

    #Get the number of each time group:
    #Morning Rush Hour:
    #NOTE: We consider any trip that starts or ends in rush hour time to be part of rush hour, as it would be skewed by rush hour traffic.
    start = '7:00'
    end = '9:00'
    start =pd.to_datetime(start).time()
    end =pd.to_datetime(end).time()
    mask = (df['start_time'].between(start, end, inclusive='both')) | (df['end_time'].between(start, end, inclusive='both')) #Where the trip starts or ends in morning rush hour
    df_rush_hour_1 = df[mask]
    df = df[~mask]
    #Evening Rush Hour
    start = '15:00'
    end = '20:00'
    start =pd.to_datetime(start).time()
    end =pd.to_datetime(end).time()
    mask = (df['start_time'].between(start, end, inclusive='both')) | (df['end_time'].between(start, end, inclusive='both')) #Where the trip starts or ends in evening rush hour
    df_rush_hour_2 = df[mask]
    df = df[~mask]

    #Mid Day:
    start = '9:00'
    end = '15:00'
    start =pd.to_datetime(start).time()
    end =pd.to_datetime(end).time()
    mask = (df['start_time'].between(start, end, inclusive='both')) | (df['end_time'].between(start, end, inclusive='both')) #Where the trip starts or ends in evening rush hour
    df_mid_day = df[mask]
    df = df[~mask]


    #Night Shift (the remaining data):
    df_night_shift = df[~mask]


    #Get the distribution of each group
    #Lengths of each group:
    length_rush_hour = len(df_rush_hour_1) + len(df_rush_hour_2) #+ len(df_rush_hour_3) + len(df_rush_hour_4)
    length_mid_day = len(df_mid_day)
    length_night_shift = len(df_night_shift)

    print("Total Number of trips")
    print(f"Total {length}")
    print(f"Rush Hour: {length_rush_hour}")
    print(f"Mid Day: {length_mid_day}")
    print(f"Night Shift: {length_night_shift}")

    print("//////////////////////////////////")

    print("Percentage Distribution of Data")
    print(f"Rush Hour: {length_rush_hour/length}")
    print(f"Mid Day: {length_mid_day/length}")
    print(f"Night Shift: {length_night_shift/length}")

    return None
    
'''
THIS ALSO DOESN'T WORK
def debug_Manhattan_Distance():
    trip_df = pd.read_csv("Debug/taxitripdict.csv")
    trip_df['Dist'] = 0.0
    for i in range(len(trip_df)):
        lat1=trip_df.loc[i,"Start_Lat"] 
        lon1=trip_df.loc[i,"Start_Lon"] 
        lat2=trip_df.loc[i,"End_Lat"]  
        lon2=trip_df.loc[i,"End_Lon"] 
        dist_lat = abs(distance_between_latitudes_feet(lat1, lat2))
        dist_lon = abs(distance_between_longitudes_feet(lat1, lon1, lon2))
        trip_df.loc[i,'Dist'] = dist_lat+dist_lon

    trip_df.to_csv("Debug/taxitripdictdist.csv", index=False)

#distance between two latitudes
def distance_between_latitudes_feet(lat1, lat2):
    """
    Calculate the distance in feet between two latitudes.

    Args:
        lat1: Latitude of the first point in decimal degrees.
        lat2: Latitude of the second point in decimal degrees.

    Returns:
        The distance between the two latitudes in feet.
    """
    # Earth's radius in feet
    earth_radius_feet = 20902231 

    # Convert latitude differences to radians
    dlat = radians(lat2 - lat1)

    # Calculate distance using the arc length formula
    distance = earth_radius_feet * dlat
    
    return distance

def distance_between_longitudes_feet(lat, lon1, lon2):
    """
    Calculates the distance in feet between two longitudes at a given latitude.

    Args:
        lat_deg: Latitude in degrees.
        lon1_deg: Longitude 1 in degrees.
        lon2_deg: Longitude 2 in degrees.

    Returns:
        Distance in feet between the two longitudes.
    """
    # Earth's radius in feet
    earth_radius_feet = 20902231

    # Convert degrees to radians
    lat_rad = radians(lat)
    lon1_rad = radians(lon1)
    lon2_rad = radians(lon2)

    # Calculate the difference in longitude
    delta_lon = abs(lon1_rad - lon2_rad)

    # Calculate the distance
    distance = earth_radius_feet * cos(lat_rad) * delta_lon

    return distance
'''
'''
API DOESN'T WORK
#Use Google's Distance Matrix API to get the distance of all trips in our dictionary
def debug_api_Call_on_data():
    trip_df = pd.read_csv("Debug/taxitripdict.csv")
    dist_list = []
    time_list = []
    #iterate through each row in the dataframe
    for i in range(0,1):#len(trip_df)):
        origins= str(trip_df.loc[i,"Start_Lat"]) + ',' + str(trip_df.loc[i,"Start_Lon"])
        destinations = str(trip_df.loc[i,"End_Lat"])+','+str(trip_df.loc[i,"End_Lon"])
        print(origins, destinations)
        #Call the API and add the results each 25 trips (this is the most that can be handled in one call by the api)
        #NOTE: We use walking instead of driving to prevent U-turns when the lat/long isn't perfectly on a road.  The API follows roads either way.
        #matrix = gmaps.distance_matrix(origins, destinations, mode='walking')
        matrix = requests.get(url + 'origins=' + origins +
               '&destinations=' + destinations +
               '&mode=walking' +
               '&key=' + api_key)
        #append results to our list, convert distance from meters to feet and time from second to minutes
        print(matrix)
        matrix = matrix.json()
        print(matrix)
        for row in matrix['rows']:
            print(row)
            for element in row['elements']:
                dist_list.append((element['distance']['value'])*3.281)
                time_list.append((element['duration']['value'])/60)

    print(dist_list)
    print(time_list)

    return None
'''

def main():
    
    '''
    #Generate nodes and edges for graph of New York City streets
    nodes_df, edges_df = read_graphml_to_dataframe()
    #Get the Speed Limit of each street
    Speed_Limit_df = read_SpeedLimit_data()
    #Get Traffic Modifier of each street
    Traffic_df = read_Traffic_data()
    #Get Accident Modifier of each street
    Accidents_df = read_Accident_data()

    #Append speed limit, travel time, and modifiers to edges dataframe
    edges_df = preprocess_add_speed_limits_to_edges(edges_df, Speed_Limit_df)
    edges_df = preprocess_add_traffic_to_edges(edges_df, Traffic_df)
    edges_df = preprocess_add_accidents_to_edges(edges_df, Accidents_df)

    #Print Nodes and Edges to csv
    nodes_df.to_csv('Debug/Nodes.csv', index=False)
    edges_df.to_csv('Debug/Edges.csv', index=False)
    '''

    #Debuging functions
    year = 2024
    #debug_process_taxizone_data()
    #debug_process_taxizone_alltrips()
    #debug_dist_calulation()
    #debug_data_distribution(year)
    
    #Read and clean taxi trip data
    
    #years = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    #for year in years:
    #Debugging print
    print(f"Processing Data for year {year}")
    #Get Trip data for each year
    Trip_df = read_Trip_data(year)
    #Clean Trip Data
    Trip_df = preprocess_remove_na(Trip_df)
    #Remove trips that are out of bounds.
    #NOTE: All region ids for 2011 onward are automatically in bounds, so we do not have to further process.
    if year <2011:
        Trip_df = preprocess_remove_out_of_bound(Trip_df)
    #print data to Cleaned folder
    file = os.path.join(data_dir, output_dir, f'TripData_{year}.csv')
    Trip_df.to_csv(file, index=False)
    

if __name__ == "__main__":
    main()