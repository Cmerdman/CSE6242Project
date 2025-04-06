# ==========================
# IMPORT REQUIRED LIBRARIES
# ==========================
import heapq
import geopy.distance
import networkx as nx
import pandas as pd
import os
import geopy
from math import radians, sin, cos, sqrt, atan2, floor, ceil
import warnings


# ==========================
# DEFINE FILE PATHS
# ==========================
data_dir = "Data"
taxi_dir = "Raw"
output_dir = "Cleaned"

#Preprocessing of Data
#Data Files (Stored in Data folder):
#   Raw/yellow_tripdata_20XX_01.parquet: Files of the actual taxi trips from our dataset, all from the yellow zone, in January, for years 2011-2024


#Functions to read the various data files.
def read_Trip_data(year):
    """
    Reads a the trip data file for the current year and returns data as Pandas DataFrames.
    
    Args: 
        - Year: The Current Year selected by the user
    Returns:
        - Cleaned DataFrame of the trip data for the current year.
    """

    #Read File
    file_path = os.path.join(data_dir,taxi_dir,f'yellow_tripdata_{year}-01.parquet')

    if not os.path.exists(file_path):
        print(f"Error: Taxi Trip data file not found for year {year}!")
        return None

    df = pd.read_parquet(file_path)

    #Drop rows where the pickup or dropoff id is greater than 263 (as this is outside of our range)
    df = df[df['PULocationID'] <= 263]
    df = df[df['DOLocationID'] <= 263]
    #Drop rows where the pickup or dropp id is negative 
    df = df[df['PULocationID'] > 0]
    df = df[df['DOLocationID'] > 0]
    #Drop non-trips (where distance is 0 or negative, where fare amount is negative, and where passenger count is 0)
    df = df[df['trip_distance']>0]
    df = df[df['fare_amount']>0]
    df = df[df['passenger_count']>0]
    #Drop non-trips (where the payment type is 3= No charge, 4= Dispute, 5= Unknown, or 6 = Voided trip)
    df = df[df['payment_type']<3]
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
    df_rush_hour = pd.concat([df_rush_hour_1, df_rush_hour_2], ignore_index=True)
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

    #keep first 2 million points distributed using original distribution
    timezone_per_two_million = {"RushHour": 420700*2, "MidDay": 298440*2, "NightShift": 280859*2}
    df_zone_split = pd.read_csv("Debug/pickup_distribution.csv", dtype=str)
     
    #Distribute the data on original distributions 
    df = pd.DataFrame()
    for row in df_zone_split.iterrows():
        zone_id = int(row[1][0])
        zone_percent = float(row[1][2])
        #Get the number of rows to keep for this zone
        num_rows_night = floor(zone_percent* timezone_per_two_million["NightShift"])
        num_rows_mid = floor(zone_percent* timezone_per_two_million["MidDay"])
        num_rows_rush = floor(zone_percent* timezone_per_two_million["RushHour"])
        #Get the rows for this zone
        df_Nightzone = df_night_shift[df_night_shift['PULocationID'] == zone_id]
        df_Midzone = df_mid_day[df_mid_day['PULocationID'] == zone_id]
        df_Rushzone = df_rush_hour[df_rush_hour['PULocationID'] == zone_id]
        #Keep the first num_rows rows for this zone
        df_Nightzone = df_Nightzone.head(num_rows_night)
        df_Midzone = df_Midzone.head(num_rows_mid)
        df_Rushzone = df_Rushzone.head(num_rows_rush)
        #Add the rows for this zone to the final dataframe
        df = pd.concat([df, df_Nightzone, df_Midzone, df_Rushzone], ignore_index=True)

    #They made a typo naming the airport fee column for 2024 only
    if year == 2024:
        df = df.rename(columns={"Airport_fee":"airport_fee"})

    #Add the latitude and longitude of the pickup and dropoff locations
    #NOTE: This function works because the coordinate lookup data is stored in order of the LocationID.  However, because python zero indexes, we have to subtract 1 from the ids.
    
    df['startend'] = df['PULocationID'].astype(str) + df['DOLocationID'].astype(str)
    
    #ensure both are strings
    df['startend'] = df['startend'].astype(str)

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
    
    #Calculate trip time in minutes
    df['Trip_Time'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds()/60


    df = df.reset_index()

    #Normalize Data
    df = normalize_data(df)

    df = drop_outliers(df)

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
    #Remove rows where norm_trip_distance, norm_fare_amount, or norm_Trip_Time are NaN
    df_input = df_input.dropna(how = 'any', axis = 'rows', subset=['norm_trip_distance', 'norm_fare_amount', 'norm_Trip_Time'])
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

def drop_outliers(df):
  #Check for outliers
  df_filtered = df[df['norm_fare_amount'] < 3]
  df_filtered = df_filtered[df_filtered['norm_trip_distance'] < 3]
  df_filtered = df_filtered[df_filtered['norm_Trip_Time'] < 3]

  #Limit the cap for fare, time and distance for plotting
  df_filtered = df_filtered[df_filtered['fare_amount'] < 500] #500 dollar cap for fare
  df_filtered = df_filtered[df_filtered['trip_distance'] < 60] #60 mile cap for distance
  df_filtered = df_filtered[df_filtered['Trip_Time'] < 600] #600 minute cap for time

  return df_filtered


def main():
    warnings.filterwarnings("ignore")
    
    #Read and clean taxi trip data
    years = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    for year in years:
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
        print(f"Final length: {len(Trip_df)}")
        file = os.path.join(data_dir, output_dir, f'TripData_{year}.csv')
        Trip_df.to_csv(file, index=False)
    

if __name__ == "__main__":
    main()