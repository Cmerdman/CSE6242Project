# ==========================
# IMPORT REQUIRED LIBRARIES
# ==========================
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import pickle
import matplotlib.pyplot as plt #Used for Debug plots
import os
import numpy as np
import seaborn as sns
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import warnings
warnings.filterwarnings('ignore')
    


#Define directories
data_dir = "Data"
cleaned_dir = "Cleaned"
output_dir = "Analyzed"


#Performs DBSCAN and assigns clusters for trip distance and total fare.
def run_DBSCAN(year, epsilon):
    infile = os.path.join(f'TripData_{year}.csv')
    df = pd.read_csv(infile)
    df["cluster"] = 0
    #Switch time zones to numbers
    df = df.replace("Rush Hour", 1)
    df = df.replace("Mid Day", 2)
    df = df.replace("Night Shift", 3)

    #convert store and fwd flag to binary
    df['store_and_fwd_flag'] = df['store_and_fwd_flag'].replace('N', 0)
    df['store_and_fwd_flag'] = df['store_and_fwd_flag'].replace('Y', 1)

    df = df.select_dtypes(include=np.number)

   # Drop rows where infinite
    df = df[~np.isinf(df['norm_trip_distance'])]
    df = df[~np.isinf(df['norm_fare_amount'])]
    df = df[~np.isinf(df['norm_Trip_Time'])]

    for time in range(1,4):
        if time == 1:
            data = df[df['Time Zone'] == 1]
            timezone = "Rush Hour"
        if time == 2:
            data = df[df['Time Zone'] == 2]
            timezone = "Mid Day"
        if time == 3:
            data = df[df['Time Zone'] == 3]
            timezone = "Night Shift"
        for i in data['startend'].unique():
            zonedata = data[data['startend'] == i]
            if(len(zonedata)>0):
              data1 = zonedata[["norm_trip_distance","norm_fare_amount","norm_Trip_Time"]]
              samples = np.ceil(len(zonedata)/3)
              samples = int(samples)
              dbscan = DBSCAN(eps=epsilon, min_samples=samples)
              clusters1 = dbscan.fit_predict(zonedata[["norm_trip_distance","norm_fare_amount","norm_Trip_Time"]])
              data1_copy = zonedata.copy()
              data1_copy["cluster"] = clusters1
              data1_copy = data1_copy.drop(labels=["norm_trip_distance", "norm_fare_amount","norm_Trip_Time"], axis=1)
              df["cluster"].update(data1_copy["cluster"])

    #convert time zones back to words only in time zone column
    df['Time Zone'] = df['Time Zone'].replace(1, "Rush Hour")
    df['Time Zone'] = df['Time Zone'].replace(2, "Mid Day")
    df['Time Zone'] = df['Time Zone'].replace(3, "Night Shift")


    #Save data to analyzed folder
    outfile = os.path.join(data_dir, output_dir, f'TripData_{year}.csv')
    df.to_csv(outfile, index=False)
 

#Statitstics for each year
def run_stats(year):
  infile = os.path.join( f'TripData_{year}.csv')
  df = pd.read_csv(infile)
  df_abnormal = df[df['cluster']==-1]
  df_normal = df[df['cluster']>-1]

  print(f"Year {year}:")
  print(f"Percentage of abnormal trips: {100*(len(df_abnormal)/len(df))}%")
  print(f"Total abnormal trips: {len(df_abnormal)}")
  #Get most common abnormal trip route
  df_abnormal2 = df_abnormal.groupby(['PULocationID', 'DOLocationID']).size().reset_index(name='count')
  df_abnormal2 = df_abnormal2.sort_values(by=['count'], ascending=False)
  print(f"Most common abnormal trip route: {df_abnormal2.iloc[0]['PULocationID']} to {df_abnormal2.iloc[0]['DOLocationID']} with {df_abnormal2.iloc[0]['count']} trips")
  #get most common abnormal trip route by time zone
  #print(df_abnormal['Time Zone'])
  df_ab_rush = df_abnormal[df_abnormal['Time Zone'] == "Rush Hour"]
  df_ab_mid = df_abnormal[df_abnormal['Time Zone'] == "Mid Day"]
  df_ab_night = df_abnormal[df_abnormal['Time Zone'] == "Night Shift"]
  df_ab_rush = df_ab_rush.groupby(['PULocationID', 'DOLocationID']).size().reset_index(name='count')
  df_ab_rush = df_ab_rush.sort_values(by=['count'], ascending=False)
  df_ab_mid = df_ab_mid.groupby(['PULocationID', 'DOLocationID']).size().reset_index(name='count')
  df_ab_mid = df_ab_mid.sort_values(by=['count'], ascending=False)
  df_ab_night = df_ab_night.groupby(['PULocationID', 'DOLocationID']).size().reset_index(name='count')
  df_ab_night = df_ab_night.sort_values(by=['count'], ascending=False)
  print(f"Most common abnormal trip route in Rush Hour: {df_ab_rush.iloc[0]['PULocationID']} to {df_ab_rush.iloc[0]['DOLocationID']} with {df_ab_rush.iloc[0]['count']} trips")
  print(f"Most common abnormal trip route in Mid Day: {df_ab_mid.iloc[0]['PULocationID']} to {df_ab_mid.iloc[0]['DOLocationID']} with {df_ab_mid.iloc[0]['count']} trips")
  print(f"Most common abnormal trip route in Night Shift: {df_ab_night.iloc[0]['PULocationID']} to {df_ab_night.iloc[0]['DOLocationID']} with {df_ab_night.iloc[0]['count']} trips")
  #Average abnormal trip distance
  print(f"Average abnormal trip distance: {df_abnormal['trip_distance'].mean()} miles")
  #Average abnormal trip time
  print(f"Average abnormal trip time: {df_abnormal['Trip_Time'].mean()} minutes")
  #Average fare per mile for abnormal trips
  df_abnormal['fare_per_mile'] = df_abnormal['fare_amount']/df_abnormal['trip_distance']
  print(f"Average fare per mile for abnormal trips: ${df_abnormal['fare_per_mile'].mean():.2f}")
  print("\n")


  #Normal Trips
  print(f"Percentage of normal trips: {100*(len(df_normal)/len(df))}%")
  print(f"Total normal trips: {len(df_normal)}")
  df_normal2 = df_normal.groupby(['PULocationID', 'DOLocationID']).size().reset_index(name='count')
  df_normal2 = df_normal2.sort_values(by=['count'], ascending=False)
  print(f"Most common normal trip route: {df_normal2.iloc[0]['PULocationID']} to {df_normal2.iloc[0]['DOLocationID']} with {df_normal2.iloc[0]['count']} trips")
  df_normal_rush = df_normal[df_normal['Time Zone'] == "Rush Hour"]
  df_normal_mid = df_normal[df_normal['Time Zone'] == "Mid Day"]
  df_normal_night = df_normal[df_normal['Time Zone'] == "Night Shift"]
  df_normal_rush = df_normal_rush.groupby(['PULocationID', 'DOLocationID']).size().reset_index(name='count')
  df_normal_rush = df_normal_rush.sort_values(by=['count'], ascending=False)
  df_normal_mid = df_normal_mid.groupby(['PULocationID', 'DOLocationID']).size().reset_index(name='count')
  df_normal_mid = df_normal_mid.sort_values(by=['count'], ascending=False)
  df_normal_night = df_normal_night.groupby(['PULocationID', 'DOLocationID']).size().reset_index(name='count')
  df_normal_night = df_normal_night.sort_values(by=['count'], ascending=False)
  print(f"Most common normal trip route in Rush Hour: {df_normal_rush.iloc[0]['PULocationID']} to {df_normal_rush.iloc[0]['DOLocationID']} with {df_normal_rush.iloc[0]['count']} trips")
  print(f"Most common normal trip route in Mid Day: {df_normal_mid.iloc[0]['PULocationID']} to {df_normal_mid.iloc[0]['DOLocationID']} with {df_normal_mid.iloc[0]['count']} trips")
  print(f"Most common normal trip route in Night Shift: {df_normal_night.iloc[0]['PULocationID']} to {df_normal_night.iloc[0]['DOLocationID']} with {df_normal_night.iloc[0]['count']} trips")
  #average normal trip distance
  print(f"Average normal trip distance: {df_normal['trip_distance'].mean()} miles")
  #average normal trip time
  print(f"Average normal trip time: {df_normal['Trip_Time'].mean()} minutes")
  #Average fer per mile for normal trips
  df_normal['fare_per_mile'] = df_normal['fare_amount']/df_normal['trip_distance']
  print(f"Average fare per mile for normal trips: ${df_normal['fare_per_mile'].mean():.2f}")

  print("\n")

  #Percentage of abnormal trips by vendor
  #Abnormal trips by vendorID
  df_abnormal_vendor = df_abnormal.groupby(['VendorID']).size().reset_index(name='count')
  df_abnormal_vendor_dict = df_abnormal_vendor.to_dict()
  #total trips by vendorID
  df_vendor = df.groupby(['VendorID']).size().reset_index(name='count')
  df_vendor_dict = df_vendor.to_dict()
  print(f"Percentage of abnormal trips for Vendor 1: {100*(df_abnormal_vendor_dict['count'][0]/df_vendor_dict['count'][0])}%")
  print(f"Percentage of abnormal trips for Vendor 2: {100*(df_abnormal_vendor_dict['count'][1]/df_vendor_dict['count'][1])}%")

  #Number of trips where pickup zone is dropoff zone
  df_pickup_dropoff = df[df['PULocationID'] == df['DOLocationID']]
  print(f"Percentage of trips where pickup zone is dropoff zone: {100*(len(df_pickup_dropoff)/len(df))}%")

  #Explain Abnormal data causes (excessive fare, excessive time, excessively slow, etc)
  #Get mean and standard deviation of normal fare and speed
  mean_fare = df_normal['fare_amount'].mean()
  std_fare = df_normal['fare_amount'].std()
  #speed is distance divided by time
  #drop trip time of 0
  df_normal = df_normal[df_normal['Trip_Time'] > 0]
  df_normal['speed'] = df_normal['trip_distance']/df_normal['Trip_Time']
  df_abnormal['speed'] = df_abnormal['trip_distance']/df_abnormal['Trip_Time']
  mean_speed = df_normal['speed'].mean()
  std_speed = df_normal['speed'].std()

  #lump all fees together
  df_normal['fees'] = df_normal['extra'] + df_normal['tolls_amount'] + df_normal['congestion_surcharge'] + df_normal['airport_fee']
  df_abnormal['fees'] = df_abnormal['extra'] + df_abnormal['tolls_amount'] + df_abnormal['congestion_surcharge'] + df_abnormal['airport_fee']
  mean_fees = df_normal['fees'].mean()
  std_fees = df_normal['fees'].std()

  #tips
  df_normal['tip_amount'] = df_normal['tip_amount'].fillna(0)
  df_abnormal['tip_amount'] = df_abnormal['tip_amount'].fillna(0)
  mean_tip = df_normal['tip_amount'].mean()
  std_tip = df_normal['tip_amount'].std()

  #Excessive time: Whenever the trip takes more than 2 hours, regardless of distance (the lengthiest trip in our dataset during rush hour takes about 90 minutes. This threshold represents where the end_time was not correctly recorded)
  df_etime = df_abnormal[df_abnormal['Trip_Time'] > 120]
  #remove these rows from the abnormal dataset so they aren't calculated twice
  df_abnormal = df_abnormal[df_abnormal['Trip_Time'] <= 120]
  #Excessive fare: Whenever the fare is more than 2 standard deviations away from the normal mean
  df_efare = df_abnormal[df_abnormal['fare_amount'] > mean_fare + 2*std_fare]
  #remove these rows from the abnormal dataset so they aren't calculated twice
  df_abnormal = df_abnormal[df_abnormal['fare_amount'] <= mean_fare + 2*std_fare]
  #Abnormally small fare:
  df_asfare = df_abnormal[df_abnormal['fare_amount'] < mean_fare - 2*std_fare]
  #remove these rows from the abnormal dataset so they aren't calculated twice
  df_abnormal = df_abnormal[df_abnormal['fare_amount'] >= mean_fare - 2*std_fare]
  #Excessively slow:
  df_eslow = df_abnormal[df_abnormal['speed'] < mean_speed - 2*std_speed]
  #remove these rows from the abnormal dataset so they aren't calculated twice
  df_abnormal = df_abnormal[df_abnormal['speed'] >= mean_speed - 2*std_speed]
  #Excessively fast:
  df_efast = df_abnormal[df_abnormal['speed'] > mean_speed + 2*std_speed]
  #remove these rows from the abnormal dataset so they aren't calculated twice
  df_abnormal = df_abnormal[df_abnormal['speed'] <= mean_speed + 2*std_speed]
  #extra fees:
  df_efees = df_abnormal[df_abnormal['fees'] > mean_fees + 2*std_fees]
  #remove rows
  df_abnormal = df_abnormal[df_abnormal['fees'] <= mean_fees + 2*std_fees]
  #excessive tips
  df_etips = df_abnormal[df_abnormal['tip_amount'] > mean_tip + 2*std_tip]
  #remove rows
  df_abnormal = df_abnormal[df_abnormal['tip_amount'] <= mean_tip + 2*std_tip]


  causes_dict = {}
  causes_dict['Running Timer'] = len(df_etime)
  causes_dict['High Fare'] = len(df_efare)
  causes_dict['Low Fare'] = len(df_asfare)
  causes_dict['Slow'] = len(df_eslow)
  causes_dict['Fast'] = len(df_efast)
  causes_dict['High Fees'] = len(df_efees)
  causes_dict['High Tips'] = len(df_etips)
  #causes_dict['Unexplained'] = len(df_abnormal)
  print("Causes for abnormality")
  for key, value in causes_dict.items():
    print(f"{key}: {value}")

  print("\n")
  #plot causes_dict
  plt.figure(figsize=(8, 6))
  plt.bar(causes_dict.keys(), causes_dict.values())
  plt.title(f'Causes for abnormality for {year}')
  plt.xlabel('Cause')
  plt.ylabel('Count')
  plt.show()


#Statistics plots 
def Stats_Plots(year):
  infile = os.path.join( f'TripData_{year}.csv')
  df = pd.read_csv(infile)
  df["dist_zone_cluster"] = 0
  df['time_zone_cluster'] = 0

  for time in range(1,2):
        if time == 1:
            data = df[df['Time Zone'] == "Rush Hour"]
            timezone = "Rush Hour"
            data = data[['startend', 'PULocationID', 'DOLocationID']].reset_index()
        elif time == 2:
            data = df[df['Time Zone'] == "Mid Day"]
            timezone = "Mid Day"
            data = data[['startend', 'PULocationID', 'DOLocationID']].reset_index()
        elif time == 3:
            data = df[df['Time Zone'] == "Night Shift"]
            timezone = "Night Shift"
            data = data[['startend', 'PULocationID', 'DOLocationID']].reset_index()
        data1 = data.groupby(['startend']).count().reset_index()
        data2 = data.groupby(['PULocationID', 'DOLocationID']).size().reset_index(name='count')

        #3D plot of trips by pu and do
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data2['PULocationID'], data2['DOLocationID'], data2['count'], color='blue')
        ax.set_xlabel('Pickup Location ID')
        ax.set_ylabel('Dropoff Location ID')
        ax.set_zlabel('Num Trips')
        plt.title(f'Trips by Pickup and Dropoff Location for {timezone} for {year}')
        plt.show()

        #3D plot of normal and abnormal trips
        df_abnormal = df[df['cluster']==-1]
        df_normal = df[df['cluster']>-1]
        data_normal = df_normal.groupby(['PULocationID', 'DOLocationID']).size().reset_index(name='count')
        data_abnormal = df_abnormal.groupby(['PULocationID', 'DOLocationID']).size().reset_index(name='count')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data_normal['PULocationID'], data_normal['DOLocationID'], data_normal['count'], color='blue')
        ax.scatter(data_abnormal['PULocationID'], data_abnormal['DOLocationID'], data_abnormal['count'], color='red')
        ax.set_xlabel('Pickup Location ID')
        ax.set_ylabel('Dropoff Location ID')
        ax.set_zlabel('Num Trips')
        plt.title(f'Number of normal and abnormal trips by trip start and end locations for {timezone} for {year}')
        plt.show()

        #Bar Chart of zones with the most abnormalities
        df_abnormal2 = df_abnormal.groupby(['PULocationID']).size().reset_index(name='abnormal_count')
        df_normal2 = df_normal.groupby(['PULocationID']).size().reset_index(name='normal_count')
        #merge the dataframes
        df_merged = pd.merge(df_abnormal2, df_normal2, on='PULocationID', how='outer')
        df_merged = df_merged.fillna(0)
        #Get percentage of abnormal trips for each PULocationID as total number of abnormal trips for ID divided by all trips for ID
        df_merged['percent_abnormal'] = df_merged['abnormal_count']/(df_merged['abnormal_count']+df_merged['normal_count'])
        #plot the data
        plt.figure(figsize=(8, 6))
        plt.bar(df_merged['PULocationID'], df_merged['percent_abnormal'])
        plt.title(f'Percentage of abnormal trips by pickup location for {timezone} for {year}')
        plt.xlabel('Pickup Location ID')
        plt.ylabel('Percentage of abnormal trips')
        plt.show()

        #3d bar chart of trips with the most abnormalities
        df_abnormal2 = df_abnormal.groupby(['PULocationID', 'DOLocationID']).size().reset_index(name='abnormal_count')
        df_normal2 = df_normal.groupby(['PULocationID', 'DOLocationID']).size().reset_index(name='normal_count')
        #merge the dataframes
        df_merged = pd.merge(df_abnormal2, df_normal2, on=['PULocationID', 'DOLocationID'], how='outer')
        df_merged = df_merged.fillna(0)
        #Get percentage of abnormal trips for each PULocationID as total number of abnormal trips for ID divided by all trips for ID
        df_merged['percent_abnormal'] = df_merged['abnormal_count']/(df_merged['abnormal_count']+df_merged['normal_count'])
        #sort the data by percent_abnormal
        df_merged = df_merged.sort_values(by=['percent_abnormal'], ascending=False)
        #plot the data
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.bar3d(df_merged['PULocationID'], df_merged['DOLocationID'], df_merged['percent_abnormal'], 0.5,0.5,0.5, color='red',shade=True)
        ax.set_xlabel('Pickup Location ID')
        ax.set_ylabel('Dropoff Location ID')
        ax.set_zlabel('Percentage of abnormal trips')
        plt.title(f'Percentage of abnormal trips by pickup and dropoff location for {timezone} for {year}')
        plt.show()

        #3d graph of fare vs time vs distance
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df_abnormal['trip_distance'], df_abnormal['Trip_Time'], df_abnormal['fare_amount'], color='red')
        ax.scatter(df_normal['trip_distance'], df_normal['Trip_Time'], df_normal['fare_amount'], color='blue')
        ax.set_xlabel('Trip Distance')
        ax.set_ylabel('Trip Time')
        ax.set_zlabel('Fare')
        plt.title(f'Fare vs Time vs Distance for {timezone} for {year}')
        plt.show()

#Linear regression models
def regression(year):
  infile = os.path.join( f'TripData_{year}.csv')
  df = pd.read_csv(infile)
  #get normal data
  df_normal = df[df['cluster']>-1]

  #Train to calculate fare amount on pickup id, dropoff id, distance, and time of day
  df_normal['fare_per_mile'] = df_normal['fare_amount']/df_normal['trip_distance']
  dftemp = df_normal[df_normal['fare_per_mile'] <= 15]
  X = dftemp[['PULocationID', 'DOLocationID', 'Time Zone', 'trip_distance']]
  #y is fare per mile
  y = dftemp['fare_per_mile']
  #convert y to log scale
  y = np.log(y)

  #Convert shift to numeric representation
  X = X.replace("Rush Hour", 1)
  X = X.replace("Mid Day", 2)
  X = X.replace("Night Shift", 3)

  #split into train and test set
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  #train model
  model = MLPRegressor(max_iter=100, hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=42) 
  model.fit(X_train, y_train)

  #test model
  y_pred = model.predict(X_test)
  #convert back to normal
  y_pred = np.exp(y_pred)
  y_test = np.exp(y_test)
  print(y_pred)
  mse = mean_squared_error(y_test, y_pred)
  print(f"Mean Squared Error for fare calculation for {year}: {mse}")
  #accuracy
  r2 = r2_score(y_test, y_pred)
  print(f"R-squared for fare calculation for {year}: {r2}")

  #Save model to pkl
  model_filename = f'model_{year}_fare.pkl'
  with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

  #Train to calculate trip time
  dftemp = df_normal[df_normal['Trip_Time'] <= 60]
  dftemp = dftemp[dftemp['Trip_Time'] >= 0]
  dftemp.dropna()
  X = dftemp[['PULocationID', 'DOLocationID', 'Time Zone', 'trip_distance']]
  y = dftemp['Trip_Time']
  #replace 0 minutes with 6 seconds
  y = y.replace(0, 0.1)
  #convert y to log scale
  y = np.log(y)
  #Convert shift
  X = X.replace("Rush Hour", 1)
  X = X.replace("Mid Day", 2)
  X = X.replace("Night Shift", 3)

  #split into train and test set
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  #Train model
  model = MLPRegressor(max_iter=100, hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=42) 
  model.fit(X_train, y_train)

  #test model
  y_pred = model.predict(X_test)
  #convert back to normal 
  y_pred = np.exp(y_pred)
  y_test = np.exp(y_test)
  mse = mean_squared_error(y_test, y_pred)
  print(f"Mean Squared Error for time calculation for {year}: {mse}")
  #accuracy
  r2 = r2_score(y_test, y_pred)
  print(f"R-squared for time calculation for {year}: {r2}")

  #Save model to pkl
  model_filename = f'model_{year}_time.pkl'
  with open(model_filename, 'wb') as file:
    pickle.dump(model, file)


def main():
    years = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    for year in years:
        print(f"Running DBSCAN with epsilon = 1.0 for {year}")
        run_DBSCAN(year, 1.0)
        #run_stats(year)
        #Stats_Plots(year)
        regression(year)
    


if __name__ == "__main__":
    main()