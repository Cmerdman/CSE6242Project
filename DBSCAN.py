# ==========================
# IMPORT REQUIRED LIBRARIES
# ==========================

import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt #Used for Debug plots
import os
import numpy as np
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    


#Define directories
data_dir = "Data"
cleaned_dir = "Cleaned"
output_dir = "Analyzed"


#Performs DBSCAN and assigns clusters for trip distance and total fare.
def Run_DBSCAN(year, zoneStart, zoneEnd):
    #Set the final end zone number to itself +1 to avoid missing the final zone in the for loop
    zoneEnd = zoneEnd+1
    #Read data file
    infile = os.path.join(data_dir, cleaned_dir, f'TripData_{year}.csv')
    df = pd.read_csv(infile)
    df["dist_zone_cluster"] = 0
    df['time_zone_cluster'] = 0

    

    #Get the information for the relevant pickup zone ID
    #Loops through the time zones:
    for time in range(1,4):
        if time == 1:
            data = df[df['Time Zone'] == "Rush Hour"]
            timezone = "Rush Hour"
        elif time == 2:
            data = df[df['Time Zone'] == "Mid Day"]
            timezone = "Mid Day"
        elif time == 3:
            data = df[df['Time Zone'] == "Night Shift"]
            timezone = "Night Shift"
    
        #Loops through all specified zones:
        for i in range(zoneStart, zoneEnd):
            zonedata = data[data['PULocationID'] == i]
            if(len(zonedata)>0):
                data1 = zonedata[["norm_trip_distance","norm_fare_amount"]]
                data2 = zonedata[["norm_trip_distance","norm_Trip_Time"]]

                #print(len(zonedata))
                min_sample_num = np.ceil(len(zonedata)/3)
                min_sample_num = int(min_sample_num)

                #Cluster data with DBSCAN
                dbscan = DBSCAN(eps=0.75, min_samples=min_sample_num) # We can adjust epsilon and the min samples as needed.  I chose 0.75 and 1/3 of the length of the zone's data initially

                clusters1 = dbscan.fit_predict(data1)
                clusters2 = dbscan.fit_predict(data2)

                #Save clisters to the original dataframe
                #data1["dist_zone_cluster"] = clusters1
                #data2["time_zone_cluster"] = clusters2
                data1_copy = data1.copy()
                data1_copy["dist_zone_cluster"] = clusters1
                data2_copy = data2.copy()
                data2_copy["time_zone_cluster"] = clusters2
                #data1.loc[:, 'dist_zone_cluster'] = clusters1
                #data2.loc[:, 'time_zone_cluster'] = clusters2

                #print(clusters1)
                #print(clusters2)
                

                data1_copy = data1_copy.drop(labels=["norm_trip_distance", "norm_fare_amount"], axis=1)
                data2_copy = data2_copy.drop(labels=["norm_trip_distance", "norm_Trip_Time"], axis=1)
                #Update original dataframe
                df["dist_zone_cluster"].update(data1_copy["dist_zone_cluster"])
                df["time_zone_cluster"].update(data2_copy["time_zone_cluster"])
                #df = pd.merge(df, data, left_index=True, right_index=True, how='left')

                #DEBUG: Plot
                #DebugPlot(df, year, timezone, i)


    #Save data to analyzed folder
    outfile = os.path.join(data_dir, output_dir, f'TripData_{year}.csv')
    df.to_csv(outfile, index=False)
 


#Debugging plotting:
def DebugPlot(df, year, timezone, ID):
    #Get only the data we need for the plot
    df = df[df['Time Zone']==timezone]
    df = df[df['PULocationID']==ID]
    clusters = df['dist_zone_cluster'].tolist()
    unique_labels = np.unique(clusters)
    figure, axis = plt.subplots(1, 2)
    figure.suptitle(f'Dataframe Clusters for {year} for {timezone} at Pickup zone {ID}')
    for label in unique_labels:
        if label == -1:
            # Plot abnormal points as black
            axis[0].scatter(df[df['dist_zone_cluster'] == label]['trip_distance'], df[df['dist_zone_cluster'] == label]['fare_amount'], color='black', label='Abnormal')
            axis[1].scatter(df[df['dist_zone_cluster'] == label]['DOLocationID'], df[df['dist_zone_cluster'] == label]['fare_amount'], color='black', label='Abnormal')
        else:
            axis[0].scatter(df[df['dist_zone_cluster'] == label]['trip_distance'], df[df['dist_zone_cluster'] == label]['fare_amount'], label=f'Normal Cluster {label}')
            axis[1].scatter(df[df['dist_zone_cluster'] == label]['DOLocationID'], df[df['dist_zone_cluster'] == label]['fare_amount'], label=f'Normal Cluster {label}')
    
    #plt.scatter(df['trip_distance'], df['fare_amount'], c=df['dist_zone_cluster'], cmap='viridis', label='Cluster Data', s=50)
    
    axis[0].set_xlabel('trip_distance')
    axis[0].set_ylabel('fare_amount')
    axis[0].legend()
    axis[1].set_xlabel('DO Zone')
    axis[1].set_ylabel('fare_amount')
    axis[1].legend()
    plt.show()

def DebugPlot_zone(df, year, timezone, ID):
    #Get only the data we need for the plot
    df = df[df['Time Zone']==timezone]
    df = df[df['PULocationID']==ID]
    clusters = df['dist_zone_cluster'].tolist()
    unique_labels = np.unique(clusters)
    plt.figure(figsize=(8, 6))
    for label in unique_labels:
        if label == -1:
            # Plot abnormal points as black
            plt.scatter(df[df['dist_zone_cluster'] == label]['DOLocationID'], df[df['dist_zone_cluster'] == label]['fare_amount'], color='black', label='Abnormal')
        else:
            plt.scatter(df[df['dist_zone_cluster'] == label]['DOLocationID'], df[df['dist_zone_cluster'] == label]['fare_amount'], label=f'Normal Cluster {label}')
    
    #plt.scatter(df['trip_distance'], df['fare_amount'], c=df['dist_zone_cluster'], cmap='viridis', label='Cluster Data', s=50)
    plt.title(f'Dataframe Clusters for {year} for {timezone} at Pickup zone {ID}')
    plt.xlabel('DO Zone')
    plt.ylabel('fare_amount')
    plt.legend()
    plt.show()


#Anomoly detection in dataset using z scores:
def DetectAnomoly(year):
     #Read data file
    infile = os.path.join(data_dir, cleaned_dir, f'TripData_{year}.csv')
    df = pd.read_csv(infile)
    df["total_amount"] = df["total_amount"] - df["tip_amount"]
    dftemp = df.copy()
    
    #Change trip distance to 1 where it is 0 so we don't get a division by 0 error
    dftemp["trip_distance"] = dftemp["trip_distance"].replace(0, 1)
    df["fare_per_mile"] = dftemp["total_amount"]/dftemp["trip_distance"]
    df['z_score'] = np.abs(stats.zscore(df['fare_per_mile']))
    #df['z_score'] = df['fare_per_mile'].where(df['fare_per_mile'] != 0).apply(lambda x: stats.zscore([x])[0] if pd.notna(x) else 1)
    df = df[df['trip_distance']>1]
    #dftemp['z_score'] = np.abs(stats.zscore(dftemp['fare_per_mile']))
    df['z_score'] = np.abs(stats.zscore(df['fare_per_mile']))

    # Set a threshold for anomaly detection
    threshold = 2

    # Identify anomalies
    df['anomalies'] = 0
    df.loc[df['z_score'] > threshold, 'anomalies'] = 1

    '''
    #DEBUG: Plot
    # Custom colormap from a list of colors
    colors = ['cyan', 'black']
    cmap_custom = LinearSegmentedColormap.from_list('my_cmap', colors, N=len(colors))
    plt.figure(figsize=(8, 6))
    plt.scatter(df['trip_distance'], df['total_amount'], c=df['anomalies'], cmap=cmap_custom, s=50)
    plt.title(f'Anamolies for {year}')
    plt.xlabel('trip_distance')
    plt.ylabel('total_amount')
    #plt.legend(*plt.gca().get_legend_handles_labels())
    plt.show()
    '''

    '''
    plt.figure(figsize=(8, 6))
    plt.scatter(df['trip_distance'], df['z_score'], c=df['anomalies'], cmap='viridis', s=50)
    plt.title(f'Anamolies for {year}')
    plt.xlabel('trip_distance')
    plt.ylabel('z_score')
    #plt.legend(*plt.gca().get_legend_handles_labels())
    plt.show()
    '''



def main():
    startZone = 1
    endZone = 263
    #1 to 263 is the range of all zones in the dataset
    
    Run_DBSCAN(2024, startZone, endZone)
    
    #years = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    #for year in years:
    #    Run_DBSCAN(year, startZone, endZone)
        #DetectAnomoly(year)
    #Run_DBSCAN(2012, startZone, endZone)

if __name__ == "__main__":
    main()