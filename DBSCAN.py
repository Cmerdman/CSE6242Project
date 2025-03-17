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
    df["zone_cluster"] = 0

    dbscan = DBSCAN(eps=1.75, min_samples=20) # We can adjust epsilon and the min samples as needed.  I chose 1.75 and 20 initially

    #Get the information for the relevant pickup zone ID
    #Loops through all specified zones
    for i in range(zoneStart, zoneEnd):
        data = df[df['PULocationID'] == i]
        if(len(data)>0):
            data = data[["trip_distance","fare_amount"]]
            #Debug: Try with smaller dataset
            #data = data.head(10000)

            #We can scale the data if we want, but I do not beleive it is necessary for this project.
            #scaler = StandardScaler()
            #scaled_data = scaler.fit_transform(data)

            #Cluster data with DBSCAN
            #clusters = dbscan.fit_predict(scaled_data)
            clusters = dbscan.fit_predict(data)

            #Save clisters to the original dataframe
            data["zone_cluster"] = clusters
            #DEBUG: Plot
            #DebugPlot(data, year, 24)
            data = data.drop(labels=["trip_distance", "fare_amount"], axis=1)
            #Update original dataframe
            df["zone_cluster"].update(data["zone_cluster"])
            #df = pd.merge(df, data, left_index=True, right_index=True, how='left')


    #Save data to analyzed folder
    outfile = os.path.join(data_dir, output_dir, f'TripData_{year}.csv')
    df.to_csv(outfile, index=False)
 


#Debugging plotting:
def DebugPlot(df, year, ID):
    plt.figure(figsize=(8, 6))
    plt.scatter(df['trip_distance'], df['fare_amount'], c=df['default_cluster'], cmap='viridis', s=50)
    plt.title(f'Dataframe Clusters for {year} for Pickup zone {ID}')
    plt.xlabel('trip_distance')
    plt.ylabel('fare_amount')
    #plt.legend(*plt.gca().get_legend_handles_labels())
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
    years = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    for year in years:
        Run_DBSCAN(year, startZone, endZone)
        #DetectAnomoly(year)
    #Run_DBSCAN(2012, startZone, endZone)

if __name__ == "__main__":
    main()