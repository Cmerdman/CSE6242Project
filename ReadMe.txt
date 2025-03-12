ReadMe.txt For Data Visualization Project:

Utilized python Libraries: 
heapq
networkx 
pandas 
math
os



Note on data folder structure:
The Data folder contains many subfolders for our taxi input data.  This is in case we have to make any changes to how we prepare or analyze the data.  The structure is as follows:
- Raw (Raw taxi trip data from the dataset)
- Cleaned (The Raw data with any unusable values removed)
- Dijkstra (The Cleaned data with the calculated distance and time for the shortest, safest, and most traffic efficient path added.)
- Analyzed (The Dijkstra appended data with a flag to indicate if the taxi trip may be fradulent.)
The remaining files are the speed limits of each street, accident information, traffic information, and the edge node graph of New York used to calculate route distances.

Note on the Debug folder:
This contains the graph data for New York used by the Dijkstra calculation, as well as the taxi zone lookup table (only taxizonelookup.csv is used)
We can move Edges.csv and Nodes.csv to the Data folder if we want.  We're treating the graph as static (we are not considering if the street was built in our observed timeframe), so the same nodes and edges are present for the entire timeframe
If you do so, be sure to update the output for them in Data.py and the input for Dijkstra.py

Note on taxi trip data structure:
It changes by year, so I had to make all dataframes have the same columns
Columns:
VendorID, tpep_pickup_datetime, tpep_dropoff_datetime, passenger_count, trip_distance, RatecodeID, store_and_fwd_flag, PULocationID, DOLocationID, payment_type, fare_amount, extra, mta_tax, tip_amount, tolls_amount, improvement_surcharge, total_amount, congestion_surcharge, airport_fee 
Added:
pickup_latitude
pickup_longitude
dropoff_latitude
dropoff_longitude
trip_time (in minutes)

Note: This standard started in 2011.  The following changes were made for 2009 and 2010
Changes:
vendor_name -> Changed to VendorID 1= Creative Mobile Technologies, LLC; 2= VeriFone Inc, 3 = Other (NOTE: This is 3 for all values in 2009 and 2010)
Trip_Pickup_DateTime -> Changed to tpep_pickup_datetime
Trip_Dropoff_DateTime -> Changed to tpep_dropoff_datetime
Passenger_Count -> Changed to passenger_count	
Trip_Distance -> Changed to trip_distance
Rate_Code -> Changed to RatecodeID
Start_Lon -> Changed to pickup_longitude
Start_Lat -> Changed to pickup_latitude
Rate_Code -> Changed to RateCodeID
store_and_forward -> Changed to store_and_fwd_flag
End_Lon -> Changed to dropoff_longitude
End_Lat -> Changed to dropoff_latitude
Payment_Type -> Changed to payment_type (int) using dictionary	
Fare_Amt -> Changed to fare_amount
surcharge -> Changed to extra 
mta_tax -> Not changed
Tip_Amt -> Changed to tip_amount
Tolls_Amt -> Changed to tolls_amount
Total_Amt -> Changed to total_amount

Additions (all with default 0 values)
improvement_surcharge 
congestion_surcharge
airport_fee 


NOTHING WORKS FOR CALCULATING DISTANCES AND NOBODY IS GETTING BACK TO ME
I CAN'T ACCESS THE GOOGLE API
I CAN'T UES DIJKSTRA CAUSE THERE'S 142,000 STREETS IN NEW YORK
NOT EVEN MANHATTAN DISTANCE BETWEEN START AND ENDING CORDINATES IS RIGHT
