ReadMe.txt For Data Visualization Project:

- Project Description:

This project is centered around taxi trip data from 2011-2024 in New York City. The core objective of this study is to understand
trends within taxi data across various measures of time (by year or shift). This includes identifying abnormality within the data set as well as understanding
and interpolating new data points based on patterns in the data.

Within the visualization, there are three major components that achieve the above mentioned goal. 

First, there is a visualization of the DBSCAN (Density-Based Spatial Clustering of Applications with Noise).
This is an unsupervised algorithm that examines the proximity of data points and clusters as is appropriate. The parameters fed into this algorithm are:
trip time, trip distance, and fare amount. This is displayed as a 3d graph, and the user can customize the data by selected a specific area of nyc
(borough) and time of day.

Second, there is a coropleth graph and a bar chart that respectively displays the average trip time and trip frequency. 
These figures can be filtered by borough and shift.

Third, there is the option for a user to select pickup and dropoff locations to create their own trip. Once the user selects two points,
OpenRouteService's directions API will be called to find the distance between the two points (by car). Then, the fare per mile and trip time will
be estimated using two separate Multilayer Perceptron models. 

- Project Installation Instructions:

(Recommended)
Required Data: Data cleaned and analyzed using DBSCAN: https://drive.google.com/file/d/1R6pNJQO8p5SQQ8RU68xBpO6DRd55ARWV/view?usp=sharing
Download and unzip the file and store in the root of this folder. The folder should be named "Analyzed", as this relative path will be called by
the visualization script.

To run the visualization:
The visualization requires the analyzed data to be downloaded. Ensure that the paths defined in the visualization script match their referenced 
counterparts. To run and view the script, is recommended to use VScode with a Jupiter extension. From there, the required packages need to be downloaded.
This can be done easily by importing the dependencies from requirements.txt into your virtual environment (venv). Then, run the notebook in its entirety ("run all"). 
View this link on more details on how to download dependencies inside a python environment: 
https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#using-a-requirements-file. Unless manually changed, this link 
will take you to your local dashboard instance. http://127.0.0.1:8053/

In order to use the Multilayer perceptron map, You must create an account with OpenRouteService. To find more information, 
please visit https://api.openrouteservice.org. You must sign up for an account with them, and obtain an API key. 
Once this is completed, please place the API key within the designated variable in Visualization_dashboard.ipynb. The variable is named 'ors_key'.

(Optional, advised against)
To preprocess the raw taxi trip data:
Download the Yellow Taxi Trip Records parquet files for January of 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 
2022, 2023, and 2024 from https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
Store downlaoded parquet files in the Data/Raw subfolder.
Run Data.py to clean the data. Cleaned data is stored in the Data/Cleaned subfolder. This process takes approximately 2 hours.
Run DBSCAN.py to analyze and cluster the data. Clustered data is stored in the Data/Analyzed subfolder.  This process takes approximately 4 hours.
NOTE: It is advised to grab the cleaned an analyzed data from the link provided at the top of this document to avoid the six hours needed for preprocessing.


- Project Folder Structure Overview:

Project folder contains the following subfolders:
  - CODE (All project code) 
  - DOC (Project documents) 

  Code folder contains the required code for the project and subfolders for the data and machine learning models.
   - [Code] Data.py is utilized to clean raw data files.
   - [Code] DBSCAN.py is used to analyze and cluster the cleaned data files.
   - [Code] Visualization_dashboard is used to run the visualization.
   - requirements.txt is a list of dependencies that can be loaded into your virtual environment

  Code folder contains the following subfolders:
   - "Data/"  Used to store the data for this project.  
   - "Models/"  Used to store the trained machine learning models for this project.
   - "tax_zones/" contains shape files, used to apply geometry to taxi data

  Data Subfolder contains the Link_to_data.txt file containing relevant links to the data used for the project and the following subfolders: 
   - Raw (Raw taxi trip data from the dataset)
   - Cleaned (The raw data cleaned by Data.py)
   - Analyzed (The dataset after being analyzed and clustered using DBSCAN.py.  This is the final data used for the visualization.)

  Models subfolder contains the trained linear regression models used for the interactive portion of this project.
   - Models are saved in the format of [year]_model_[time/fare].pkl, for estimating time and fare amount respectively for each year in the data.
