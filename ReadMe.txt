ReadMe.txt For Data Visualization Project:

Please set your python environment utilizing the provided environment.yaml file.

(Recommended)
Required Data: Data cleaned and analyzed using DBSCAN: https://drive.google.com/file/d/1R6pNJQO8p5SQQ8RU68xBpO6DRd55ARWV/view?usp=sharing
Download and unzip the file and store in the Data/Analyzed subfolder.

To run the visualization:
Requires the analyzed data to be downloaded and the paths to that as well as the shape file data to be updated. From there, the required packages need to be downloaded and the notebook run in its entirety. Unless manually changed, this link will take you to your local dashboard instance. http://127.0.0.1:8053/

**In order to use the Multilayer perceptron map, You must create an account with OpenRouteService. To find more information, please visit [https://api.openrouteservice.org](this link). You must sign up for an account with them, and obtain an API key. Once this is completed, please place the API key within the desginated variable in Visualization_dashboard.ipynb. The variable is named 'ors_key'.**


(Optional, advised against)
To preprocess the raw taxi trip data:
Download the Yellow Taxi Trip Records parquet files for January of 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, and 2024 from https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
Store downlaoded parquet files in the Data/Raw subfolder.
Run Data.py to clean the data.  Cleaned data is stored in the Data/Cleaned subfolder.  This process takes approximately 2 hours.
Run DBSCAN.py to analyze and cluster the data.  Clustered data is stored in the Data/Analyzed subfolder.  This process takes approximately 4 hours.
NOTE: It is advized to grab the cleaned an analyzed data from the link provided at the top of this document to avoid the six hours needed for preprocessing.


Project Folder Structure:
Project folder contains the code required for the project and subfolders for the data and linear regression models.
 - Code Data.py is utilized to clean raw data files.
 - Code DBSCAN.py is used to analyze and cluster the cleaned data files.
 - Code Visualization_dashboard is used to run the visualization.
 - Environment.yaml is used to set the python environment for the project.
Project folder contains the following subfolders:
 - Data.  Used to store the data for this project.  
 - Models.  Used to store the trained linear regression models for this project.

  -> Data Subfolder contains the Link_to_data.txt file containing relevant links to the data used for the project and the following subfolders: 
    - Raw (Raw taxi trip data from the dataset)
    - Cleaned (The raw data cleaned by Data.py)
    - Analyzed (The dataset after being analyzed and clustered using DBSCAN.py.  This is the final data used for the visualization.)

  ->Models subfolder contains the trained linear regression models used for the interactive portion of this project.
    - Models are saved in the format of [year]_model_[time/fare].pkl, for estimating time and fare amount respectively for each year in the data.
