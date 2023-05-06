# !pip install folium
# !pip install numpy
# !pip install pandas
# !pip install matplotlib
# !pip install scipy
# !pip install seaborn
# !pip install geopandas
# !pip install haversine

import os
import numpy as np
import pandas as pd
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import scipy as sp
from scipy.stats import pearsonr
import folium
import glob
import seaborn as sns
import unicodedata
import haversine as hs

rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

font_path = fm.findfont('AppleGothic')

if not font_path:
    print('Warning: AppleGothic font not found')
else:
    print("AppleGothic font found at ", font_path)

font_prop = fm.FontProperties(fname=font_path, size=12)

# checking current directory
print(os.getcwd())

dataset = "./dataset/"
META = dataset + "META/"
TRAIN_AWS = dataset + "TRAIN_AWS/"
TRAIN_PM = dataset + "TRAIN/"
TEST_AWS = dataset + "TEST_AWS/"
TEST_PM = dataset + "TEST_INPUT/"
AWS_CITY_VARIABLE = dataset + "ALL_CITY_VARIABLE/AWS_TRAIN_VARIABLE/"
PM_CITY_VARIABLE = dataset + "ALL_CITY_VARIABLE/PM_TRAIN_VARIABLE/"
AWS_CITY_YEAR = dataset + "CITY_YEAR/AWS_TRAIN_CITY_YEAR/"
PM_CITY_YEAR = dataset + "CITY_YEAR/PM_TRAIN_CITY_YEAR/"
Linear_Interpolate = dataset + "Linear_Interpolate_Filled/"
LI_AWS = Linear_Interpolate + "Linear_TRAIN_AWS/"

# ------------------------------------------------------------------------------------

map_Kor = folium.Map(location=(36.62, 126.984873), zoom_start = 9, tiles="Stamen Terrain")
map_Kor.save("Climate_Map.html")

# reading map data csv files.
awsmap_csv = pd.read_csv(META + "awsmap.csv", encoding="UTF-8")
pmmap_csv = pd.read_csv(META + "pmmap.csv", encoding="UTF-8")

# allocating each columns into list variable.
aws_loc = awsmap_csv["Location"]
aws_lat = awsmap_csv["Latitude"]
aws_lng = awsmap_csv["Longitude"]

def obs_distance(df1, loc1, df2, loc2):
    for n in range(len(df1["Location"])):
        df1_loc = df1["Location"][i]
        df2_loc = df2["Location"][j]
        df1_lat = df1["Latitude"][i]
        df2_lat = df2["Latitude"][j]
        df1_lng = df1["Longitude"][i]
        df2_lng = df2["Longitude"][j]

        point_1 = (df1_lat, df1_lng)
        point_2 = (df2_lat, df2_lng)
        distance = hs.haversine(point_1, point_2)
        where = f"{df1_loc} and {df2_loc}"
        return (f"{where} = {distance}")


pm_loc = pmmap_csv["Location"]
pm_lat = pmmap_csv["Latitude"]
pm_lng = pmmap_csv["Longitude"]

new = []
for i in aws_loc:
    new.append(unicodedata.normalize('NFC', i))
aws_loc.columns = new

new = []
for i in pm_loc:
    new.append(unicodedata.normalize('NFC', i))
pm_loc.columns = new

# printing out the location on map, using folium.

aws_num = 0
while aws_num < len(aws_loc):
    folium.Marker(location=[aws_lat[aws_num], aws_lng[aws_num]], popup=aws_loc[aws_num],
                  icon=folium.Icon(color="blue")).add_to(map_Kor)
    aws_num += 1

pm_num = 0
while pm_num < len(pm_loc):
    folium.Marker(location=[pm_lat[pm_num], pm_lng[pm_num]], popup=pm_loc[pm_num],
                  icon=folium.Icon(color="red")).add_to(map_Kor)
    pm_num += 1

map_Kor.save("Climate_Map.html")

# ------------------------------------------------------------------------------------
# Map done #
# ------------------------------------------------------------------------------------

# reading All Test & Train csv files.

file_locations = {
    'meta' : META,
    'train_aws': TRAIN_AWS,
    'train_pm': TRAIN_PM,
    'test_aws': TEST_AWS,
    'test_pm': TEST_PM
}

all_file_locations = {}
for key, value in file_locations.items():
    all_file_locations[key] = glob.glob(value + "*.csv")
'''
type all_file_locations["key"] to call
keys are "train_aws", "train_pm", "test_aws", "test_pm"
'''

# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------

# Converting all column names into English

for files in all_file_locations['train_aws']:
    train_aws_files = pd.read_csv(files)
    train_aws_files.rename(columns={"연도":"Year", "일시":"DateTime", "지점":"Observatory","기온(°C)":"Temperature", "풍향(deg)":"Wind_Direction", "풍속(m/s)":"Wind_Speed", "강수량(mm)":"Precipitation", "습도(%)":"Humidity"}, inplace=True)
    train_aws_files.to_csv(files, index=False)


for files in all_file_locations['test_aws']:
    test_aws_files = pd.read_csv(files)
    test_aws_files.rename(columns={"연도":"Year", "일시":"DateTime", "지점":"Observatory","기온(°C)":"Temperature", "풍향(deg)":"Wind_Direction", "풍속(m/s)":"Wind_Speed", "강수량(mm)":"Precipitation", "습도(%)":"Humidity"}, inplace=True)
    test_aws_files.to_csv(files, index=False)

for files in all_file_locations['train_pm']:
    train_pm_files = pd.read_csv(files)
    train_pm_files.rename(columns={"연도":"Year", "일시":"DateTime", "측정소":"Observatory"}, inplace=True)
    train_pm_files.to_csv(files, index=False)


for files in all_file_locations['test_pm']:
    test_pm_files = pd.read_csv(files)
    test_pm_files.rename(columns={"연도":"Year", "일시":"DateTime", "측정소":"Observatory"}, inplace=True)
    test_pm_files.to_csv(files, index=False)

# ------------------------------------------------------------------------------------

# create new folders to store all csv files (per city, per year)
if not os.path.exists(dataset + "CITY_YEAR"):
    os.mkdir(dataset + "CITY_YEAR")

if not os.path.exists(AWS_CITY_YEAR):
    os.mkdir(AWS_CITY_YEAR)

if not os.path.exists(PM_CITY_YEAR):
    os.mkdir(PM_CITY_YEAR)

if not os.path.exists(dataset + "ALL_CITY_VARIABLE"):
    os.mkdir(dataset + "ALL_CITY_VARIABLE")

if not os.path.exists(AWS_CITY_VARIABLE):
    os.mkdir(AWS_CITY_VARIABLE)

if not os.path.exists(PM_CITY_VARIABLE):
    os.mkdir(PM_CITY_VARIABLE)

if not os.path.exists(dataset + "Linear_Interpolate_Filled"):
    os.mkdir(dataset + "Linear_Interpolate_Filled")

if not os.path.exists(LI_AWS):
    os.mkdir(LI_AWS)


# ------------------------------------------------------------------------------------
# separate csv file by city and years
# selecting each files within the TRAIN_AWS folder

for train_aws_file in all_file_locations['train_aws']:
    # read csv file
    df_aws = pd.read_csv(train_aws_file)
    # get location name from file name
    location = os.path.splitext(os.path.basename(train_aws_file))[0]
    # separate by year and save as separate csv files
    for year in range(4):
        year_df = df_aws[df_aws['Year'] == year]
        year_filename = AWS_CITY_YEAR + f"train_aws_{location}_{year}.csv"
        year_df.to_csv(year_filename, index=False)

for train_pm_file in all_file_locations['train_pm']:
    # read csv file
    df_pm = pd.read_csv(train_pm_file)
    # get location name from file name
    location = os.path.splitext(os.path.basename(train_pm_file))[0]
    # separate by year and save as separate csv files
    for year in range(4):
        year_df = df_pm[df_pm['Year'] == year]
        year_filename = PM_CITY_YEAR + f"train_pm_{location}_{year}.csv"
        year_df.to_csv(year_filename, index=False)

# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------


column_names = {
    "Temperature": 3,
    "Wind_Direction": 4,
    "Wind_Speed": 5,
    "Precipitation": 6,
    "Humidity": 7
}

for column_name, col_idx in column_names.items():
    temp_list = []
    for train_aws_file in all_file_locations['train_aws']:
        file_name = train_aws_file.split("/")[-1].split(".")[0]
        df_cols_only = pd.read_csv(train_aws_file, usecols=[0, 1, col_idx])
        temp_list.append((file_name, df_cols_only))

    AWS_TRAIN_total = pd.concat([df_temp[1][column_name] for df_temp in temp_list], axis=1)
    column_name = column_name.replace("/", "_")
    AWS_TRAIN_total.columns = [df_temp_new[0] for df_temp_new in temp_list]

    AWS_TRAIN_total.insert(loc=0, column='Year', value=temp_list[0][1]["Year"])
    AWS_TRAIN_total.insert(loc=1, column='DateTime', value=temp_list[0][1]["DateTime"])

    AWS_TRAIN_total.to_csv(AWS_CITY_VARIABLE + f"{column_name}.csv", index=False)

# ------------------------------------------------------------------------------------

# created new dataframe to store same columns per city
PM_TRAIN_total = pd.DataFrame()

# selecting each files within the TRAIN_AWS folder
for train_pm_file in all_file_locations['train_pm']:
    # read csv file
    df_train_pm = pd.read_csv(train_pm_file)
    # get location name from file name
    column_name = train_pm_file.split("/")[-1].split(".")[0]
    # separate PM column from each file and merge it into one file, save as csv file.
    column_data = df_train_pm.iloc[:, 3]
    column_data.name = column_name
    PM_TRAIN_total[column_name] = column_data

PM_TRAIN_total['Year'] = df_train_pm.iloc[:, 0]
PM_TRAIN_total['DateTime'] = df_train_pm.iloc[:, 1]
PM_TRAIN_total = PM_TRAIN_total.set_index(["Year", "DateTime"])

PM_TRAIN_total.to_csv(PM_CITY_VARIABLE + "PM2_5.csv", index=True)

# ------------------------------------------------------------------------------------

# finding wind-direction correlation between cities.
# One of the file created from above will be used.

rel_wind_dir = pd.read_csv(AWS_CITY_VARIABLE + "Wind_Direction.csv")

locs_aws = list(rel_wind_dir.columns)
new = []
for i in locs_aws:
    new.append(unicodedata.normalize('NFC', i))
rel_wind_dir.columns = new
rel_w_d = rel_wind_dir.iloc[:, 2:]

# ------------------------------------------------------------------------------------

# Calculate the correlations between each column
corr_matrix = rel_w_d.corr()

# Plot a heatmap of the correlation matrix
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix of AWS Variables")
plt.show()

# ------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------

# Filling in Missing Values

AWS_paths = all_file_locations["train_aws"]

# 1. Linear Interpolation

for train_aws_file in all_file_locations['train_aws']:
    train_aws_file = unicodedata.normalize('NFC', train_aws_file)

# Selecting each files within the TRAIN_AWS folder.
for train_aws_file in all_file_locations['train_aws']:
    # Read each file.
    data = pd.read_csv(train_aws_file)
    # Added one more line of code to get the location name for future usage.
    location_name = train_aws_file.split("/")[-1].split(".")[0]
    '''
    Dead Codes (No Longer Used, Keeping it for future reference.)
    # Splitting DateTime into separate columns of Date and Time.
    data[["Date", "Time"]] = data["DateTime"].str.split(" ", expand=True)
    # Convert time into Date_Range
    date_range = list(data["Time"][0:])
    '''
    # Interpolate the data and replace the old columns.
    for columns_each in data[data.columns[3:8]]:
        data[columns_each] = data[columns_each].interpolate()
    # Export the data in .csv format to a new designated folder.
    data.to_csv(LI_AWS + f"{location_name}_filled.csv", index=True)

# ------------------------------------------------------------------------------------

aws_distance = []
for i in range(len(awsmap_csv["Location"])):
    for j in range(len(awsmap_csv["Location"])):
        if awsmap_csv["Location"][i] != awsmap_csv["Location"][j]:
            distance = obs_distance(awsmap_csv, awsmap_csv["Location"][i], awsmap_csv, awsmap_csv["Location"][j])
            aws_distance.append(distance)

print(aws_distance)

print("done")

