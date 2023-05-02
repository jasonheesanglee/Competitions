# !pip install folium
# !pip install numpy
# !pip install pandas
# !pip install matplotlib
# !pip install scipy
# !pip install seaborn

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import folium
import math
import itertools
import glob
import seaborn as sns
import matplotlib.font_manager as fm
plt.rcParams['font.family'] = 'AppleGothic'
font_path = fm.findfont('AppleGothic')
if not font_path:
    print('Warning: AppleGothic font not found')
else:
    print("AppleGothic font found at ", font_path)

font_prop = fm.FontProperties(fname=font_path, size=12)



print(os.getcwd())
dataset = "./dataset/"
META = dataset + "META/"
TRAIN_AWS = dataset + "TRAIN_AWS/"
TRAIN_PM = dataset + "TRAIN/"
TEST_AWS = dataset + "TEST_AWS/"
TEST_PM = dataset + "TEST_INPUT/"


# checking path of this CURRENT document

map_Kor = folium.Map(location=(36.62, 126.984873), zoom_start = 9, tiles="Stamen Terrain")
map_Kor.save("Climate_Map.html")

# reading map data csv files.
awsmap_csv = pd.read_csv(META + "awsmap.csv")
pmmap_csv = pd.read_csv(META + "pmmap.csv")

# Map saved

# allocating each columns into list variable.
aws_loc = awsmap_csv["Location"]
aws_lat = awsmap_csv["Latitude"]
aws_lng = awsmap_csv["Longitude"]

pm_loc = pmmap_csv["Location"]
pm_lat = pmmap_csv["Latitude"]
pm_lng = pmmap_csv["Longitude"]

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

# create new folders to store all csv files (per city, per year)
if not os.path.exists(dataset + "CITY_YEAR"):
    os.mkdir(dataset + "CITY_YEAR")

if not os.path.exists(dataset + "CITY_YEAR/AWS_TRAIN_CITY_YEAR"):
    os.mkdir(dataset + "CITY_YEAR/AWS_TRAIN_CITY_YEAR")

if not os.path.exists(dataset + "CITY_YEAR/PM_TRAIN_CITY_YEAR"):
    os.mkdir(dataset + "CITY_YEAR/PM_TRAIN_CITY_YEAR")

# ------------------------------------------------------------------------------------

# separate csv file by city and years

# selecting each files within the TRAIN_AWS folder

for train_aws_file in all_file_locations['train_aws']:
    # read csv file
    df = pd.read_csv(train_aws_file)
    # get location name from file name
    location = os.path.splitext(os.path.basename(train_aws_file))[0]
    # separate by year and save as separate csv files
    for year in range(4):
        year_df = df[df['연도'] == year]
        year_filename = dataset + f"CITY_YEAR/AWS_TRAIN_CITY_YEAR/train_aws_{location}_{year}.csv"
        year_df.to_csv(year_filename, index=False)

for train_pm_file in all_file_locations['train_pm']:
    # read csv file
    df = pd.read_csv(train_pm_file)
    # get location name from file name
    location = os.path.splitext(os.path.basename(train_pm_file))[0]
    # separate by year and save as separate csv files
    for year in range(4):
        year_df = df[df['연도'] == year]
        year_filename = dataset + f"CITY_YEAR/PM_TRAIN_CITY_YEAR/train_pm_{location}_{year}.csv"
        year_df.to_csv(year_filename, index=False)

# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------

# create new folders to store all csv files (per variable, all cities combined)
if not os.path.exists(dataset + "ALL_CITY_VARIABLE"):
    os.mkdir(dataset + "ALL_CITY_VARIABLE")

if not os.path.exists(dataset + "ALL_CITY_VARIABLE/AWS_TRAIN_VARIABLE"):
    os.mkdir(dataset + "ALL_CITY_VARIABLE/AWS_TRAIN_VARIABLE")

if not os.path.exists(dataset + "ALL_CITY_VARIABLE/PM_TRAIN_VARIABLE"):
    os.mkdir(dataset + "ALL_CITY_VARIABLE/PM_TRAIN_VARIABLE")

# ------------------------------------------------------------------------------------
# 기온
# created new dataframe to store same columns per city
new_list = []

for train_aws_file in all_file_locations['train_aws']:
    column_name = train_aws_file.split("/")[-1].split(".")[0]
    df = pd.read_csv(train_aws_file, usecols=[0, 1, 3])
    new_list.append((column_name, df))

AWS_TRAIN_total = pd.concat([df[1]['기온(°C)'] for df in new_list], axis=1)
AWS_TRAIN_total.columns = [df[0] for df in new_list]
print(AWS_TRAIN_total)

AWS_TRAIN_total.insert(loc=0, column='연도', value=new_list[0][1]["연도"])
AWS_TRAIN_total.insert(loc=1, column='일시', value=new_list[0][1]["일시"])


AWS_TRAIN_total.to_csv(dataset + f"ALL_CITY_VARIABLE/AWS_TRAIN_VARIABLE/temperature.csv", index=False)
# ------------------------------------------------------------------------------------------------------
new_list = []

for train_aws_file in all_file_locations['train_aws']:
    column_name = train_aws_file.split("/")[-1].split(".")[0]
    df = pd.read_csv(train_aws_file, usecols=[0, 1, 4])
    new_list.append((column_name, df))

AWS_TRAIN_total = pd.concat([df[1]['풍향(deg)'] for df in new_list], axis=1)
AWS_TRAIN_total.columns = [df[0] for df in new_list]
print(AWS_TRAIN_total)

AWS_TRAIN_total.insert(loc=0, column='연도', value=new_list[0][1]["연도"])
AWS_TRAIN_total.insert(loc=1, column='일시', value=new_list[0][1]["일시"])


AWS_TRAIN_total.to_csv(dataset + f"ALL_CITY_VARIABLE/AWS_TRAIN_VARIABLE/wind_dir.csv", index=False)

# ------------------------------------------------------------------------------------------------------
new_list = []

for train_aws_file in all_file_locations['train_aws']:
    column_name = train_aws_file.split("/")[-1].split(".")[0]
    df = pd.read_csv(train_aws_file, usecols=[0, 1, 5])
    new_list.append((column_name, df))

AWS_TRAIN_total = pd.concat([df[1]['풍속(m/s)'] for df in new_list], axis=1)
AWS_TRAIN_total.columns = [df[0] for df in new_list]
print(AWS_TRAIN_total)

AWS_TRAIN_total.insert(loc=0, column='연도', value=new_list[0][1]["연도"])
AWS_TRAIN_total.insert(loc=1, column='일시', value=new_list[0][1]["일시"])


AWS_TRAIN_total.to_csv(dataset + f"ALL_CITY_VARIABLE/AWS_TRAIN_VARIABLE/wind_speed.csv", index=False)


# ------------------------------------------------------------------------------------------------------
new_list = []

for train_aws_file in all_file_locations['train_aws']:
    column_name = train_aws_file.split("/")[-1].split(".")[0]
    df = pd.read_csv(train_aws_file, usecols=[0, 1, 6])
    new_list.append((column_name, df))

AWS_TRAIN_total = pd.concat([df[1]['강수량(mm)'] for df in new_list], axis=1)
AWS_TRAIN_total.columns = [df[0] for df in new_list]
print(AWS_TRAIN_total)

AWS_TRAIN_total.insert(loc=0, column='연도', value=new_list[0][1]["연도"])
AWS_TRAIN_total.insert(loc=1, column='일시', value=new_list[0][1]["일시"])


AWS_TRAIN_total.to_csv(dataset + f"ALL_CITY_VARIABLE/AWS_TRAIN_VARIABLE/rain_amount.csv", index=False)

# ------------------------------------------------------------------------------------------------------
new_list = []

for train_aws_file in all_file_locations['train_aws']:
    column_name = train_aws_file.split("/")[-1].split(".")[0]
    df = pd.read_csv(train_aws_file, usecols=[0, 1, 7])
    new_list.append((column_name, df))

AWS_TRAIN_total = pd.concat([df[1]['습도(%)'] for df in new_list], axis=1)
AWS_TRAIN_total.columns = [df[0] for df in new_list]
print(AWS_TRAIN_total)

AWS_TRAIN_total.insert(loc=0, column='연도', value=new_list[0][1]["연도"])
AWS_TRAIN_total.insert(loc=1, column='일시', value=new_list[0][1]["일시"])


AWS_TRAIN_total.to_csv(dataset + f"ALL_CITY_VARIABLE/AWS_TRAIN_VARIABLE/humidity.csv", index=False)

# ------------------------------------------------------------------------------------
new_list = []
column_names = {
    "기온(°C)": 3,
    "풍향(deg)": 4,
    "풍속(m/s)": 5,
    "강수량(mm)": 6,
    "습도(%)": 7
}

for column_name, col_idx in column_names.items():
    temp_list = []
    for train_aws_file in all_file_locations['train_aws']:
        file_name = train_aws_file.split("/")[-1].split(".")[0]
        df = pd.read_csv(train_aws_file, usecols=[0, 1, col_idx])
        temp_list.append((file_name, df))

    AWS_TRAIN_total = pd.concat([df[1][column_name] for df in temp_list], axis=1)
    column_name = column_name.replace("/", "_")
    AWS_TRAIN_total.columns = [df[0] for df in temp_list]

    AWS_TRAIN_total.insert(loc=0, column='연도', value=temp_list[0][1]["연도"])
    AWS_TRAIN_total.insert(loc=1, column='일시', value=temp_list[0][1]["일시"])

    AWS_TRAIN_total.to_csv(dataset + f"ALL_CITY_VARIABLE/AWS_TRAIN_VARIABLE/{column_name}.csv", index=False)

# ------------------------------------------------------------------------------------

# created new dataframe to store same columns per city
PM_TRAIN_total = pd.DataFrame()

# selecting each files within the TRAIN_AWS folder
for train_pm_file in all_file_locations['train_pm']:
    # read csv file
    df = pd.read_csv(train_pm_file)
    # get location name from file name
    column_name = train_pm_file.split("/")[-1].split(".")[0]
    # separate PM column from each file and merge it into one file, save as csv file.
    column_data = df.iloc[:, 3]
    column_data.name = column_name
    PM_TRAIN_total[column_name] = column_data

PM_TRAIN_total['연도'] = df.iloc[:, 0]
PM_TRAIN_total['일시'] = df.iloc[:, 1]
PM_TRAIN_total = PM_TRAIN_total.set_index(["연도", "일시"])

PM_TRAIN_total.to_csv(dataset + f"ALL_CITY_VARIABLE/PM_TRAIN_VARIABLE/PM2_5.csv", index=True)

# ---------------------------well---------------------------------------------------------

# finding wind-direction correlation between cities.
# One of the file created from above will be used.

rel_wind_dir = pd.read_csv(dataset + "ALL_CITY_VARIABLE/AWS_TRAIN_VARIABLE/풍향(deg).csv")

corr_matrix = rel_wind_dir.corr()

plt.plot(corr_matrix['풍향(deg)'], corr_matrix.index, 'o')
plt.title('Correlation with Relative Wind Direction', fontproperties=font_prop)
plt.xlabel('Correlation Coefficient', fontproperties=font_prop)
plt.ylabel('Variables', fontproperties=font_prop)

plt.show()

print("Done")
# print(PM_TRAIN_total)

