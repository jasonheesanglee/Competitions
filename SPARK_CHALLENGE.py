# !pip install folium
# !pip install numpy
# !pip install pandas
# !pip install matplotlib
# !pip install scipy
# !pip install seaborn

import os
import numpy as np
import pandas as pd
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import scipy as sp
import folium
import math
import itertools
import glob
import seaborn as sns
import unicodedata



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

if not os.path.exists(AWS_CITY_YEAR):
    os.mkdir(AWS_CITY_YEAR)

if not os.path.exists(PM_CITY_YEAR):
    os.mkdir(PM_CITY_YEAR)

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
        year_filename = AWS_CITY_YEAR + f"train_aws_{location}_{year}.csv"
        year_df.to_csv(year_filename, index=False)

for train_pm_file in all_file_locations['train_pm']:
    # read csv file
    df = pd.read_csv(train_pm_file)
    # get location name from file name
    location = os.path.splitext(os.path.basename(train_pm_file))[0]
    # separate by year and save as separate csv files
    for year in range(4):
        year_df = df[df['연도'] == year]
        year_filename = PM_CITY_YEAR + f"train_pm_{location}_{year}.csv"
        year_df.to_csv(year_filename, index=False)

# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------

# create new folders to store all csv files (per variable, all cities combined)
if not os.path.exists(dataset + "ALL_CITY_VARIABLE"):
    os.mkdir(dataset + "ALL_CITY_VARIABLE")

if not os.path.exists(AWS_CITY_VARIABLE):
    os.mkdir(AWS_CITY_VARIABLE)

if not os.path.exists(PM_CITY_VARIABLE):
    os.mkdir(PM_CITY_VARIABLE)

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

    AWS_TRAIN_total.to_csv(AWS_CITY_VARIABLE + f"{column_name}.csv", index=False)

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

PM_TRAIN_total.to_csv(PM_CITY_VARIABLE + "PM2_5.csv", index=True)

# ------------------------------------------------------------------------------------

# finding wind-direction correlation between cities.
# One of the file created from above will be used.

rel_wind_dir = pd.read_csv(AWS_CITY_VARIABLE + "풍향(deg).csv")
print(rel_wind_dir)
locs_aws = list(rel_wind_dir.columns)
new = []
for i in locs_aws:
    new.append(unicodedata.normalize('NFC', i))
rel_wind_dir.columns = new

rel_w_d = rel_wind_dir.iloc[:, 2:]
print(rel_w_d)

# ------------------------------------------------------------------------------------

# Calculate the correlations between each column
corr_matrix = rel_w_d.corr()

# Plot a heatmap of the correlation matrix
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix of AWS Variables")
plt.show()

# ------------------------------------------------------------------------------------



# ------------------------------------------------------------------------------------

# finding locations that has comparatively high correlation values.

upper_tri = np.triu(corr_matrix, k=1)

row, col = np.where(upper_tri > 0.55)

loc_corr_list = []

for i in range(len(row)):
    loc_a = locs_aws[row[i]+2]
    loc_b = locs_aws[col[i] + 2]
    corr = upper_tri[row[i], col[i]]
    loc_corr_list.append((loc_a, loc_b, corr))

print("Location pairs with correlation greater than 0.55: ")

pre_km_a = []
pre_km_b = []

for loc_a, loc_b, corr in loc_corr_list:
    print(loc_a, loc_b, corr)

# ------------------------------------------------------------------------------------

# finding distance in km between loc_a and loc_b.

    pre_km_a_pre = df.loc[loc_a, [aws_lat, aws_lng]]
    pre_km_b_pre = df.loc[loc_b, [aws_lat, aws_lng]]
    pre_km_a.append(pre_km_a_pre)
    pre_km_b.append(pre_km_b_pre)



# dlon =


print("Done")


