import os
import numpy as np
import pandas as pd
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import scipy as sp
from scipy.stats import pearsonr
import folium
import math
import itertools
import glob
import seaborn as sns
import unicodedata
# import geopandas as gpd


# AWS 결측치 채우기. 결측치 index[i-1,i+1].mean() 으로 채우기

def fillAwsAllMdata(df):
        df = pd.DataFrame(df)
        count = 0
        for k in range(3, 8, 1):
            df_tmp = df.iloc[:, k]
            nullIndex = df_tmp.index[df.isnull().any(axis=1)].tolist()
            for i in range(len(nullIndex)):
                if i + 1 == len(nullIndex):
                    fillData = float(df.iloc[nullIndex[i] - count-1, k])
                    df.iloc[nullIndex[i] - count-1:, k] = df.iloc[nullIndex[i] - count-1:, k].fillna(fillData)
                elif nullIndex[i + 1] - nullIndex[i] == 1:
                    count += 1
                else:
                    if count==0:
                        fillData = float((df.iloc[nullIndex[i] - 1,k]+df.iloc[nullIndex[i] + 1, k]) / 2)
                        df.iloc[nullIndex[i], k] = fillData
                    else:
                        fillData = float((df.iloc[nullIndex[i]-count-1,k]+df.iloc[nullIndex[i]+1,k])/2)
                        df.iloc[nullIndex[i]-count-1:nullIndex[i]+1,k] = df.iloc[nullIndex[i]-count-1:nullIndex[i]+1,k].fillna(fillData)
                        count = 0
        return df


# PM 결측치 채우기. 결측치 index[i-1,i+1].mean() 으로 채우기
def fillPmAllMdata(df):
        df = pd.DataFrame(df)
        count = 0
        k = 3
        nullIndex = df.index[df.isnull().any(axis=1)].tolist()
        for i in range(len(nullIndex)):
                if i + 1 == len(nullIndex):
                    fillData = float(df.iloc[nullIndex[i] - count-1, k])
                    df.iloc[nullIndex[i] - count-1:, k] = df.iloc[nullIndex[i] - count-1:, k].fillna(fillData)
                elif nullIndex[i + 1] - nullIndex[i] == 1:
                    count += 1
                else:
                    if count==0:
                        fillData = float((df.iloc[nullIndex[i] - 1, k]+df.iloc[nullIndex[i] + 1, k]) / 2)
                        df.iloc[nullIndex[i], k] = fillData
                    else:
                        if df.isnull().iloc[nullIndex[i] - count, k]:
                             fillData = df.iloc[nullIndex[i] + 2, k]
                             df.iloc[:nullIndex[i] + 1, k] = df.iloc[:nullIndex[i] + 1, k].fillna(fillData)
                        else:
                            fillData = float((df.iloc[nullIndex[i] - count-1, k] + df.iloc[nullIndex[i] + 1, k]) / 2)
                            df.iloc[nullIndex[i] - count-1:nullIndex[i] + 1,k] = df.iloc[nullIndex[i] - count-1:nullIndex[i] + 1, k].fillna(fillData)
                        count = 0
        return df

awsPath = "TRAIN_AWS/*.csv"
pmPath = "TRAIN/*.csv"
AWS_paths = glob.glob(awsPath)
PM_paths = glob.glob(pmPath)
AWS_paths.sort()
PM_paths.sort()
conAWS_List = list()
conPM_List = list()

for i in range(len(AWS_paths)):
    AWS_paths[i] = unicodedata.normalize('NFC', AWS_paths[i])


# AWS 한글 지역명으로 경로값 가져오기
for i in range(len(AWS_paths)):
    globals()[f'{AWS_paths[i][10:-4]}'] = pd.read_csv(f'{AWS_paths[i]}')
# AWS 결측치 채우기
for i in range(len(AWS_paths)):
    tmp = AWS_paths[i][10:-4]
    globals()[f'{AWS_paths[i][10:-4]}_AWS'] = fillAwsAllMdata(eval(tmp))


# 지점 라벨링
for i in range(len(AWS_paths)):
    globals()[f'{AWS_paths[i][10:-4]}_AWS']['지점'] = i
    if i == 0:
        globals()[f'{AWS_paths[i][10:-4]}_AWS_CON'] = globals()[f'{AWS_paths[i][10:-4]}_AWS']
        conAWS_List.append(globals()[f'{AWS_paths[i][10:-4]}_AWS_CON'])
    elif i > 0:
        globals()[f'{AWS_paths[i][10:-4]}_AWS_CON'] = globals()[f'{AWS_paths[i][10:-4]}_AWS'].drop('일시',axis=1)
        globals()[f'{AWS_paths[i][10:-4]}_AWS_CON'] = globals()[f'{AWS_paths[i][10:-4]}_AWS_CON'].drop('연도',axis=1)
        conAWS_List.append(globals()[f'{AWS_paths[i][10:-4]}_AWS_CON'])
merged_AWS_df = pd.concat(conAWS_List,axis=1)

for i in range(len(PM_paths)):
    PM_paths[i] = unicodedata.normalize('NFC', PM_paths[i])


# PM 한글 지역명으로 경로값 가져오기
for i in range(len(PM_paths)):
    globals()[f'{PM_paths[i][6:-4]}'] = pd.read_csv(f'{PM_paths[i]}')


# PM 지역명으로 결측치 채우기
for i in range(len(PM_paths)):
    tmp = PM_paths[i][6:-4]
    globals()[f'{PM_paths[i][6:-4]}_PM'] = fillPmAllMdata(eval(tmp))


# 측정소 라벨링
for i in range(len(PM_paths)):
    globals()[f'{PM_paths[i][6:-4]}_PM']['측정소'] = i + 30
    if i == 0:
        globals()[f'{PM_paths[i][6:-4]}_PM_CON'] = globals()[f'{PM_paths[i][6:-4]}_PM']
        conPM_List.append(globals()[f'{PM_paths[i][6:-4]}_PM_CON'])
    elif i > 0:
        globals()[f'{PM_paths[i][6:-4]}_PM_CON'] = globals()[f'{PM_paths[i][6:-4]}_PM'].drop('일시',axis=1)
        globals()[f'{PM_paths[i][6:-4]}_PM_CON'] = globals()[f'{PM_paths[i][6:-4]}_PM_CON'].drop('연도',axis=1)
        conPM_List.append(globals()[f'{PM_paths[i][6:-4]}_PM_CON'])
merged_PM_df = pd.concat(conPM_List,axis=1)


def missAllDrange(df):
    nList = []
    for k in range(3,8,1):
        df_tmp = df.iloc[:,k]
        nullIndex = df_tmp.index[df.isnull().any(axis=1)].tolist()
        count = 0
        for i in range(len(nullIndex)):
            if i + 1 == len(nullIndex):
                if count > 0:
                    nList.append(f'{nullIndex[i]-count}:{k} ~ {nullIndex[i]}:{k}')
                else:
                     nList.append(f'{nullIndex[i]}:{k}')
            elif nullIndex[i+1] - nullIndex[i] == 1:
                count += 1
            else:
                if count==0:
                    nList.append(f'{nullIndex[i]}:{k}')
                else:
                    nList.append(f'{nullIndex[i]-count}:{k} ~ {nullIndex[i]}:{k}')
                    count = 0
        return nList

def obs_distance(df1, loc1, df2, loc2):
    for i in range(len(df1)):
        for j in range(len(df2)):
            if i != j:
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


print("done")