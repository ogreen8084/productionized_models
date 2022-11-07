import pandas as pd 
import numpy as np 
import random

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import scipy.stats as stats
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

df = pd.read_csv('./datasets/train.csv')

Xtrain = df.drop(columns=['HEATING_LOAD'])
ytrain = df[['HEATING_LOAD']]
#these variables are another order of magnitude, need to make smaller
LARGE_VARIABLES = ['SURFACE_AREA', 'WALL_AREA', 'ROOF_AREA']

std_scaler = StandardScaler()

preprocessor = ColumnTransformer([('std_scale', StandardScaler(), LARGE_VARIABLES)], remainder='passthrough')
pipeline = Pipeline([('preprocessing', preprocessor), ('xgboost', XGBRegressor(random_state=1233))])

