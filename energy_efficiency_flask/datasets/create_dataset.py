import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_excel("ENERGY_EFFICIENCY.xlsx")
import random

random.seed(1233)

Xtrain, Xtest = train_test_split(df, random_state=1233, test_size=0.25)

Xtrain.to_csv("train.csv", index=False)
Xtest.to_csv("test.csv", index=False)
