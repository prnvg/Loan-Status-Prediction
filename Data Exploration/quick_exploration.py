import pandas as pd
import numpy as np
import matplotlib as plt

df = pd.read_csv("train.csv")
print(df.head(10))
print(df.describe())
print(df['Property_Area'].value_counts())