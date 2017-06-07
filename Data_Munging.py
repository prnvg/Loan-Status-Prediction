import pandas as pd
import numpy as np
import matplotlib as plt

df = pd.read_csv("train.csv")

df.apply(lambda x: sum(x.isnull()),axis=0) 
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
df['Self_Employed'].fillna('No',inplace=True)
df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])
print(df.head())