import pandas as pd
import numpy as np
import matplotlib as plt

df = pd.read_csv("train.csv")

#print(df.apply(lambda x: sum(x.isnull()),axis=0) )
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
df['Self_Employed'].fillna('No',inplace=True)
df['Gender'].fillna('Male',inplace=True)
df['Married'].fillna('Yes',inplace=True)
df['Dependents'].fillna('0',inplace=True)
df['Credit_History'].fillna(1.0,inplace=True)
df['Loan_Amount_Term'].fillna(360.0,inplace=True)
df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])

from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i])
df.dtypes


from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

def classification_model(model, data, predictors, outcome):
  
  model.fit(data[predictors],data[outcome])
  predictions = model.predict(data[predictors])
  
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print "Accuracy : %s" % "{0:.3%}".format(accuracy)

  kf = KFold(data.shape[0], n_folds=5)
  error = []
  for train, test in kf:
   
    train_predictors = (data[predictors].iloc[train,:])
    
    train_target = data[outcome].iloc[train]
    
    model.fit(train_predictors, train_target)
    
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
 
  print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))
  
model = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)
predictor_var = ['TotalIncome_log','LoanAmount_log','Credit_History','Dependents','Property_Area']
outcome_var = 'Loan_Status'
classification_model(model, df,predictor_var,outcome_var)
