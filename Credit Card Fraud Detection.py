import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

# READING DATASET

Data = pd.read_csv("creditcard.csv")

df = pd.DataFrame(Data)
# print(df)

# DESCRIBE AND UNDERSTAND THE DATA

print(df.shape)
print(df.describe())

#COLUMNS

print(df.columns)

# IMBALANCE IN THE DATA

fraud = df[df['Class'] == 1]
valid = df[df['Class'] == 0]
print(fraud.shape)
print(valid.shape)

# OUTLIER FRACTION

Outlier_Fraction = len(fraud)/len(valid)
print(Outlier_Fraction)
'''
Only 0.17% fraudulent transaction out all the transactions.
The data is highly unbalanced.
'''

# PLOTTING THE CORRELATION MATRIX
'''
The correlation matrix graphically gives us an idea of how features corelate with each other and can help us predict 
what are the features that are most relevant for the prediction.
'''

Correlation_Matrix = df.corr()
fig = plt.figure(figsize=(12,9))
sns.heatmap(Correlation_Matrix, vmax=0.8, square=True)
plt.show()

'''
In the heatmap we can clearly see that most of the features do not correlate to other features but there are some 
features that either has a positive or a negative correlation with each other.
For example: V2 and V5 are highly negatively correlated with the feature called Amount. We also see some correlation 
with V20 and Amount. This gives us a deeper understanding of the Data available to us.
'''

# SPLITING THE DATA TO X AND Y

X = df.drop(['Class'], axis=1)
Y = df['Class']
print(X.shape)
print(Y.shape)

# TRAINING AND TESTING THE DATA TO BUILD THE MODEL

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)

# BUILDING MODEL

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
model = rfc.fit(X_train,Y_train)

# PREDICTIONS

y_pred = rfc.predict(X_test)

# MODEL EVALUATION

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

cm = confusion_matrix(Y_test,y_pred)
print(cm)

accuracy = accuracy_score(Y_test,y_pred)
print("The accuracy score is: ",accuracy)

precision = precision_score(Y_test,y_pred)
print("The precision score is: ",precision)

recall = recall_score(Y_test,y_pred)
print("The recall score is: ",recall)

f1 = f1_score(Y_test,y_pred)
print("The f1-score is: ",f1)

report = classification_report(Y_test,y_pred, labels=[1,0])
print(report)

# PREDICTION OF MODEL

Prediction = rfc.predict([[18,0.247491128,0.277665627,1.185470842,-0.09260255,-1.314393979,-0.150115998,-0.94636495,-1.617935051,1.544071402,-0.829880601,-0.583199527,0.524933232,-0.453375297,0.081393088,1.555204196,-1.396894893,0.783130838,0.436621214,2.177807168,-0.230983143,1.650180361,0.200454091,-0.185352508,0.423073148,0.820591262,-0.227631864,0.336634447,0.250475352,22.75]])
print(Prediction) #prediction should be zero