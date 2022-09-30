import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Determine feature importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix

dataset_path = "C:/Users/rsers/OneDrive/Documents/Github/ML RandomForest/stroke-data-main.csv"
dataset = pd.read_csv(dataset_path)
print('Dataset shape:', dataset.shape)
dataset.head(5)

# Data Exploration

# Rows containing duplicate data - Remove if required
dup_rows_df = dataset[dataset.duplicated()]
print("Number of duplicate rows: ", dup_rows_df.shape)
print("-------------------------------------------------------------------------")
# Find null values
print('Null Values:\n\n', dataset.isnull().sum())
print("-------------------------------------------------------------------------")
print('Data Types:\n\n', dataset.dtypes)
print("-------------------------------------------------------------------------")

# Percentage of mising BMI and Smoking values
SmSt_mv = (dataset['smoking_status'].isnull().sum() / len(dataset['smoking_status'])*100)
print('Smoking Status missing values:', round(SmSt_mv, 1),'%')

# Percentage of missing values
SmSt_mv = (dataset['bmi'].isnull().sum() / len(dataset['bmi'])*100)
print('BMI missing values:', round(SmSt_mv, 1),'%')

# Drop smoking status & impute missing BMI values (too many missing values to impute smoking status)
df = dataset.drop(['smoking_status'], 1)
print('Dataset shape:', df.shape)
df.head(5)

# Assess BMI column distribution - Create histograms & boxplot on BMI feature
print('Distribution Plot')
df['bmi'].hist(figsize=(5,5), bins = 50, color = "c", edgecolor='black')
# plt.show()
print("-------------------------------------------------------------------------")
print('Box Plot')
# sns.boxplot(dataset['bmi'])

# Due to the posetive skew in the data & many high BMI outliers that would negatively effect parametric imputation,
# Utilisation of the median shall be used for this numerical column (non-parametric)

# Impute missing BMI level rows with BMI median
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

# Check the imputation of the dataset
print('Null Values:\n\n', df.isnull().sum())

# Convert categorical data into numerical data - Binary categorical fields were re-labelled as 0's and 1's &
# fields with more than 2 unique values were labelled with one hot encoding

# Binary/nominal variables - ever_married, Residence_type
# categorical - One hot variables - work_type, smoking_status, gender

# transform nominal variables that only have 2 values
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['ever_married']))}
print(class_mapping)
df['ever_married'] = df['ever_married'].map(class_mapping)

class_mapping = {label: idx for idx, label in enumerate(np.unique(df['Residence_type']))}
print(class_mapping)
df['Residence_type'] = df['Residence_type'].map(class_mapping)

class_mapping = {label: idx for idx, label in enumerate(np.unique(df['gender']))}
print(class_mapping)
df['gender'] = df['gender'].map(class_mapping)

# transform nominal variables that have more than 2 values
df[['work_type']] = df[['work_type']].astype(str)

# concatenate nominal variables from pd.getdummies &
transpose = pd.get_dummies(df[['work_type']])

# And the ordinal variables to form the final dataset
df = pd.concat([df,transpose], axis=1)[['id','age','hypertension','heart_disease','ever_married','Residence_type',
                                        'avg_glucose_level','bmi','gender','work_type_Govt_job','work_type_Never_worked',
                                        'work_type_Private','work_type_children','work_type_Self-employed','stroke']]
df.head(5)

# Define label vector
y = df[['stroke']]

# Define feature array
X = df.drop(['id','stroke'], 1)

# Define random forest model
model = RandomForestClassifier(n_estimators = 100)
model.fit(X, y)

# Get importance
importance = model.feature_importances_

# Summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
 
# Plot feature importance
# pyplot.barh([x for x in range(len(importance))], importance)
# pyplot.show()

# So, Age, Glucose level & BMI are the most significant features
# Consider dropping additional variables in future analyses

# Display label counts - The imbalanced data problem
df['stroke'].value_counts()

# Standardize data
ss = StandardScaler()

X = ss.fit_transform(X)

# Addressing the data imbalenced problem in the dataset
# Currently the ouput label 'stroke' has a 98.2:1.8 ratio of non-stroke:stroke events which will negatively bias
# any model trained using the dataset as is. Rebalancing the dataset must be performed before using the data to 
# train any models

# Define scoring function
def classification_eval(y_test, y_pred):
    print(f'accuracy  = {np.round(accuracy_score(y_test, y_pred), 3)}')
    print(f'precision = {np.round(precision_score(y_test, y_pred), 3)}')
    print(f'recall    = {np.round(recall_score(y_test, y_pred), 3)}')
    print(f'f1-score  = {np.round(f1_score(y_test, y_pred), 3)}')
    print(f'roc auc   = {np.round(roc_auc_score(y_test, y_pred), 3)}')

# Instantiate oversampler 
rs = RandomOverSampler()

# Run model on dataset
X, y = rs.fit_resample(X, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)

# Instantiate model
rf = RandomForestClassifier()

# Train model
rf.fit(X_train, y_train)

# Predict on test data
y_pred = rf.predict(X_test)

# Assess accuracy
classification_eval(y_test, y_pred)

# Plot confusion matrix
confusion_matrix(y_test, y_pred) 