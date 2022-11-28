# IMPORT LIBRARIES

# major libs
import numpy as np
import pandas as pd
import sklearn

print(np.__version__, pd.__version__, sklearn.__version__) # useful for defining requirements.txt

# data preparation
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# model creation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# model eval
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score
from sklearn.metrics import ConfusionMatrixDisplay

# charts
import matplotlib.pyplot as plt

# serialiser
import pickle
from pickle import dump, load

###

# CREATE DATAFRAME (CSV AS PD DF)

df = pd.read_csv("Churn_Modelling.csv")

# DEFINE X AND y

X = df.iloc[:, 3:-1] 

y = df.iloc[:, -1] 

# DROP UNWANTED COLS

X = df.drop(['RowNumber', 'CustomerId', 'Surname', 'Gender', 'Exited'], axis='columns') # drop cols 

# SPLIT AND RANDOMISE THE DATASET

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# MAKE COL TRANSFORMER - FOR ENCODING AND SCALING

ct = make_column_transformer(
    ( OneHotEncoder(handle_unknown='ignore', sparse=False), ['Geography'] ),
    ( StandardScaler(), [ 'CreditScore', 'Age',  'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary' ] )
)

# FIT AND TRAIN CT ON X_TRAIN ONLY

X_train = ct.fit_transform(X_train) 

# TRANSFORM X_TEST

X_test = ct.transform(X_test) 

# TEST DIFFERENT MODELS WITH SKLEARN
'''
models = {}

# Logistic Regression
from sklearn.linear_model import LogisticRegression
models['Logistic Regression'] = LogisticRegression(random_state=1)

# Support Vector Machines
from sklearn.svm import LinearSVC
models['Support Vector Machines'] = LinearSVC(random_state=1)

# Decision Trees
from sklearn.tree import DecisionTreeClassifier
models['Decision Trees'] = DecisionTreeClassifier(random_state=1)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
models['Random Forest'] = RandomForestClassifier(random_state=1)

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
models['Naive Bayes'] = GaussianNB()

# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
models['K-Nearest Neighbor'] = KNeighborsClassifier()

# EVALUATE EACH MODEL AGAINST y_PRED

accuracy, precision, recall = {}, {}, {}

for key in models.keys():
    
    # Fit the classifier model
    models[key].fit(X_train, y_train)
    
    # Prediction 
    y_pred = models[key].predict(X_test)
    
    # Calculate Accuracy, Precision and Recall Metrics  
    accuracy[key] = accuracy_score(y_test, y_pred)
    precision[key] = precision_score(y_test, y_pred)
    recall[key] = recall_score(y_test, y_pred)

df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall'])
df_model['Accuracy'] = accuracy.values()
df_model['Precision'] = precision.values()
df_model['Recall'] = recall.values()

print(df_model)'''

# SELECT RANDOM FOREST FOR OUR CHURN PREDICTION MODEL

from sklearn.ensemble import RandomForestClassifier

churn_model_sklearn = RandomForestClassifier(random_state=1, n_estimators=100) # n_estimators=100 is default

churn_model_sklearn.fit(X_train, y_train) # fit the model before making predictions

model_pred = churn_model_sklearn.predict(X_test)

# EVALUATE OUR MODEL USING CONF MATRIX, ACCURACY, PRECISION AND RECALL SCORES FOR SELECTED MODEL

print(confusion_matrix(y_test, model_pred)) # Note: TN=top_left, FN=btm_lft, FP=top_rgt, TP=btm_rgt

print(accuracy_score(y_test, model_pred))

print(precision_score(y_test, model_pred))

print(recall_score(y_test, model_pred))

# SAVE CT AND MODEL AS PICKLE FILE

pickle.dump(ct, open('scaler.pkl', 'wb'))

pickle.dump(churn_model_sklearn, open("churn_model_sklearn.pkl", "wb")) 

# LOAD AND MAKE PRED WITH PICKLE CT AND MODEL

churn_model_sklearn_pkl = pickle.load(open('churn_model_sklearn.pkl', 'rb'))

scaler_pkl = pickle.load(open('scaler.pkl', 'rb'))

# TEST THE MODEL

#new_data_np = np.array([[ 600, "France", 40, 3, 60000, 2, 1, 1, 50000 ]]) # new data as np array # truth=0
#new_data_np = np.array([[ 645, "Spain", 44, 8, 113755.78, 2, 1, 0, 149756.71 ]]) # new data as np array # truth=0
#new_data_np = np.array([[ 725, "Germany", 19, 0, 75888.2, 1, 0, 0, 45613.75 ]]) # new data as np array # truth=1
#new_data_np = np.array([[ 622, "Spain", 46, 4, 107073.27, 2, 1, 1, 30984.59 ]]) # new data as np array # truth=1
new_data_np = np.array([[ 524, "Germany", 31, 8, 107818.63, 1, 1, 0, 199725.39 ]]) # new data as np array # truth=1

new_data_pd_df = pd.DataFrame(new_data_np, columns=["CreditScore", "Geography", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]) # new data conv to pd dataframe
#print(new_data_pd_df)

new_data_scaled = scaler_pkl.transform(new_data_pd_df) # scale the new data
#print(new_data_scaled)

new_data_pred = churn_model_sklearn_pkl.predict(new_data_scaled)

print(new_data_pred, type(new_data_pred)) # new_data_pred is a np array scalar e.g. [1]

new_data_pred = int(new_data_pred) # convert it into an integer

print(new_data_pred, type(new_data_pred)) # new_data_pred is now an int e.g. 1

json_return=dict.fromkeys(["prediction"], new_data_pred) # convert it into a python "dictionary" or "dict" which can be read with javascript as a "json object literal". fromkeys() lets me specify the name of the key in the returned key:value pair.

print(json_return)

print(1)


# APPLY THRESHOLDING ie 1 if pred > 2.0

'''Thresholding for RandomForest: "In simple words, this means that the actual RF output is [p0, p1] (assuming binary classification), from which the predict method simply returns the class with the highest value, i.e. 0 if p0 > p1 and 1 otherwise." https://stackoverflow.com/questions/57307095/change-threshold-value-for-random-forest-classifier '''

new_data_pred_prob = churn_model_sklearn_pkl.predict_proba(new_data_scaled)

print(new_data_pred_prob)
# [[0.18 0.82]] 
# new_data_pred[0][0]==0.18 and new_data_pred[0][1]==0.82
# currently returns [1] coz that is index of the largest number (ie argmax)

threshold = 0.3
# define threshold here

# if the num on the right of the pred probability (1 or exited) is larger than the threshold, then prediction must always be 1 even if num on left (0 or remain) is larger. 
# For instance: say probability_array = [[ 0.3, 0.2 ]], if we did not use thresholding, coz 0.3 is the arger number, result would be 0. If we apply threshold of IF probability_array[0][1] >= 0.2 THEN we can make the prediction 1. 
# This is to boost recall score ie lower false negatives (person left - 1 - even though predicted to stay - 0).
# Now IF BOTH scores are under the threshold, AND [0][0] is > [0][1] then we make prediction = 0.
# ie say Thres for [0][1] >= 0.3
# If [0][1] >= 0.3 
# [[ 0.39 0.31 ]] 
# pred = 1
# If [0][1] < 0.3 and [0][1] > [0][0]
# [[0.20 0.29]]
# pred = 1
# If [0][1] < 0.3 and [0][1] < [0][0]
# [[0.29 0.2]]
# pred = 0

if new_data_pred_prob[0][1] >= threshold:
    final_pred_thresh = 1 
elif new_data_pred_prob[0][1] < threshold and new_data_pred_prob[0][1] > new_data_pred_prob[0][0]:
    final_pred_thresh = 1
elif new_data_pred_prob[0][1] < threshold and new_data_pred_prob[0][1] < new_data_pred_prob[0][0]:
    final_pred_thresh = 0

print(final_pred_thresh)

print(2)

####### TESTING THRESHOLDS ON MODEL PREDS TO SEE WHICH IS BEST #######

### METHOD: i wanna see new conf matrix based on adding a classif thresh to the model preds:
# 1. use smaller 2D array from predict_proba() result 
# 2. apply the classif thresh to each result
# 3. add result of classif thres to new 1D arr [n, n, n, n...] 
# 4. apply the above onto model preds across X_test
# 5. create conf matrix and apply recall score

# 1.

'''##test_pred_probs = churn_model_sklearn_pkl.predict_proba(X_test)
##print(test_pred_probs[:10,:]) # see array of pred probs
##test_working_arr = test_pred_probs[:10,:]'''

test_working_arr = [[0.20, 0.29], [0.29, 0.2]] # note that both arrays are under the Thres of 0.3

# 2. and 3.

test_arr_with_classif_thres_res = []

test_threshold = 0.3

for i in range(len(test_working_arr)):
    if test_working_arr[i][1] >= test_threshold:
        test_arr_with_classif_thres_res.append(1)
    elif test_working_arr[i][1] < test_threshold and test_working_arr[i][1] > test_working_arr[i][0]:
        test_arr_with_classif_thres_res.append(1)
    elif test_working_arr[i][1] < test_threshold and test_working_arr[i][1] < test_working_arr[i][0]:
        test_arr_with_classif_thres_res.append(0)

print('test_arr_with_classif_thres_res ' + str(test_arr_with_classif_thres_res))
# result: "test_arr_with_classif_thres_res [1, 0]"
# so even though both are under thres of 0.3, the larger of the two nums will be selected 

'''##actual_first_ten_labels = y_test[:10]
##print(actual_first_ten_labels) # just to compare with ground truth labels of first ten labels'''

# 4.

print(3)

model_pred_2 = churn_model_sklearn.predict_proba(X_test)

#print(len(model_pred_2), model_pred_2.shape)
# 2000 (2000, 2)

print(model_pred_2)

arr_with_classif_thres_res_2  = []

threshold_2 = 0.3

for i in range(len(model_pred_2)):
    if model_pred_2[i][1] >= threshold_2:
        arr_with_classif_thres_res_2.append(1)
    elif model_pred_2[i][1] < threshold_2 and model_pred_2[i][1] > model_pred_2[i][0]:
        arr_with_classif_thres_res_2.append(1)
    elif model_pred_2[i][1] < threshold_2 and model_pred_2[i][1] < model_pred_2[i][0]:
        arr_with_classif_thres_res_2.append(0)
    
'''previously: 
for i in range(len(model_pred_2)):
    if model_pred_2[i][1] >= threshold_2:
        arr_with_classif_thres_res_2.append(1)
    elif model_pred_2[i][1] < threshold_2:
        arr_with_classif_thres_res_2.append(0)
'''
print(arr_with_classif_thres_res_2)

# 5.

classif_thres_conf_matrix = confusion_matrix(y_test, arr_with_classif_thres_res_2)

print(classif_thres_conf_matrix)

print(accuracy_score(y_test, arr_with_classif_thres_res_2))

print(precision_score(y_test, arr_with_classif_thres_res_2))

print(recall_score(y_test, arr_with_classif_thres_res_2))

'''
threshold_2 = 0.2
0.759
0.4461538461538462
0.7876543209876543

threshold_2 = 0.25
0.791
0.48914858096828046
0.7234567901234568

threshold_2 = 0.3
0.8205
0.5445736434108527
0.6938271604938272
# increased recall score from 0.5 to 0.7 with classif thresh of 0.3 instead of default of 0.5 and accuracy still good !
# let's use this threshold 

threshold_2 = 0.5
0.8635
0.726027397260274
0.5234567901234568
compare with model results using .predict() with default 0.5 thres
ie model_pred = churn_model_sklearn.predict(X_test)

'''


#print(df.iloc[2670,:]) corresponding to row of test data index number 2670