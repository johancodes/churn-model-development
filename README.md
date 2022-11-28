# churn-model-development

This machine learning model predicts if a customer will leave a particular bank or not. It's trained on the bank's public dataset of 8000 rows, with parameters specific to the bank. The model has an accuracy of 80%. 

Notes:
- This is a Logistic Regression Model to predict if a customer will leave a bank or not.
- Different SKlearn algorithms were tested and Random Forest Classifier was selected.
- Thresholding was applied on the prediction values. We reduced the classification threshold. This decreases false negatives (customer predicted to stay even though they leave).  See: https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall#:~:text=False%20positives%20increase%2C%20and%20false%20negatives%20decrease. 
