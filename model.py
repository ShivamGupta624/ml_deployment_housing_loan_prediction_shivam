import pickle
import pandas as pd
import numpy as np
data = pd.read_csv('House_loan_Prediction_Assignment.csv')
X=data.drop("Loan_Status",axis = 1)
X = X.values
Y=data[["Loan_Status"]]
Y = Y.values
#split the data set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=.60)
#Using random forest model algorithm
from sklearn import tree
classifier=tree.DecisionTreeClassifier()
classifier.fit(x_train,y_train)
predictions=classifier.predict(x_test)
#export the model
pickle.dump(classifier, open('model.pkl','wb'))
#load the model and test with a custom input
model = pickle.load( open('model.pkl','rb'))
x = [[1,0,0,0,0,128.0,360.0,1.0,5849.0]]
predict = model.predict(x)
print("Hello Worlds")
print(predict[0])