#Importing the important library.

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings

warnings.filterwarnings("ignore")

#-----------------------------------------------

hw_df=pd.read_csv(r"C:\sudhanshu_projects\project-task-training-course\HeightWeightRatio.csv")

#-----------------------------------------------
#Now we have to check there is any null value or not.

hw_df.isnull().sum()

#Conclusion: As there is no null value.

#-----------------------------------------------
#Now we have to check the datatype of column.

hw_df.dtypes

#Conclusion: All columns are numeric in nature.

#-------------------------------------------------
#Now we have to check the information about the dataset.

hw_df.info()

#Conclusion: Here we get the information about no. of null values,datatype of column.

#-------------------------------------------------
#Now here we have to get the description of dataset.

hw_df.describe()

#Conclusion: Here we get all the statistical measures on the dataset.

#--------------------------------------------------
#Now here we decide the independent and dependent feature.

hw_df.drop("Index",axis=1,inplace=True)

x=hw_df.drop("Weight(Pounds)",axis=1) #Here we take height as independent feature.

y=hw_df.drop("Height(Inches)",axis=1) #Here we take weight as dependent feature.

#Conclusion: Here we want to calculate weight based on height.

#----------------------------------------------------
#Now here we want to split the datset into train and test.

#---------------------------------------------
#Here we import train_test_split library.
from sklearn.model_selection import train_test_split

#---------------------------------------------
#Here we split x-train/test and y-train/test.
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


#-----------------------------------------------
#Here we can see how two features depend on each other.

plt.scatter(x=hw_df["Height(Inches)"],y=hw_df["Weight(Pounds)"])
plt.title("Height Vs Weight")
plt.xlabel("Height in Inches")
plt.ylabel("Weights in Pounds")
plt.show()

#Conclusion: Here we seen that the data is clustered,but has positive correlation.

sns.boxenplot(data=hw_df,x="Height(Inches)",y="Weight(Pounds)")

#--------------------------------------------------------
#Now here we create linear regression models.

from sklearn.linear_model import LinearRegression

#------------------------------------------
#Here we create linear regression model object.
lr=LinearRegression()

#------------------------------------------
#Here we train the lr model.
lr.fit(x_train,y_train)

#------------------------------------------
#Here we predict the values using lr model.
y_pred_lr=lr.predict(x_test)

#------------------------------------------
#Here we find the accuracy of lr model.
score_lr=lr.score(x_test,y_test)
print(f"The accuracy of lr model is {score_lr}.")

#---------------------------------------------------------
#Now we can do the scaling of data.

#-----------------------------------------
#Here we import standard scaler library.
from sklearn.preprocessing import StandardScaler

#-----------------------------------------
#Now here we create standard scaler model object.
sc=StandardScaler()

#-----------------------------------------
#Now here we fit and transform x_train.
x_train_sc=sc.fit_transform(x_train)

#-----------------------------------------
#Now here we transform the x_test.
x_test_sc=sc.transform(x_test)

#--------------------------------------------------------
#Here we train lr model on scaled data.

#---------------------------------------
#Here we build linear regression model object.
lr1=LinearRegression()

#---------------------------------------
#Here we train the lr1 model.
lr1.fit(x_train_sc,y_train)

#---------------------------------------
#Here we predict the values using lr1 model.
y_pred_lr1=lr1.predict(x_test_sc)

#---------------------------------------
#Here we find the accuracy of lr1 model.
score_lr1=lr1.score(x_test_sc,y_test)
print(f"Th accuracy of lr1 model is {score_lr1}.")

#Here also the accuracy is very less.

#------------------------------------------------------
#Now here we build non-linear regression model.

#--------------------------------------------
#Here we import svr model.
from sklearn.svm import SVR

#--------------------------------------------
#Here we create svr model object.
svr=SVR()

#--------------------------------------------
#Here we train the svr model.
svr.fit(x_train,y_train)

#--------------------------------------------
#Here we predict the values using svr model.
y_pred_svr=svr.predict(x_test)

#--------------------------------------------
#Here we find the accuracy of svr model.
score_svr=svr.score(x_test,y_test)
print(f"The accuracy of svr model is {score_svr}.")

#----------------------------------------------
#Here we import knr model.
from sklearn.neighbors import KNeighborsRegressor

#----------------------------------------------
#Here we build knr model object.
knr=KNeighborsRegressor()

#----------------------------------------------
#Here we train the knr model.
knr.fit(x_train,y_train)

#----------------------------------------------
#Here we predict the values using knr model.
y_pred_knr=knr.predict(x_test)

#----------------------------------------------
#Here we find the accuracy of knr model.
score_knr=knr.score(x_test,y_test)
print(f"The accuracy of knr model is {score_knr}.")

#Conclusion: Here the accuracy more decreases.

#-----------------------------------------------
#Now here we use dtr model.
from sklearn.tree import DecisionTreeRegressor

#-------------------------------------------
#Here we create dtr model object.
dtr=DecisionTreeRegressor()

#-------------------------------------------
#Here we train the dtr model.
dtr.fit(x_train,y_train)

#-------------------------------------------
#Here we predict the values using dtr model.
y_pred_dtr=dtr.predict(x_test)

#-------------------------------------------
#Here we find the accuracy of dtr model.
score_dtr=dtr.score(x_test,y_test)
print(f"The accuracy of dtr model is {score_dtr}.")

#-----------------------------------------------
#dtr model on scaled data.

#-------------------------------------------
#Here we create dtr model object.
dtr1=DecisionTreeRegressor()

#-------------------------------------------
#Here we train the dtr1 model.
dtr1.fit(x_train_sc,y_train)

#-------------------------------------------
#Here we predict the values using dtr1 model.
y_pred_dtr1=dtr1.predict(x_test_sc)

#-------------------------------------------
#Here we find the accuracy of dtr1 model.
score_dtr1=dtr1.score(x_test_sc,y_test)
print(f"The accuracy of dtr1 model is {score_dtr1}.")

#Conclusion: The Accuracy of model is very less.

#--------------------------------------------------------
#Now here we use rfr model.

from sklearn.ensemble import RandomForestRegressor

#----------------------------------------------
#Here we create rfr model object.
rfr=RandomForestRegressor()

#----------------------------------------------
#Here we train the rfr model.
rfr.fit(x_train,y_train)

#----------------------------------------------
#Here we predict the values using rfr model.
y_pred_rfr=rfr.predict(x_test)

#----------------------------------------------
#Here we find the accuracy of rfr model.
score_rfr=rfr.score(x_test,y_test)
print(f"The accuracy of rfr model is {score_rfr}.")

#--------------------------------------------------------
#Now apply the model on scaled data.

#----------------------------------------------
#Here we create rfr model object.
rfr1=RandomForestRegressor()

#----------------------------------------------
#Here we train the rfr1 model.
rfr1.fit(x_train_sc,y_train)

#----------------------------------------------
#Here we predict the values using rfr1 model.
y_pred_rfr1=rfr1.predict(x_test_sc)

#----------------------------------------------
#Here we find the accuracy of rfr1 model.
score_rfr1=rfr1.score(x_test_sc,y_test)
print(f"The accuracy of rfr1 model is {score_rfr1}.")


#------------------------------------------------------------
#Now we save the lr model.

pickle.dump(lr,open(r"C:\sudhanshu_projects\project-task-training-course\HeightWeightRatio.pkl","wb"))
