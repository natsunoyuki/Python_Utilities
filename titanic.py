# -*- coding: utf-8 -*-
#kaggle titanic disaster competition
from scipy import *
import matplotlib.pyplot as plt
import pandas as pd

directory = "/Users/yumi/Downloads/titanic/"

#function list:
def getHonorific(DF,i):
    return DF["Name"].loc[i].split(",")[1].split(".")[0].strip()

def loadData(directory):
    TRAIN = pd.read_csv(directory+"train.csv")
    TEST = pd.read_csv(directory+"test.csv")
    return TRAIN, TEST
    
def extractHonorifics(TRAIN,TEST):
    honorifics=[]
    for i in range(len(TRAIN["Name"])):
        honorifics.append(getHonorific(TRAIN,i))
    for i in range(len(TEST["Name"])):
        honorifics.append(getHonorific(TEST,i))
    toMr=["Capt","Col","Don","Dr","Jonkheer","Major","Rev","Sir"]
    toMrs=["Dona","Lady","Mme","the Countess"]
    toMs=["Mlle","Miss"]
    for i in range(len(honorifics)):
        if(honorifics[i] in toMr):
            honorifics[i] = "Mr"
        elif(honorifics[i] in toMrs):
            honorifics[i] = "Mrs"
        elif(honorifics[i] in toMs):
            honorifics[i] = "Ms"
    return honorifics
    
def scatterPlot(X,feature1,feature2):
    plt.scatter(X[feature1],X[feature2])  
    plt.xlabel(feature1)
    plt.ylabel2(feature2)
    plt.show()  
         
#end function list
        
TRAIN,TEST = loadData(directory)

#The Name column contains possibly important information in the form of the 
#honorifics in the passenger's name. Extract that information:
honorifics = extractHonorifics(TRAIN,TEST)
TRAIN["Honorific"] = pd.DataFrame(honorifics[:len(TRAIN)])
TEST["Honorific"] = pd.DataFrame(honorifics[len(TRAIN):])

#extract features and target to train the machine with
features = ["Pclass","Sex","Age","SibSp","Parch","Ticket","Fare","Embarked","Honorific"]
X_train = TRAIN[features]
y_train = TRAIN["Survived"]
X_test = TEST[features]

