# -*- coding: utf-8 -*-
#kaggle titanic disaster competition
from scipy import *
import matplotlib.pyplot as plt
import pandas as pd

directory = "/Users/yumi/Downloads/titanic/"

#function list:
def getHonorific(DF,i):
    #get the honorific from Name
    return DF["Name"].loc[i].split(",")[1].split(".")[0].strip()

def loadData(directory):
    TRAIN = pd.read_csv(directory+"train.csv")
    TEST = pd.read_csv(directory+"test.csv")
    return TRAIN, TEST
    
def extractHonorifics(TRAIN,TEST):
    #we simplify the entire honorific set to a simplified version of just
    #Master, Ms, Mr and Mrs
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

def checkForNan(DF,COLUMN):
    X = pd.isnull(DF[COLUMN])
    
         
#end function list
      
#load the necessary data          
TRAIN,TEST = loadData(directory)

#Feature engineering:
honorifics = extractHonorifics(TRAIN,TEST)
TRAIN["Honorific"] = pd.DataFrame(honorifics[:len(TRAIN)])
TEST["Honorific"] = pd.DataFrame(honorifics[len(TRAIN):])

TRAIN["Companions"] = TRAIN["SibSp"] + TRAIN["Parch"]
TEST["Companions"] = TEST["SibSp"] + TEST["Parch"]

CABINS = []
for i in range(len(TRAIN.Cabin)):
    CABINS.append(TRAIN.Cabin[i])
for i in range(len(TEST.Cabin)):
    CABINS.append(TEST.Cabin[i])

for i in range(len(CABINS)):
    if(not(pd.isnull(CABINS[i]))):
        CABINS[i] = CABINS[i][0]
    else:
        #replace NaN with N
        CABINS[i] = "N"

print "Unique cabin letters: {}".format(unique(CABINS))

TRAIN["Cabin_letter"] = pd.DataFrame(CABINS[:len(TRAIN.Cabin)])
TEST["Cabin_letter"] = pd.DataFrame(CABINS[len(TRAIN.Cabin):])

#extract features and target to train the machine with
features = ["Pclass","Sex","Age","SibSp","Parch","Companions","Cabin_letter","Fare","Embarked","Honorific"]
X_train = TRAIN[features]
y_train = TRAIN["Survived"]
X_test = TEST[features]

#pd.get_dummies() treats all numerical data as continuous and will not create
#dummy variables for them. Therefore we need to force them into strings
X_train.Pclass = X_train.Pclass.astype(str)
X_test.Pclass = X_test.Pclass.astype(str)

X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

