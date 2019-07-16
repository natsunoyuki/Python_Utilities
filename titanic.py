# -*- coding: utf-8 -*-
#kaggle titanic disaster competition

from scipy import *
import matplotlib.pyplot as plt
import pandas as pd

directory = "/Users/yumi/Downloads/titanic/"

def getHonorific(DF,i):
    return DF["Name"].loc[i].split(",")[1].split(".")[0].strip()

TRAIN = pd.read_csv(directory+"train.csv")
TEST = pd.read_csv(directory+"test.csv")

honorifics=[]

for i in range(len(TRAIN["Name"])):
    honorifics.append(getHonorific(TRAIN,i))

for i in range(len(TEST["Name"])):
    honorifics.append(getHonorific(TEST,i))

#print "All passenger honorifics: {}".format(unique(honorifics))

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
        
#print "Reduced passenger honorifics: {}".format(unique(honorifics))

TRAIN["Honorific"] = pd.DataFrame(honorifics[:len(TRAIN)])
TEST["Honorific"] = pd.DataFrame(honorifics[len(TRAIN):])

features = ["Pclass","Sex","Age","SibSp","Parch","Ticket","Fare","Embarked","Honorific"]
X_train = TRAIN[features]
y_train = TRAIN["Survived"]
X_test = TEST[features]

