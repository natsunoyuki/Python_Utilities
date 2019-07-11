#placeholder main file for loading and calling SKLEARN machine learning
#algorithms and functions.
from scipy import *
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split #deprecated
#from sklearn.model_selection import train_test_split #newer version
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from sklearn.datasets import make_moons
#from sklearn.datasets import fetch_lfw_people
from sklearn.datasets import load_digits
from sklearn.datasets import load_iris
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
#from scipy.cluster.hierarchy import dendrogram, ward
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
#make_pipeline generates names for steps automatically.
from sklearn.pipeline import make_pipeline
import os
#For reference this line is 80 characters long#################################

DATA_PATH='./'

def scatterPlot(X,y,feature1,feature2,targetNames):
    for i in range(len(targetNames)):
        plt.scatter(X[y==i,feature1],X[y==i,feature2])  
    plt.legend(targetNames)
    plt.show()

def visualize_classifier(model,X,y,ax=None,cmap='rainbow'):
    ax=ax or plt.gca()
    ax.scatter(X[:,0],X[:,1],c=y,s=30,cmap=cmap,clim=(y.min(),y.max()),zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim=ax.get_xlim()
    ylim=ax.get_ylim()
    model.fit(X,y)
    xx,yy=meshgrid(linspace(*xlim,num=200),linspace(*ylim,num=200))
    Z=model.predict(c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
    n_classes=len(unique(y))
    contours=ax.contourf(xx,yy,Z,alpha=0.3,levels=arange(n_classes+1)-0.5,cmap=cmap,clim=(y.min(),y.max()),zorder=1)
    ax.set(xlim=xlim,ylim=ylim)
    
def eval_on_features(features,target,regressor):
    n_train=184
    X_train,X_test=features[:n_train],features[n_train:]
    y_train,y_test=target[:n_train],target[n_train:]
    regressor.fit(X_train,y_train)
    print("Test-set R^2: {:.2f}".format(regressor.score(X_test,y_test)))
    y_pred=regressor.predict(X_test)
    y_pred_train=regressor.predict(X_train)
    plt.figure(figsize=(10,3))
    plt.xticks(range(0,len(X),8),xticks.strftime("%a %m-%d"),rotation=90,ha='left')
    plt.plot(range(n_train),y_train,label='train')
    plt.plot(range(n_train,len(y_test)+n_train),y_test,'-',label='test')
    plt.plot(range(n_train),y_pred_train,'--',label='prediction train')
    plt.plot(range(n_train,len(y_test)+n_train),y_pred,'--',label='prediction test')
    #plt.legend(loc=(1.01,0))
    plt.legend(loc=(0,1))
    plt.xlabel('Date')
    plt.ylabel('Rentals')    
    
def load_citibike():
    data_mine = pd.read_csv(os.path.join(DATA_PATH, "citibike.csv"))
    data_mine['one'] = 1 #adds a new column to the data frame 'one'
    #this column serves as a time stamp, in this case the a counter
    #for the number of bike rentals within a 3h block
    data_mine['starttime'] = pd.to_datetime(data_mine.starttime)
    #originally, starttime is in the 8/31/2015 23:53:03 format. This changes
    #it to the 2015-08-31 23:53:03 format.
    data_starttime = data_mine.set_index("starttime")
    #Set the DataFrame index (row labels) using one or more existing columns.
    #This sets the indexing column of the data frame to starttime rather than
    #the standard numerical index
    data_resampled = data_starttime.resample("3h").sum().fillna(0)
    #Downsample the series into 3 minute bins and sum the values of the 
    #timestamps falling into a bin.
    #somehow .resample() removes stoptime, end station name, start station
    #name and usertype from the data frame
    #.sum() sums the cumulative time stamp values within each 3h block
    #.fillna(0) fills NA/NaN values using the specified method, 0 in this case
    return data_resampled.one    

def load_citibike_dense():
    #resamples to 1h instead of 3h for a denser data set
    data_mine = pd.read_csv(os.path.join(DATA_PATH, "citibike.csv"))
    data_mine['one'] = 1
    data_mine['starttime'] = pd.to_datetime(data_mine.starttime)
    data_starttime = data_mine.set_index("starttime")
    data_resampled = data_starttime.resample("1h").sum().fillna(0)
    return data_resampled.one    

def make_signals():
    # fix a random state seed
    rng = random.RandomState(42)
    n_samples = 2000
    time = linspace(0, 8, n_samples)
    # create three signals
    s1 = sin(2 * time)  # Signal 1 : sinusoidal signal
    s2 = sign(sin(3 * time))  # Signal 2 : square signal
    s3 = signal.sawtooth(2 * pi * time)  # Signal 3: saw tooth signal

    # concatenate the signals, add noise
    S = c_[s1, s2, s3]
    S += 0.2 * rng.normal(size=S.shape)

    S /= S.std(axis=0)  # Standardize data
    S -= S.min()
    return S

def make_forge():
    # a carefully hand-designed dataset lol
    X, y = make_blobs(centers=2, random_state=4, n_samples=30)
    y[array([7, 27])] = 0
    mask = ones(len(X), dtype=bool)
    mask[array([0, 1, 5, 26])] = 0
    X, y = X[mask], y[mask]
    return X, y

def make_wave(n_samples=100):
    rnd = random.RandomState(42)
    x = rnd.uniform(-3, 3, size=n_samples)
    y_no_noise = (sin(4 * x) + x)
    y = (y_no_noise + rnd.normal(size=len(x))) / 2
    return x.reshape(-1, 1), y

def load_extended_boston():
    boston = load_boston()
    X = boston.data

    X = MinMaxScaler().fit_transform(boston.data)
    X = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)
    return X, boston.target
    
#X,y=make_blobs(n_samples=1000,centers=10,random_state=0,cluster_std=1)
#Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,random_state=0,train_size=0.5)

#rfc=RandomForestClassifier(n_estimators=100,random_state=0)
#rfc.fit(Xtrain,ytrain)
#yfit=rfc.predict(Xtest)

#forest=RandomForestRegressor(100)
#forest.fit(Xtrain,ytrain)
#yfit=forest.predict(Xtest)

#either use this:
#visualize_classifier(rfc,X,y)
#or this:
#plt.scatter(Xtrain[:,0],Xtrain[:,1],c=ytrain,cmap='rainbow')
#plt.scatter(Xtest[:,0],Xtest[:,1],c=yfit,alpha=0.5,cmap='rainbow')

#mat=confusion_matrix(ytest,yfit)
#sns.heatmap(mat.T,square=True,annot=True,fmt='d',cbar=False)
#plt.xlabel('true label')
#plt.ylabel('predicted label')


#svc=SVC()
#svc.fit(Xtrain,ytrain)
#plt.figure()
#visualize_classifier(svc,X,y)