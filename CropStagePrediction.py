
# Regression Example With Boston Dataset: Baseline
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from numpy import load
import os
import pandas as pd
from keras.metrics import binary_accuracy
import tensorflow as tf
# IMPORT NPZ LOAD

#General concept, organize images by date and then use SAVI and NDRE to predict if its 1-5 image throughout the year. I can roughly translate this
# NDRE we have
# I could just use NDRE and NDVI for this, save myself so much time, and memory
# SAVI requires NIR and Red so we can use it, I just have not yet extracted it.

"DECIDED I WILL USE NDVI AND NDRE FOR THIS"

# organize data as simply 9 values, first 6 are mean, 25 percentile, median, 75 percentile for ndvi and ndre, and then 9th is "stage"
# it should do this pretty roughly
# input as simple classification (NOT REGRESSION) between 4 (or 5 stages)
#

def LoadData(crop):
    # create a panda data set that we will be appending to
    columns = 9
    count = 0
    df = pd.DataFrame(columns=range(columns))
    mpath = os.path.join(crop, 'data/train')
    for item in os.listdir('Corn/data/train'):
        p1 = os.path.join(mpath, item)
        for fold in os.listdir(p1): # were at county level now
            p2 = os.path.join(p1, fold)
            for date in os.listdir(p2):
                if date != 'YIELD.txt':
                    tlist = []
                    indexreal = True
                    p3 = os.path.join(p2, date)
                    for index in os.listdir(p3):
                        
                    
                        print(index)
                        if index == 'NDRE.npz':
                            print("truendre")
                            l = load(os.path.join(p3, 'NDRE.npz'))
                            l = l['arr_0']
                            if l.size!=0:
                                ndre = [np.average(l), np.quantile(l, .25), np.quantile(l, .5), np.quantile(l, .75)]
                                tlist.extend(ndre)
                                #indexreal = True
                                #load
                            else:
                                indexreal=False
                
                        if index == 'NDVI.npz':
                            print("truendvi")
                            l = load(os.path.join(p3, 'NDVI.npz'))
                            l = l['arr_0']
                            if l.size!=0:
                                ndvi = [np.average(l), np.quantile(l, .25), np.quantile(l, .5), np.quantile(l, .75)] 
                                print(len(ndvi), "length of ndvi")
                                tlist.extend(ndvi)
                                #indexreal = True
                            else:
                                indexreal = False
                
                    # append to a row in the 
                    print(date, "fold")
                    if '201903' in date:
                        tlist.extend([1]) # append to the last column in the appropraite row
                    if '201904' in date:
                        tlist.extend([1]) # th
                    if '201905' in date:
                        tlist.extend([2]) # th
                    if '201906' in date:
                        tlist.extend([3]) # th
                    if '201907' in date:
                        tlist.extend([4]) # th
                    if '201908' in date:
                        tlist.extend([4]) # th
              #      if indexreal == True:
                    print(len(tlist), "length of tlist")
                    if indexreal==True:
                        df.loc[count] = tlist
                        count = count+1
                        print(count)
    savedf(df, crop)
    return df, columns

def CSPTrain(crop, save = True):
    if save == False:
        cf, dim = LoadData(crop)
    else:
        cf = loaddf('{crop}CStageData.npz'.format(crop=crop))
    
    dim = dim = 8
    print(dim)
    X = cf.iloc[:, 0:dim]
    Y = cf.iloc[:, dim]
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=10)
    results = cross_val_score(pipeline, X, Y, cv=kfold)
    print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    print(results)


# should be multiclass
#but mean squared error is also useful



def ogbaseline_model():
	# create model
	dim = 8
	model = Sequential()
	model.add(Dense(dim, input_dim=dim, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

def baseline_model():
	# create model
	dim = 8
	model = Sequential()
	model.add(Dense(dim, input_dim=dim, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mse', optimizer='adam', metrics = [tf.keras.metrics.CategoricalAccuracy()])
	return model

def savedf(df, crop):
	filename = '{crop}CStageData.npz'.format(crop=crop)
	df.to_pickle(filename)
	return filename
def loaddf(filename):
	cf = pd.read_pickle(filename)
	return cf	


CSPTrain('Corn', save=True)




"for tomorrow"
"README"
"Actually add shapefile/coordinate/tif extraction"
"Upload to github"
"Understand how MSE cross validation scoring is happening"
"How to score with multi-class"
"How to improve results"
"Clean up functions, code, everything"


#Standardized: -1.03 (0.46) MSE
#[-0.5959164  -0.52216208 -0.86101267 -0.80812834 -1.14755144 -2.10266247
# -0.88318952 -0.71464738 -1.59003106 -1.03631257]