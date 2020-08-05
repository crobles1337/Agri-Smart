
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
from numpy import genfromtxt
import pandas as pd
import os


# FOR CORN

"NEXT TASKS"

"DEAL WITH THE FACT THAT YOU ARE CREATING A LIST OF LISTS, SO YOU NEED TO LIST COMPREHENSION AND FOR LOOP, INSTEAD OF APPENDING A LIST TO A LIST"
"LOOK UP LIST COMPREHENSION"
"KERAS WORKS VIA https://stackoverflow.com/questions/62465620/error-keras-requires-tensorflow-2-2-or-higher"




# load dataset
#dataframe = read_csv("housing.csv", delim_whitespace=True, header=None)
def OTraining(crop):
	c = 0
	emptycount =0
	# crop = 'Corn'
	mpath = os.path.join(crop, 'data/train')
	tarlist = list()
	hwlist = list()
	sclist = list()
	sslist = list()
	sulist = list()
	npilist = list()

	for fil in os.listdir(mpath): # fil = input
		p1 = os.path.join(mpath, fil)
		for codir in os.listdir(p1): # codir = 2019co
			if not 'YIELD' in codir:
				print("not yield in ", codir)
				p2 = os.path.join(p1, codir)
				tarray = [] # there is one tarray per county
				if 'YIELD.txt' in os.listdir(p2): # check if dataset is labeled
					print("yield.txt labeled in ", p2)
					with open(os.path.join(p2, 'YIELD.txt'), 'r') as y:
						yval = y.read()
					empty = False
					for date in os.listdir(p2): # date  = dated image
						if date != 'YIELD.txt':
							p3 = os.path.join(p2, date)
							for npz in os.listdir(p3):			
								print(os.path.splitext(npz)[0], "npz1")
								if os.path.splitext(npz)[1] == '.npz':
								
									ld = load(os.path.join(p3, npz))
									npi = ld['arr_0']
									print(npi.shape, "shape of ", p3, npz)
									if npi.shape[1] != 0:
										tarray.extend([np.average(npi), np.median(npi), np.quantile(npi, .25), np.quantile(npi, .75)]) # should end at size 4*16*5 = 320
										
										print("succes npz loading")
									else:
										print("shape is 0 for", p3, npz, print(npi.shape))
										empty = True
					npilist.append(len(tarray))
					if empty == False:
						wv = True
						# for each county we will also append
						# GET WEATHER HERE!!!
						p2 = str.replace(p2, 'Corn', 'Wheat', 1)
						p2 = str.replace(p2, 'CoCorn', 'WWWheat')
					
						print(p2)
						if not os.path.exists(p2): 
							print(p2, "doesn't exist")
							p2 = str.replace(p2, 'WWW', 'SWW')
						if not os.path.exists(p2):
							print(p2, "doesn't exist")
							p2 = str.replace(p2, 'SWW', 'DWW')
						if os.path.exists(p2):	
							listdir = os.listdir(p2)				
							print(p2, "this is the new p2")
							if 'HistWeather.csv' in listdir:
								print("truehweathercsv")
								hw = genfromtxt(os.path.join(p2, 'HistWeather.csv'), delimiter=',', skip_header=1, usecols=(1,2,3,4,5))
								Fhw = hw.flatten('F')
								tarray.extend(Fhw)
								hwlist.append(len(Fhw))
								# extract and FLATTEN
							if 'Stress_Count.csv' in listdir:
								print("stresscounttrue")
								sc = genfromtxt(os.path.join(p2, 'Stress_Count.csv'), delimiter=',', skip_header=1)
								Fsc = sc.flatten('F')
								tarray.extend(Fsc)
								sclist.append(len(Fsc))
							if 'SoilStats.npz' in listdir:
								print("soilstatstrue")
								ss0 = load(os.path.join(p2, 'SoilStats.npz'))
								ss = ss0['arr_0']
								Fss = ss.flatten('F')
								tarray.extend(Fss)
								sslist.append(len(Fss))
							if 'SoilUncertainty.npz' in listdir:
								print("soiluncertaintytrue")
								su0 = load(os.path.join(p2, 'SoilUncertainty.npz'))
								su = su0['arr_0']
								Fsu = su.flatten('F')
								tarray.extend(Fsu)
								sulist.append(len(Fsu))
					
							print(len(tarray), "length of tarray")
							#print(tarray.shape)
							"currently the tarray is empty??"
				
							#print(tarray.shape, "shape of tarray")
							tarray.append(float(yval))
						else: 
							print("CANNOT FIND WEATHERSTATS")
							wv = False

						if wv == True:
							if c == 0:
								cf = pd.DataFrame(columns = range(len(tarray)))
								cf.loc[c] = tarray
								c = c + 1
								dim = len(tarray)
							else:
								c = c+1
								cf.loc[c] = tarray
					else:
						print("error due to a value being empty")
						emptycount = emptycount+1
				else:
					print(p2, "yield text file doesn't exist and will not be added to the training set. Current length of training set == ", c)
				tarlist.append(len(tarray))
	print("tarlist", tarlist)
	print("count of 1205's in tarlist", np.count_nonzero(tarlist==1205))
	print("hwlist", hwlist)
	print("sclist", sclist)
	print("sulist", sulist)
	print("sslist", sslist)
	print("npilist", npilist)
	print(len(npilist))
	print(emptycount)
	print(c)
	print(dim)
	print("cf", cf, )#######################################
	savedf(df=cf, crop = crop)
	return cf, dim
# there are some errors b/c there are a lot of empty values rn
#I SHOULD MAKE A NEW FILE THAT DOES SAVE CF, B/C THIS TAKES LIKE 8 MIN TO MAKE EVERY TIME

def savedf(df, crop):
	filename = '{crop}TData.npz'.format(crop=crop)
	df.to_pickle(filename)
	return filename
def loaddf(filename):
	cf = pd.read_pickle(filename)
	return cf	

def runML(crop, cfexists = True):
	if cfexists ==False:
		cf, dim = OTraining(crop)
	else:
		cf = loaddf('{crop}TData.npz'.format(crop=crop))
		dim = len(cf.columns)
	dim = dim-1
	print(dim)
	#X = cf[:, 0:dim]
	#Y = cf[:,dim]
	X = cf.iloc[:, 0:dim]
	Y = cf.iloc[:, dim]
	#bsmodel = baseline_model(dim)
	print("bsmodel")
	estimators = []
	print("estimators")
	estimators.append(('standardize', StandardScaler()))
	print("estiators.append")
	estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=200, batch_size=16, verbose=0)))
	print("estimators.append(mlp")
	pipeline = Pipeline(estimators)
	print("pipeline")
	kfold = KFold(n_splits=10)
	print("kfold")
	results = cross_val_score(pipeline, X, Y, cv=kfold)
	print("results")
	print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))


def baseline_model():
	# create model
	dim = 1204
	model = Sequential()
	model.add(Dense(dim, input_dim=dim, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

runML('Corn')



def saveML():
	#https://machinelearningmastery.com/save-load-keras-deep-learning-models/#:~:text=Save%20Your%20Neural%20Network%20Model%20to%20JSON&text=Keras%20provides%20the%20ability%20to,model%20from%20the%20JSON%20specification.

	None
	return None





# Record : 
# Standardized: -22987.22 (5522.86) MSE w/ batch Size : 8, epocghs 50
# Standardized: -21874.35 (5369.04) MSE w/ batch size: 4, epochs 200