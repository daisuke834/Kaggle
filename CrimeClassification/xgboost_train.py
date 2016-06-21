import pandas as pd
import numpy as np
from sklearn import grid_search
import xgboost as xgb
#from sklearn.svm import LinearSVC
from datetime import datetime as dt
from six.moves import cPickle as pickle
from sklearn.preprocessing import StandardScaler
import os
import time
import matplotlib.pyplot as plt
import csv
import gc
import sys

def data_perapraiton(_df, _features):
	_df['year'] = _df['Dates'].apply(lambda _x: dt.strptime(_x, '%Y-%m-%d %H:%M:%S').year)
	_df['month'] = _df['Dates'].apply(lambda _x: dt.strptime(_x, '%Y-%m-%d %H:%M:%S').month)
	_df['day'] = _df['Dates'].apply(lambda _x: dt.strptime(_x, '%Y-%m-%d %H:%M:%S').day)
	_df['hour'] = _df['Dates'].apply(lambda _x: dt.strptime(_x, '%Y-%m-%d %H:%M:%S').hour)
	_df = _df.drop('Dates', axis=1)

	_df = _df.join(pd.get_dummies(_df['DayOfWeek']))
	_df = _df.drop('DayOfWeek', axis=1)
	_df = _df.join(pd.get_dummies(_df['PdDistrict']))
	_df = _df.drop('PdDistrict', axis=1)
	return np.array(_df[_features])

if __name__ == '__main__':
	pd.set_option('display.width', 200)
	_file_name_data = 'data.pickle'
	_y_names = ['ARSON', 'ASSAULT', 'BAD CHECKS', 'BRIBERY', 'BURGLARY', 'DISORDERLY CONDUCT', 'DRIVING UNDER THE INFLUENCE', \
		'DRUG/NARCOTIC', 'DRUNKENNESS', 'EMBEZZLEMENT', 'EXTORTION', 'FAMILY OFFENSES', 'FORGERY/COUNTERFEITING', 'FRAUD', \
		'GAMBLING', 'KIDNAPPING', 'LARCENY/THEFT', 'LIQUOR LAWS', 'LOITERING', 'MISSING PERSON', 'NON-CRIMINAL', 'OTHER OFFENSES', \
		'PORNOGRAPHY/OBSCENE MAT', 'PROSTITUTION', 'RECOVERED VEHICLE', 'ROBBERY', 'RUNAWAY', 'SECONDARY CODES', 'SEX OFFENSES FORCIBLE', \
		'SEX OFFENSES NON FORCIBLE', 'STOLEN PROPERTY', 'SUICIDE', 'SUSPICIOUS OCC', 'TREA', 'TRESPASS', 'VANDALISM', 'VEHICLE THEFT', \
		'WARRANTS', 'WEAPON LAWS']
	_features = ['year', 'month', 'day', 'hour', \
		'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', \
		'SOUTHERN', 'MISSION', 'NORTHERN', 'BAYVIEW', 'CENTRAL', 'TENDERLOIN', 'INGLESIDE', 'TARAVAL', 'PARK', 'RICHMOND', \
		'X', 'Y' ]

	if os.path.exists(_file_name_data):
		with open(_file_name_data, "rb") as _fh:
			_all_data = pickle.load(_fh)
			_X = _all_data['_X']
			_X_test = _all_data['_X_test']
			_y = _all_data['_y']
	else:
		_train_df = pd.read_csv('train.csv')
		_test_df = pd.read_csv('test.csv')
		_train_df['label'] = _train_df['Category'].apply(lambda _x: _y_names.index(_x))
		_y = np.array(_train_df['label'], dtype=np.int)
		_X = data_perapraiton(_train_df, _features)
		_index_data = np.random.permutation(len(_X))
		_X = _X[_index_data]
		_y = _y[_index_data]
		_X_test = data_perapraiton(_test_df, _features)
		_scaler = StandardScaler()
		_scaler.fit(_X)
		_X = _scaler.transform(_X)
		_X_test = _scaler.transform(_X_test)
		_all_data = {'_X':_X, '_X_test':_X_test, '_y':_y}
		with open(_file_name_data, 'wb') as _fh:
			pickle.dump(_all_data, _fh, pickle.HIGHEST_PROTOCOL)
		del(_train_df); del(_test_df); del(_index_data);

	print '_X',_X.shape
	print '_X_test', _X_test.shape
	print '_y',_y.shape
	#_X = _X[:10000]
	#_y = _y[:10000]

	_learning_rate_list = [0.3]
	_n_estimators_list = [100, 300, 1000, 3000]
	#_learning_rate_list = [0.1, 0.3, 1.0]
	#_n_estimators_list = [20, 50, 100]
	#_learning_rate_list = [0.01, 0.03 ,0.1, 0.3, 1.0]
	#_n_estimators_list = [40, 60, 80, 100]

	gc.collect()
	if len(_learning_rate_list) * len(_n_estimators_list) == 1:
		_best_learning_rate = _learning_rate_list[0]
		_best_n_estimators = _n_estimators_list[0]
		print 'Does not perform Grid Search'
	else:
		print '***Start Grid Search***'
		sys.stdout.flush()
		_time_start = time.time()
		_param_grid = {'learning_rate':_learning_rate_list, 'n_estimators':_n_estimators_list}
		_grid = grid_search.GridSearchCV(xgb.XGBClassifier(), param_grid=_param_grid, verbose=2, n_jobs=1, cv=5)
		_grid.fit(_X, _y)
		with open(__file__.split('.')[0]+'_grid_search.pickle', 'wb') as _fh:
			pickle.dump(_grid, _fh, pickle.HIGHEST_PROTOCOL)
		_time_GS_end = time.time()
		print 'time for Grid Search:'+str(_time_GS_end-_time_start)+'sec ('+str((_time_GS_end-_time_start)/60.0)+'min)'
		print 'Best Score (Valid Set)='+str(_grid.best_score_)+', Best Param='+str(_grid.best_params_)
		_best_learning_rate = _grid.best_params_['learning_rate']
		_best_n_estimators = _grid.best_params_['n_estimators']

		_scores = np.ndarray((len(_learning_rate_list), len(_n_estimators_list)), dtype=float)
		for _numest in _n_estimators_list:
			for _parameters, _mean_validation_score, _cv_validation_scores in _grid.grid_scores_:
				print '\t',_parameters, '\t', _mean_validation_score
				_scores[_learning_rate_list.index(_parameters['learning_rate']), _n_estimators_list.index(_parameters['n_estimators'])] = _mean_validation_score
		sys.stdout.flush()

		for _index in range(len(_learning_rate_list)):
			plt.plot(_n_estimators_list, _scores[_index,:], label='alpha='+str(_learning_rate_list[_index]))
		plt.legend(loc='lower right')
		plt.title('Accuracy at Validation Set')
		plt.xscale('log')
		plt.xlabel('# of estiamtors')
		plt.ylabel('Accuracy at Validation Set')
		plt.savefig(__file__.split('.')[0]+'_accuracy.png')
		plt.clf()
		with open(__file__.split('.')[0]+'_scores.pickle', 'wb') as _fh:
			pickle.dump(_scores, _fh, pickle.HIGHEST_PROTOCOL)
		del(_grid); del(_scores);
		gc.collect()
		print '***End Grid Search***'
		sys.stdout.flush()

	print '***Start Learning on Best Parameter***'
	_time_Learn_start = time.time()
	print '_best_learning_rate', _best_learning_rate
	print '_best_n_estimators', _best_n_estimators
	_model = xgb.XGBClassifier(learning_rate=_best_learning_rate, n_estimators=_best_n_estimators)
	_model.fit(_X, _y)
	_time_Learn_end   = time.time()
	print 'time for Learning:'+str(_time_Learn_end-_time_Learn_start)+'sec ('+str((_time_Learn_end-_time_Learn_start)/60.0)+'min)'
	with open(__file__.split('.')[0]+'_model.pickle', 'wb') as _fh:
		pickle.dump(_model, _fh, pickle.HIGHEST_PROTOCOL)
	print '***End Learning on Best Parameter***'
	sys.stdout.flush()

	print '***Start Predicting***'
	_time_prediction_start = time.time()
	_predictions_proba = _model.predict_proba(_X_test).T
	_output_filename = __file__.split('.')[0] + '.csv'
	_csvWriter = csv.writer(open(_output_filename, 'wb'))
	_csvWriter.writerow(['Id']+_y_names)
	_time_prediction_end = time.time()
	print len(_predictions_proba)
	for _index in range(len(_predictions_proba)):
		_elem = [_index]
		for _col in range(len(_y_names)):
			_elem.append(_predictions_proba[_index,_col])
		_csvWriter.writerow(_elem)
	print 'time for prediction:'+str(_time_prediction_end-_time_prediction_start)+'sec ('+str((_time_prediction_end-_time_prediction_start)/60.0)+'min)'
