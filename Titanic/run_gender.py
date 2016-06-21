import numpy as np
import pandas as pd
from sklearn import grid_search
from sklearn import ensemble
import matplotlib.pyplot as plt
import csv
import time

def onehotendoding(_data, _name):
	_data = pd.concat([_data, pd.get_dummies(_data[_name], prefix=_name)], axis=1)
	_data = _data.drop(_name, axis=1)
	return _data

if __name__ == '__main__':
	#_feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Fare', 'Embarked']
	_feature_names = ['Sex']
	#_dtype = {'Age':np.float, 'Fare':np.float}
	_train_pd = (pd.read_csv('train.csv'))[['Survived']+_feature_names]
	_test_pd = pd.read_csv('test.csv')[['PassengerId']+_feature_names]
	
	_train_pd = onehotendoding(_train_pd, 'Sex')
	_test_pd  = onehotendoding(_test_pd, 'Sex')

	_y_train = np.array(_train_pd['Survived'])
	_X_train = np.array(_train_pd.drop('Survived', axis=1))
	_IDs_test= np.array(_test_pd['PassengerId'], dtype=np.int)
	_X_test  = np.array(_test_pd.drop('PassengerId', axis=1))

	_time_start = time.time()
	_num_of_tree_list = [30,40, 50, 60, 70, 80,90, 100, 110, 120, 130, 140, 150, 200, 250, 300, 1000]
#	_num_of_tree_list = [_i for _i in range(20,100,4)]
#	_num_of_tree_list = [_i for _i in range(10,20,1)]
	print 'n_estimators:', len(_num_of_tree_list), _num_of_tree_list
	_param_grid = {'n_estimators':_num_of_tree_list}
	_grid = grid_search.GridSearchCV(ensemble.RandomForestClassifier(), param_grid=_param_grid, verbose=2, n_jobs=4, cv=10)
	_grid.fit(_X_train, _y_train)
	_time_end = time.time()
	print 'time for learning:'+str(_time_end-_time_start)+'sec ('+str((_time_end-_time_start)/60.0)+'min)'
	print 'Best Score (Valid Set)='+str(_grid.best_score_)+', Best Param='+str(_grid.best_params_)
	_num_of_tree_maxAccuracy = _grid.best_params_['n_estimators']
	_model = _grid.best_estimator_

	_scores = np.ndarray(len(_num_of_tree_list), dtype=float)
	for _parameters, _mean_validation_score, _cv_validation_scores in _grid.grid_scores_:
		print '\t',_parameters, '\t', _mean_validation_score
		_scores[_num_of_tree_list.index(_parameters['n_estimators'])] = _mean_validation_score

	plt.plot(_num_of_tree_list, _scores)
	plt.title('Accuracy at Validation Set: Random Forest')
	plt.xscale('log')
	plt.xlabel('Number of trees')
	plt.ylabel('Accuracy at Validation Set')
	plt.show()

	_predictions = _model.predict(_X_test)
	_csvWriter = csv.writer(open('output.csv', 'wb'))
	_csvWriter.writerow(['PassengerId','Survived'])
	for _index in range(len(_IDs_test)):
		_csvWriter.writerow([_IDs_test[_index],_predictions[_index]])
