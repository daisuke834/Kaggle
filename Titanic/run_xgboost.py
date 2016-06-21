import numpy as np
import pandas as pd
from sklearn import grid_search
import xgboost as xgb
import matplotlib.pyplot as plt
import csv
import time

def onehotendoding(_data, _name):
	_data = pd.concat([_data, pd.get_dummies(_data[_name], prefix=_name)], axis=1)
	_data = _data.drop(_name, axis=1)
	return _data

if __name__ == '__main__':
	pd.set_option('display.width', 200)
	#_feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Fare', 'Embarked']
	_feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Fare', 'Embarked']
	_dtype = {'Age':np.float, 'Fare':np.float, 'SibSp':np.float}
	_train_pd = (pd.read_csv('train.csv', dtype=_dtype))[['Survived']+_feature_names]
	_test_pd = pd.read_csv('test.csv', dtype=_dtype)[['PassengerId']+_feature_names]

	#_train_pd = onehotendoding(_train_pd, 'Pclass')
	#_test_pd  = onehotendoding(_test_pd, 'Pclass')
	_train_pd['Sex'] = _train_pd.apply(lambda _x: 1 if _x['Sex']=='male' else 0, axis=1)
	_test_pd['Sex'] = _test_pd.apply(lambda _x: 1 if _x['Sex']=='male' else 0, axis=1)
	#_train_pd = onehotendoding(_train_pd, 'Embarked')
	#_test_pd  = onehotendoding(_test_pd, 'Embarked')
	_train_pd['Age']  = _train_pd['Age'].fillna('-1').astype(np.float)
	_test_pd['Age']   = _test_pd['Age'].fillna('-1').astype(np.float)
	_train_pd['Fare'] = _train_pd['Fare'].fillna('-1').astype(np.float)
	_test_pd['Fare']  = _test_pd['Fare'].fillna('-1').astype(np.float)
	#_train_pd['Age_valid'] = _train_pd.apply(lambda _x: 0 if _x['Age']==-1 else 1, axis=1)
	#_test_pd['Age_valid']  =  _test_pd.apply(lambda _x: 0 if _x['Age']==-1 else 1, axis=1)
	#_train_pd['Fare_valid']= _train_pd.apply(lambda _x: 0 if _x['Fare']==-1 else 1, axis=1)
	#_test_pd['Fare_valid'] =  _test_pd.apply(lambda _x: 0 if _x['Fare']==-1 else 1, axis=1)
	_mean_age = (_train_pd[_train_pd['Age']>0])['Age'].median()
	_mean_fare = (_train_pd[_train_pd['Fare']>=0])['Fare'].median()
	_train_pd['Age'] = _train_pd.apply(lambda _x: _mean_age   if _x['Age']<=0 else _x['Age'], axis=1)
	_test_pd['Age']  = _test_pd.apply( lambda _x: _mean_age   if _x['Age']<=0 else _x['Age'], axis=1)
	_train_pd['Fare'] = _train_pd.apply(lambda _x: _mean_fare if _x['Fare']<0 else _x['Fare'], axis=1)
	_test_pd['Fare']  = _test_pd.apply( lambda _x: _mean_fare if _x['Fare']<0 else _x['Fare'], axis=1)
	#_train_pd['SibSp'] = _train_pd.apply(lambda _x: 1   if _x['SibSp']>0 else 0, axis=1)
	#_test_pd['SibSp']  = _test_pd.apply( lambda _x: 1   if _x['SibSp']>0 else 0, axis=1)
	#_train_pd = _train_pd.drop('Fare', axis=1)
	#_test_pd = _test_pd.drop('Fare', axis=1)
	_train_pd = _train_pd.drop('SibSp', axis=1)
	_test_pd = _test_pd.drop('SibSp', axis=1)
	_train_pd = _train_pd.drop('Embarked', axis=1)
	_test_pd = _test_pd.drop('Embarked', axis=1)
	#Embarked_C Embarked_Q Embarked_S
	#Pclass_1 Pclass_2 Pclass_3
	#SibSp
	_y_train = np.array(_train_pd['Survived'])
	_X_train = np.array(_train_pd.drop('Survived', axis=1))
	_IDs_test= np.array(_test_pd['PassengerId'], dtype=np.int)
	_X_test  = np.array(_test_pd.drop('PassengerId', axis=1))
	print 'train', _X_train.shape
	print 'test', _X_test.shape

	_time_start = time.time()
	_learning_rate_list = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.4]
	#_n_estimators_list = [2, 5, 10, 50, 80, 100, 150, 200, 300, 500]
	_n_estimators_list = [40, 60, 80, 100]
	_param_grid = {'learning_rate':_learning_rate_list, 'n_estimators':_n_estimators_list}
	_grid = grid_search.GridSearchCV(xgb.XGBClassifier(), param_grid=_param_grid, verbose=2, n_jobs=1, cv=3)
	_grid.fit(_X_train, _y_train)
	_time_end = time.time()
	print 'time for learning:'+str(_time_end-_time_start)+'sec ('+str((_time_end-_time_start)/60.0)+'min)'
	print 'Best Score (Valid Set)='+str(_grid.best_score_)+', Best Param='+str(_grid.best_params_)
	_num_of_tree_maxAccuracy = _grid.best_params_['learning_rate']
	_model = _grid.best_estimator_

	_scores = np.ndarray((len(_learning_rate_list), len(_n_estimators_list)), dtype=float)
	for _numest in _n_estimators_list:
		for _parameters, _mean_validation_score, _cv_validation_scores in _grid.grid_scores_:
			print '\t',_parameters, '\t', _mean_validation_score
			_scores[_learning_rate_list.index(_parameters['learning_rate']), _n_estimators_list.index(_parameters['n_estimators'])] = _mean_validation_score

	for _index in range(len(_n_estimators_list)):
		plt.plot(_learning_rate_list, _scores[:, _index], label=str(_n_estimators_list[_index]))
	plt.title('Accuracy at Validation Set')
	plt.xscale('log')
	plt.xlabel('Learning Rate')
	plt.ylabel('Accuracy at Validation Set')
	plt.legend()
	plt.show()

	_predictions = _model.predict(_X_test)
	_output_filename = __file__.split('.')[0] + '.csv'
	_csvWriter = csv.writer(open(_output_filename, 'wb'))
	_csvWriter.writerow(['PassengerId','Survived'])
	for _index in range(len(_IDs_test)):
		_csvWriter.writerow([_IDs_test[_index],_predictions[_index]])
