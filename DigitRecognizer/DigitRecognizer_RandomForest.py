import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn import grid_search
import csv
import time
import matplotlib.pyplot as plt
from matplotlib import cm

if __name__ == '__main__':
	_train = np.array(pd.read_csv('train.csv'))
	_X_train = _train[:,1:]
	_y_train = _train[:,0]
	del(_train)

	#_X_train = _X_train[:1000]
	#_y_train = _y_train[:1000]

	_pixel_depth = 255.0
	_X_train_norm = _X_train/_pixel_depth

	print 'X(train):', _X_train.shape
	print 'y(train):', _y_train.shape

	_time_start = time.time()
	_num_of_tree_list = [100, 300, 1000, 3000]
	print 'n_estimators:', len(_num_of_tree_list), _num_of_tree_list
	_param_grid = {'n_estimators':_num_of_tree_list}
	_grid = grid_search.GridSearchCV(ensemble.RandomForestClassifier(), param_grid=_param_grid, verbose=2, n_jobs=4)
	_grid.fit(_X_train_norm, _y_train)
	_time_end = time.time()
	print 'time for learning:'+str(_time_end-_time_start)+'sec ('+str((_time_end-_time_start)/60.0)+'min)'
	print 'Best Score (Valid Set)='+str(_grid.best_score_)+', Best Param='+str(_grid.best_params_)
	_num_of_tree_maxAccuracy = _grid.best_params_['n_estimators']
	_model = _grid.best_estimator_

	_scores = np.ndarray(len(_num_of_tree_list), dtype=float)
	for _parameters, _mean_validation_score, _cv_validation_scores in _grid.grid_scores_:
		print '\t',_parameters, '\t', _mean_validation_score
		_scores[_num_of_tree_list.index(_parameters['n_estimators'])] = _mean_validation_score

	_X_test = np.array(pd.read_csv('test.csv'))
	_X_test_norm = _X_test/_pixel_depth
	_test_predict = _model.predict(_X_test_norm)
	print 'X(test):', _X_test.shape
	_output_filename = __file__.split('.')[0] + '.csv'
	_writecsv = csv.writer(file(_output_filename, 'w'), lineterminator='\n')
	_writecsv.writerow(['ImageId','Label'])
	for _index, _predict in enumerate(_test_predict):
		_writecsv.writerow([_index+1,_predict])

	plt.plot(_num_of_tree_list, _scores)
	plt.title('Accuracy at Validation Set: Random Forest')
	plt.xscale('log')
	plt.xlabel('Number of trees')
	plt.ylabel('Accuracy at Validation Set')
	plt.show()

	_p = np.random.random_integers(0, len(_X_test), 25)
	_samples = _X_test[_p]
	for _index, _sample in enumerate(_samples):
		plt.subplot(5,5,_index+1)
		plt.axis('off')
		plt.imshow(_sample.reshape(28,28), cmap=cm.gray_r, interpolation='nearest')
		_predict = _model.predict(_sample.reshape(1,-1)/_pixel_depth)
		plt.title(str(int(_predict)), color='red')
	plt.show()
