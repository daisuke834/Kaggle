import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime as dt
from six.moves import cPickle as pickle
import os
import time
import matplotlib.pyplot as plt
import csv
import gc
import sys

_batch_size = 128
_max_epoch = 20
_num_steps = int(878049/_batch_size*_max_epoch)
#_num_steps = 21
_num_hidden1 = 64
_num_hidden2 = 64
_batch_size_eval = 8192


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
	_time_start = time.time()
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

	with open(_file_name_data, "rb") as _fh:
		_all_data = pickle.load(_fh)
	_X = _all_data['_X']
	_X_test = _all_data['_X_test']
	_y = _all_data['_y']
	_num_labels = len(_y_names)
	_y = (np.arange(_num_labels) == _y[:,None]).astype(np.float32)

	print '_X',_X.shape
	print '_X_test', _X_test.shape
	print '_y',_y.shape

	_num_of_trainvalidset = _X.shape[0]
	_num_of_features = _X.shape[1]
	_num_test_dataset = _X_test.shape[0]

	_subset_offset = [0]
	_num_of_folds = 5
	for _fold_index in range(_num_of_folds):
		_subset_offset.append(int(_num_of_trainvalidset*(_fold_index+1)/_num_of_folds))
	print _subset_offset

	#_meta_features = None
	_X_meta_features = None
	_X_test_meta_features = np.zeros((_num_test_dataset, _num_labels))
	_meta_features_name = []
	for _name in _y_names:
		_meta_features_name.append('meta_'+_name)
	_validation_loss = np.zeros(_num_of_folds)

	for _fold_index in range(_num_of_folds+1):
		if _fold_index==_num_of_folds:
			# Training and Meta Feature of Test Set
			print 'Training with all data set'
			_X_train = _X.copy()
			_y_train = _y.copy()
			_X_valid = None
			_y_valid = None
			print 'X_train:', _X_train.shape
			print 'y_train:', _y_train.shape
			print 'X_test', _X_test.shape
			_num_train_dataset = _X_train.shape[0]
			_num_valid_dataset = None
			_current_meta_features = None
			gc.collect()
		else:
			# Cross Validation and Meta Feature of training set
			print 'n-folds Cross Validation'
			_valid_start_offset = _subset_offset[_fold_index]
			_valid_end_offset = _subset_offset[_fold_index+1]
			_X_train = _X[0:_valid_start_offset]
			_X_train = np.vstack((_X_train, _X[_valid_end_offset:]))
			_y_train = _y[0:_valid_start_offset]
			_y_train = np.vstack((_y_train, _y[_valid_end_offset:]))
			_X_valid = _X[_valid_start_offset:_valid_end_offset]
			_y_valid = _y[_valid_start_offset:_valid_end_offset]
			print 'X_train:', _X_train.shape
			print 'y_train:', _y_train.shape
			print 'X_valid:', _X_valid.shape
			print 'y_valid:', _y_valid.shape
			print 'total', _X_train.shape[0]+_X_valid.shape[0]
			_num_train_dataset = _X_train.shape[0]
			_num_valid_dataset = _X_valid.shape[0]
			_current_meta_features = np.zeros((_num_valid_dataset, _num_labels))
			gc.collect()

		_graph = tf.Graph()
		with _graph.as_default():
			_tf_alpha = tf.placeholder(tf.float32)
			_tf_keep_prob = tf.placeholder(tf.float32)
			_tf_X = tf.placeholder(tf.float32, shape=(_batch_size, _num_of_features))
			_tf_y = tf.placeholder(tf.float32, shape=(_batch_size, _num_labels))

			_tf_X_eval = tf.placeholder(tf.float32, shape=(_batch_size_eval, _num_of_features))
			_tf_y_eval = tf.placeholder(tf.float32, shape=(_batch_size_eval, _num_labels))

			_weights1 = tf.Variable(tf.truncated_normal([_num_of_features, _num_hidden1], stddev=0.1), dtype=tf.float32)
			_biases1 = tf.Variable(tf.constant(1.0, shape=[_num_hidden1]), dtype=tf.float32)
			_weights2 = tf.Variable(tf.truncated_normal([_num_hidden1, _num_hidden2], stddev=0.1), dtype=tf.float32)
			_biases2 = tf.Variable(tf.constant(1.0, shape=[_num_hidden2]), dtype=tf.float32)
			_weights3 = tf.Variable(tf.truncated_normal([_num_hidden2, _num_labels], stddev=0.1), dtype=tf.float32)
			_biases3 = tf.Variable(tf.constant(1.0, shape=[_num_labels]), dtype=tf.float32)

			def model():
				_fully_connect1 = tf.nn.relu(tf.matmul(_tf_X, _weights1)+_biases1)
				_fully_connect2 = tf.nn.dropout(tf.nn.relu(tf.matmul(_fully_connect1, _weights2)+_biases2), _tf_keep_prob)
				_read_out = tf.matmul(_fully_connect2, _weights3)+_biases3
				return _read_out
			def model_eval():
				_fully_connect1 = tf.nn.relu(tf.matmul(_tf_X_eval, _weights1)+_biases1)
				_fully_connect2 = tf.nn.relu(tf.matmul(_fully_connect1, _weights2)+_biases2)
				_read_out = tf.matmul(_fully_connect2, _weights3)+_biases3
				return _read_out
			def predict_proba():
				return tf.nn.softmax(model_eval())
			def calc_loss(_num_of_data):
				_predicts = tf.slice(model_eval(), [0,0], [_num_of_data,-1])
				_ys = tf.slice(_tf_y_eval, [0,0], [_num_of_data,-1])
				return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(_predicts,_ys))

			#_tf_global_step = tf.Variable(0, trainable=False)
			#_rate = tf.train.exponential_decay(_tf_alpha, global_step=_tf_global_step, decay_steps=10, decay_rate=0.99)
			_tf_logits = model()
			_tf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(_tf_logits, _tf_y))
			#_tf_optimizer = tf.train.AdamOptimizer(_rate).minimize(_tf_loss)
			_tf_optimizer = tf.train.AdamOptimizer(_tf_alpha).minimize(_tf_loss)
			#_tf_optimizer = tf.train.RMSPropOptimizer(_tf_alpha).minimize(_tf_loss)
			_tf_prediction = tf.nn.softmax(_tf_logits)

		#***Learning****************************
		print '*****************************'
		print 'Start Learning'
		_time_start = time.time()
		_learning_curve_mini_batch_count = []
		_learning_curve_Deviation = []

		with tf.Session(graph=_graph) as _session:
			_session.run(tf.initialize_all_variables())
			for _step in range(_num_steps):
				_offset = (_step * _batch_size) % (_num_train_dataset - _batch_size)
				_epoch = float(_step) * _batch_size / _num_train_dataset
				if _offset==0:
					_array_index_rand = np.random.permutation(len(_X_train))
					_X_train = _X_train[_array_index_rand]
					_y_train = _y_train[_array_index_rand]
				_batch_X = _X_train[_offset:(_offset + _batch_size), :]
				_batch_y = _y_train[_offset:(_offset + _batch_size)]
				_feed_dict = {_tf_X:_batch_X, _tf_y:_batch_y, _tf_alpha:0.001, _tf_keep_prob:0.5}
				_, _loss, _predictions = _session.run([_tf_optimizer, _tf_loss, _tf_prediction], feed_dict=_feed_dict)
				if _step%100 ==0:
					print 'Learning Curve\t', _fold_index, '\t', _step, '\t', _epoch, '\t', _loss
				_learning_curve_mini_batch_count.append(_epoch)
				_learning_curve_Deviation.append(_loss)

			if _fold_index!=_num_of_folds:
				print '***Prediction of ValidSet***'
				_num_of_step_valid_prediction = int((_num_valid_dataset+_batch_size_eval-1)/_batch_size_eval)
				_sum_loss = 0.0
				for _step in range(_num_of_step_valid_prediction):
					_offset = _step * _batch_size_eval
					_batch_X = _X_valid[_offset:(_offset + _batch_size_eval), :]
					_batch_y = _y_valid[_offset:(_offset + _batch_size_eval)]
					_real_batch_size = len(_batch_X)
					if _real_batch_size != _batch_size_eval:
						_padding = np.zeros((_batch_size_eval-_real_batch_size, _num_of_features))
						_batch_X = np.vstack((_batch_X, _padding))
						_padding = np.zeros((_batch_size_eval-_real_batch_size, _num_labels))
						_batch_y = np.vstack((_batch_y, _padding))
					_feed_dict = {_tf_X_eval:_batch_X, _tf_keep_prob:1}
					_batch_predict = _session.run(predict_proba(), feed_dict=_feed_dict)[:_real_batch_size]
					_current_meta_features[_offset:(_offset + _real_batch_size)] = _batch_predict
					_feed_dict = {_tf_X_eval:_batch_X, _tf_y_eval:_batch_y, _tf_keep_prob:1}
					_loss = _session.run(calc_loss(_real_batch_size), feed_dict=_feed_dict)
					_sum_loss = _sum_loss + _loss * float(_real_batch_size)/float(_batch_size_eval)
				_validation_loss[_fold_index] = _sum_loss/_num_of_step_valid_prediction
				if _X_meta_features is None:
					_X_meta_features = _current_meta_features.copy()
				else:
					_X_meta_features = np.vstack((_X_meta_features, _current_meta_features))
				plt.plot(_learning_curve_mini_batch_count, _learning_curve_Deviation, label=str(_fold_index))
				plt.xlabel('Epoch')
				plt.ylabel('Cross Entropy')
				plt.title('Learning Curve of training sets')
				plt.legend()
				plt.savefig('LearningCurve.png')
			else:
				print '***Prediction of TestSet***'
				_num_of_step_test_prediction = int((_num_test_dataset+_batch_size_eval-1)/_batch_size_eval)
				for _step in range(_num_of_step_test_prediction):
					_offset = _step * _batch_size_eval
					_batch_X = _X_test[_offset:(_offset + _batch_size_eval), :]
					_real_batch_size = len(_batch_X)
					if _real_batch_size != _batch_size_eval:
						_padding = np.zeros((_batch_size_eval-_real_batch_size, _num_of_features))
						_batch_X = np.vstack((_batch_X, _padding))
					_feed_dict = {_tf_X_eval:_batch_X, _tf_keep_prob:1}
					_batch_predict = _session.run(predict_proba(), feed_dict=_feed_dict)[:_real_batch_size]
					_X_test_meta_features[_offset:(_offset + _real_batch_size)] = _batch_predict

	plt.clf()

	for _fold_index in range(_num_of_folds):
		print 'fold='+str(_fold_index), '\tValidLoss='+str(_validation_loss[_fold_index])

	_X_meta = np.hstack((_X, _X_meta_features))
	_X_test_meta = np.hstack((_X_test, _X_test_meta_features))
	_all_data_meta = {'_X_meta':_X_meta, '_X_test_meta':_X_test_meta}
	with open('X_meta.pickle', 'wb') as _fh:
		pickle.dump(_all_data_meta, _fh, pickle.HIGHEST_PROTOCOL)


	print '***Start Predicting***'
	_output_filename = 'nn_train.csv'
	_csvWriter = csv.writer(open(_output_filename, 'wb'))
	_csvWriter.writerow(['Id']+_y_names)
	_time_end = time.time()
	print len(_X_test_meta_features)
	for _index in range(len(_X_test_meta_features)):
		_elem = [_index]
		for _col in range(len(_y_names)):
			_elem.append(_X_test_meta_features[_index,_col])
		_csvWriter.writerow(_elem)
	print 'time for prediction:'+str(_time_end-_time_start)+'sec ('+str((_time_end-_time_start)/60.0)+'min)'
