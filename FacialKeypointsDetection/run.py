import numpy as np
import pandas as pd
import tensorflow as tf
import csv
import time
import matplotlib.pyplot as plt
from matplotlib import cm
import json
import gc



if __name__ == '__main__':
	pd.set_option('display.width', 200)
	_image_size = 96
	_pixel_depth = 255.0
	_y_name_all = ['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y', 'left_eye_inner_corner_x', 'left_eye_inner_corner_y', \
		'left_eye_outer_corner_x', 'left_eye_outer_corner_y', 'right_eye_inner_corner_x', 'right_eye_inner_corner_y', 'right_eye_outer_corner_x', 'right_eye_outer_corner_y', \
		'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y', 'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y', 'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y', \
		'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y', 'nose_tip_x', 'nose_tip_y', 'mouth_left_corner_x', 'mouth_left_corner_y', 'mouth_right_corner_x', 'mouth_right_corner_y', \
		'mouth_center_top_lip_x', 'mouth_center_top_lip_y', 'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y']
	_dict_grid_param = json.loads(open('parameters.json', 'r').read())
	with open('parameters_back.json', 'w') as _fh:
		json.dump(_dict_grid_param, _fh, indent=4)
	#_y_name_all = ['left_eye_inner_corner_x']
	_train_df = pd.read_csv('training.csv')
	_test_df = pd.read_csv('test.csv')
	_ImageId_test = list(np.array(_test_df['ImageId'], dtype=np.int))
	_X_test = np.array([_x.split(' ') for _x in np.array(_test_df['Image'])], dtype=np.float)
	_num_test_dataset = len(_test_df)
	del(_test_df)
	print '_X_test', _X_test.shape
	_X_test_norm = (_X_test - _X_test.mean(axis=1).reshape(-1,1) ) / _X_test.std(axis=1).reshape(-1,1)
	_X_test_norm = _X_test_norm.reshape(-1, _image_size,_image_size, 1)
	_predicts_test = np.ndarray((_num_test_dataset, len(_y_name_all)), dtype=np.float)

	_y_mean = dict()

	#***************************************************************************
	for _y_index, _y_name in enumerate(_y_name_all):
		print '***********************'
		#_num_y=len(_y_name)
		_num_y=1
		print 'y_name=', _y_name
		_ctrain_df = _train_df[[_y_name] + ['Image']].dropna()
		_num_dataset = len(_ctrain_df)
		_X = np.array([_x.split(' ') for _x in np.array(_ctrain_df['Image'])], dtype=np.float)
		_y = np.array(_ctrain_df[_y_name]).reshape(_num_dataset,_num_y)
		_y_mean[_y_name] = np.mean(_y)
		print '_X', _X.shape
		print '_y', _y.shape
		del(_ctrain_df)
		gc.collect()

		#Data Preparation
		_num_dataset = len(_X)
		_valid_start_index = int(_num_dataset*0.8)
		_array_index_rand = np.random.permutation(_num_dataset)
		_X_train = _X[_array_index_rand[:_valid_start_index]]
		_y_train = _y[_array_index_rand[:_valid_start_index]]
		_X_valid = _X[_array_index_rand[_valid_start_index:]]
		_y_valid = _y[_array_index_rand[_valid_start_index:]]
		del(_array_index_rand);
		del(_X);
		del(_y);
		_X_train_norm = (_X_train-_X_train.mean(axis=1).reshape(-1,1)) / _X_train.std(axis=1).reshape(-1,1)
		_X_valid_norm = (_X_valid-_X_valid.mean(axis=1).reshape(-1,1)) / _X_valid.std(axis=1).reshape(-1,1)
		_X_train_norm = _X_train_norm.reshape(-1, _image_size,_image_size, 1)
		_X_valid_norm = _X_valid_norm.reshape(-1, _image_size,_image_size, 1)
		_y_train_norm = _y_train/float(_image_size)
		_y_valid_norm = _y_valid/float(_image_size)
		del(_X_train); del(_X_valid);
		del(_y_train); del(_y_valid);
		gc.collect()

		_num_train_dataset = len(_X_train_norm)
		_num_valid_dataset = len(_X_valid_norm)
		print 'number of train dataset:', _num_train_dataset
		print 'number of valid dataset:', _num_valid_dataset

		#*********Definition ********************
		#_num_steps = 3001
		_num_steps_grid_search = 1501
		_num_steps_training = 1501
		#_num_steps = 11
		_batch_size = 128
		#_dummy_batch_labeldata = np.zeros((_batch_size,_num_y))
		_patch_size = 5
		#_patch_size = 10
		_depth = 32
		_num_hidden = 64
		#_num_hidden = 512
		_graph = tf.Graph()
		with _graph.as_default():
			_tf_lambda = tf.placeholder(tf.float32)
			_tf_alpha = tf.placeholder(tf.float32)
			_tf_keep_prob = tf.placeholder(tf.float32)
			_tf_X = tf.placeholder(tf.float32, shape=(_batch_size, _image_size, _image_size, 1))
			_tf_y = tf.placeholder(tf.float32, shape=(_batch_size, _num_y))

			_weights1 = tf.Variable(tf.truncated_normal([_patch_size, _patch_size, 1, _depth], stddev=0.1), dtype=tf.float32)
			_biases1 = tf.Variable(tf.zeros([_depth]), dtype=tf.float32)
			_weights2 = tf.Variable(tf.truncated_normal([_patch_size, _patch_size, _depth, _depth], stddev=0.1), dtype=tf.float32)
			_biases2 = tf.Variable(tf.constant(1.0, shape=[_depth]), dtype=tf.float32)
			_weights3 = tf.Variable(tf.truncated_normal([_image_size//4 * _image_size//4 * _depth, _num_hidden], stddev=0.1), dtype=tf.float32)
			_biases3 = tf.Variable(tf.constant(1.0, shape=[_num_hidden]), dtype=tf.float32)
			_weights4 = tf.Variable(tf.truncated_normal([_num_hidden, _num_y], stddev=0.1), dtype=tf.float32)
			_biases4 = tf.Variable(tf.constant(1.0, shape=[_num_y]), dtype=tf.float32)

			def model():
				_conv1 = tf.nn.relu(tf.nn.conv2d(_tf_X,  _weights1, strides=[1,1,1,1], padding='SAME') + _biases1)
				_pool1 = tf.nn.max_pool(_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
				_conv2 = tf.nn.relu(tf.nn.conv2d(_pool1, _weights2, strides=[1,1,1,1], padding='SAME') + _biases2)
				_pool2 = tf.nn.max_pool(_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
				_shape = _pool2.get_shape().as_list()
				_pool2_flat = tf.reshape(_pool2, [_shape[0], _shape[1]*_shape[2]*_shape[3]])
				_fully_connect1 = tf.nn.dropout(tf.nn.relu(tf.matmul(_pool2_flat, _weights3)+_biases3), _tf_keep_prob)
				_read_out = tf.matmul(_fully_connect1, _weights4)+_biases4
				return _read_out
			def calc_loss(_num_of_data):
				_predicts = tf.slice(model(), [0,0], [_num_of_data,-1])
				_ys = tf.slice(_tf_y, [0,0], [_num_of_data,-1])
				return tf.reduce_mean(tf.square(_predicts - _ys))

			_tf_prediction = model()
			_tf_loss = tf.reduce_mean(tf.square(_tf_prediction - _tf_y))/2.0
			_tf_loss_reg = _tf_loss + 0.5 * _tf_lambda * (tf.nn.l2_loss(_weights1) + tf.nn.l2_loss(_weights2) + tf.nn.l2_loss(_weights3) + tf.nn.l2_loss(_weights4))
			#_tf_optimizer = tf.train.AdagradOptimizer(_tf_alpha).minimize(_tf_loss_reg)
			_tf_optimizer = tf.train.AdamOptimizer(_tf_alpha).minimize(_tf_loss_reg)

		#***Learning****************************
		print '*****************************'
		print 'Start Learning'
		_time_start = time.time()
		_alpha_list = _dict_grid_param[_y_name][0]
		_lambda_list = _dict_grid_param[_y_name][1]
		#_alpha_list = np.array([0.03, 0.06, 0.1, 0.3])
		#_lambda_list = np.array([1e-5, 3e-5, 6e-5, 1e-4])
		#_alpha_list = np.logspace(-2,1,7)
		#_lambda_list = np.logspace(-5,-1,9)
		if len(_alpha_list)*len(_lambda_list) == 1:
			print 'Only one parameter set is selected.'
			_best_Deviation_valid = None
			_best_alpha = _alpha_list[0]
			_best_lambda = _lambda_list[0]
		else:
			print '***Grid Search***'
			_Deviation_list = np.ndarray( (len(_alpha_list), len(_lambda_list) ), dtype=float)
			for _al_index, _alpha in enumerate(_alpha_list):
				for _lam_index, _lambda in enumerate(_lambda_list):
					with tf.Session(graph=_graph) as _session:
						_session.run(tf.initialize_all_variables())
						for _step in range(_num_steps_grid_search):
							_offset = (_step * _batch_size) % (_num_train_dataset - _batch_size)
							if _offset==0:
								_array_index_rand = np.random.permutation(len(_X_train_norm))
								_X_train_norm = _X_train_norm[_array_index_rand]
								_y_train_norm = _y_train_norm[_array_index_rand]
							_batch_X = _X_train_norm[_offset:(_offset + _batch_size), :]
							_batch_y = _y_train_norm[_offset:(_offset + _batch_size)]
							_feed_dict = {_tf_X:_batch_X, _tf_y:_batch_y, _tf_lambda:_lambda, _tf_alpha:_alpha, _tf_keep_prob:0.5}
							_, _loss= _session.run([_tf_optimizer, _tf_loss_reg], feed_dict=_feed_dict)
							if _step!=0 and _step%10==0:
								_feed_dict = {_tf_X:_batch_X, _tf_y:_batch_y, _tf_keep_prob:1}
								_c_loss = _session.run(calc_loss(_batch_size), feed_dict=_feed_dict)
								print '\t', _step, '\tDeviation=', _c_loss * _image_size
						_loss_sum = 0.0
						for _step in range(int((_num_valid_dataset+_batch_size-1)/_batch_size)):
							_offset = _step * _batch_size
							_batch_X = _X_valid_norm[_offset:(_offset + _batch_size)]
							_batch_y = _y_valid_norm[_offset:(_offset + _batch_size)]
							_real_batch_size = len(_batch_X)
							if _real_batch_size != _batch_size:
								_padding = np.zeros((_batch_size-_real_batch_size, _image_size,_image_size, 1))
								_batch_X = np.vstack((_batch_X, _padding))
								_padding = np.zeros((_batch_size-_real_batch_size, _num_y))
								_batch_y = np.vstack((_batch_y, _padding))
							_feed_dict = {_tf_X:_batch_X, _tf_y:_batch_y, _tf_keep_prob:1}
							_c_loss = _session.run(calc_loss(_real_batch_size), feed_dict=_feed_dict)
							#print '\t', _c_loss
							_loss_sum = _loss_sum +  _c_loss*(float(_real_batch_size)/_batch_size)
						_ave_Deviation = _loss_sum / float((_num_valid_dataset+_batch_size-1)/_batch_size) * _image_size
						_Deviation_list[_al_index, _lam_index] = _ave_Deviation
					print 'alpha='+str(_alpha)+',\tlambda='+str(_lambda)+',\tAveDeviation='+str(_ave_Deviation)+',\tNum_of_points='+str(_num_valid_dataset)

			_best_Deviation_valid = None
			_best_alpha = None
			_best_lambda = None
			for _al_index, _alpha in enumerate(_alpha_list):
				for _lam_index, _lambda in enumerate(_lambda_list):
					if _best_Deviation_valid is None or _Deviation_list[_al_index, _lam_index]<_best_Deviation_valid:
						_best_Deviation_valid = _Deviation_list[_al_index, _lam_index]
						_best_alpha = _alpha_list[_al_index]
						_best_lambda = _lambda_list[_lam_index]
						_dict_grid_param[_y_name]=[[_best_alpha],[_best_lambda]]
			print '***Grid Search End***'
		print '***Selected Parameter***'
		print _y_name
		print 'Validation Best Deviation:', _best_Deviation_valid
		print 'Validation Best Alpha:',		_best_alpha
		print 'Validation Best Lambda:',	_best_lambda
		with open('parameters.json', 'w') as _fh:
			json.dump(_dict_grid_param, _fh, indent=4)

		_learning_curve_mini_batch_count = []
		_learning_curve_Deviation = []
		_X_train_norm = np.vstack((_X_train_norm, _X_valid_norm))
		_y_train_norm = np.vstack((_y_train_norm, _y_valid_norm))
		_num_train_dataset = _X_train_norm.shape[0]
		with tf.Session(graph=_graph) as _session:
			_session.run(tf.initialize_all_variables())
			print '***Training by Selected Parameter***'
			for _step in range(_num_steps_training):
				_offset = (_step * _batch_size) % (_num_train_dataset - _batch_size)
				if _offset==0:
					_array_index_rand = np.random.permutation(len(_X_train_norm))
					_X_train_norm = _X_train_norm[_array_index_rand]
					_y_train_norm = _y_train_norm[_array_index_rand]
				_batch_X = _X_train_norm[_offset:(_offset + _batch_size), :]
				_batch_y = _y_train_norm[_offset:(_offset + _batch_size)]
				_feed_dict = {_tf_X:_batch_X, _tf_y:_batch_y, _tf_lambda:_best_lambda, _tf_alpha:_best_alpha, _tf_keep_prob:0.5}
				_, _loss = _session.run([_tf_optimizer, _tf_loss_reg], feed_dict=_feed_dict)
				if _step!=0 and _step%10==0:
					_feed_dict = {_tf_X:_batch_X, _tf_y:_batch_y, _tf_keep_prob:1}
					_c_loss = _session.run(calc_loss(_batch_size), feed_dict=_feed_dict)
					print '\t', _step, '\tDeviation=', _c_loss * _image_size
					_learning_curve_mini_batch_count.append(_step)
					_learning_curve_Deviation.append(_c_loss * _image_size)
			plt.plot(_learning_curve_mini_batch_count, _learning_curve_Deviation)
			plt.xlabel('Batch Count')
			plt.ylabel('Deviation')
			plt.yscale('log')
			plt.title('Learning Curve, '+_y_name+', alpha='+str(_best_alpha)+', lambda='+str(_best_lambda))
			plt.savefig('Learn_'+str(int(_y_index))+'_'+_y_name+'.png')
			plt.clf()

			print '***Prediction by Selected Parameter***'
			for _step in range(int((_num_test_dataset+_batch_size-1)/_batch_size)):
				_offset = _step * _batch_size
				_batch_X = _X_test_norm[_offset:(_offset + _batch_size), :]
				_real_batch_size = len(_batch_X)
				if _real_batch_size != _batch_size:
					_padding = np.zeros((_batch_size-_real_batch_size, _image_size,_image_size, 1))
					_batch_X = np.vstack((_batch_X, _padding))
				_feed_dict = {_tf_X:_batch_X, _tf_keep_prob:1}
				_batch_predict = _session.run(model(), feed_dict=_feed_dict)[:_real_batch_size]
				_batch_predict = _batch_predict.reshape(_real_batch_size)
				_batch_predict = _batch_predict * _image_size
				_predicts_test[_offset:(_offset + _real_batch_size), _y_index] = _batch_predict
		del(_X_train_norm); del(_y_train_norm);
		del(_X_valid_norm); del(_y_valid_norm);
		del(_learning_curve_mini_batch_count); del(_learning_curve_Deviation);
		gc.collect()
		_time_end = time.time()
		print 'End Learning:',_y_name
		print 'time for learning:', str(_time_end-_time_start) + 'sec\t(' + str((_time_end-_time_start)/60.0) + 'min)'
		print '**********************************************************'
		print '**********************************************************'
		print ''

	print ''
	print '**********************************************************'
	print '**********************************************************'
	print '**********************************************************'
	print '**********************************************************'
	print '**********************************************************'
	print 'Predicted Y'
	_anomaly_count_list_per_name = np.zeros(len(_y_name_all), dtype=np.int)
	_anomaly_count_list_per_Image = np.zeros(len(_predicts_test[:,0]))
	_temp_str = 'Row\t'
	for _column, _temp_name in enumerate(_y_name_all):
		_temp_str = _temp_str + _temp_name
		if _column != len(_y_name_all)-1:
			_temp_str = _temp_str + '\t'
	print _temp_str
	for _row in range(len(_predicts_test[:,0])):
		_temp_str = str(_row) + '\t'
		for _column in range(len(_y_name_all)):
			_temp_str = _temp_str + str(int(_predicts_test[_row, _column]))
			if _column != len(_y_name_all)-1:
				_temp_str = _temp_str + '\t'
			if _predicts_test[_row, _column] <0 or _predicts_test[_row, _column]>_image_size:
				_anomaly_count_list_per_name[_column] = _anomaly_count_list_per_name[_column]+1
				_anomaly_count_list_per_Image[_row]   = _anomaly_count_list_per_Image[_row]+1
		print _temp_str

	print ''
	print '***********************************'
	print 'Creating Submit File'
	_output_filename = __file__.split('.')[0] + '.csv'
	_writecsv = csv.writer(file(_output_filename, 'w'), lineterminator='\n')
	_writecsv.writerow(['RowId','Location'])
	_lookup = pd.read_csv('IdLookupTable.csv')
	_RowIds = np.array(_lookup['RowId'], dtype=np.int)
	_ImageIds = np.array(_lookup['ImageId'], dtype=np.int)
	_FeatureNames = np.array(_lookup['FeatureName'])
	for _index in range(len(_RowIds)):
		_row	= _ImageId_test.index(_ImageIds[_index])
		_column = _y_name_all.index(_FeatureNames[_index])
		_predict = _predicts_test[_row, _column]
		_predict = int(_predict)
		if _predict<0 or _predict>_image_size:
			_predict = _y_mean[_FeatureNames[_index]]
		_writecsv.writerow([_RowIds[_index],_predict])

	print ''
	print '***********************************'
	print 'Anomalies per y-name'
	_index_rank_anomaly = _anomaly_count_list_per_name.argsort()[-1::-1]
	for _index in range(len(_y_name_all)):
		_temp_name = _y_name_all[_index_rank_anomaly[_index]]
		_count = _anomaly_count_list_per_name[_index_rank_anomaly[_index]]
		if _count>0:
			print _temp_name+'\t'+str(int(_count))
	print ''
	print '***********************************'
	print 'Anomalies per image'
	_sorted_index_list = _anomaly_count_list_per_Image.argsort()[-1::-1]
	for _index in range(100):
		print str(_sorted_index_list[_index])+'\t'+str(int(_anomaly_count_list_per_Image[_sorted_index_list[_index]]))
	print '***********************************'

'''
		plt.subplot(1, 2, 1)
		for _al_index, _alpha in enumerate(_alpha_list):
			plt.plot(_lambda_list, _Deviation_list[_al_index, :], label='Alpha='+str(_alpha))
		plt.title('Accuracy at Valid Set')
		plt.xscale('log')
		plt.yscale('log')
		plt.xlabel('lambda')
		plt.ylabel('Loss at Valid Set')
		plt.legend()
		plt.subplot(1, 2, 2)
		for _lam_index, _lambda in enumerate(_lambda_list):
			plt.plot(_alpha_list, _Deviation_list[:, _lam_index], label='lambda='+str(_lambda))
		plt.title('Loss at Valid Set')
		plt.xscale('log')
		plt.yscale('log')
		plt.xlabel('alpha')
		plt.ylabel('Accuracy at Valid Set')
		plt.legend()
		plt.show()
'''

'''
'''

'''
	_p = np.random.random_integers(0, _num_test_dataset, 25)
	_samples_X = _X_test[_p]
	#_samples_Y	= _y_predict[_p]
	for _index in range(len(_samples_X)):
		plt.subplot(5,5,_index+1)
		plt.axis('off')
		plt.imshow(_samples_X[_index].reshape(_image_size,_image_size), cmap=cm.gray_r, interpolation='nearest')
		#plt.title(str(_samples_Y[_index]), color='red')
	plt.show()
'''
