import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import time
import csv


if __name__ == '__main__':
	_num_labels=10
	_image_size = 28
	_pixel_depth = 255.0
	_train = np.array(pd.read_csv('train.csv'))
	_num_dataset = len(_train)
	_valid_start_index = int(_num_dataset*0.8)
	_array_index_rand = np.random.permutation(_num_dataset)
	_X_train = _train[_array_index_rand[:_valid_start_index],1:].reshape(-1, _image_size,_image_size, 1)
	_y_train = _train[_array_index_rand[:_valid_start_index],0]
	_X_valid = _train[_array_index_rand[_valid_start_index:],1:].reshape(-1, _image_size,_image_size, 1)
	_y_valid = _train[_array_index_rand[_valid_start_index:],0]
	del(_train)
	_X_test = np.array(pd.read_csv('test.csv')).reshape(-1, _image_size,_image_size, 1)

	_X_train_norm = _X_train/_pixel_depth
	_X_valid_norm = _X_valid/_pixel_depth
	_X_test_norm = _X_test/_pixel_depth

	_y_train = (np.arange(_num_labels) == _y_train[:,None]).astype(np.float32)
	_y_valid = (np.arange(_num_labels) == _y_valid[:,None]).astype(np.float32)
	_num_train_dataset = len(_X_train)
	_num_valid_dataset = len(_X_valid)
	_num_test_dataset = len(_X_test)
	print 'number of train dataset:', _num_train_dataset
	print 'number of valid dataset:', _num_valid_dataset
	del(_array_index_rand)

	#*********Definition ********************
	_num_steps = 3001
	#_num_steps = 101
	#_num_steps = 11
	_batch_size = 128
	_dummy_batch_labeldata = np.zeros((_batch_size,_num_labels))
	_patch_size = 5
	_depth = 16
	_num_hidden = 64
	_graph = tf.Graph()
	with _graph.as_default():
		_tf_lambda = tf.placeholder(tf.float32)
		_tf_alpha = tf.placeholder(tf.float32)
		_tf_keep_prob = tf.placeholder(tf.float32)
		_tf_X = tf.placeholder(tf.float32, shape=(_batch_size, _image_size, _image_size, 1))
		_tf_y = tf.placeholder(tf.float32, shape=(_batch_size, _num_labels))

		_weights1 = tf.Variable(tf.truncated_normal([_patch_size, _patch_size, 1, _depth], stddev=0.1), dtype=tf.float32)
		_biases1 = tf.Variable(tf.zeros([_depth]), dtype=tf.float32)
		_weights2 = tf.Variable(tf.truncated_normal([_patch_size, _patch_size, _depth, _depth], stddev=0.1), dtype=tf.float32)
		_biases2 = tf.Variable(tf.constant(1.0, shape=[_depth]), dtype=tf.float32)
		_weights3 = tf.Variable(tf.truncated_normal([_image_size//4 * _image_size//4 * _depth, _num_hidden], stddev=0.1), dtype=tf.float32)
		_biases3 = tf.Variable(tf.constant(1.0, shape=[_num_hidden]), dtype=tf.float32)
		_weights4 = tf.Variable(tf.truncated_normal([_num_hidden, _num_labels], stddev=0.1), dtype=tf.float32)
		_biases4 = tf.Variable(tf.constant(1.0, shape=[_num_labels]), dtype=tf.float32)

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
		def predict():
			return tf.argmax(tf.nn.softmax(model()),1)
		def num_of_errors(_num_of_data):
			_predicts = tf.slice(tf.argmax(tf.nn.softmax(model()),1), [0], [_num_of_data])
			_labels = tf.slice(tf.argmax(_tf_y, 1), [0], [_num_of_data])
			return tf.reduce_sum(tf.cast(tf.not_equal(_predicts, _labels), tf.int32))

		_tf_logits = model()
		_tf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(_tf_logits, _tf_y))
		_tf_loss_reg = _tf_loss + 0.5 * _tf_lambda * (tf.nn.l2_loss(_weights1) + tf.nn.l2_loss(_weights2) + tf.nn.l2_loss(_weights3) + tf.nn.l2_loss(_weights4))
		_tf_optimizer = tf.train.AdagradOptimizer(_tf_alpha).minimize(_tf_loss_reg)
		_tf_prediction = tf.nn.softmax(_tf_logits)

	#***Learning****************************
	print '*****************************'
	print 'Start Learning'
	_time_start = time.time()
	_alpha_list = np.array([0.06, 0.1, 0.3])
	_lambda_list = np.array([1e-4, 3e-4, 1e-3])
	#_alpha_list = np.array([0.1])
	#_lambda_list = np.array([3e-4])
	#_alpha_list = np.array([0.03, 0.06, 0.1, 0.3])
	#_lambda_list = np.array([1e-5, 3e-5, 6e-5, 1e-4])
	#_alpha_list = np.logspace(-2,1,7)
	#_lambda_list = np.logspace(-5,-1,9)
	_scores = np.ndarray( (len(_alpha_list), len(_lambda_list) ), dtype=float)
	for _al_index, _alpha in enumerate(_alpha_list):
		for _lam_index, _lambda in enumerate(_lambda_list):
			with tf.Session(graph=_graph) as _session:
				_session.run(tf.initialize_all_variables())
				for _step in range(_num_steps):
					_offset = (_step * _batch_size) % (_num_train_dataset - _batch_size)
					_batch_data		= _X_train_norm[_offset:(_offset + _batch_size), :]
					_batch_labels	= _y_train[_offset:(_offset + _batch_size)]
					_feed_dict = {_tf_X:_batch_data, _tf_y:_batch_labels, _tf_lambda:_lambda, _tf_alpha:_alpha, _tf_keep_prob:0.5}
					_, _l, _predictions = _session.run([_tf_optimizer, _tf_loss_reg, _tf_prediction], feed_dict=_feed_dict)
				_num_of_errors = 0
				for _step in range(int((_num_valid_dataset+_batch_size-1)/_batch_size)):
					_offset = _step * _batch_size
					_batch_data		= _X_valid_norm[_offset:(_offset + _batch_size), :]
					_batch_labels	= _y_valid[_offset:(_offset + _batch_size)]
					_real_batch_size = len(_batch_data)
					if _real_batch_size != _batch_size:
						_padding = np.zeros((_batch_size-_real_batch_size, _image_size,_image_size, 1))
						_batch_data = np.vstack((_batch_data, _padding))
						_padding = np.zeros((_batch_size-_real_batch_size, _num_labels))
						_batch_labels = np.vstack((_batch_labels, _padding))
					_feed_dict = {_tf_X:_batch_data, _tf_y:_batch_labels, _tf_keep_prob:1}
					_current_num_of_errors = _session.run(num_of_errors(_real_batch_size), feed_dict=_feed_dict)
					_num_of_errors = _num_of_errors + _current_num_of_errors
				_accuracy_valid = float(_num_valid_dataset - _num_of_errors) / _num_valid_dataset
				_scores[_al_index, _lam_index] = _accuracy_valid
			print 'alpha='+str(_alpha)+',\tlambda='+str(_lambda)+',\tValidAccuracy='+str(_accuracy_valid)+',\tNum_of_errors='+str(_num_of_errors)+',\tNum_of_points='+str(_num_valid_dataset)

	_best_accuracy_valid = None
	_best_alpha = None
	_best_lambda = None
	for _al_index, _alpha in enumerate(_alpha_list):
		for _lam_index, _lambda in enumerate(_lambda_list):
			if _best_accuracy_valid is None or _scores[_al_index, _lam_index]>_best_accuracy_valid:
				_best_accuracy_valid = _scores[_al_index, _lam_index]
				_best_alpha = _alpha_list[_al_index]
				_best_lambda = _lambda_list[_lam_index]

	_y_predict = np.ndarray(len(_X_test_norm), dtype=np.int)
	with tf.Session(graph=_graph) as _session:
		_session.run(tf.initialize_all_variables())
		for _step in range(_num_steps):
			_offset = (_step * _batch_size) % (_num_train_dataset - _batch_size)
			_batch_data		= _X_train_norm[_offset:(_offset + _batch_size), :]
			_batch_labels	= _y_train[_offset:(_offset + _batch_size)]
			_feed_dict = {_tf_X:_batch_data, _tf_y:_batch_labels, _tf_lambda:_best_lambda, _tf_alpha:_best_alpha, _tf_keep_prob:0.5}
			_, _l, _predictions = _session.run([_tf_optimizer, _tf_loss_reg, _tf_prediction], feed_dict=_feed_dict)
		_num_of_errors = 0
		for _step in range(int((_num_test_dataset+_batch_size-1)/_batch_size)):
			_offset = _step * _batch_size
			_batch_data		= _X_test_norm[_offset:(_offset + _batch_size), :]
			_real_batch_size = len(_batch_data)
			if _real_batch_size != _batch_size:
				_padding = np.zeros((_batch_size-_real_batch_size, _image_size,_image_size, 1))
				_batch_data = np.vstack((_batch_data, _padding))
			_feed_dict = {_tf_X:_batch_data, _tf_keep_prob:1}
			_batch_predict = _session.run(predict(), feed_dict=_feed_dict)[:_real_batch_size]
			_y_predict[_offset:(_offset + _real_batch_size)] = _batch_predict
	_output_filename = __file__.split('.')[0] + '.csv'
	_writecsv = csv.writer(file(_output_filename, 'w'), lineterminator='\n')
	_writecsv.writerow(['ImageId','Label'])
	for _index, _predict in enumerate(_y_predict):
		_writecsv.writerow([_index+1,_predict])
	_time_end = time.time()
	print 'End Learning'
	print '*****************************'
	print 'time for learning:', str(_time_end-_time_start) + 'sec\t(' + str((_time_end-_time_start)/60.0) + 'min)'
	print 'Validation Best Accuracy:', _best_accuracy_valid
	print 'Validation Best Error Rate:', (1.0-_best_accuracy_valid)
	print 'Validation Best Alpha:', _best_alpha
	print 'Validation Best Lambda:', _best_lambda

	plt.subplot(1, 2, 1)
	for _al_index, _alpha in enumerate(_alpha_list):
		plt.plot(_lambda_list, _scores[_al_index, :], label='Alpha='+str(_alpha))
	plt.title('Accuracy at Valid Set')
	plt.xscale('log')
	plt.xlabel('lambda')
	plt.ylabel('Accuracy at Valid Set')
	plt.legend(loc='lower right')

	plt.subplot(1, 2, 2)
	for _lam_index, _lambda in enumerate(_lambda_list):
		plt.plot(_alpha_list, _scores[:, _lam_index], label='lambda='+str(_lambda))
	plt.title('Accuracy at Valid Set')
	plt.xscale('log')
	plt.xlabel('alpha')
	plt.ylabel('Accuracy at Valid Set')
	plt.legend(loc='lower right')
	plt.show()

	_p = np.random.random_integers(0, len(_X_test), 25)
	_samples_X = _X_test[_p]
	_samples_Y	= _y_predict[_p]
	for _index in range(len(_samples_Y)):
		plt.subplot(5,5,_index+1)
		plt.axis('off')
		plt.imshow(_samples_X[_index].reshape(28,28), cmap=cm.gray_r, interpolation='nearest')
		plt.title(str(_samples_Y[_index]), color='red')
	plt.show()
