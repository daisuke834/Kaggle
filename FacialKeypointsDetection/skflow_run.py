import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.learn as skflow
from sklearn import metrics
from sklearn.learning_curve import learning_curve
import csv
#from sklearn import utils
import time
import matplotlib.pyplot as plt
from matplotlib import cm
import json
import gc
import os
from six.moves import cPickle as pickle
import sys

_file_name_train_data = 'data/training.csv'
_file_name_test_data = 'data/test.csv'
_file_name_lookup = 'data/IdLookupTable.csv'
_file_name_data = 'data/data.pickle'

_y_name_all = ['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y', 'left_eye_inner_corner_x', 'left_eye_inner_corner_y', \
	'left_eye_outer_corner_x', 'left_eye_outer_corner_y', 'right_eye_inner_corner_x', 'right_eye_inner_corner_y', 'right_eye_outer_corner_x', 'right_eye_outer_corner_y', \
	'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y', 'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y', 'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y', \
	'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y', 'nose_tip_x', 'nose_tip_y', 'mouth_left_corner_x', 'mouth_left_corner_y', 'mouth_right_corner_x', 'mouth_right_corner_y', \
	'mouth_center_top_lip_x', 'mouth_center_top_lip_y', 'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y']
_num_y = len(_y_name_all)

_num_epochs = 100
_batch_size = 64
_eval_batch_size = 64
_image_size = 96
_pixel_depth = 255
_patch_size1 = 3
_depth1 = 32
_patch_size2 = 2
_depth2 = 64
_patch_size3 = 2
_depth3 = 128
_num_hidden_fc1 = 1024
_validation_size = 100
_early_stop_patience = 100

if os.path.exists(_file_name_data):
	with open(_file_name_data, "rb") as _fh:
		_all_data = pickle.load(_fh)
		_X_train_norm	= _all_data['_X_train_norm']
		_y_train_norm	= _all_data['_y_train_norm']
		_X_valid_norm	= _all_data['_X_valid_norm']
		_y_valid_norm	= _all_data['_y_valid_norm']
		_X_test			= _all_data['_X_test']
		_y_mean			= _all_data['_y_mean']
		_ImageId_test	= _all_data['_ImageId_test']
else:
	_train_df = pd.read_csv(_file_name_train_data)
	_test_df = pd.read_csv(_file_name_test_data)

	_ImageId_test = list(np.array(_test_df['ImageId'], dtype=np.int))
	_X_test = np.array([_x.split(' ') for _x in np.array(_test_df['Image'])], dtype=np.float)
	_num_test_dataset = len(_test_df)
	del(_test_df)
	print '_X_test', _X_test.shape
	_X_test_norm = (_X_test - _X_test.mean(axis=1).reshape(-1,1) ) / _X_test.std(axis=1).reshape(-1,1)
	_X_test_norm = _X_test_norm.reshape(-1, _image_size,_image_size, 1)
	_predicts_test = np.ndarray((_num_test_dataset, len(_y_name_all)), dtype=np.float)

	_y_mean = dict()

	_train_df = _train_df.dropna()
	_num_dataset = len(_train_df)
	_X = np.array([_x.split(' ') for _x in np.array(_train_df['Image'])], dtype=np.float)
	_y = np.array(_train_df[_y_name_all]).reshape(_num_dataset,_num_y)
	_y_mean = np.mean(_y, axis=0)

	#_valid_start_index = int(_num_dataset*0.8)
	_valid_start_index = int(_num_dataset-_validation_size)
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

	_all_data = { \
		'_X_train_norm':_X_train_norm, '_y_train_norm':_y_train_norm, \
		'_X_valid_norm':_X_valid_norm, '_y_valid_norm':_y_valid_norm, \
		'_X_test':_X_test, \
		'_y_mean':_y_mean, '_ImageId_test':_ImageId_test \
		}
	with open(_file_name_data, 'wb') as _fh:
		pickle.dump(_all_data, _fh, pickle.HIGHEST_PROTOCOL)

_train_size = _y_train_norm.shape[0]
_valid_size = _y_valid_norm.shape[0]
print 'train size:', _train_size
print 'valid size:', _valid_size
gc.collect()

def max_pool_2x2(_t):
    return tf.nn.max_pool(_t, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def exp_decay(_global_step):
    return tf.train.exponential_decay(learning_rate=1e-3, global_step=_global_step, decay_steps=_train_size, decay_rate=0.98)

def cnn_model(_X, _y):
    with tf.variable_scope('conv_layer1'):
        _h_conv1 = skflow.ops.conv2d(_X, n_filters=_depth1, filter_shape=[_patch_size1, _patch_size1], bias=True, activation=tf.nn.relu)
        _h_pool1 = max_pool_2x2(_h_conv1)
    with tf.variable_scope('conv_layer2'):
        _h_conv2 = skflow.ops.conv2d(_h_pool1, n_filters=_depth2, filter_shape=[_patch_size2, _patch_size2], bias=True, activation=tf.nn.relu)
        _h_pool2 = max_pool_2x2(_h_conv2)
    with tf.variable_scope('conv_layer3'):
        _h_conv3 = skflow.ops.conv2d(_h_pool2, n_filters=_depth3, filter_shape=[_patch_size3, _patch_size3], bias=True, activation=tf.nn.relu)
        _h_pool3 = max_pool_2x2(_h_conv3)
        _h_pool3_flat = tf.reshape(_h_pool3, [-1, _image_size // 8 * _image_size // 8 * _depth3])
    _h_fc1 = skflow.ops.dnn(_h_pool3_flat, [_num_hidden_fc1], activation=tf.nn.relu, dropout=0.5)
    return skflow.models.linear_regression(_h_fc1, _y)

#_classifier = skflow.TensorFlowEstimator(model_fn=cnn_model, n_classes=0, batch_size=_batch_size, \
#	early_stopping_rounds=_early_stop_patience, steps=10, optimizer='Adam', \
#	learning_rate=exp_decay, continue_training=True)
_classifier = skflow.TensorFlowEstimator(model_fn=cnn_model, n_classes=0, batch_size=_batch_size, \
	steps=10, optimizer='Adam', \
	learning_rate=exp_decay, continue_training=True)

_title = 'learning curve for cnn'
# to plot the learning curve, continue_training must be set False
# generate_learning_curve(_classifier, _title, 'mean_squared_error', _X_train_norm, _y_train_norm)

for _index in xrange(0, _num_epochs):
    _classifier.fit(_X_train_norm, _y_train_norm, logdir='log')
    _score = metrics.mean_squared_error(_y_valid_norm, _classifier.predict(_X_valid_norm))
    print 'mean squared error:', _score
    sys.stdout.flush()

_y_test_norm_predict = _classifier.predict(_X_test_norm, batch_size=_eval_batch_size)
make_submission(_y_test_norm_predict * _image_size)

print ''
print '***********************************'
print 'Creating Submit File'
_output_filename = __file__.split('.')[0] + '.csv'
_writecsv = csv.writer(file(_output_filename, 'w'), lineterminator='\n')
_writecsv.writerow(['RowId','Location'])
_lookup = pd.read_csv(_file_name_lookup)
_RowIds = np.array(_lookup['RowId'], dtype=np.int)
_ImageIds = np.array(_lookup['ImageId'], dtype=np.int)
_FeatureNames = np.array(_lookup['FeatureName'])
for _index in range(len(_RowIds)):
	_row	= _ImageId_test.index(_ImageIds[_index])
	_column = _y_name_all.index(_FeatureNames[_index])
	_predict = _y_test_norm_predict[_row, _column]
	_predict = int(_predict)
	if _predict<0 or _predict>_image_size:
		_predict = _y_mean[_column]
	_writecsv.writerow([_RowIds[_index],_predict])
