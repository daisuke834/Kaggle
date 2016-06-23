import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import SGD
from keras.models import model_from_json
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras import backend
from sklearn.cross_validation import train_test_split
from six.moves import cPickle as pickle
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import csv
from datetime import datetime
from glob import glob
import gc
import h5py

_ctime = datetime.now()
_timestr = str(_ctime.year) +'_' + ('%02d'%_ctime.month) +'_' + ('%02d'%_ctime.day) +'_' + ('%02d%02d%02d'%(_ctime.hour,_ctime.minute,_ctime.second)) +'_'

_file_train = 'data/training.csv'
_file_test = 'data/test.csv'
_file_name_lookup		= 'data/IdLookupTable.csv'
_output_filename1		= 'output/' + _timestr + 'output_float.csv'
_output_filename2		= 'output/' + _timestr + 'output_int.csv'
_file_learning_curve	= 'output/' + _timestr + 'model2_hist.png'
_file_premodel_weights		= 'output/' + _timestr + 'model_weights.h5'
_srch_premodel_weights		= 'output/' + '*' +		 'model_weights.h5'
_file_model_predict		= 'output/' + _timestr + 'model_predict.png'
_file_output_binary		= 'output/' + _timestr + 'output_binary.pickle'
_image_size = 96

_rand_seed = 42
#_num_of_epoch = 3
_num_of_epoch = 1000
_learning_rate_start = 0.03
_learning_rate_end = 0.001

_y_name_all = ['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y', 'left_eye_inner_corner_x', 'left_eye_inner_corner_y', \
	'left_eye_outer_corner_x', 'left_eye_outer_corner_y', 'right_eye_inner_corner_x', 'right_eye_inner_corner_y', 'right_eye_outer_corner_x', 'right_eye_outer_corner_y', \
	'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y', 'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y', 'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y', \
	'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y', 'nose_tip_x', 'nose_tip_y', 'mouth_left_corner_x', 'mouth_left_corner_y', 'mouth_right_corner_x', 'mouth_right_corner_y', \
	'mouth_center_top_lip_x', 'mouth_center_top_lip_y', 'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y']

_data_pairs = [
	dict(
		columns=(
			'left_eye_center_x', 'left_eye_center_y',
			'right_eye_center_x', 'right_eye_center_y',
			),
		flip_indices=((0, 2), (1, 3)),
		),
	dict(
		columns=(
			'nose_tip_x', 'nose_tip_y',
			),
		flip_indices=(),
		),
	dict(
		columns=(
			'mouth_left_corner_x', 'mouth_left_corner_y',
			'mouth_right_corner_x', 'mouth_right_corner_y',
			'mouth_center_top_lip_x', 'mouth_center_top_lip_y',
			),
		flip_indices=((0, 2), (1, 3)),
        ),
	dict(
        columns=(
			'mouth_center_bottom_lip_x',
			'mouth_center_bottom_lip_y',
			),
		flip_indices=(),
		),
	dict(
		columns=(
			'left_eye_inner_corner_x', 'left_eye_inner_corner_y',
			'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
			'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
			'right_eye_outer_corner_x', 'right_eye_outer_corner_y',
			),
		flip_indices=((0, 2), (1, 3), (4, 6), (5, 7)),
		),
	dict(
		columns=(
			'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
			'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
			'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
			'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',
			),
		flip_indices=((0, 2), (1, 3), (4, 6), (5, 7)),
		),
	]

def load(_test=False, _cols=None):
	_fname = _file_test if _test else _file_train
	_df = read_csv(os.path.expanduser(_fname))
	_df['Image'] = _df['Image'].apply(lambda _x: np.fromstring(_x, sep=' '))
	if _cols:
		_df = _df[list(_cols)+['Image']]
	print _df.count()
	_df = _df.dropna()
	
	_X_norm = np.vstack(_df['Image'].values) / 255.0
	_X_norm = _X_norm.astype(np.float32)
	
	if not _test:
		_y_norm = _df[_df.columns[:-1]].values
		_y_norm = (_y_norm -48.0) / 48.0
		_X_norm, _y_norm = shuffle(_X_norm, _y_norm, random_state=_rand_seed)
		_y_norm = _y_norm.astype(np.float32)
	else:
		_y_norm = None
		
	return _X_norm, _y_norm

def load2d(_test=False, _cols=None):
	_X_norm, _y_norm = load(_test, _cols)
	_X_norm = _X_norm.reshape(-1,1,96, 96)
	return _X_norm, _y_norm

def flip_image(_X, _y, _flip_indices):
	_X_flipped = np.array(_X[:,:,:, ::-1])
	_y_flipped = np.array(_y)
	_y_flipped[:, ::2] = _y_flipped[:, ::2] * -1
	for _index in range(_y.shape[0]):
		for _index_a, _index_b in _flip_indices:
			_y_flipped[_index, _index_a], _y_flipped[_index, _index_b] = (_y_flipped[_index, _index_b], _y_flipped[_index, _index_a])
	return _X_flipped, _y_flipped


_X_norm, _y_norm = load2d()
_y_mean = (_y_norm*48.0+48.0).mean(axis=0)
print 'X_norm.shape=', _X_norm.shape
print 'X_norm.min=', _X_norm.min()
print 'X_norm.max=', _X_norm.max()
print 'y_norm.shape=', _y_norm.shape
print 'y_norm.min=', _y_norm.min()
print 'y_norm.max=', _y_norm.max()

_X_test_norm, _ = load2d(_test=True)
_predicts_test_norm = np.ndarray((len(_X_test_norm), len(_y_name_all)), dtype=np.float)

for _index, _cpair in enumerate(_data_pairs):
	print ''
	print '**********************************************************'
	print '**********************************************************'
	gc.collect()
	_cols = _cpair['columns']
	print '***', _index, _cols, '***'
	_file_cmodel_weights	= 'output/' + _timestr + 'model_weights_'+str(_index)+'.h5'
	_srch_cmodel_weights	= 'output/' + '*' +		 'model_weights_'+str(_index)+'.h5'

	_flip_indices = _cpair['flip_indices']
	_X_norm, _y_norm = load2d(_cols=_cols)
	_X_train, _X_val, _y_train, _y_val = train_test_split(_X_norm, _y_norm, test_size=0.1, random_state=_rand_seed)

	_X_train_flipped, _y_train_flipped = flip_image(_X_train, _y_train, _flip_indices)
	_X_train = np.vstack((_X_train, _X_train_flipped))
	_y_train = np.vstack((_y_train, _y_train_flipped))
	
	_X_val_flipped, _y_val_flipped = flip_image(_X_val, _y_val, _flip_indices)
	_X_val = np.vstack((_X_val, _X_val_flipped))
	_y_val = np.vstack((_y_val, _y_val_flipped))

	del(_X_train_flipped); del(_y_train_flipped);
	del(_X_val_flipped); del(_y_val_flipped);
	del(_X_norm); del(_y_norm);
	gc.collect()

	print '_X_train.shape=', _X_train.shape
	print '_y_train.shape=', _y_train.shape
	print '_X_val.shape=', _X_val.shape
	print '_y_val.shape=', _y_val.shape
	
	#For Debug
	#if _X_train.shape[0] > 200: _X_train = _X_train[:200]
	#if _y_train.shape[0] > 200: _y_train = _y_train[:200]
	
	print 'Building model'
	_model = Sequential()

	_model.add(Convolution2D(32,3,3, input_shape=(1,96,96), name='conv1'))
	_model.add(Activation('relu'))
	_model.add(MaxPooling2D(pool_size=(2,2)))
	_model.add(Dropout(0.1))

	_model.add(Convolution2D(64,2,2, name='conv2'))
	_model.add(Activation('relu'))
	_model.add(MaxPooling2D(pool_size=(2,2)))
	_model.add(Dropout(0.2))

	_model.add(Convolution2D(128,2,2, name='conv3'))
	_model.add(Activation('relu'))
	_model.add(MaxPooling2D(pool_size=(2,2)))
	_model.add(Dropout(0.3))

	_model.add(Flatten())
	_model.add(Dense(500, name='dense1'))
	_model.add(Activation('relu'))
	_model.add(Dropout(0.5))

	_model.add(Dense(500, name='dense2'))
	_model.add(Activation('relu'))
	_model.add(Dense(len(_cols)))
	
	#_srch_cmodel_weights	= 'output/' + '*' +	'model_weights_'+str(_index)+'.h5'
	#_srch_premodel_weights	= 'output/' + '*' +	'model_weights.h5'
	_list_cmodel_weights = glob(_srch_cmodel_weights)
	_list_premodel_weights = glob(_srch_premodel_weights)
	if len(_list_cmodel_weights)>0:
		_loaded_file = _list_cmodel_weights[-1]
		print 'loading', _loaded_file
		_model.load_weights(_loaded_file)
	elif len(_list_premodel_weights)>0:
		_loaded_file = _list_premodel_weights[-1]
		print 'loading', _loaded_file
		with h5py.File(_loaded_file) as _fh:
			_layer_names = [_tmp.decode('utf8') for _tmp in _fh.attrs['layer_names']]
			_weight_value_tuples = []
			for _layer_index, _layer_name in enumerate(_layer_names):
				if _layer_index >= len(_model.layers)-1:
					break
				_loaded_weight = _fh[_layer_name]
				_weight_names = [_tmp.decode('utf8') for _tmp in _loaded_weight.attrs['weight_names']]
				if len(_weight_names):
					_weight_values = [_loaded_weight[_weight_name] for _weight_name in _weight_names]
					_layer = _model.layers[_layer_index]
					_symbolic_weights = _layer.trainable_weights + _layer.non_trainable_weights
					if len(_weight_values) != len(_symbolic_weights):
						raise Exception('Numbers of elements does not match between model and loaded file.\n' \
							+ 'Model: Layer #'+str(_layer_index) +' ('+_layer.name+'):\t' + str(len(_symbolic_weights)) +'\n'\
							+ 'Loaded: Layer ' + _layer_name + '\t' + str(len(_weight_values)))
					_weight_value_tuples += zip(_symbolic_weights, _weight_values)
			backend.batch_set_value(_weight_value_tuples)
			print('len layer names:', len(_layer_names))
			print('len model layers:', len(_model.layers))
			print('layer names:', _layer_names)
			print('model layer names:', [_layer.name for _layer in _model.layers])
	else:
		raise Exception('File of saved weights doesnot exist.')
	print 'Weights loaded'

	#Freezing 1st and 2nd convolution layer
	for _layer in _model.layers[:8]:
		_layer.trainable = False
	
	_sgd = SGD(lr=_learning_rate_start, momentum=0.9, nesterov=True)
	_model.compile(loss='mean_squared_error', optimizer=_sgd)

	_learning_rates = np.linspace(_learning_rate_start, _learning_rate_end, _num_of_epoch)
	_change_learning_rate = LearningRateScheduler(lambda _epoch: float(_learning_rates[_epoch]))
	_early_stop = EarlyStopping(patience=100)

	print 'Training models'
	_hist = _model.fit(_X_train, _y_train, nb_epoch=_num_of_epoch, validation_data=(_X_val, _y_val), callbacks=[_change_learning_rate, _early_stop])

	_ctime2 = datetime.now()
	_timestr2 = str(_ctime2.year) +'_' + ('%02d'%_ctime2.month) +'_' + ('%02d'%_ctime2.day) +'_' + ('%02d%02d%02d'%(_ctime2.hour,_ctime2.minute,_ctime2.second)) +'_'

	_file_cmodel_weights	= 'output/' + _timestr2 + 'model_weights_'+str(_index)+'.h5'
	_model.save_weights(_file_cmodel_weights)

	plt.plot(_hist.history['loss'], linewidth=3, label=str(_index)+'_train')
	plt.plot(_hist.history['val_loss'], linewidth=3, label=str(_index)+'_valid')

	_y_test_norm = _model.predict(_X_test_norm)

	for _cCol_offset, _cCol_name in enumerate(_cols):
		_cCol_index = _y_name_all.index(_cCol_name)
		_predicts_test_norm[:,_cCol_index] = _y_test_norm[:,_cCol_offset]
	print 'End Learning of current pair'
		
plt.grid()
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.ylim(1e-4, 1e-2)
plt.yscale('log')
plt.savefig(_file_learning_curve)
plt.clf()

def plot_sample(_X, _y, _axis):
	_img = _X.reshape(96, 96)
	_axis.imshow(_img, cmap='gray')
	_axis.scatter(_y[0::2] * 48 + 48, _y[1::2] * 48 + 48, marker='x', s=10)

_predicts_test = _predicts_test_norm*48.0+48.0
    
_fig = plt.figure(figsize=(6,6))
_fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for _i in range(16):
	_axis = _fig.add_subplot(4,4, _i+1, xticks=[], yticks=[])
	plot_sample(_X_test_norm[_i], _predicts_test_norm[_i], _axis)
plt.savefig(_file_model_predict)
plt.clf()

_output_binary = dict()
_output_binary['_predicts_test']=_predicts_test
with open(_file_output_binary, 'wb') as _fh:
	pickle.dump(_output_binary, _fh, pickle.HIGHEST_PROTOCOL)

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
_temp_str=''
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
_writecsv = csv.writer(file(_output_filename1, 'w'), lineterminator='\n')
_writecsv.writerow(['RowId','Location'])
_lookup = read_csv(_file_name_lookup)
_RowIds = np.array(_lookup['RowId'], dtype=np.int)
_ImageIds = np.array(_lookup['ImageId'], dtype=np.int)
_FeatureNames = np.array(_lookup['FeatureName'])
for _index in range(len(_RowIds)):
	_row = _ImageIds[_index]-1
	_column = _y_name_all.index(_FeatureNames[_index])
	_cpredict = _predicts_test[_row, _column]
	if _cpredict<0 or _cpredict>_image_size:
		_cpredict = _y_mean[_column]
	_writecsv.writerow([_RowIds[_index],_cpredict])
print '***********************************'
print 'Creating Submit File (int)'
_writecsv = csv.writer(file(_output_filename2, 'w'), lineterminator='\n')
_writecsv.writerow(['RowId','Location'])
_lookup = read_csv(_file_name_lookup)
_RowIds = np.array(_lookup['RowId'], dtype=np.int)
_ImageIds = np.array(_lookup['ImageId'], dtype=np.int)
_FeatureNames = np.array(_lookup['FeatureName'])
for _index in range(len(_RowIds)):
	_row	= _ImageIds[_index]-1
	_column = _y_name_all.index(_FeatureNames[_index])
	_cpredict = _predicts_test[_row, _column]
	_cpredict = int(_cpredict)
	if _cpredict<0 or _cpredict>_image_size:
		_cpredict = int(_y_mean[_column])
	_writecsv.writerow([_RowIds[_index],_cpredict])

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
	_ccount_anomaly = int(_anomaly_count_list_per_Image[_sorted_index_list[_index]])
	if _ccount_anomaly>0:
		print str(_sorted_index_list[_index])+'\t'+str(_ccount_anomaly)
print '***********************************'

