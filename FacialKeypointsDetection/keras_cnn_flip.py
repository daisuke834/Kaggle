import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import SGD
from keras.models import model_from_json
from keras.callbacks import LearningRateScheduler, EarlyStopping
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

_ctime = datetime.now()
_timestr = str(_ctime.year) +'_' + ('%02d'%_ctime.month) +'_' + ('%02d'%_ctime.day) +'_' + ('%02d%02d%02d'%(_ctime.hour,_ctime.minute,_ctime.second)) +'_'

_file_train = 'data/training.csv'
_file_test = 'data/test.csv'
_file_name_lookup		= 'data/IdLookupTable.csv'
_output_filename1		= 'output/' + _timestr + 'output_float.csv'
_output_filename2		= 'output/' + _timestr + 'output_int.csv'
_file_learning_curve	= 'output/' + _timestr + 'model2_hist.png'
_file_model_arch_jsn	= 'output/' + _timestr + 'model_architecture.json'
_file_model_weights		= 'output/' + _timestr + 'model_weights.h5'
_srch_model_arch_jsn	= 'output/' + '*' +		 'model_architecture.json'
_srch_model_weights		= 'output/' + '*' +		 'model_weights.h5'
_file_model_predict		= 'output/' + _timestr + 'model_predict.png'
_file_output_binary		= 'output/' + _timestr + 'output_binary.pickle'
_image_size = 96

_rand_seed = 42
_num_of_epoch = 1000
_learning_rate_start = 0.03
_learning_rate_end = 0.001

_y_name_all = ['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y', 'left_eye_inner_corner_x', 'left_eye_inner_corner_y', \
	'left_eye_outer_corner_x', 'left_eye_outer_corner_y', 'right_eye_inner_corner_x', 'right_eye_inner_corner_y', 'right_eye_outer_corner_x', 'right_eye_outer_corner_y', \
	'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y', 'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y', 'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y', \
	'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y', 'nose_tip_x', 'nose_tip_y', 'mouth_left_corner_x', 'mouth_left_corner_y', 'mouth_right_corner_x', 'mouth_right_corner_y', \
	'mouth_center_top_lip_x', 'mouth_center_top_lip_y', 'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y']

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

def flip_image(_X, _y):
	_flip_indices = [
		(0, 2), (1, 3),
		(4, 8), (5, 9), (6, 10), (7, 11),
		(12, 16), (13, 17), (14, 18), (15, 19),
		(22, 24), (23, 25),
		]
	_X_flipped = np.array(_X[:,:,:, ::-1])
	_y_flipped = np.array(_y)
	_y_flipped[:, ::2] = _y_flipped[:, ::2] * -1
	for _index in range(_y.shape[0]):
		for _index_a, _index_b in _flip_indices:
			_y_flipped[_index, _index_a], _y_flipped[_index, _index_b] = (_y_flipped[_index, _index_b], _y_flipped[_index, _index_a])
	return _X_flipped, _y_flipped

_X_norm, _y_norm = load2d()
_y_mean = (_y_norm*48.0+48.0).mean(axis=0)
_X_train, _X_val, _y_train, _y_val = train_test_split(_X_norm, _y_norm, test_size=0.1, random_state=_rand_seed)

_X_train_flipped, _y_train_flipped = flip_image(_X_train, _y_train)
_X_train = np.vstack((_X_train, _X_train_flipped))
_y_train = np.vstack((_y_train, _y_train_flipped))

_X_val_flipped, _y_val_flipped = flip_image(_X_val, _y_val)
_X_val = np.vstack((_X_val, _X_val_flipped))
_y_val = np.vstack((_y_val, _y_val_flipped))

del(_X_train_flipped); del(_y_train_flipped);
del(_X_val_flipped); del(_y_val_flipped);
del(_X_norm); del(_y_norm);
gc.collect()

print 'Num of Epoch=', _num_of_epoch
print ''
print '_X_train.shape=', _X_train.shape
print '_X_train.min=', _X_train.min()
print '_X_train.max=', _X_train.max()
print '_y_train.shape=', _y_train.shape
print '_y_train.min=', _y_train.min()
print '_y_train.max=', _y_train.max()
print ''
print '_X_val.shape=', _X_val.shape
print '_X_val.min=', _X_val.min()
print '_X_val.max=', _X_val.max()
print '_y_val.shape=', _y_val.shape
print '_y_val.min=', _y_val.min()
print '_y_val.max=', _y_val.max()

_model = Sequential()

_model.add(Convolution2D(32,3,3, input_shape=(1,96,96)))
_model.add(Activation('relu'))
_model.add(MaxPooling2D(pool_size=(2,2)))
_model.add(Dropout(0.1))

_model.add(Convolution2D(64,2,2))
_model.add(Activation('relu'))
_model.add(MaxPooling2D(pool_size=(2,2)))
_model.add(Dropout(0.2))

_model.add(Convolution2D(128,2,2))
_model.add(Activation('relu'))
_model.add(MaxPooling2D(pool_size=(2,2)))
_model.add(Dropout(0.3))

_model.add(Flatten())
_model.add(Dense(500))
_model.add(Activation('relu'))
_model.add(Dropout(0.5))

_model.add(Dense(500))
_model.add(Activation('relu'))
_model.add(Dense(30))

_list_model_arch_jsn = glob(_srch_model_arch_jsn)
if len(_list_model_arch_jsn)>0:
	_loaded_file = _list_model_arch_jsn[-1]
	print 'loading', _loaded_file
	_model = model_from_json(open(_loaded_file).read())
	
_list_model_weights = glob(_srch_model_weights)
if len(_list_model_weights)>0:
	_loaded_file = _list_model_weights[-1]
	print 'loading', _loaded_file
	_model.load_weights(_loaded_file)

_sgd = SGD(lr=_learning_rate_start, momentum=0.9, nesterov=True)
_model.compile(loss='mean_squared_error', optimizer=_sgd)

_learning_rates = np.linspace(_learning_rate_start, _learning_rate_end, _num_of_epoch)
_change_learning_rate = LearningRateScheduler(lambda _epoch: float(_learning_rates[_epoch]))
_early_stop = EarlyStopping(patience=100)

_hist = _model.fit(_X_train, _y_train, nb_epoch=_num_of_epoch, validation_data=(_X_val, _y_val), callbacks=[_change_learning_rate, _early_stop])
_json_string = _model.to_json()
with open(_file_model_arch_jsn, 'w') as _fh:
	_fh.write(_json_string)
_model.save_weights(_file_model_weights)

plt.plot(_hist.history['loss'], linewidth=3, label='train')
plt.plot(_hist.history['val_loss'], linewidth=3, label='valid')
plt.grid()
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.ylim(1e-4, 1e-2)
plt.yscale('log')
plt.savefig(_file_learning_curve)
plt.clf()

_X_test_norm, _ = load2d(_test=True)
_y_test_norm = _model.predict(_X_test_norm)
_y_test = _y_test_norm*48.0+48.0

def plot_sample(_X, _y, _axis):
	_img = _X.reshape(96, 96)
	_axis.imshow(_img, cmap='gray')
	_axis.scatter(_y[0::2] * 48 + 48, _y[1::2] * 48 + 48, marker='x', s=10)
    
_fig = plt.figure(figsize=(6,6))
_fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for _i in range(16):
	_axis = _fig.add_subplot(4,4, _i+1, xticks=[], yticks=[])
	plot_sample(_X_test_norm[_i], _y_test_norm[_i], _axis)
plt.savefig(_file_model_predict)
plt.clf()

_output_binary = dict()
_output_binary['_y_test']=_y_test
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
_anomaly_count_list_per_Image = np.zeros(len(_y_test[:,0]))
_temp_str = 'Row\t'
for _column, _temp_name in enumerate(_y_name_all):
	_temp_str = _temp_str + _temp_name
	if _column != len(_y_name_all)-1:
		_temp_str = _temp_str + '\t'
print _temp_str
_temp_str=''
for _row in range(len(_y_test[:,0])):
	#_temp_str = str(_row) + '\t'
	for _column in range(len(_y_name_all)):
		#_temp_str = _temp_str + str(int(_y_test[_row, _column]))
		#if _column != len(_y_name_all)-1:
		#	_temp_str = _temp_str + '\t'
		if _y_test[_row, _column] <0 or _y_test[_row, _column]>_image_size:
			_anomaly_count_list_per_name[_column] = _anomaly_count_list_per_name[_column]+1
			_anomaly_count_list_per_Image[_row]   = _anomaly_count_list_per_Image[_row]+1
	#print _temp_str

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
	_cpredict = _y_test[_row, _column]
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
	_cpredict = _y_test[_row, _column]
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

