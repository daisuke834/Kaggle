import numpy as np
import pandas as pd
import csv
import time
import matplotlib.pyplot as plt
from matplotlib import cm
import gc

def Img_and_y_Shift(_X, _Y, _max):
	assert len(_X.shape) ==4
	_N_of_points = _X.shape[0]
	_image_size = _X.shape[1]
	_shift1 = np.random.randint(-_max, _max+1,_N_of_points)
	_shift2 = np.random.randint(-_max, _max+1,_N_of_points)
	_X_shifted = np.zeros((_N_of_points, _image_size, _image_size, 1))
	for _index in range(_N_of_points):
		if _shift1[_index]>0:
			if _shift2[_index]>0:
				_X_shifted[_index, _shift1[_index]:, _shift2[_index]:, 0] = _X[_index, :_image_size-_shift1[_index], :_image_size-_shift2[_index], 0]
			else:
				_X_shifted[_index, _shift1[_index]:, :_image_size+_shift2[_index], 0] = _X[_index, :_image_size-_shift1[_index], -_shift2[_index]:, 0]
		else:
			if _shift2[_index]>0:
				_X_shifted[_index, :_image_size+_shift1[_index], _shift2[_index]:, 0] = _X[_index, -_shift1[_index]:, :_image_size-_shift2[_index], 0]
			else:
				_X_shifted[_index, :_image_size+_shift1[_index], :_image_size+_shift2[_index], 0] = _X[_index, -_shift1[_index]:, -_shift2[_index]:, 0]
		for _y_idx in range(_Y.shape[1]):
			if _y_idx%2==0:
				_Y[_index, _y_idx] = _Y[_index, _y_idx]+_shift1[_index]
			else:
				_Y[_index, _y_idx] = _Y[_index, _y_idx]+_shift2[_index]
	return _X_shifted


if __name__ == '__main__':
	pd.set_option('display.width', 200)
	_image_size = 96
	_pixel_depth = 255.0
	_y_name_all = ['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y', 'left_eye_inner_corner_x', 'left_eye_inner_corner_y', \
		'left_eye_outer_corner_x', 'left_eye_outer_corner_y', 'right_eye_inner_corner_x', 'right_eye_inner_corner_y', 'right_eye_outer_corner_x', 'right_eye_outer_corner_y', \
		'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y', 'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y', 'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y', \
		'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y', 'nose_tip_x', 'nose_tip_y', 'mouth_left_corner_x', 'mouth_left_corner_y', 'mouth_right_corner_x', 'mouth_right_corner_y', \
		'mouth_center_top_lip_x', 'mouth_center_top_lip_y', 'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y']
	_dict_grid_param = { \
		'left_eye_center_x':[[0.1],[0.0001]] ,
		'left_eye_center_y':[[0.1],[0.0001]] ,
		'right_eye_center_x':[[0.1],[0.0003]] ,
		'right_eye_center_y':[[0.3],[0.0001]] ,
		'left_eye_inner_corner_x':[[0.5],[0.003]] ,
		'left_eye_inner_corner_y':[[0.1],[0.001]] , \
		'left_eye_outer_corner_x':[[0.5],[0.0003]] , \
		'left_eye_outer_corner_y':[[0.3],[0.0001]] , \
		'right_eye_inner_corner_x':[[0.5],[0.001]] , \
		'right_eye_inner_corner_y':[[0.3],[0.001]] , \
		'right_eye_outer_corner_x':[[0.1],[0.0001]] , \
		'right_eye_outer_corner_y':[[0.5],[0.0001]] , \
		'left_eyebrow_inner_end_x':[[0.1],[0.0001]] , \
		'left_eyebrow_inner_end_y':[[0.3],[0.0003]] , \
		'left_eyebrow_outer_end_x':[[0.3],[0.0003]] , \
		'left_eyebrow_outer_end_y':[[0.1],[0.001]] , \
		'right_eyebrow_inner_end_x':[[0.3],[0.0003]] , \
		'right_eyebrow_inner_end_y':[[0.3],[0.001]], \
		'right_eyebrow_outer_end_x':[[0.1],[0.0003]] , \
		'right_eyebrow_outer_end_y':[[0.3],[0.0001]] , \
		'nose_tip_x':[[0.1],[0.001]] , \
		'nose_tip_y':[[0.1],[0.0003]] , \
		'mouth_left_corner_x':[[0.5],[0.001]] , \
		'mouth_left_corner_y':[[0.1],[0.0001]] , \
		'mouth_right_corner_x':[[0.1],[0.0003]] , \
		'mouth_right_corner_y':[[0.1],[0.001]] , \
		'mouth_center_top_lip_x':[[0.3],[0.0003]] , \
		'mouth_center_top_lip_y':[[0.1],[0.001]] , \
		'mouth_center_bottom_lip_x':[[0.1],[0.001]] , \
		'mouth_center_bottom_lip_y':[[0.1],[0.0001]]
		}
	#_y_name_all = ['left_eye_inner_corner_x']
	#_train_df = pd.read_csv('training.csv')
	_test_df = pd.read_csv('test.csv')
	#_ImageId_test = list(np.array(_test_df['ImageId'], dtype=np.int))
	_X_test = np.array([_x.split(' ') for _x in np.array(_test_df['Image'])], dtype=np.float)
	#_num_test_dataset = len(_test_df)
	#del(_test_df)
	#print '_X_test', _X_test.shape
	_X_test = _X_test.reshape(-1, _image_size,_image_size, 1)
	#_X_test_norm = _X_test/_pixel_depth
	#_predicts_test = np.ndarray((_num_test_dataset, len(_y_name_all)), dtype=np.float)

	def ImgShift(_X, _max):
		assert len(_X.shape) ==4
		_N_of_points = _X.shape[0]
		_image_size = _X.shape[1]
		_shift1 = np.random.randint(-_max, _max+1,_N_of_points)
		_shift2 = np.random.randint(-_max, _max+1,_N_of_points)
		_X_shifted = np.zeros((_N_of_points, _image_size, _image_size, 1))
		for _index in range(_N_of_points):
			if _shift1[_index]>0:
				if _shift2[_index]>0:
					_X_shifted[_index, _shift1[_index]:, _shift2[_index]:, 0] = _X[_index, :_image_size-_shift1[_index], :_image_size-_shift2[_index], 0]
				else:
					_X_shifted[_index, _shift1[_index]:, :_image_size+_shift2[_index], 0] = _X[_index, :_image_size-_shift1[_index], -_shift2[_index]:, 0]
			else:
				if _shift2[_index]>0:
					_X_shifted[_index, :_image_size+_shift1[_index], _shift2[_index]:, 0] = _X[_index, -_shift1[_index]:, :_image_size-_shift2[_index], 0]
				else:
					_X_shifted[_index, :_image_size+_shift1[_index], :_image_size+_shift2[_index], 0] = _X[_index, -_shift1[_index]:, -_shift2[_index]:, 0]
		return _X_shifted



	#_X_train = np.array([_x.split(' ') for _x in np.array(_train_df['Image'])], dtype=np.float)
	#_num_train_dataset = len(_train_df)
	#_p = np.random.random_integers(0, _num_test_dataset, 25)
	#_samples_X = _X_train[:25]
	#_p = np.random.random_integers(0, _num_test_dataset, 25)
	#_samples_X = _X_test[_p]
	_X_test = ImgShift(_X_test, 20)
	_samples_X = _X_test[[1518, 1094, 932, 154, 167, 1467, 513, 1007, 1259, 356, 1355, 414, 1699, 41, 200, 239, 1591]]



	for _index in range(len(_samples_X)):
		plt.subplot(5,5,_index+1)
		plt.axis('off')
		_a = _samples_X[_index]
		_mean = _a.mean()
		_std = _a.std()
		plt.imshow(_samples_X[_index].reshape(_image_size,_image_size), cmap=cm.gray_r, interpolation='nearest')
		#plt.imshow((_samples_X[_index].reshape(_image_size,_image_size)-_mean)/_std, cmap=cm.gray_r, interpolation='nearest')
		#plt.title(str(_samples_Y[_index]), color='red')
	plt.show()
	#plt.savefig('a.png')
	for _index in range(len(_samples_X)):
		plt.subplot(5,5,_index+1)
		plt.axis('off')
		_a = _samples_X[_index]
		_mean = _a.mean()
		_std = _a.std()
		print _mean, '\t', _std
		#plt.imshow(_samples_X[_index].reshape(_image_size,_image_size), cmap=cm.gray_r, interpolation='nearest')
		_a = (_samples_X[_index].reshape(_image_size,_image_size)-_mean)/_std
		_X_norm = np.zeros(_a.shape)
		_X_norm[:_image_size-20,:_image_size-20]=_a[20:,20:]
		plt.imshow(_X_norm, cmap=cm.gray_r, interpolation='nearest')
		#plt.title(str(_samples_Y[_index]), color='red')
	#plt.show()
	plt.savefig('b.png')
