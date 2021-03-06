import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.models import model_from_json
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os

_file_train = 'training.csv'
_file_test = 'test.csv'

def load(_test=False, _cols=None):
	_fname = _file_test if _test else _file_train
	_df = read_csv(os.path.expanduser(_fname))
	_df['Image'] = _df['Image'].apply(lambda _x: np.fromstring(_x, sep=' '))
	if _cols:
		_df = _df[list(_cols)+['Image']]
	print _df.count()
	_df = _df.dropna()
	
	_X = np.vstack(_df['Image'].values) / 255.0
	_X = _X.astype(np.float32)
	
	if not _test:
		_y = _df[_df.columns[:-1]].values
		_y = (_y -48.0) / 48.0
		_X, _y = shuffle(_X, _y, random_state=42)
		_y = _y.astype(np.float32)
	else:
		_y = None
		
	return _X, _y

_X, _y = load()
print 'X.shape=', _X.shape
print 'X.min=', _X.min()
print 'X.max=', _X.max()
print 'y.shape=', _y.shape
print 'y.min=', _y.min()
print 'y.max=', _y.max()

_model = Sequential()
_model.add(Dense(100, input_dim=9216))
_model.add(Activation('relu'))
_model.add(Dense(30))

_sgd = SGD(lr='0.01', momentum=0.9, nesterov=True)
_model.compile(loss='mean_squared_error', optimizer=_sgd)
_hist = _model.fit(_X, _y, nb_epoch=100, validation_split=0.2)
_json_string = _model.to_json()
with open('model1_architecture.json', 'w') as _fh:
	_fh.write(_json_string)
_model.save_weights('model1_weights.h5')

#_model = model_from_json(open('model1_architecture.json').read())
#_model.load_weights('model1_weights.h5')

plt.plot(_hist.history['loss'], linewidth=3, label='train')
plt.plot(_hist.history['val_loss'], linewidth=3, label='valid')
plt.grid()
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.ylim(1e-3, 1e-1)
plt.yscale('log')
plt.savefig('hist.png')
plt.clf()

_X_test, _ = load(_test=True)
_y_test = _model.predict(_X_test)


def plot_sample(_X, _y, _axis):
	_img = _X.reshape(96, 96)
	_axis.imshow(_img, cmap='gray')
	_axis.scatter(_y[0::2] * 48 + 48, _y[1::2] * 48 + 48, marker='x', s=10)
    
_fig = plt.figure(figsize=(6,6))
_fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for _i in range(16):
	_axis = _fig.add_subplot(4,4, _i+1, xticks=[], yticks=[])
	plot_sample(_X_test[_i], _y_test[_i], _axis)
plt.savefig('predict.png')
plt.clf()


