##########
# Import #
##########

import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters
np.set_printoptions(threshold=np.inf)
from matplotlib.backends.backend_pdf import PdfPages
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('ignore')

###########
# Matplot #
###########

sns.set(style = 'whitegrid', palette = 'muted', font_scale = 1.2)

COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F00FF"]

sns.set_palette(sns.color_palette(COLORS_PALETTE))

rcParams['figure.figsize'] = (16, 10)
register_matplotlib_converters()

########
# Data #
########

np.random.seed(7)
data = pd.read_csv("../data/time_series_covid19_confirmed_global.csv")
data

data= data.drop(columns = ['Province/State', 'Lat','Long'])
country_data = data.groupby(['Country/Region'], as_index = False).agg('sum')
country_data

country_data.loc['Global'] = country_data.agg('sum')
country_data

#######################
# Optimizer and loss　#
######################

global_data = country_data.iloc[-1:]
global_data = global_data.drop(columns = ['Country/Region'])
global_data = global_data.astype('float64')
global_dataset = global_data.to_numpy() #convert dataframe to numpy
global_dataset

def create_dataset_global(dataset, look_back = 1):
    dataX, dataY = [], []
    for j in range(dataset.shape[0]):
        for i in range(dataset.shape[1] - look_back-1):
            a = dataset[j,i:(i + look_back)]
            if a.any():
                dataX.append(a)
                dataY.append(dataset[j,i + look_back])
    return np.array(dataX), np.array(dataY)

scaler_global = MinMaxScaler(feature_range=(0,1))
global_dataset_scale = scaler_global.fit_transform(global_dataset.flatten()[:, np.newaxis])
global_dataset_scale = np.reshape(global_dataset_scale,(global_dataset.shape[0],global_dataset.shape[1]))

# Loss functions
from keras.losses import mean_squared_error, mean_absolute_error, huber_loss, logcosh

loss_dict = {'mse': mean_squared_error,
             'mae': mean_absolute_error,
             'huber_loss': huber_loss,
             'logcosh': logcosh
             }
'''
  choose loss function: 
    mse, mae, huber, logcosh 
  optimizer: 
    adam, sgd, adadelta, adagrad, rmsprop
'''
loss_l = ['huber_loss', 'logcosh', 'mse','mae']
opt_l = ['adam', 'sgd', 'adadelta', 'adagrad', 'rmsprop']

class LOMSTT:
  # Loss func name
  loss_name = ''
  # Optimizer name
  opt_name = ''
  # Model list
  models = []
  # Score list
  scores = []
  # TrainPredict list
  trainPredict = []
  # TestPredict list
  testPredict = []

  def __init__(self, loss_name, opt_name, models, scores):
    self.loss_name = loss_name
    self.opt_name = opt_name
    self.models = models
    self.scores = scores

test_data_size = 10
train = global_dataset_scale[:, :-test_data_size]
test = global_dataset_scale[:, -test_data_size:]

# Reshape into X=t and Y=t+1
look_back = 4
trainX, trainY = create_dataset_global(train, look_back)
testX, testY = create_dataset_global(test, look_back)

# Reshape input to be [samples, time_steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

epochs = 20
batch_size = 1
loms_l = []

for i, oname in enumerate(opt_l):
  temp_l = []
  for j, lname in enumerate(loss_l):
    # Create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape = (1, look_back)))
    model.add(Dense(1))
    # Loss name, Optimizer name
    model.compile(loss = lname, optimizer = oname)

    scores = [0] * epochs
    models = [0] * epochs

    # Record models and evaluate scores
    for k in range(epochs):
      model.fit(trainX, trainY, epochs=1, batch_size = batch_size, verbose = 2)
      models[k] = model
      scores[k] = model.evaluate(testX, testY, batch_size = batch_size)
    #  Add to list
    temp_l.append(LOMSTT(lname, oname, models, scores))
  loms_l.append(temp_l)

for i in range(len(loss_l)):
  for j in range(len(opt_l)):
    plt.plot(loms_l[j][i].scores, label = loms_l[j][i].opt_name)
  plt.xlabel('Number of epochs')
  plt.ylabel('Test loss')
  plt.legend(fontsize = 20)
  plt.title(loms_l[j][i].loss_name, fontsize = 'large', y = -0.1)
  plt.figure()
  pdf = PdfPages('../opt and loss.pdf')
  pdf.savefig()

# Create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=50, batch_size=1, verbose=2)

epochs_array = [30, 40 , 50, 60, 70, 80, 90, 100]
test_score = []
for e in epochs_array:
    # Create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=e, batch_size=1, verbose=2)
    test_score.append(model.evaluate(testX, testY, batch_size=1))

plt.plot(epochs_array, test_score)

lookback_array = [3,4,5,6,7,8]
test_accuracy = []
for look in lookback_array:
    trainX, trainY = create_dataset_global(train, look)
    testX, testY = create_dataset_global(test, look)
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)
    test_accuracy.append(model.evaluate(testX, testY, batch_size = 1))

plt.plot(lookback_array,test_accuracy)

###############
# prediction　#
##############

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

cases = country_data.loc[country_data['Country/Region'] == 'United Kingdom']
cases = cases.drop(columns = ['Country/Region'])
cases = cases.astype('float64')
dataset = cases.to_numpy()[0]

scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset[:, np.newaxis])

test_data_size = 5
train = dataset[:-test_data_size]
test = dataset[-test_data_size:]

# reshape into X=t and Y=t+1
look_back = 4
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=70, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

