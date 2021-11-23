import tensorflow as tf
from keras.models import Sequential, load_model, Model
from keras.layers import InputLayer, Conv2D, UpSampling2D

def create_model_alpha():
  model = Sequential()
  model.add(InputLayer(input_shape=(256, 256, 1)))
  model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
  model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
  model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
  model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
  model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
  model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
  model.add(UpSampling2D((2, 2)))
  model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
  model.add(UpSampling2D((2, 2)))
  model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
  model.add(UpSampling2D((2, 2)))
  model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
  opt = tf.keras.optimizers.RMSprop(
    learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False,
    name='RMSprop',
  )
  model.compile(optimizer=opt, loss='mse')
  return model