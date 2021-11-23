from keras.preprocessing.image import ImageDataGenerator
from skimage.color import rgb2lab

def train(model,Xtrain, spe, ep):
    
  # Image transformer
  datagen = ImageDataGenerator(
          shear_range=0.2,
          zoom_range=0.2,
          rotation_range=20,
          horizontal_flip=True)

  # Generate training data
  batch_size = 10
  def image_a_b_gen(batch_size):
      for batch in datagen.flow(Xtrain, batch_size=batch_size):
          lab_batch = rgb2lab(batch)
          X_batch = lab_batch[:,:,:,0]
          Y_batch = lab_batch[:,:,:,1:] / 128
          yield (X_batch.reshape(X_batch.shape+(1,)), Y_batch)

  # Train model      
#   tensorboard = TensorBoard(log_dir="output/first_run")
  # model.fit(image_a_b_gen(batch_size), callbacks=[tensorboard], epochs=20, steps_per_epoch=10)
  history = model.fit(image_a_b_gen(batch_size), batch_size=batch_size,steps_per_epoch = spe,  epochs=ep, verbose=1)
  return history, model
# history, model = train(model)