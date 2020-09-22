import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os

train_dir = 'cats_and_dogs/train'
validation_dir = 'cats_and_dogs/validation'
test_dir = 'cats_and_dogs/test'

# Get number of files in each directory. The train and validation directories
# each have the subdirecories "dogs" and "cats".
total_train = sum([len(files) for r, d, files in os.walk(train_dir)])
total_val = sum([len(files) for r, d, files in os.walk(validation_dir)])
total_test = len(os.listdir(test_dir))

# Variables for pre-processing and training.
batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150


train_image_generator = ImageDataGenerator(rescale= 1./255)
validation_image_generator =  ImageDataGenerator(rescale= 1./255)
test_image_generator =  ImageDataGenerator(rescale= 1./255)

train_data_gen =  train_image_generator.flow_from_directory(directory=train_dir, target_size = (IMG_HEIGHT,IMG_WIDTH),
                                                            batch_size = batch_size, class_mode = 'binary')
val_data_gen = validation_image_generator.flow_from_directory(directory= validation_dir,
                                                              target_size = (IMG_HEIGHT,IMG_WIDTH),
                                                              batch_size = batch_size, class_mode = 'binary')
test_data_gen = test_image_generator.flow_from_directory(directory= 'cats_and_dogs',
                                                         target_size = (IMG_HEIGHT, IMG_WIDTH), batch_size = 1,
                                                         classes = ['test'], class_mode = None, shuffle=False, )





train_image_generator = ImageDataGenerator(rescale= 1./255,
                                            rotation_range=90,
                                            width_shift_range=1.0,
                                            height_shift_range=1.0,
                                            shear_range=0.2,
                                            zoom_range=0.2,
                                           vertical_flip= True)



train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='binary')



model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(optimizer= 'adam', 
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

model.summary()


history = model.fit(train_data_gen, steps_per_epoch=None, epochs=epochs, validation_data=val_data_gen,
                    validation_steps=None)

model.save('trained_model')

model.predict(test_data_gen)

probabilities = model.predict(test_data_gen)
prediction = model.predict_classes(test_data_gen)

print(probabilities)
print(prediction)