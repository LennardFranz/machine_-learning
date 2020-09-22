import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator




test_image_generator =  ImageDataGenerator(rescale= 1./255)
test_data_gen = test_image_generator.flow_from_directory(directory= 'cats_and_dogs', target_size=(150, 150),
                                                         batch_size=1,  classes = ['test'], class_mode=None,
                                                         shuffle=False )

trained_model = tf.keras.models.load_model('trained_model')

trained_model.summary()

probabilities = trained_model.predict(test_data_gen)
# only works until 31.12.2020
prediction = trained_model.predict_classes(test_data_gen)

#in 2021 use the following
prediction = (trained_model.predict(test_data_gen) > 0.5).astype('int32')

#showing the probabilities for the classification as a dog and the prediction (1 means dog and 0 cat)
print(probabilities)
print(prediction)
