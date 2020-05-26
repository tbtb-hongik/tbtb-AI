import pytesseract
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
#
# print(pytesseract.image_to_string('img_08.png'))

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
train_x, train_y = train_x / 255.0, train_y / 255.0
test_x, test_y = test_x / 255.0, test_y / 255.0

image = train_x[0]
#(batch_size, height, width, channel)
image = image[tf.newaxis, ..., tf.newaxis]

#Convolution
conv2d = tf.keras.layers.Conv2D(filers=3, tkernel_size=(3, 3), strides=(1, 1), padding='SAME')
# conv2d2 = tf.keras.layers.Conv2D(3, 3, 1, "SAME", "relu")

#Visualization
# print(image.dtype)
image = tf.cast(image, dtype= tf.float32)
layer = conv2d
o

# plt.subplot(1,2,1)
# plt.imshow(image[0, :, :, 0], 'gray')
# plt.subplot(1,2,2)
# plt.imshow(output[0, :, :, 0], 'gray')
# plt.show()
#
# print(np.min(image), np.max(image))
# print(np.min(output), np.max(output))


weight = layer.get_weights()
# print(weight[0].shape, weight[1].shape)

# plt.figure(figsize= (15,5))
# plt.subplot(1,3,1)
# plt.hist(output.numpy().ravel(), range= [-2, 2])
# plt.ylim(0, 100)
# plt.subplot(1,3,2)
# plt.title(weight[0].shape)
# plt.imshow(weight[0][:,:,0,0], 'gray')
# plt.subplot(1,3,3)
# plt.title(output.shape)
# plt.imshow(output[0, :, :, 0], 'gray')
# plt.colorbar()
# plt.show()


#Activation Function
act_layer = tf.keras.layers.ReLU()
act_output = act_layer(output)
# print(act_output.shape)
# plt.imshow(act_output[0, :, :, 0], 'gray')
# plt.show()

#Pooling
pool_layers = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='SAME')
pool_output = pool_layers(act_output)

# plt.figure(figsize= (15, 5))
# plt.subplot(121)
# plt.hist(pool_output.numpy().ravel(), range= [-2,2])
# plt.ylim(5)
# plt.subplot(122)
# plt.imshow(pool_output[0,:,:,0], 'gray')
# plt.colorbar()
# plt.show()


#Fully Connected
flat_layer = tf.keras.layers.Flatten()
flatten_output = flat_layer(pool_output)
# print(flatten_output.shape)
# plt.figure(figsize=(10,5))
# plt.subplot(211)
# plt.hist(flatten_output.numpy().ravel())
# plt.subplot(212)
# plt.imshow(flatten_output[:, :100])
# plt.show()

#Dense
dense_layer = tf.keras.layers.Dense(32, activation='relu') #32개에 연결하겠다.
dense_output = dense_layer(flatten_output)
# print(dense_output.shape)
dense_layer_ex = tf.keras.layers.Dense(10, activation='relu')
dense_test = dense_layer_ex(flatten_output)
# print(dense_test.shape)


#Dropout
#학습할 때만
dropout_layer = tf.keras.layers.Dropout(0.7)
dropout_output = dropout_layer(dense_output)





#---------------------------------------------------------------------------------------------------------------
#최종, layer를 쌓는 방법은 케라스 공홈에 있다.
input_shape = (28, 28, 1)
num_classes = 10

from tensorflow.keras import layers
inputs = layers.Input(shape= input_shape) #input입구 만들어주기

#Feature Extraction
net = layers.Conv2D(32, 3, padding= "same")(inputs)
net = layers.Activation('relu')(net)
net = layers.Conv2D(32, 3, padding= "same")(net)
net = layers.Activation('relu')(net)
net = layers.MaxPool2D((2,2))(net)
net = layers.Dropout(0.25)(net)

net = layers.Conv2D(64, 3, padding= "same")(net)
net = layers.Activation('relu')(net)
net = layers.Conv2D(64, 3, padding= "same")(net)
net = layers.Activation('relu')(net)
net = layers.MaxPool2D((2,2))(net)
net = layers.Dropout(0.25)(net)

#Fully Connect
net = layers.Flatten()(net)
net = layers.Dense(512)(net)
net = layers.Activation('relu')(net)
net = layers.Dropout(0.25)(net)
net = layers.Dense(10)(net) #mnist가 10개이기 때문에, 내보내는 노드수도 10개로 내보내야 한다.
net = layers.Activation('softmax')(net) #정규분포값으로 만들어줌

# model = tf.keras.Model(input=inputs, output= net, name= "Basic_CNN")
model = tf.keras.Model(inputs, net, "Basic_CNN")
model.build()
print(model.summary())



cv2.findContours()



cv2.threshold()












