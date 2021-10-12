import tensorflow as tf 
from tensorflow import keras
import matplotlib.pyplot as plt 
import numpy as np
n=int(input("enter the no of images to be seen"))
(xtrain,ytrain),(xtest,ytest)=keras.datasets.mnist.load_data()
for i in range(n):
    plt.figure()
    plt.imshow(xtest[i])
    plt.colorbar()
    plt.grid(False)
    plt.show()
xtrain=xtrain/255
xtest=xtest/255
xflat=xtrain.reshape(len(xtrain),28*28)
xtflat=xtest.reshape(len(xtest),28*28)
model=keras.Sequential([
    keras.layers.Dense(100,input_shape=(784,),activation='relu'),
    keras.layers.Dense(200,input_shape=(100,),activation='relu'),
    keras.layers.Dense(300,input_shape=(200,),activation='relu'),
    keras.layers.Dense(10,activation='sigmoid')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']

)
model.fit(xflat,ytrain,epochs=5)
model.evaluate(xtflat,ytest)
y=model.predict(xtflat)
y=[np.argmax(i)for i in y]
print(y[:5])