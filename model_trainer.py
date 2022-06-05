from xml.dom.minidom import Element
from PIL import Image
import mnist
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

x_train= mnist.train_images()
y_train= mnist.train_labels()
x_test= mnist.test_images()
y_test= mnist.test_labels()

x_train=x_train.reshape(-1,28*28)
x_test=x_test.reshape(-1,28*28)

x_train=(x_train/255)
x_test=(x_test/255)

model=MLPClassifier(solver='adam', activation='relu',hidden_layer_sizes=(64,64))
model.fit(x_train,y_train)

filename=('model1.sav')
joblib.dump(model,filename)

prediction= model.predict(x_test)

err= confusion_matrix(y_test,prediction)


# -----function to determine accuracy

def accuracy(cm):
    diagonal = cm.trace()
    element=cm.sum()
    return diagonal/element

print(accuracy(err))
