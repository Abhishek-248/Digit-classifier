import joblib
from sklearn.neural_network import MLPClassifier
from PIL import Image
import numpy as np
img= Image.open(r"C:\Users\Abhishek Kashyap\Downloads\test\six.png")

data=list(img.getdata())

for i in range(len(data)):
    data[i]=255-data[i]
data=np.array(data)/255
data=data.reshape(1,-1)

model=joblib.load(r"C:\Users\Abhishek Kashyap\Downloads\test\model.sav")

ans=model.predict(data)
print(ans)