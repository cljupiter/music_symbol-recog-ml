import os
import numpy as np
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.externals import joblib
from PIL import Image



array = []

# prepare data
for folder in os.listdir('data/'):
    for filename in os.listdir('data/'+folder):
        im = Image.open("data/{}/{}".format(folder, filename))
        array.append((list(im.getdata()), folder))

array = shuffle(array)
array = zip(*array)

X = np.array(array[0])
Y = np.array(array[1])

validation_size = 0.2
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(
                                        X, Y, test_size=validation_size)

model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

ac = accuracy_score(Y_validation, predictions)
print("accuracy: " + str(ac)[:5])

joblib.dump(model, 'm_symbols.pkl')
