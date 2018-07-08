import os
import numpy as np
from sklearn.externals import joblib
from PIL import Image
import sys



model = joblib.load('m_symbols.pkl') 

def recognize_image(filepath):
    img = Image.open(filepath).convert('1')

    # transform image to appropriate size
    layer = Image.new('1', (55,55), 255)
    layer.paste(img, tuple(map(lambda x:(x[0]-x[1])/2, zip((55,55), img.size))))

    pred = model.predict(np.array([list(layer.getdata())]))

    return pred[0]


if len(sys.argv) > 1:

    for path in sys.argv[1:]:
        if os.path.isfile(path):
            # process single image
            print(recognize_image(path))
        else:
            # or process all images in the folder
            for f in os.listdir(path):
                if os.path.isfile(path+'/'+f):
                    print(recognize_image(path+'/'+f))
else:
    print('Error: the path to the image as an argument is required')
