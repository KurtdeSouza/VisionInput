import warnings
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense
from keras.applications.resnet50 import ResNet50
from PIL import Image
import os
import numpy as np
training_img  = []
solution = []
for path in os.listdir('training_data/Ground'):
    img = Image.open('./training_data/Ground/' + path)
    training_img.append(np.array(img))
    solution.append(0)
    img.close()
for path in os.listdir('training_data/Air'):
    img = Image.open('./training_data/Air/' + path)
    training_img.append(np.array(img))
    solution.append(1)
    img.close()
#add paths for ground and air training sets with solutions as 1/0 (air/ground)
warnings.filterwarnings('ignore')

