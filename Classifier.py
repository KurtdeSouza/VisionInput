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
train_x  = []
train_y = []
for path in os.listdir('training_data/Ground'):
    img = Image.open('./training_data/Ground/' + path)
    train_x.append(np.array(img))
    train_y.append(0)
    img.close()
for path in os.listdir('training_data/Air'):
    img = Image.open('./training_data/Air/' + path)
    train_x.append(np.array(img))
    train_y.append(1)
    img.close()
warnings.filterwarnings('ignore')

