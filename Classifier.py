import warnings
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense
from keras.applications.resnet50 import ResNet50
from keras.optimizers import SGD
from PIL import Image
import os
import numpy as np
warnings.filterwarnings('ignore')
training_imgs = []
labels = []
for path in os.listdir('training_data/Ground'):
    img = Image.open('./training_data/Ground/' + path)
    training_imgs.append(np.array(img))
    labels.append(0)
    img.close()
for path in os.listdir('training_data/Air'):
    img = Image.open('./training_data/Air/' + path)
    training_imgs.append(np.array(img))
    labels.append(1)
    img.close()
training_imgs = np.array(training_imgs)
training_imgs = training_imgs.astype(np.float32)
training_imgs /=255.0
labels = np.array(labels)

strat = tf.distribute.MirroredStrategy()
BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strat.num_replicas_in_sync

STEP = 19647//64
with strat.scope():
    base_model = ResNet50(include_top=False)
    x= base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    pred = Dense(1, activation='sigmoid')(x)
    model = Model(inputs = base_model.input, outputs=pred)
    opt = SGD(lr = 0.0001, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=opt,metrix=['accuracy'])
    train_data = tf.data.Dataset.from_tensor_slices((training_imgs,labels))
    train_data = train_data.cache().shuffle(10000).batch(BATCH_SIZE)
    model.fit(train_data, epochs=30)
    model.save('jump_class_weights.h5')