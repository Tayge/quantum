import random
import numpy as np
from keras import backend as K
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint,  ReduceLROnPlateau
import preprocessing
from model_unet import classic_unet
from model_unet_resnet import unet_resnet
import zipfile

with zipfile.ZipFile('./stage_1.zip', 'r') as zip_ref:
    zip_ref.extractall('./data')
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
TRAIN_PATH = './data/stage1_train/'
TEST_PATH = './data/stage1_test/'
seed = 42
random.seed = seed
np.random.seed = seed

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

X_train, Y_train = preprocessing.get_and_resize_train(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, TRAIN_PATH)

model_selection = input('Select and write a model: Unet or Unet_resnet?')
if model_selection == 'Unet_resnet':
    model = unet_resnet()
    name_model = 'model_unet_resnet.h5'
else: 
    model = classic_unet()
    name_model = 'model_classic_unet.h5'

model = model.build_model(IMG_WIDTH=IMG_WIDTH, IMG_HEIGHT=IMG_HEIGHT, IMG_CHANNELS=IMG_CHANNELS)
model.compile(optimizer='adam', loss=binary_crossentropy, metrics = [dice_coef, 'acc', 'mse'])

early_stopping = EarlyStopping(patience=10, verbose=1)


model_checkpoint = ModelCheckpoint(name_model, save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(factor=0.1, patience=4, min_lr=0.00001, verbose=1)

epochs = 50
batch_size = 16



results = model.fit(X_train, Y_train, validation_split=0.1,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping, model_checkpoint, reduce_lr],shuffle=True)