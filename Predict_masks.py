from train import dice_coef
import preprocessing

import keras
from keras.models import Model, load_model
import numpy as np
import pandas as pd
import os
from skimage.transform import resize
from skimage.morphology import label

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
TRAIN_PATH = './data/stage1_train/'
TEST_PATH = './data/stage1_test/'

X_train, Y_train = preprocessing.get_and_resize_train(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, TRAIN_PATH)
X_test = preprocessing.get_and_resize_train(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, TEST_PATH)

model_selection = input('Select and write a model for predict: Unet or Unet_resnet?')
if model_selection == 'Unet_resnet':
    name_model = './model_unet_resnet.h5'
else: 
    name_model = './model_classic_unet.h5'

model = load_model(name_model, custom_objects={'dice_coef': dice_coef})
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Create list of upsampled test masks
preds_test_upsampled = []
test_ids = next(os.walk(TEST_PATH))[1]
sizes_test = preprocessing.size_test(test_ids, IMG_CHANNELS, TEST_PATH)
for i in range(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]), 
                                       (sizes_test[i][0], sizes_test[i][1]), 
                                       mode='constant', preserve_range=True))

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

new_test_ids = []
rles = []
for n, id_ in enumerate(test_ids):
    rle = list(prob_to_rles(preds_test_upsampled[n]))
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))

# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('predict test classic unet.csv', index=False)