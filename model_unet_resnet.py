import tensorflow as tf
import keras
from keras.models import Model, load_model
from keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout, UpSampling2D, MaxPooling2D, Concatenate, Activation
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.applications.resnet50 import ResNet50


class unet_resnet():
    
    def __init__(self):
        self.model = None
    def build_model(self, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS):
        base_model = ResNet50(weights='imagenet', input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), include_top=False)

        conv1 = base_model.get_layer('conv1_relu').output #128, 128, 64
        conv2 = base_model.get_layer('conv2_block1_out').output #(None, 64, 64, 256)
        conv3 = base_model.get_layer('conv3_block3_out').output #(None, 32, 32, 512)
        conv4 = base_model.get_layer('conv4_block6_out').output #(None, 16, 16, 1024) 
        conv5 = base_model.get_layer('conv5_block3_out').output #(None, 8, 8, 2048) 

        inp = base_model.get_layer(base_model.layers[0].name).output 

        up1 = Conv2DTranspose(2048, (2, 2), strides=(2, 2), padding='same') (conv5)
        conc_1 = Concatenate()([up1, conv4]) #2048+1024
        conv_conc_1 = Conv2D(1024, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conc_1)
        conv_conc_1 = Dropout(0.1) (conv_conc_1)

        up2 = Conv2DTranspose(1024, (2, 2), strides=(2, 2), padding='same') (conv_conc_1)
        conc_2 = Concatenate()([up2, conv3]) #1024+512
        conv_conc_2 = Conv2D(512, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conc_2)
        conv_conc_2 = Dropout(0.1) (conv_conc_2)

        up3 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same') (conv_conc_2)
        conc_3 = Concatenate()([up3, conv2]) #512+256
        conv_conc_3 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conc_3)
        conv_conc_3 = Dropout(0.1) (conv_conc_3)

        up4 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (conv_conc_3)
        conc_4 = Concatenate()([up4, conv1]) #256 + 64
        conv_conc_4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conc_4)
        conv_conc_4 = Dropout(0.1) (conv_conc_4)

        up5 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (conv_conc_4)#256, 256, 32
        conv_conc_5 = Conv2D(1, (1, 1), padding='same')(up5)
        conv_conc_5 = Activation('sigmoid')(conv_conc_5)

        self.model = Model(inputs=base_model.input, outputs=conv_conc_5)   
        return self.model


