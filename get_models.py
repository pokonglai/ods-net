from __future__ import print_function
import keras

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.metrics import *
from keras import backend as K
from keras.regularizers import l2

import tensorflow as tf
import numpy as np

from custom_func import *

from keras.utils import plot_model
	
def create_model_ODS_Net(img_input_shape, hyper_params):
    print("img_input_shape=",img_input_shape)

    base_depth = 64
    dropout_rate = 0.5

    first_input = Input(shape=img_input_shape)
    second_input = Input(shape=img_input_shape)

    concatenated_images = concatenate([first_input, second_input])

    conv1 = SeparableConv2D(base_depth, (3, 3), padding='same')(concatenated_images)
    conv1 = Activation('relu')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = SeparableConv2D(base_depth, (3, 3), padding='same')(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = SeparableConv2D(base_depth*2, (3, 3), padding='same')(pool1)
    conv2 = Activation('relu')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = SeparableConv2D(base_depth*2, (3, 3), padding='same')(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = SeparableConv2D(base_depth*4, (3, 3), padding='same')(pool2)
    conv3 = Activation('relu')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = SeparableConv2D(base_depth*4, (3, 3), padding='same')(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = SeparableConv2D(base_depth*8, (3, 3), padding='same')(pool3)
    conv4 = Activation('relu')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = SeparableConv2D(base_depth*8, (3, 3), padding='same')(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = SeparableConv2D(base_depth*16, (3, 3), padding='same')(pool4)
    conv5 = Activation('relu')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = SeparableConv2D(base_depth*16, (3, 3), padding='same')(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = concatenate([Conv2DTranspose(base_depth*8, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(base_depth*8, (3, 3), padding='same')(up6)
    conv6 = Activation('relu')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = SeparableConv2D(base_depth*8, (3, 3), padding='same')(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Dropout(rate=dropout_rate)(conv6)

    up7 = concatenate([Conv2DTranspose(base_depth*4, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(base_depth*4, (3, 3), padding='same')(up7)
    conv7 = Activation('relu')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = SeparableConv2D(base_depth*4, (3, 3), padding='same')(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Dropout(rate=dropout_rate)(conv7)

    up8 = concatenate([Conv2DTranspose(base_depth*2, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(base_depth*2, (3, 3), padding='same')(up8)
    conv8 = Activation('relu')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = SeparableConv2D(base_depth*2, (3, 3), padding='same')(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Dropout(rate=dropout_rate)(conv8)

    up9 = concatenate([Conv2DTranspose(base_depth*2, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(base_depth*2, (3, 3), padding='same')(up9)
    conv9 = Activation('relu')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = SeparableConv2D(base_depth*2, (3, 3), padding='same')(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Dropout(rate=dropout_rate)(conv9)

    ## increase FCL when compared to v2
    dense1 = Dense(base_depth*4,  activation='relu')(conv9)
    dense1 = Dropout(rate=dropout_rate)(dense1)
    dense1 = Dense(base_depth*2,  activation='relu')(dense1)
    dense1 = Dropout(rate=dropout_rate)(dense1)
    dense1 = Dense(base_depth, )(dense1)
    dense1 = Dense(base_depth//2, )(dense1)

    output_depth = Dense(1, name="depth_map")(dense1)
    output_norms = Dense(3, name="normal_map")(dense1)

    model = Model(inputs=[first_input, second_input], outputs=[output_depth, output_norms])
    model.summary()

    model_optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=model_optimizer, 
                  loss=berHu_loss_elementwise,
                  metrics=[rmse, median_dist, log10_error])
    return model

def create_model_ODS_Net_w_borderLoss(img_input_shape, hyper_params):
    print("img_input_shape=",img_input_shape)

    base_depth = 64
    dropout_rate = 0.5

    first_input = Input(shape=img_input_shape)
    second_input = Input(shape=img_input_shape)

    concatenated_images = concatenate([first_input, second_input])

    conv1 = SeparableConv2D(base_depth, (3, 3), padding='same')(concatenated_images)
    conv1 = Activation('relu')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = SeparableConv2D(base_depth, (3, 3), padding='same')(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = SeparableConv2D(base_depth*2, (3, 3), padding='same')(pool1)
    conv2 = Activation('relu')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = SeparableConv2D(base_depth*2, (3, 3), padding='same')(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = SeparableConv2D(base_depth*4, (3, 3), padding='same')(pool2)
    conv3 = Activation('relu')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = SeparableConv2D(base_depth*4, (3, 3), padding='same')(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = SeparableConv2D(base_depth*8, (3, 3), padding='same')(pool3)
    conv4 = Activation('relu')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = SeparableConv2D(base_depth*8, (3, 3), padding='same')(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = SeparableConv2D(base_depth*16, (3, 3), padding='same')(pool4)
    conv5 = Activation('relu')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = SeparableConv2D(base_depth*16, (3, 3), padding='same')(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = concatenate([Conv2DTranspose(base_depth*8, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(base_depth*8, (3, 3), padding='same')(up6)
    conv6 = Activation('relu')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = SeparableConv2D(base_depth*8, (3, 3), padding='same')(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Dropout(rate=dropout_rate)(conv6)

    up7 = concatenate([Conv2DTranspose(base_depth*4, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(base_depth*4, (3, 3), padding='same')(up7)
    conv7 = Activation('relu')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = SeparableConv2D(base_depth*4, (3, 3), padding='same')(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Dropout(rate=dropout_rate)(conv7)

    up8 = concatenate([Conv2DTranspose(base_depth*2, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(base_depth*2, (3, 3), padding='same')(up8)
    conv8 = Activation('relu')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = SeparableConv2D(base_depth*2, (3, 3), padding='same')(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Dropout(rate=dropout_rate)(conv8)

    up9 = concatenate([Conv2DTranspose(base_depth*2, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(base_depth*2, (3, 3), padding='same')(up9)
    conv9 = Activation('relu')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = SeparableConv2D(base_depth*2, (3, 3), padding='same')(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Dropout(rate=dropout_rate)(conv9)

    ## increase FCL when compared to v2
    dense1 = Dense(base_depth*4,  activation='relu')(conv9)
    dense1 = Dropout(rate=dropout_rate)(dense1)
    dense1 = Dense(base_depth*2,  activation='relu')(dense1)
    dense1 = Dropout(rate=dropout_rate)(dense1)
    dense1 = Dense(base_depth, )(dense1)
    dense1 = Dense(base_depth//2, )(dense1)

    output_depth = Dense(1, name="depth_map")(dense1)
    output_norms = Dense(3, name="normal_map")(dense1)

    model = Model(inputs=[first_input, second_input], outputs=[output_depth, output_norms])
    model.summary()

    model_optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=model_optimizer, 
                  loss=berHu_loss_elementwise_w_border,
                  metrics=[rmse, median_dist, log10_error])
    return model



def get_model(model_config_name, img_input_shape, hyper_params):
    model = None

    if model_config_name == "ODS_Net": model = create_model_ODS_Net(img_input_shape, hyper_params) 
    elif  model_config_name == "ODS_Net_borderloss": model = create_model_ODS_Net_w_borderLoss(img_input_shape, hyper_params) 
    
    print("Estimated memory usage for parameters: ", get_model_memory_usage(hyper_params["batch_size"], model), "gbytes")
    return model

def get_model_memory_usage(batch_size, model):
    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes
