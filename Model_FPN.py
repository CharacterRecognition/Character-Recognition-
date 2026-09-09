import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import numpy as np
from tensorflow.keras.optimizers import Adam

from Classificaltion_Evaluation import net_evaluation


def residual_block(x, filters, stride=1):
    """Basic Residual Block"""
    shortcut = x
    x = layers.Conv2D(filters, (3, 3), strides=stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, (3, 3), strides=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x


def backbone(input_tensor):
    """Backbone Network"""
    x = layers.Conv2D(64, (7, 7), strides=2, padding='same', use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    c1 = residual_block(x, 64)
    c2 = residual_block(c1, 128, stride=2)
    c3 = residual_block(c2, 256, stride=2)
    c4 = residual_block(c3, 512, stride=2)

    return [c1, c2, c3, c4]


def fpn(features):
    """Feature Pyramid Network"""
    p4 = layers.Conv2D(256, 1, padding="same")(features[3])
    p3 = layers.Add()([layers.Conv2D(256, 1, padding="same")(features[2]), layers.UpSampling2D(size=2)(p4)])
    p2 = layers.Add()([layers.Conv2D(256, 1, padding="same")(features[1]), layers.UpSampling2D(size=2)(p3)])
    p1 = layers.Add()([layers.Conv2D(256, 1, padding="same")(features[0]), layers.UpSampling2D(size=2)(p2)])

    # Upsample to the original input shape (512, 512, 3)
    p1 = layers.UpSampling2D(size=2)(p1)
    p1 = layers.UpSampling2D(size=2)(p1)
    p1 = layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same")(p1)  # Final Single Image Output
    return p1


def fpn_model(input_shape):
    inputs = Input(shape=input_shape)
    backbone_features = backbone(inputs)
    output_image = fpn(backbone_features)

    model = Model(inputs, output_image)
    return model


def Model_FPN(Data):
    input_shape = (512, 512, 3)
    Data_resized = np.resize(Data, (Data.shape[0], input_shape[0], input_shape[1], input_shape[2]))
    Target_resized = np.resize(Data, (Data.shape[0], input_shape[0], input_shape[1], input_shape[2]))
    model = fpn_model(input_shape)
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse', metrics=['accuracy'])
    model.summary()

    model.fit(Data_resized, Target_resized, epochs=50, batch_size=32, validation_split=0.25)
    Predict = model.predict(Data_resized)
    Eval = [net_evaluation(Target_resized[n].astype('uint8'), Predict[0][n].astype('uint8')) for n in
            range(Predict[0].shape[0])]
    EVAl = np.mean(Eval, axis=0)
    return EVAl, Predict

