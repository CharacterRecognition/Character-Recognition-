import keras
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Add, Concatenate, UpSampling2D
import numpy as np
from Classificaltion_Evaluation import net_evaluation


def conv_bn_leaky(x, filters, kernel_size, strides=1):
    """Conv + BN + LeakyReLU"""
    x = Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x


def csp_block(x, filters, n=1):
    """Cross Stage Partial block (used in YOLOv5 backbone)"""
    route = x
    x = conv_bn_leaky(x, filters // 2, 1)
    route_1 = x
    x = conv_bn_leaky(x, filters // 2, 1)
    for _ in range(n):
        y = conv_bn_leaky(x, filters // 2, 1)
        y = conv_bn_leaky(y, filters // 2, 3)
        x = Add()([x, y])
    x = conv_bn_leaky(x, filters // 2, 1)
    x = Concatenate()([x, route_1])
    x = conv_bn_leaky(x, filters, 1)
    return x


#  YOLOv5 model definition

def yolo_v5_model(input_size=(256, 256, 3), num_classes=3):
    inputs = Input(input_size)

    # Backbone (CSPDarknet-like)
    x = conv_bn_leaky(inputs, 32, 3)  # P1
    x = conv_bn_leaky(x, 64, 3, strides=2)  # Downsample
    x = csp_block(x, 64, n=1)

    x = conv_bn_leaky(x, 128, 3, strides=2)  # P2
    x = csp_block(x, 128, n=3)

    x = conv_bn_leaky(x, 256, 3, strides=2)  # P3
    c3 = csp_block(x, 256, n=3)  # Small-scale features

    x = conv_bn_leaky(c3, 512, 3, strides=2)  # P4
    c4 = csp_block(x, 512, n=3)  # Medium-scale features

    x = conv_bn_leaky(c4, 1024, 3, strides=2)  # P5
    c5 = csp_block(x, 1024, n=1)  # Large-scale features

    # --- Neck (PANet-like feature fusion) ---
    p5 = conv_bn_leaky(c5, 512, 1)
    up5 = UpSampling2D(2)(p5)
    p4 = conv_bn_leaky(c4, 512, 1)
    p4 = Concatenate()([p4, up5])
    p4 = csp_block(p4, 512, n=1)

    up4 = UpSampling2D(2)(p4)
    p3 = conv_bn_leaky(c3, 256, 1)
    p3 = Concatenate()([p3, up4])
    p3 = csp_block(p3, 256, n=1)

    # Head (Detection layers)
    # YOLOv5 outputs 3 scales (P3, P4, P5)
    detect_small = Conv2D(num_classes + 5, 1, activation='sigmoid')(p3)
    detect_medium = Conv2D(num_classes + 5, 1, activation='sigmoid')(p4)
    detect_large = Conv2D(num_classes + 5, 1, activation='sigmoid')(p5)

    model = Model(inputs, [detect_small, detect_medium, detect_large])
    return model


def Model_YOLOv5(Images, HN=None, sol=None):
    if sol is None:
        sol = [4, 50, 0, 5, 0]
    if HN is None:
        HN = 64

    IMG_SIZE = 256
    classes = 3
    optimizer_list = ['SGD', 'Adam', 'RMSprop', 'Adagrad', 'Adadelta']
    input_shape = (IMG_SIZE, IMG_SIZE, 3)

    # Resize inputs
    Train_Temp = np.zeros((Images.shape[0], *input_shape))
    for i in range(Images.shape[0]):
        Train_Temp[i, :] = np.resize(Images[i], input_shape)
    Train_X = Train_Temp

    Train_Temp = np.zeros((Images.shape[0], *input_shape))
    for i in range(Images.shape[0]):
        Train_Temp[i, :] = np.resize(Images[i], input_shape)
    Train_Y = Train_Temp

    # Build YOLOv5 model
    model = yolo_v5_model(input_size=input_shape, num_classes=classes)

    model.compile(
        optimizer=optimizer_list[int(sol[2])],
        loss=keras.losses.binary_crossentropy,
        metrics=['accuracy']
    )
    model.summary()
    model.fit(Train_X, Train_Y, epochs=sol[1], steps_per_epoch=2, verbose="auto")
    Predict = model.predict(Train_X)
    Eval = [net_evaluation(Train_Y[n].astype('uint8'), Predict[0][n].astype('uint8')) for n in
            range(Predict[0].shape[0])]
    EVAl = np.mean(Eval, axis=0)
    return Predict, EVAl
