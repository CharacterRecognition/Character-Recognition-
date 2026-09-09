import numpy as np
from tensorflow import keras
from keras import layers
import tensorflow as tf
from Classificaltion_Evaluation import net_evaluation


def enhanced_attention(x, reduction=4, sol=None):
    if sol is None:
        sol = [5, 5, 100]
    channels = x.shape[-1]

    # Channel Attention (SE) with hidden neurons
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(channels // reduction, activation="relu")(se)
    se = layers.Dense(int(sol[1]), activation="relu")(se)  # Added hidden layer with more neurons
    se = layers.Dense(channels, activation="sigmoid")(se)
    se = layers.Reshape((1, 1, channels))(se)
    x_ca = layers.Multiply()([x, se])

    # Spatial Attention
    avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
    concat = tf.concat([avg_pool, max_pool], axis=-1)

    sa = layers.Conv2D(1, kernel_size=7, padding="same", activation="sigmoid")(concat)
    x_sa = layers.Multiply()([x, sa])

    # Fuse channel + spatial
    out = layers.Add()([x_ca, x_sa])
    return out


def transformer_block(x, num_heads=4, key_dim=32, ff_dim=128):
    # Flatten tokens
    h, w, c = x.shape[1], x.shape[2], x.shape[3]
    x_norm = layers.LayerNormalization()(x)
    qkv = layers.Dense(key_dim * num_heads * 3)(x_norm)
    qkv = layers.Reshape((h * w, 3 * num_heads, key_dim))(qkv)
    q, k, v = tf.split(qkv, 3, axis=2)
    attn = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(float(key_dim))
    attn = tf.nn.softmax(attn, axis=-1)
    out = tf.matmul(attn, v)
    out = layers.Reshape((h, w, num_heads * key_dim))(out)
    out = layers.Conv2D(c, kernel_size=1)(out)
    ff = layers.Dense(ff_dim, activation="relu")(out)
    ff = layers.Dense(c)(ff)
    return layers.Add()([x, ff])


# 3. YOLOv8 Convolution
def conv_block(x, filters, kernel=3, stride=1, act="silu", shortcut=True):
    y = layers.Conv2D(filters, kernel, stride, padding="same")(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation(tf.nn.silu if act == "silu" else act)(y)
    if shortcut:
        if x.shape[-1] != filters or stride != 1:
            x = layers.Conv2D(filters, 1, stride, padding="same")(x)
            x = layers.BatchNormalization()(x)
        y = layers.Add()([x, y])
    return y


# C2f Block + Transformer Attention + Enhanced Attention (EA)
def c2f_transformer_block(x, filters, n=2):
    path_list = []
    split = filters // 2
    # 1st conv
    p = conv_block(x, split, 1, 1, shortcut=False)
    path_list.append(p)
    # internal transformations
    for _ in range(n):
        p = conv_block(path_list[-1], split, 3, 1, shortcut=False)
        # Add Transformer attention
        p = transformer_block(p)
        # Add Enhanced Attention
        p = enhanced_attention(p)
        path_list.append(p)
    # Concatenate + fuse
    x = layers.Concatenate()(path_list)
    x = conv_block(x, filters, 1, 1, shortcut=False)
    return x


# 5. SPPF
def sppf(x, filters):
    x1 = layers.MaxPooling2D(5, 1, padding="same")(x)
    x2 = layers.MaxPooling2D(5, 1, padding="same")(x1)
    x3 = layers.MaxPooling2D(5, 1, padding="same")(x2)
    x = layers.Concatenate()([x, x1, x2, x3])
    return conv_block(x, filters, 1, 1, shortcut=False)


def AMSSTD_TYOLOv8_EA(input_shape, num_classes, sol=None):
    if sol is None:
        sol = [5, 5, 100]
    inp = keras.Input(shape=input_shape)

    # Backbone (enhanced)
    x = conv_block(inp, 32, 3, 1, shortcut=False)
    x = conv_block(x, 64, 3, 2)
    P2 = c2f_transformer_block(x, 64)

    x = conv_block(P2, 128, 3, 2)
    P3 = c2f_transformer_block(x, 128)

    x = conv_block(P3, 256, 3, 2)
    P4 = c2f_transformer_block(x, 256)

    x = conv_block(P4, 512, 3, 2)
    P5 = c2f_transformer_block(x, 512)

    x = sppf(P5, 512)

    # Neck
    x = conv_block(x, 256, 1, 1, shortcut=False)
    x = layers.UpSampling2D()(x)
    x = layers.Concatenate()([x, P4])
    x = c2f_transformer_block(x, 256)

    x = conv_block(x, int(sol[0]), 1, 1, shortcut=False)
    x = layers.UpSampling2D()(x)
    x = layers.Concatenate()([x, P3])
    x = c2f_transformer_block(x, 128)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inp, out)


def Model_AMSSTD_TYOLOv8_EA(Data, BS=4, sol=None):
    if sol is None:
        sol = [5, 5, 100]
    input_shape = (256, 256, 3)
    num_classes = 3

    # Resize input
    Train = np.zeros((Data.shape[0], *input_shape))
    for i in range(Data.shape[0]):
        Train[i] = np.resize(Data[i], input_shape)
    Test = np.zeros((Data.shape[0], *input_shape))
    for i in range(Data.shape[0]):
        Test[i] = np.resize(Data[i], input_shape)

    model = AMSSTD_TYOLOv8_EA(input_shape, num_classes, sol=sol)
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(Train, Test, batch_size=BS, epochs=10, steps_per_epoch=int(sol[2]), verbose=2)
    score = model.predict(Train)
    Eval = np.mean([net_evaluation(Test[n].astype('uint8'), score[n].astype('uint8')) for n in range(score.shape[0])],
                   axis=0)
    return Eval, score
