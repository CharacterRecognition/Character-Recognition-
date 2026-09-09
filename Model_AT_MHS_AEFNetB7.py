import numpy as np
import tensorflow as tf
from keras import Model
from keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    Reshape, GlobalMaxPooling1D, MultiHeadAttention, Add
)
from keras.applications import EfficientNetB7
from Classificaltion_Evaluation import net_evaluation


# Custom Swin Transformer Block

def SwinTransformerBlock(inputs, num_heads=4, mlp_dim=256, dropout=0.1):
    # Multi-Head Self Attention
    x = LayerNormalization()(inputs)
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(x, x)
    x = Add()([inputs, Dropout(dropout)(attn_output)])

    # MLP
    y = LayerNormalization()(x)
    y = Dense(mlp_dim, activation='relu')(y)
    y = Dense(inputs.shape[-1])(y)
    x = Add()([x, Dropout(dropout)(y)])
    return x


def Model_AT_MHS_AEFNetB7(Train_Data):

    IMG_SIZE = (32, 32, 3)
    num_classes = 3
    Train_Data = np.array([
        tf.image.resize(img, IMG_SIZE[:2]).numpy()
        for img in Train_Data
    ])

    Train_Target = np.array([
        tf.image.resize(img, IMG_SIZE[:2]).numpy()
        for img in Train_Data
    ])

    # EfficientNetB7 Base Model
    eff_model = EfficientNetB7(
        include_top=False,
        input_shape=IMG_SIZE,
        weights="imagenet"
    )
    eff_model.trainable = False

    # Model Input
    inputs = Input(shape=IMG_SIZE)
    eff_features = eff_model(inputs)  # (B, H, W, C)
    h, w, c = eff_features.shape[1], eff_features.shape[2], eff_features.shape[3]

    # Flatten spatial dimensions → sequence
    x = Reshape((h * w, c))(eff_features)  # (B, H*W, C)

    # Swin Transformer Block(s)
    for _ in range(2):  # number of layers
        x = SwinTransformerBlock(x, num_heads=4, mlp_dim=256, dropout=0.1)

    x = LayerNormalization()(x)
    x = Dropout(0.1)(x)

    # Global Max Pooling over sequence
    x = GlobalMaxPooling1D()(x)

    # Fully Connected + Output
    x = Dense(128, activation="relu")(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    # Build & Compile Model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    # Training
    model.fit(Train_Data, Train_Target, epochs=10, batch_size=4, validation_split=0.25)

    # Evaluation
    pred = model.predict(Train_Data)
    Eval = [net_evaluation(Train_Target[n].astype("uint8"),pred[n].astype("uint8"))for n in range(pred.shape[0])]
    EVAl = np.mean(Eval, axis=0)
    return EVAl, pred



