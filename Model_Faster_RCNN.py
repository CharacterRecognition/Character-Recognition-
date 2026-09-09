import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from Classificaltion_Evaluation import net_evaluation


def load_fasterrcnn_model(num_classes):
    base_model = tf.keras.applications.VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=(None, None, 3)
    )
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model


def Model_Faster_RCNN(Data):
    IMG_SIZE = 32
    NUM_CLASSES = 3
    X = np.zeros((Data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(Data.shape[0]):
        temp = np.resize(Data[i], (IMG_SIZE * IMG_SIZE, 3))
        X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))
    Target = np.zeros((Data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(Data.shape[0]):
        temp = np.resize(Data[i], (IMG_SIZE * IMG_SIZE, 3))
        Target[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    model = load_fasterrcnn_model(NUM_CLASSES)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(X, Target, epochs=10, batch_size=32, validation_split=0.25)
    pred = model.predict(X)
    pred_labels = np.zeros_like(pred)
    pred_labels[np.arange(pred.shape[0]), np.argmax(pred, axis=1)] = 1
    Eval = [net_evaluation(Target[n].astype('uint8'),pred_labels[n].astype('uint8'))for n in range(pred.shape[0])]
    return Eval, pred


