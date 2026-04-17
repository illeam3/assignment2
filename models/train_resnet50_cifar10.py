import os
import random
import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10
from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator

# 재현성용 seed
SEED = 1
random.seed(SEED)
np.random.seed(SEED)
tf.set_random_seed(SEED)

# 저장 폴더
os.makedirs("models", exist_ok=True)

# 설정
NUM_CLASSES = 10
BATCH_SIZE = 64
EPOCHS = 10
INPUT_SHAPE = (32, 32, 3)
MODEL_PATH = "models/resnet50_cifar10_model1.h5"

# CIFAR-10 로드
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 정규화
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# one-hot
y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

# ResNet50 기본 모델
inputs = Input(shape=INPUT_SHAPE)
base_model = ResNet50(
    include_top=False,
    weights=None,
    input_tensor=inputs
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
outputs = Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

model = Model(inputs=inputs, outputs=outputs)

# 컴파일
optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# 데이터 증강
datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)

# 최고 성능 모델 저장
checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_acc",
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

# 학습
model.fit_generator(
    datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(x_train) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_test, y_test),
    callbacks=[checkpoint],
    verbose=1
)

# 최종 평가
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", loss)
print("Test accuracy:", acc)
print("Saved to:", MODEL_PATH)