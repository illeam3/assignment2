import os
import numpy as np

from keras.models import load_model
from keras.datasets import cifar10
from keras.utils import to_categorical

MODEL1_PATH = r"C:\Users\ram\Desktop\학교\4-1\신인지\assignment2\models\resnet50_cifar10_model1.h5"
MODEL2_PATH = r"C:\Users\ram\Desktop\학교\4-1\신인지\assignment2\models\resnet50_cifar10_model2.h5"

NUM_CLASSES = 10

if not os.path.exists(MODEL1_PATH):
    raise FileNotFoundError("model1 없음: " + MODEL1_PATH)

if not os.path.exists(MODEL2_PATH):
    raise FileNotFoundError("model2 없음: " + MODEL2_PATH)

print("Loading models...")
model1 = load_model(MODEL1_PATH)
model2 = load_model(MODEL2_PATH)
print("Models loaded.")

(_, _), (x_test, y_test) = cifar10.load_data()

x_test = x_test.astype("float32") / 255.0
y_test_cat = to_categorical(y_test, NUM_CLASSES)

loss1, acc1 = model1.evaluate(x_test, y_test_cat, verbose=0)
loss2, acc2 = model2.evaluate(x_test, y_test_cat, verbose=0)

print("Model1 test loss:", loss1)
print("Model1 test acc :", acc1)
print("Model2 test loss:", loss2)
print("Model2 test acc :", acc2)

pred1 = np.argmax(model1.predict(x_test, verbose=0), axis=1)
pred2 = np.argmax(model2.predict(x_test, verbose=0), axis=1)
y_true = y_test.reshape(-1)

disagree = np.where(pred1 != pred2)[0]
print("Disagreement count:", len(disagree))

correct1 = np.sum(pred1 == y_true)
correct2 = np.sum(pred2 == y_true)

print("Model1 correct:", int(correct1))
print("Model2 correct:", int(correct2))

import matplotlib.pyplot as plt

save_dir = r"C:\Users\ram\Desktop\학교\4-1\신인지\assignment2\results\disagreements"
os.makedirs(save_dir, exist_ok=True)

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

max_save = 5
for i, idx in enumerate(disagree[:max_save]):
    plt.figure(figsize=(4, 4))
    plt.imshow(x_test[idx])
    plt.title(
        "true={0}, model1={1}, model2={2}".format(
            class_names[y_true[idx]],
            class_names[pred1[idx]],
            class_names[pred2[idx]]
        )
    )
    plt.axis("off")
    out_path = os.path.join(save_dir, "disagreement_{0}.png".format(i))
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

print("Saved disagreement images to:", save_dir)