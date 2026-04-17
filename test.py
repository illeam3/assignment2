import os
import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras import backend as K

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL1_PATH = os.path.join(BASE_DIR, "models", "resnet50_cifar10_model1.h5")
MODEL2_PATH = os.path.join(BASE_DIR, "models", "resnet50_cifar10_model2.h5")

RESULT_DIR = os.path.join(BASE_DIR, "results", "disagreements")
SUMMARY_PATH = os.path.join(BASE_DIR, "results", "summary.txt")

NUM_CLASSES = 10
BATCH_SIZE = 128

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def get_relu_layers(model):
    relu_layers = []
    for layer in model.layers:
        if layer.__class__.__name__ == "Activation":
            config = layer.get_config()
            if config.get("activation") == "relu":
                relu_layers.append(layer)
    return relu_layers


def compute_neuron_coverage(model, x, batch_size=128):
    """
    Channel-level neuron coverage for ReLU activation layers.
    A channel is counted as covered if it becomes > 0 for at least one input.
    """
    relu_layers = get_relu_layers(model)

    if len(relu_layers) == 0:
        return 0.0, 0, 0

    layer_functions = []
    covered_flags = []
    total_neurons = 0

    for layer in relu_layers:
        func = K.function([model.input, K.learning_phase()], [layer.output])
        layer_functions.append(func)

        channels = int(layer.output_shape[-1])
        covered_flags.append(np.zeros(channels, dtype=np.bool_))
        total_neurons += channels

    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))
        xb = x[start:end]

        for i, func in enumerate(layer_functions):
            out = func([xb, 0])[0]

            # Conv feature map: (batch, h, w, c)
            if out.ndim == 4:
                active = np.any(out > 0, axis=(0, 1, 2))
            # Dense-like: (batch, units)
            elif out.ndim == 2:
                active = np.any(out > 0, axis=0)
            else:
                reshaped = out.reshape((out.shape[0], -1))
                active = np.any(reshaped > 0, axis=0)

            covered_flags[i] = np.logical_or(covered_flags[i], active)

    covered_neurons = int(sum(flag.sum() for flag in covered_flags))
    coverage = float(covered_neurons) / float(total_neurons)

    return coverage, covered_neurons, total_neurons


if not os.path.exists(MODEL1_PATH):
    raise FileNotFoundError("model1 없음: " + MODEL1_PATH)

if not os.path.exists(MODEL2_PATH):
    raise FileNotFoundError("model2 없음: " + MODEL2_PATH)

os.makedirs(RESULT_DIR, exist_ok=True)

print("Loading models...")
model1 = load_model(MODEL1_PATH)
model2 = load_model(MODEL2_PATH)
print("Models loaded.")

(_, _), (x_test, y_test) = cifar10.load_data()
x_test = x_test.astype("float32") / 255.0
y_test_cat = to_categorical(y_test, NUM_CLASSES)
y_true = y_test.reshape(-1)

loss1, acc1 = model1.evaluate(x_test, y_test_cat, verbose=0)
loss2, acc2 = model2.evaluate(x_test, y_test_cat, verbose=0)

print("Model1 test loss:", loss1)
print("Model1 test acc :", acc1)
print("Model2 test loss:", loss2)
print("Model2 test acc :", acc2)

pred1 = np.argmax(model1.predict(x_test, verbose=0), axis=1)
pred2 = np.argmax(model2.predict(x_test, verbose=0), axis=1)

disagree = np.where(pred1 != pred2)[0]
print("Disagreement count:", len(disagree))

correct1 = int(np.sum(pred1 == y_true))
correct2 = int(np.sum(pred2 == y_true))

print("Model1 correct:", correct1)
print("Model2 correct:", correct2)

# save 5 disagreement images
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
    out_path = os.path.join(RESULT_DIR, "disagreement_{0}.png".format(i))
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

print("Saved disagreement images to:", RESULT_DIR)

# neuron coverage
print("Computing neuron coverage...")
cov1, covered1, total1 = compute_neuron_coverage(model1, x_test, batch_size=BATCH_SIZE)
cov2, covered2, total2 = compute_neuron_coverage(model2, x_test, batch_size=BATCH_SIZE)

print("Model1 neuron coverage: {:.4f} ({}/{})".format(cov1, covered1, total1))
print("Model2 neuron coverage: {:.4f} ({}/{})".format(cov2, covered2, total2))

with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
    f.write("Model1 test loss: {}\n".format(loss1))
    f.write("Model1 test acc: {}\n".format(acc1))
    f.write("Model2 test loss: {}\n".format(loss2))
    f.write("Model2 test acc: {}\n".format(acc2))
    f.write("Disagreement count: {}\n".format(len(disagree)))
    f.write("Model1 correct: {}\n".format(correct1))
    f.write("Model2 correct: {}\n".format(correct2))
    f.write("Model1 neuron coverage: {:.4f} ({}/{})\n".format(cov1, covered1, total1))
    f.write("Model2 neuron coverage: {:.4f} ({}/{})\n".format(cov2, covered2, total2))

print("Saved summary to:", SUMMARY_PATH)