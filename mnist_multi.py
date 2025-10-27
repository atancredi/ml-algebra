# download mnist
# install dependencies (https://github.com/atancredi/idxtools)

from idxtools import BatchIdx
import numpy as np

def heaviside(x):
    return 1 if x > 0 else 0

mnist_data = BatchIdx.load_idx('mnist_data/train-images.idx3-ubyte', max_images=10000)
mnist_labels = BatchIdx.load_labels('mnist_data/train-labels.idx1-ubyte', max_labels=10000)

lr = 0.0001
epochs = 5
label_classes = range(10)
weights = {x: np.zeros(len(np.asarray(mnist_data[0]).flatten())) for x in label_classes}

# train 10 perceptrons on every specific target class
dataset = [(np.asarray(i).flatten() / 255, j) for i, j in zip(mnist_data, mnist_labels)]
print(f"start training ({len(dataset)})")
for epoch in range(epochs):
    correct_per_class = {x: 0 for x in label_classes}
    for img, label in dataset:
        for cls in label_classes:
            pred = heaviside(np.dot(weights[cls], img))
        
            binary_label = 1 if label == cls else 0 
            weights[cls] += lr * (binary_label - pred) * img

            if pred == binary_label:
                correct_per_class[cls] += 1

    acc_per_class = {x: correct_per_class[x] / len(mnist_data) for x in label_classes}
    print(f"Epoch {epoch+1}: training accuracy")
    print(acc_per_class)

# test on some examples
mnist_test_data = BatchIdx.load_idx('mnist_data/t10k-images.idx3-ubyte', max_images=1000)
mnist_test_labels = BatchIdx.load_labels('mnist_data/t10k-labels.idx1-ubyte', max_labels=1000)
test_dataset = [(np.asarray(i).flatten() / 255, j) for i, j in zip(mnist_test_data, mnist_test_labels)]
print(f"testing samples ({len(test_dataset)})")
metrics_per_class = {x: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for x in label_classes}
for img, label in test_dataset:

    for cls in label_classes:

        pred = heaviside(np.dot(weights[cls], img))
    
        binary_label = 1 if label == cls else 0  # target class: digit 0
        if pred == 1 and label == cls:
            metrics_per_class[cls]["tp"] += 1
        elif pred == 1 and label != cls:
            metrics_per_class[cls]["fp"] += 1
        elif pred == 0 and label == cls:
            metrics_per_class[cls]["fn"] += 1
        elif pred == 0 and label != cls:
            metrics_per_class[cls]["tn"] += 1
    
    # print(f"Binary target (is {target_class}?):", label, "Predicted:", pred)

acc_per_class = {x: (metrics_per_class[x]["tp"] + metrics_per_class[x]["tn"]) / len(test_dataset) for x in label_classes}
for c in acc_per_class:
    print(f"Test accuracy for class {c}:", acc_per_class[c], "tp",metrics_per_class[c]["tp"],"fp",metrics_per_class[c]["fp"],"fn",metrics_per_class[c]["fn"],"tn",metrics_per_class[c]["tn"])
