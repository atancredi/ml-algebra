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
weights = np.zeros((10, len(np.asarray(mnist_data[0]).flatten())))

# train on all classes with softmax
dataset = [(np.asarray(i).flatten() / 255, j) for i, j in zip(mnist_data, mnist_labels)]
print(f"start training ({len(dataset)})")
for epoch in range(epochs):
    correct = 0
    for img, label in dataset:
        z = np.dot(weights, img)
        pred = np.argmax(z)

        if pred == label:
            correct += 1
        else:
            weights[label] += lr * img
            weights[pred] -= lr * img

    acc = correct / len(mnist_data)
    print(f"Epoch {epoch+1}: training accuracy = {acc:.3f}")


# test on some examples
mnist_test_data = BatchIdx.load_idx('mnist_data/t10k-images.idx3-ubyte', max_images=1000)
mnist_test_labels = BatchIdx.load_labels('mnist_data/t10k-labels.idx1-ubyte', max_labels=1000)
test_dataset = [(np.asarray(i).flatten() / 255, j) for i, j in zip(mnist_test_data, mnist_test_labels)]
print(f"test samples ({len(test_dataset)})")
z_test = np.dot([x[0] for x in test_dataset], weights.T)
preds = np.argmax(z_test, axis=1)
accuracy = np.mean(preds == mnist_test_labels)
print(f"Test accuracy: {accuracy:.3f}")
