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
weights = np.zeros(len(np.asarray(mnist_data[0]).flatten()))

# train on specific target class (number 7)
target_class = 7
dataset = [(np.asarray(i).flatten() / 255, j) for i, j in zip(mnist_data, mnist_labels)]
print("start training")
for epoch in range(epochs):
    correct = 0
    for img, label in dataset:
        pred = heaviside(np.dot(weights, img))
        
        binary_label = 1 if label == target_class else 0
        weights += lr * (binary_label - pred) * img 

        if pred == binary_label:
            correct += 1

    acc = correct / len(mnist_data)
    print(f"Epoch {epoch+1}: training accuracy = {acc:.3f}")

# test on some examples
mnist_test_data = BatchIdx.load_idx('mnist_data/t10k-images.idx3-ubyte', max_images=1000)
mnist_test_labels = BatchIdx.load_labels('mnist_data/t10k-labels.idx1-ubyte', max_labels=1000)
test_dataset = [(np.asarray(i).flatten() / 255, j) for i, j in zip(mnist_test_data, mnist_test_labels)]
print(f"test samples of class {target_class} ({len(test_dataset)})")
tp = 0
fp = 0
tn = 0
fn = 0
for img, label in test_dataset:

    pred = heaviside(np.dot(weights, img))
    
    binary_label = 1 if label == target_class else 0
    if pred == 1 and label == target_class:
        tp += 1
    elif pred == 1 and label != target_class:
        fp += 1
    elif pred == 0 and label == target_class:
        fn += 1
    elif pred == 0 and label != target_class:
        tn += 1
    
    # print(f"Binary target (is {target_class}?):", label, "Predicted:", pred)

acc = (tp + tn) / len(test_dataset)
print("Test accuracy:", acc)
print("tp",tp,"fp",fp,"fn",fn,"tn",tn)