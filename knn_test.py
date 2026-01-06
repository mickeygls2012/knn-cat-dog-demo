import os
import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Parameters
IMG_SIZE = 64
DATASET_PATH = "dataset"
TEST_IMAGES = ["test1.jpg", "test2.jpg", "test3.jpg"]

def load_images(folder, label):
    data = []
    labels = []
    for file in os.listdir(folder):
        img = Image.open(os.path.join(folder, file)).convert("L")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img).flatten()
        data.append(img_array)
        labels.append(label)
    return data, labels

# Load dataset
cats, cat_labels = load_images(DATASET_PATH + "/cats", 0)
dogs, dog_labels = load_images(DATASET_PATH + "/dogs", 1)

X = np.array(cats + dogs)
y = np.array(cat_labels + dog_labels)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

# Prepare figure
plt.figure(figsize=(12, 4))

for i, file in enumerate(TEST_IMAGES):
    img = Image.open(file).convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    vector = np.array(img).flatten()

    prediction = knn.predict([vector])[0]
    label = "Cat" if prediction == 0 else "Dog"

    plt.subplot(1, 3, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"{file}\nPrediction: {label}")
    plt.axis("off")

# Save final result
plt.savefig("result.jpg")
plt.show()

print("Saved output as result.jpg")
