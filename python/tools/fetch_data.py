import pickle
import os
import umap
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import numpy as np


# Fetch the MNIST dataset
mnist = fetch_openml('mnist_784')

# Access the data and target arrays
X, y = mnist.data, mnist.target

# Print the shape of the data
print("Data shape:", X.shape)

# Print the shape of the target
print("Target shape:", y.shape)

# save the data
np.save('data.npy', X)
np.save('target.npy', y)

# plot the data as a umap using sklearn

# Create a UMAP model
filename = 'umap_model.sav'
if os.path.exists(filename):
    model = pickle.load(open(filename, 'rb'))
else:
    model = umap.UMAP(n_neighbors=5, n_components=2, random_state=42)
    pickle.dump(model, open(filename, 'wb'))

umap_data = model.fit_transform(X)

# Create a scatter plot of the UMAP data
y_values = y.values.astype(int)
plt.scatter(umap_data[:,0], umap_data[:,1], c=y_values, cmap='Spectral', s=1)
plt.title('UMAP projection of the MNIST dataset', fontsize=24)
plt.show()
