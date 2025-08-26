import matplotlib.pyplot as plt
import joblib
import numpy as np
from sklearn.datasets import load_digits

# Load dataset and trained model
digits = load_digits()
model = joblib.load("artifacts/model.joblib")

# Pick some random samples from the dataset
np.random.seed(42)
indices = np.random.choice(len(digits.data), size=10, replace=False)

for i, idx in enumerate(indices):
    image = digits.images[idx]
    true_label = digits.target[idx]
    pred_label = model.predict([digits.data[idx]])[0]

    plt.subplot(2, 5, i + 1)
    plt.imshow(image, cmap="gray")
    plt.title(f"P:{pred_label} / T:{true_label}")
    plt.axis("off")

plt.suptitle("Predicted (P) vs True (T)")
plt.tight_layout()
plt.show()
