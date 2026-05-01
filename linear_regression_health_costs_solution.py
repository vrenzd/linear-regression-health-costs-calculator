# freeCodeCamp - Linear Regression Health Costs Calculator
# This script version downloads the dataset and trains a regression model.
# Prefer running the notebook version in Google Colab.

import os
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# 1. Reproducibility
# ============================================================

tf.keras.utils.set_random_seed(42)


# ============================================================
# 2. Import data
# ============================================================

DATA_URL = "https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv"
CSV_FILE = "insurance.csv"

if not os.path.exists(CSV_FILE):
    urllib.request.urlretrieve(DATA_URL, CSV_FILE)

dataset = pd.read_csv(CSV_FILE)


# ============================================================
# 3. Preprocess data
# ============================================================

# Convert categorical columns into numeric columns.
dataset = pd.get_dummies(
    dataset,
    columns=["sex", "smoker", "region"],
    dtype=float
)

# Ensure all numeric values are float32 for TensorFlow.
dataset = dataset.astype("float32")

# Split into train and test datasets.
train_dataset = dataset.sample(frac=0.8, random_state=42)
test_dataset = dataset.drop(train_dataset.index)

# Separate labels from features.
train_labels = train_dataset.pop("expenses")
test_labels = test_dataset.pop("expenses")

# Convert labels to float32.
train_labels = train_labels.astype("float32")
test_labels = test_labels.astype("float32")


# ============================================================
# 4. Normalize features
# ============================================================

normalizer = layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_dataset))


# ============================================================
# 5. Build and train the model
# ============================================================

model = keras.Sequential([
    normalizer,
    layers.Dense(128, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(1)
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss="mae",
    metrics=["mae", "mse"]
)

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_mae",
    patience=80,
    restore_best_weights=True
)

history = model.fit(
    train_dataset,
    train_labels,
    epochs=1000,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)


# ============================================================
# 6. Evaluate model
# ============================================================

loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=2)
print("Testing set Mean Abs Error: {:5.2f} expenses".format(mae))

if mae < 3500:
    print("You passed the challenge. Great job!")
else:
    print("The Mean Abs Error must be less than 3500. Keep trying.")


# ============================================================
# 7. Plot predictions
# ============================================================

test_predictions = model.predict(test_dataset).flatten()

a = plt.axes(aspect="equal")
plt.scatter(test_labels, test_predictions)
plt.xlabel("True values (expenses)")
plt.ylabel("Predictions (expenses)")
lims = [0, 50000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.savefig("health_costs_predictions.png", bbox_inches="tight")
plt.show()
