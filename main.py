# %% [md]
"""
# Importing Libraries
"""

# %%
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# %% [md]
"""
## EDA (Exploratory Data Analysis)
"""

# %%
data = pd.read_csv("data/pokedex.csv")
print(data.head())
print(data.columns)
print(data.shape)

# %%
# A little bit of renaming and cleaning

cleaned_data = data.rename({"Index": "ID", "Type 1": "Type"}, axis=1)
cleaned_data = cleaned_data[["Image", "ID", "Name", "Type"]]

# %%
# Let's check on any random pokemon

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))

for i in range(3):
    for j in range(3):
        pkmn = cleaned_data.sample(1)
        # print(pkmn)
        img = Image.open(f"data/{pkmn.iloc[0]['Image']}")
        sprite = np.array(img)
        img.close()

        # print(sprite.shape)
        axes[i, j].imshow(sprite, interpolation="nearest", cmap=plt.get_cmap("gray"))
        axes[i, j].set_title(f"{pkmn.iloc[0]['Name']} ({pkmn.iloc[0]['Type']})")

plt.show()

# %%
# Some stats and graphs for analysis
print(cleaned_data.info())
print()
print(cleaned_data["Type"].describe())

# %% [md]
"""
We can see there are 1215 data samples, all non-null, with 18 unique types
"""

# %%
# cleaned_data["Type"].value_counts().plot(kind='pie')
cleaned_data["Type"].value_counts().plot(kind="bar")

# %% [md]
"""
The most common type is the Water Type
The least is Flying
"""
# %%
class_names = cleaned_data["Type"].unique()
print(class_names)
# %% [md]
"""
## Preprocessing
"""


# %%
def preprocess_image(path: str):
    """
    This function reads the image path and returns and np.ndarray containing the image data as an numpy array.
    """
    pth = f"data/{path}"
    data = None
    with Image.open(pth) as img:
        data = np.array(img)

    return data / 255


model_data = cleaned_data.copy()

# Read the image into the dataframe
model_data["Image"] = model_data["Image"].apply(preprocess_image)

# label encode Type
ll = LabelEncoder()
model_data["Type"] = ll.fit_transform(model_data["Type"])

print(model_data.sample())
print(model_data.info())

# %%
# Let's again check on any random pokemon
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))

for i in range(3):
    for j in range(3):
        pkmn = model_data.sample(1)
        sprite = pkmn.iloc[0]["Image"]

        # print(sprite.shape)
        axes[i, j].imshow(
            sprite,
            interpolation="nearest",
        )
        axes[i, j].set_title(f"{pkmn.iloc[0]['Name']} ({ll.inverse_transform([pkmn.iloc[0]['Type']])})")

plt.show()
# %%
# print(model_data.head())
print(model_data.columns)

# train test split
X, y = np.stack(model_data["Image"].values), model_data["Type"].values
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, train_size=0.8, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, X_val.shape, y_val.shape)

# %% [md]
"""
### Enter Easy Neural Networks!
"""

# %%
model = keras.models.Sequential()
model.add(keras.layers.Input(shape=(112, 120, 4)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(200, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(18, activation="softmax"))

# %%
model.summary()

# %%
model.layers

# %%
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

# %%
history: keras.callbacks.History = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# %%
pd.DataFrame(history.history).plot(figsize=(15, 8))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

# %%
model.evaluate(X_test, y_test)

# %%
y_prob: np.ndarray = model.predict(X_test)
y_classes = y_prob.argmax(axis=-1)
print(y_classes)

# %%
confusion_matrix = tf.math.confusion_matrix(y_test, y_classes)

# %%
axs = sns.heatmap(confusion_matrix, annot=True, fmt="g", cmap="Blues")

axs.set_xlabel("Predicted Labels")
axs.set_ylabel("True Labels")
axs.set_title("Confusion Matrix")
axs.xaxis.set_ticklabels(class_names)
axs.yaxis.set_ticklabels(class_names)
axs.figure.set_size_inches(18, 18)

plt.show()


# %% Custom Image Test
def preprocess_custom_image(image_path):
    img = Image.open(image_path)
    img = img.resize((120, 112))

    img = img.convert("RGBA")

    sprite = np.array(img)
    img.close()

    return sprite / 255


sprites = np.array(
    [
        preprocess_custom_image("psyduck.png"),
        preprocess_custom_image("golduck.png"),
        preprocess_custom_image("makuhita.png"),
        preprocess_custom_image("marshadow.webp"),
        preprocess_custom_image("shiftry.webp"),
        preprocess_custom_image("moltres.webp"),
        preprocess_custom_image("braviary.webp"),
        preprocess_custom_image("tepig.webp"),
    ]
)

sprites_t = tf.convert_to_tensor(sprites, dtype=tf.float32)

pred = model.predict(sprites_t)
predicted_classes = np.argmax(pred, axis=1)
predicted_types = ll.inverse_transform(predicted_classes)

# Visualize the test
fig, axes = plt.subplots(nrows=len(sprites), ncols=2, figsize=(15, 15))

for i, img in enumerate(sprites):
    # plot image
    axes[i, 0].imshow(img)
    axes[i, 0].set_title(f"Predicted Type: {predicted_types[i]}")
    axes[i, 0].axis("off")

    # plot distribution
    axes[i, 1].set_title("Probability distribution:\n")
    prediction_prob = []
    types = []
    for type_name, prob in zip(ll.classes_, pred[i]):
        prediction_prob.append(prob)
        types.append(type_name)

    axes[i, 1].bar(types, prediction_prob)
    axes[i, 1].tick_params(axis="x", rotation=45)
    axes[i, 1].set_ylim(0, 1)

plt.show()
