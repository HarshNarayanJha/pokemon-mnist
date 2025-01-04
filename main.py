# %% [md]
"""
# Importing Libraries
"""

# %%

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

    return data


model_data = cleaned_data.copy()

# Read the image into the dataframe
model_data["Image"] = model_data["Image"].apply(preprocess_image)

# label encode Type
ll = LabelEncoder()
model_data["Type"] = ll.fit_transform(model_data["Type"])

print(model_data.sample())
# %%
# Let's again check on any random pokemon
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))

for i in range(3):
    for j in range(3):
        pkmn = model_data.sample(1)
        sprite = pkmn.iloc[0]["Image"]

        # print(sprite.shape)
        axes[i, j].imshow(sprite, interpolation="nearest", cmap=plt.get_cmap("gray"))
        axes[i, j].set_title(f"{pkmn.iloc[0]['Name']} ({pkmn.iloc[0]['Type']})")

plt.show()
# %%
print(model_data.head())
print(model_data.columns)

# %%
# train test split
X, y = cleaned_data["Image"], cleaned_data["Type"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# %% [md]
"""
### Enter Easy Neural Networks!
"""

# %%
