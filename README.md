# Pokemon Classification using Neural Nets

This repo contains files for predicting pokemon types from sprites, using a neural network with keras.

## Datasets

1. Image with Stats Dataset
   https://www.kaggle.com/datasets/christofferms/pokemon-with-stats-and-image

## How it works

#### Exploratory Data Analysis

While exploring the data (doing EDA), we find out that we have 1215 data points with 12 columns each.
We need to convert some of these to our input and output features.

Looking at the indices, we will need only the columns 'Image', 'ID', 'Name', and 'Type'.

```python
Index(['Image', 'Index', 'Name', 'Type 1', 'Type 2', 'Total', 'HP', 'Attack', 'Defense', 'SP. Atk.', 'SP. Def', 'Speed'], dtype='object')
```

Now we will look at some of the sample inputs.
![Pokemons 1](./docs/images/pkmns1.png)
![Pokemons 2](./docs/images/pkmns2.png)

We can see different pokemons with thier respective types.

Inspecting the dataframe further, we find that we have 1215 entries (we know this already),
and 18 unique types, with type `Water` occuring the most, 150 times!
This is not good for a categorical dataset to have uneven spread of categories. It makes the model overlearn, or underlearn.
Such as in this case, it may learn that predicting `Water` everytime will give it the highest accuracy or so.

Here is a bar chart of the same

![Types Chart](./docs/images/types_chart.png)

We have these unique classes (or Pokemon Types)

#### Data Preprocessing

We have this small python function to read and preprocess the image, before converting it to a numpy array.

```python
def preprocess_image(path: str):
    """
    This function reads the image path and returns and np.ndarray containing the image data as an numpy array.
    """
    pth = f"data/{path}"
    data = None
    with Image.open(pth) as img:
        data = np.array(img)

    return data / 255
```

We then convert each of the image paths to numpy arrays within the dataframe by applying this function the to column.
Since type names are strings, we need to convert them to numbers in order for the machine to understand them.
This is were LabelEncoding jumps in. It picks each class and starts by assinging them numbers starting from 0.

```python
ll = LabelEncoder()
model_data["Type"] = ll.fit_transform(model_data["Type"])
```

We can use the same `LabelEncoder` to inverse the transformation back to the type names, say while prediction.
We will again look at some of the sample data, to make sure everything is on the right track.

![Pokemons 3](./docs/images/pkmns3.png)

#### Training and Testing

Our input feature is the numpy array's (the image), and the output feature is are the Types.
Then we can use the `train_test_split` function to split our dataset into training, testing, and validation parts.
We have their shapes as follows.

```python
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, X_val.shape, y_val.shape)
>> (680, 112, 120, 4) (365, 112, 120, 4) (680,) (365,) (170, 112, 120, 4) (170,)
```

#### The Neural Net

We create our neural network with the following layers and parameters.

```python
model = keras.models.Sequential()
model.add(keras.layers.Input(shape=(112, 120, 4)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(200, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(18, activation="softmax"))
```

```sh
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                   ┃ Output Shape          ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ flatten (Flatten)              │ (None, 53760)         │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                  │ (None, 200)           │    10,752,200 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                │ (None, 100)           │        20,100 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                │ (None, 18)            │         1,818 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
```

Then we can compile our model based on the "sparse_categorical_crossentropy" as the loss function, using the "sgd" optimizer,
and levraging "accuracy" as one of the metrics.

The training, testing, validation curve looks like this

![Train Test Val Curve](./docs/images/curve.png)

After running the tests, we save the model.

## What can we do now

A lot of things, like improving the model, adding more (and relavent) layers, and fine tuning certain parameters.
I will keep updating the README as I update the model.

## Demo

Use the included streamlit app to interact with the model.

```bash
streamlit run app.py
```
