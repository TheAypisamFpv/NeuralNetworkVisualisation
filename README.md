# Neural Network Visualization Tool

<div align="center">
    <img src="Images/DNN visual with values.png" alt="Neural Network Visualization Tool Logo" width="100%"/>
</div>

This tool allows you to visualize the inner workings of a neural network model. You can load a pre-trained model, input data, and see how the data propagates through the network layers. The tool provides a graphical interface to input data, visualize neuron activations, and connections.

## Features

- Load and visualize pre-trained neural network models.
- Input data through a graphical interface.
- Visualize neuron activations and connections.
- Cluster neurons for better visualization performance.
- Display feature importance (if feature importance file is available).

## Requirements

### Python 3.11.x - 3.12.x

Python can be downloaded from the official website: [python.org](https://www.python.org/downloads/)

## Installation

1. Clone the repository:

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. Create a Python virtual environment using the installed Python version (recommended).

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

<br>

## How to Use

### Select a Model:
- Click on the "Select Model" button to open a file dialog.
- Choose the .keras model file to load.


### Input Data:
- Enter values for each feature in the input boxes on the left side of the window.
- For categorical features, a dropdown menu will appear.

<br>

## Model Preparation

### __Mapping Values__

Create a `MappingValues.csv` file in the same directory as the model file. The `MappingValues.csv` file should contain the mapping of feature names to their possible values or ranges. This file is necessary for the tool to correctly interpret the input values for the model.

The format of the file should be as follows:

```csv
Feature1      , Feature2                  , Feature3      , ...
"[min1, max1]", "[value1, value2, value3]", "[min3, max3]", ...
```

__⚠️ Features should be in the same order as when the model was trained. ⚠️__

*Example available in `model example\` folder.*



- #### __For numerical features__:

    Use the format `"[min, max]"` to specify the range of possible values.
    
    Here, `min` will be mapped to 0, and `max` will be mapped to 1.


- #### __For categorical features__:

    Use the format `"[value1, value2, value3]"` to specify the possible values, they will be interpreted from left to right in the range [0, 1].

    Here, `value1` will be mapped to 0, `value2` to 0.5, and `value3` to 1.

<br>

---

### __Feature Importance *(Optional)*__

The feature importance file should contain the importance values for each feature in the model. It's only used to adjust the size of the input neurons in the visualization.

The format of the file should be as follows:

```csv
Feature , Importance
Feature1, 0.5
Feature2, 0.3
Feature3, 0.2
...
```
*Order does not matter.*

<br>

## Detailed Explanation

### Colors and Layout

- Colors:
    - Positive activations: <span style="color:#5BAFFC;">#5BAFFC</span> (blue)
    - Negative activations: <span style="color:#FD4F59;">#FD4F59</span> (red)

- Layout:
    - Input boxes are aligned on the left side.
    - Visualization area is on the right side.
    - The window is resizable *(with a min size)*, and the layout adjusts accordingly.

### Clustering

- Neurons are clustered using hierarchical clustering to improve visualization performance.
- Clustering can be toggled on or off *__(no clustering can lead to performance issues with large networks)__*.
- The threshold for a layer to be clustered can be adjusted *(default is 50 neurons)*.

### Visualization

- Neurons are displayed as circles.
- Connections between neurons are displayed as lines.
- The color of each connection represents the value transmitted.
- The color alpha of each connection represents the weight of the connection.
- The size of the input neurons represents their importance *(if provided)*.

<br>

# Troubleshooting

- Ensure that the model file is in the correct `.keras` format.
- Ensure that the mapping file `MappingValues.csv` is available in the same directory as the model file.
- Check the console for any error messages.
- If you are still facing issues, you can directly contact me at theaypisamfpv@gmail.com.
