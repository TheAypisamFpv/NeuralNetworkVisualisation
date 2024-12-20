# Neural Network Visualization Tool
This tool allows you to visualize the inner workings of a neural network model. You can load a pre-trained model, input data, and see how the data propagates through the network layers. The tool also provides a graphical interface to interact with the model and visualize the neurons and their connections.

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

2. Create a Python virtual environment using the installed Python version (recommended):

3. Install the required packages:

```bash
pip install -r requirements.txt
```



# Mapping Values and Feature Importance

### __Mapping Values__
The `MappingValues.csv` file should contain the mapping of feature names to their possible values or ranges.

This file is necessary for the tool to correctly interpret and display the input data.

The format of the file should be as follows:

```csv
Feature1    , Feature2                , Feature3    , ...
"[min1, max1]", "[value1, value2, value3]", "[min3, max3]", ...
```

#### For numerical features:
- Use the format `"[min, max]"` to specify the range of possible values.

#### For categorical features:
- Use the format `"[value1, value2, value3]"` to specify the possible values (they will be interpreted from left to right in the range [0, 1]). Here, `value1` will be mapped to 0, `value2` to 0.5, and `value3` to 1.

__Make Sure to use the correct feature names and values in the mapping file.__


### __Feature Importance__ (Optional)
The feature importance file should contain the importance values for each feature in the model.

It's only used to adjust the size of the input neurons in the visualization.

The format of the file should be as follows:

```csv
Feature , Importance
Feature1, 0.5
Feature2, 0.3
Feature3, 0.2
...
```

## How to Use

1. Select a Model:
    - Click on the "Select Model" button to open a file dialog.
    - Choose the .keras model file to load.
<image of selecting model>

2. Input Data:
    - Enter values for each feature in the input boxes on the left side of the window.
    - For categorical features, a dropdown menu will appear.
<image of input data>


### Visualization of the Network:
- The visualization area on the right will display the neurons and their connections.
- Neurons are color-coded based on their activation values.
- Connections are color-coded based on the weights.
<image of network visualization>

## Detailed Explanation
### Colors and Layout
- Colors:
    - Positive activations: `#5BAFFC` (blue)
    - Negative activations: `#FD4F59` (red)

- Layout:
    - Input boxes are aligned on the left side.
    - Visualization area is on the right side.
    - The window is resizable, and the layout adjusts accordingly.

### Clustering
- Neurons are clustered using hierarchical clustering to improve visualization performance.
- Clustering can be toggled on or off (No clustering can lead to performance issues with large networks).
- The threshold for clustering can be adjusted.

### Visualization
- Neurons are displayed as circles.
- Connections between neurons are displayed as lines.
- The color of each connection represents the value transmitted.
- The color alpha of each connection represents the weight of the connection. 
- The size of the input neurons represents their importance (if available).

### Interaction
Click on input boxes to enter values.
Use the dropdown menu for categorical features.
The visualization updates automatically based on the input values.
Example
Load a Model:

<image of loading model>
Input Data:

<image of input data>
Visualize the Network:

<image of network visualization>
Feature Importance:

<image of feature importance>

## Troubleshooting
- Ensure that the model file is in the correct format (`.keras`).
- Ensure that the mapping file (`MappingValues.csv`) is available in the same directory as the model file.
- Check the console for any error messages.
