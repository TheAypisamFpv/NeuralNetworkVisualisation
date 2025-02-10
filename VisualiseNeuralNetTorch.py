print("Loading dependencies...", end="\r")
from matplotlib.style import available
from sklearn.cluster import AgglomerativeClustering
from tkinter import Tk, filedialog
import pandas as pd
import numpy as np
import pyperclip
import threading
import difflib
import pygame
import torch
import torch.nn as nn
import json
import time
import csv
import os
print("Loading dependencies done.")


class NeuralNet(nn.Module):
    def __init__(self, layers, dropoutRates, l2Reg, inputActivation, hiddenActivation, outputActivation):
        super(NeuralNet, self).__init__()
        self.layers = nn.ModuleList()
        self.dropoutRates = dropoutRates
        self.l2Reg = l2Reg

        # Add input layer
        self.layers.append(nn.Linear(layers[0], layers[1]))
        self.layers.append(self.getActivation(inputActivation))
        self.layers.append(nn.BatchNorm1d(layers[1]))
        self.layers.append(nn.Dropout(dropoutRates[0]))

        # Add hidden layers
        for i in range(1, len(layers) - 2):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            self.layers.append(self.getActivation(hiddenActivation))
            self.layers.append(nn.BatchNorm1d(layers[i + 1]))
            dropoutRateIndex = min(i, len(dropoutRates) - 1)
            self.layers.append(nn.Dropout(dropoutRates[dropoutRateIndex]))

        # Add output layer
        self.layers.append(nn.Linear(layers[-2], layers[-1]))
        self.layers.append(self.getActivation(outputActivation))

    def getActivation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, nn.BatchNorm1d):
                if x.size(0) == 1:
                    continue  # Skip batch normalization if batch size is 1
            x = layer(x)
        return x


class NeuralNetApp:
    def __init__(self):
        # Colors
        self.NEGATIVE_COLOR = pygame.Color("#FD4F59")
        self.POSITIVE_COLOR = pygame.Color("#5BAFFC")
        self.TEXT_COLOR = pygame.Color("#DDDDDD")
        self.BACKGROUND_COLOR = pygame.Color("#222222")
        self.INPUT_BACKGROUND_COLOR = pygame.Color("#333333")
        self.DROP_DOWN_BACKGROUND_COLOR = pygame.Color("#444444")
        self.HOVER_COLOR = pygame.Color("#AAAAAA")
        self.ACTIVE_COLOR = pygame.Color("#555555")  # Changed ACTIVE_COLOR for better visibility
        
        # Font and input box settings
        self.FONT_SIZE = 15 # Font size for input text
        # Width and height of input boxes
        self.INPUT_BOX_TOP_MARGIN = 10
        self.INPUT_BOX_BOTTOM_MARGIN = 10
        self.INPUT_BOX_WIDTH = 140
        self.INPUT_BOX_HEIGHT = 20
        self.borderRadius = 10
        
        # Minimum window size
        self.MIN_WIDTH = 1000
        self.MIN_HEIGHT = 800
        self.TOP_BOTTOM_MARGIN = self.INPUT_BOX_TOP_MARGIN + self.INPUT_BOX_BOTTOM_MARGIN  # Margin at the top and bottom of the window
        self.MAX_HELP_TEXT_WIDTH = 400  # Maximum width for input help text
        self.INPUT_TEXT_HORIZONTAL_SPACING = 20  # Initial horizontal spacing
        self.INPUT_TEXT_VERTICAL_SPACING = 5    # Initial vertical spacing
        self.NORMALIZATION_RANGE = (-1, 1) # used to renormalize the input values to the range of the model

        # FPS setting
        self.fps = 60

        # Clustering settings
        self.clusterThreshold = 50  # Number of neurons with 
        self.enableClustering = True  # Variable to toggle clustering


        # ask user to select what device to use (don't look too closely at the code, it works and used only once)
        availableDevices = ["cpu"]
        if torch.cuda.is_available():
            availableDevices.append("cuda")
        if len(availableDevices) > 1:
            print("\nSelect the device to use:")
            for i, device in enumerate(availableDevices):
                print(f"{i + 1}: {device}")
            selectedDevice = None
            while selectedDevice is None:
                try:
                    selectedDevice = int(input("Enter the device number: "))
                    if selectedDevice < 1 or selectedDevice > len(availableDevices):
                        selectedDevice = None
                        print("Invalid device number, please try again.")
                except ValueError:
                    print("Invalid input, please enter a number.")

            if availableDevices[selectedDevice - 1] == "cuda":
                availableGpus = torch.cuda.device_count()
                if availableGpus > 1:
                    print("\nSelect the GPU to use:")
                    for i in range(availableGpus):
                        print(f"{i + 1}: GPU {i} - {torch.cuda.get_device_name(i)}")
                    selectedGpu = None
                    while selectedGpu is None:
                        try:
                            selectedGpu = int(input("Enter the GPU number: "))
                            if selectedGpu < 1 or selectedGpu > availableGpus:
                                selectedGpu = None
                                print("Invalid GPU number, please try again.")
                        except ValueError:
                            print("Invalid input, please enter a number.")
                            
                    self.device = torch.device(f"cuda:{selectedGpu - 1}")
                else:
                    self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

        # print torch parameters        
        print( "\n", "-" * 50)
        print("System Information:")
        print("PyTorch version:", torch.__version__)
        if self.device.type == "cuda":
            print(" - Current CUDA device id:", torch.cuda.current_device())
            deviceProps = torch.cuda.get_device_properties(torch.cuda.current_device())
            print(" - CUDA device name:", deviceProps.name)
            
            # Calculate total CUDA cores based on SM count and cores per SM.
            smCount = deviceProps.multi_processor_count
            # For many NVIDIA GPUs (e.g., Turing architecture used in RTX 2060), each SM has 64 cores.
            # You might need to adjust cores_per_sm based on your GPU architecture.
            coresPerSm = 64 if ("RTX" in deviceProps.name or "Turing" in deviceProps.name) else 0
            totalCudaCores = smCount * coresPerSm if coresPerSm else "Unknown"
            print(" - CUDA cores count:", totalCudaCores)

            # Add tensor core count calculation and print
            if isinstance(totalCudaCores, int):
                # Assumption: for Turing architecture or similar, each SM has 8 Tensor Cores.
                tensorCoreCount = smCount * 8
            else:
                tensorCoreCount = "Unknown"
            print(" - Tensor Core count:", tensorCoreCount)
        elif self.device.type == "cpu":
            cpuName = torch.cuda.get_device_name(0)
            print(" - CPU name:", cpuName)
        
        print("-" * 50, "\n")


        # Initialize Pygame
        pygame.init()
        pygame.display.set_caption("Neural Network Prediction")
        self.screen = pygame.display.set_mode(
            (1400, 900), pygame.RESIZABLE
        )
        self.screen.fill(self.BACKGROUND_COLOR)
        self.font = pygame.font.Font(None, self.FONT_SIZE)
        self.cursor = pygame.Rect(0, -2, 2, self.INPUT_BOX_HEIGHT - 10)
        self.clock = pygame.time.Clock()

        # Initialize variables
        self.inputBoxes = []
        self.inputValues = {}
        self.cursorVisible = True
        self.cursorTimer = 0
        self.dropdownOpen = None  # Track which dropdown is open
        self.dropdownOptions = {}  # Store dropdown options for each feature
        self.running = True
        self.prediction = None
        self.intermediateOutputs = None
        self.activeBox = None
        self.activeFeature = None
        self.modelFilePath = None
        self.model = None
        self.newModel = False
        self.mapping = None
        self.displayValues = False
        self.featuresImportance = None
        self.lastInputValues = None
        self.lastInputChangeTime = time.time()
        self.waitTime = 1 #s

        # Margins for visualization
        self.leftMargin = 600
        self.rightMargin = 100
        self.topMargin = self.INPUT_BOX_TOP_MARGIN # top margin is the same as the start of the input box so that the input boxes are aligned with the visualization
        self.bottomMargin = self.INPUT_BOX_BOTTOM_MARGIN
        self.visualisationArea = pygame.Rect(
                self.leftMargin-10,
                self.topMargin-10,
                self.screen.get_width() - self.leftMargin - self.rightMargin + 20,
                self.screen.get_height() - self.topMargin - self.bottomMargin + 20
            )

        # Threading for predictions
        self.predictionThread = None
        self.predictionLock = threading.Lock()
        self.predictionReady = False

        # Store last valid prediction
        self.lastPrediction = None
        self.lastIntermediateOutputs = None

        # Variables for text selection
        self.selectedText = ""
        self.selectionStart = None
        self.selectionEnd = None

        # Connection width settings
        self.MIN_CONNECTION_WIDTH = 1
        self.MAX_CONNECTION_WIDTH = 10


    def isFloat(self, value):
        """Check if the string can be converted to a float."""
        try:
            float(value)
            return True
        except ValueError:
            return False

    def loadModelParams(self):
        """
        Load the mapping values for the model features from a CSV file.
        """

        directory = self.modelFilePath.rsplit("/", 1)[0] + "/"
        modelName = self.modelFilePath.rsplit("/", 1)[1].rsplit(".", 1)[0]
        if not directory:
            directory = "./"

        mappingFilePath = directory + r"MappingValues.csv"
        paramsFilePath = directory + modelName + '.params'
        print(f"\nLoading model parameters from {paramsFilePath}...")
        seed = 42
        if not os.path.exists(paramsFilePath):
            print(f"No .params file found at {paramsFilePath}, using default random seed ({seed})")
        else:  
            # load the randomSeed in .params file (json format)
            with open(paramsFilePath, 'r') as file:
                params = json.load(file)
                if not 'randomSeed' in params:
                    print(f"No random seed found in {paramsFilePath}, using default seed ({seed})")
                try:
                    seed = int(params['randomSeed'])
                    print("Setting random seed to", seed)
                except:
                    print(f"Invalid random seed in {paramsFilePath}, using default seed ({seed})")
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
                        
                        
        print("\nChecking for mapping values...")
        if not os.path.exists(mappingFilePath):
            # use default values
            errorMessage = "No mapping file found, using default values (this may not work for all models)"
            print("-" * len(errorMessage))
            print(errorMessage)
            print("-" * len(errorMessage))
            mappingFilePath = r"GeneratedDataSet\MappingValues.csv"
        else:
            print(f"Loading mapping values from {mappingFilePath}...")

        
        mapping = {}
        with open(mappingFilePath, mode='r') as infile:
            reader = csv.reader(infile)
            headers = next(reader)
            values = next(reader)
            for header, value in zip(headers, values):
                if value.startswith("[") and value.endswith("]"):
                    value = eval(value)
                mapping[header] = value

        # check if the length of the mapping values is the same as the input shape of the model
        if len(mapping) != self.model.layers[0].in_features:
            raise ValueError(f"Length of mapping values ({len(mapping)}) does not match input shape of the model's input layer ({self.model.layers[0].in_features})")

        self.mapping = mapping
        print(f"Loaded mapping values from {mappingFilePath}\n")

        # Load feature importance from CSV if it exists
        importanceFilePath = os.path.join(directory, "FeatureImportance.csv")
        print(f"Checking for feature importance file at {importanceFilePath}...")
        if os.path.exists(importanceFilePath):
            self.featuresImportance = pd.read_csv(importanceFilePath, index_col=0)["Importance"].to_dict()
            print(f"Loaded feature importance from {importanceFilePath}\n")
        else:
            self.featuresImportance = {feature: 1 for feature in self.mapping.keys()}
            print("Feature importance file not found, using default neuron size\n")

        # Clear existing input boxes and values
        self.inputBoxes = []
        self.inputValues = {}
    
    def selectModelFile(self):
        """
        Open a file dialog to select a model file.
        """
        root = Tk()
        root.withdraw()  # Hide the root window
        filePath = filedialog.askopenfilename(title="Select Model File", filetypes=[("Model Files", "*.pt"), ("All Files", "*.*")])
        root.destroy()
        return filePath


    def renderDropdown(self, inputBox, options):
        """
        Render a dropdown menu for a given input box.
        """
        dropdownRect = pygame.Rect(inputBox.x, inputBox.y + self.INPUT_BOX_HEIGHT, self.INPUT_BOX_WIDTH, len(options) * self.INPUT_BOX_HEIGHT)
        pygame.draw.rect(self.screen, self.DROP_DOWN_BACKGROUND_COLOR, dropdownRect, border_radius=self.borderRadius)
        mousePos = pygame.mouse.get_pos()
        for i, option in enumerate(options):
            optionRect = pygame.Rect(inputBox.x, inputBox.y + self.INPUT_BOX_HEIGHT + i * self.INPUT_BOX_HEIGHT, self.INPUT_BOX_WIDTH, self.INPUT_BOX_HEIGHT)
            if optionRect.collidepoint(mousePos):
                pygame.draw.rect(self.screen, self.HOVER_COLOR, optionRect, border_radius=self.borderRadius)
            else:
                pygame.draw.rect(self.screen, self.DROP_DOWN_BACKGROUND_COLOR, optionRect, border_radius=self.borderRadius)
            
            optionText = self.font.render(option, True, self.TEXT_COLOR)
            self.screen.blit(optionText, (optionRect.x + 5, optionRect.y + 5))

    def wrapText(self, text:str, font, maxWidth:int):
        """
        Wrap text to fit within a given width.
        """
        words = text.split(' ')
        lines = []
        currentLine = []
        currentWidth = 0

        for word in words:
            wordWidth, _ = font.size(word + ' ')
            if currentWidth + wordWidth <= maxWidth:
                currentLine.append(word)
                currentWidth += wordWidth
            else:
                lines.append(' '.join(currentLine))
                currentLine = [word]
                currentWidth = wordWidth

        if currentLine:
            lines.append(' '.join(currentLine))

        return lines

    def getPredictionThread(self):
        """
        Thread function to get a prediction from the model.
        """
        with self.predictionLock:
            if self.predictionReady:
                return
            
            self.predictionReady = True
        
        try:
            prediction, intermediateOutputs = self.getPrediction()
            with self.predictionLock:
                self.prediction = prediction
                self.intermediateOutputs = intermediateOutputs
                self.lastPrediction = prediction
                self.lastIntermediateOutputs = intermediateOutputs
                self.predictionReady = False
        except Exception as e:
            print(f"Error getting prediction: {e}")
            with self.predictionLock:
                self.predictionReady = False    

    def getPrediction(self):
        """
        Parse input values and get a prediction from the model.
        """
        startTime = time.time()
        inputData = []
        lastInputChangeTime = time.time() - self.lastInputChangeTime

        if self.lastInputValues != self.inputValues and self.lastInputValues:
            self.lastInputChangeTime = time.time()
            self.lastInputValues = self.inputValues.copy()
            return None, None

        if (lastInputChangeTime < self.waitTime or lastInputChangeTime > self.waitTime + 0.2) and self.lastInputValues:
            return None, None

        self.lastInputValues = self.inputValues.copy()

        for inputName, inputValue in self.inputValues.items():
            if not inputValue:
                inputData.append(0)
                continue

            inputMapping = self.mapping[inputName]

            if isinstance(inputMapping[0], (int, float)):
                minValue, maxValue = inputMapping
                try:
                    inputValue = float(inputValue)
                except ValueError:
                    inputValue = 0
                normalizedValue = (float(inputValue) - minValue) / (maxValue - minValue) * (self.NORMALIZATION_RANGE[1] - self.NORMALIZATION_RANGE[0]) + self.NORMALIZATION_RANGE[0]
                normalizedValue = max(self.NORMALIZATION_RANGE[0], min(self.NORMALIZATION_RANGE[1], normalizedValue))
                inputData.append(normalizedValue)
                continue

            elif isinstance(inputMapping[0], str):
                inputMapping = [str(val).lower() for val in inputMapping]
                inputValue = inputValue.lower()
                closestMatch = difflib.get_close_matches(inputValue, inputMapping, n=1, cutoff=0.1)
                if closestMatch:
                    normalizedValue = (inputMapping.index(closestMatch[0]) / (len(inputMapping) - 1)) * (self.NORMALIZATION_RANGE[1] - self.NORMALIZATION_RANGE[0]) + self.NORMALIZATION_RANGE[0]
                    normalizedValue = max(self.NORMALIZATION_RANGE[0], min(self.NORMALIZATION_RANGE[1], normalizedValue))
                    inputData.append(normalizedValue)
                    continue
            else:
                raise ValueError(f"Invalid data type for feature mapping: {inputMapping} (got {type(inputMapping)}, expected list)")

            inputData.append(0)

        inputData = np.array([inputData])
        expectedInputShape = self.model.layers[0].in_features
        if inputData.shape[1] != expectedInputShape:
            raise ValueError(f"Expected input shape ({expectedInputShape}) does not match provided input shape ({inputData.shape[1]})")

        inputTensor = torch.FloatTensor(inputData).to(self.device)

        # Prepare dictionary with the input layer included
        intermediateOutputsTemp = {"Layers.-1": inputData}

        hooks = []
        # Register hooks for all linear modules except the output layer
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and module is not self.model.layers[-1]:
                hook = module.register_forward_hook(lambda module, inp, output, name=name: intermediateOutputsTemp.update({name: output}))
                hooks.append(hook)

        self.model.eval()
        with torch.no_grad():
            output = self.model(inputTensor)

        # Remove hooks after inference
        for h in hooks:
            h.remove()

        prediction = output.cpu().numpy()
        intermediateOutputs = {name: (tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor)
                            for name, tensor in intermediateOutputsTemp.items()}
        
        self.lastInputValues = self.inputValues.copy()
        print(f"\nPrediction took {time.time() - startTime:.2f} seconds")
        return prediction, intermediateOutputs


    def interpolateColor(self, colorA, colorB, factor:float):
        """Interpolate between two colors with a given factor (0 to 1)."""

        if np.isnan(factor):
            return pygame.Color('#666666')

        factor = max(0, min(1, factor))
        
        return pygame.Color(
            int(colorA.r + (colorB.r - colorA.r) * factor),
            int(colorA.g + (colorB.g - colorA.g) * factor),
            int(colorA.b + (colorB.b - colorA.b) * factor)
        )

    def clusterNeurons(self, layerOutputs):
        """
        Cluster neurons in a layer using hierarchical clustering.
        """
        if not self.enableClustering:
            return [(i,) for i in range(layerOutputs.shape[1])]  # No clustering if disabled

        numNeurons = layerOutputs.shape[1]
        if numNeurons < self.clusterThreshold:
            return [(i,) for i in range(numNeurons)]  # No clustering needed

        clustering = AgglomerativeClustering(n_clusters=self.clusterThreshold)
        labels = clustering.fit_predict(layerOutputs.T)
        clusters = [[] for _ in range(self.clusterThreshold)]
        for neuronIndex, clusterIndex in enumerate(labels):
            clusters[clusterIndex].append(neuronIndex)
        return clusters

    def visualize1D(self, screen, model, intermediateOutputs):
        """
        Visualize model with 1 dimensional input.
        ---
        TODO :
            combine bith passes to use only 2 for loops and not 4
        """
        print("Visualizing 1D model...")
        timeStart = time.time()
        
        # Sort intermediate outputs by the integer after "layers."
        sortedOutputs = sorted(intermediateOutputs.items(), key=lambda item: int(item[0].split(".")[1]))
        # Replace allOutputs with the sorted layer outputs in order.
        layersOutputs = [output for _, output in sortedOutputs]
        # for key, output in sortedOutputs:
        #     print(f"Shape of {key}: {output.shape}")

        numLayers = len(layersOutputs)
        maxNeurons = max([layer.shape[1] for layer in layersOutputs])

        paddingWH = [30, screen.get_height() * 0.01]
        startWidth = self.leftMargin + paddingWH[0]
            
        availableWidth = self.screen.get_width() - self.leftMargin - self.rightMargin - paddingWH[0] * 2
        layerSpacing = availableWidth / (numLayers - 1)

        startHeight = self.topMargin + 10
        availableHeight = self.screen.get_height() - self.topMargin - self.bottomMargin - paddingWH[1] * 2

        neuronRadius = max(10, int(min(layerSpacing, availableHeight / maxNeurons) / 2))
        layers = [layer.shape[1] for layer in layersOutputs]
        outputLayerSize = layers[-1]

        # Extract weights from the model (for all Linear layers)
        weights = [layer.weight.data.cpu().numpy() for layer in model.modules() if isinstance(layer, torch.nn.Linear)]

        # Get feature importance range.
        maxImportance = max(self.featuresImportance.values())
        minImportance = min(self.featuresImportance.values())
        importanceRange = maxImportance - minImportance 

        def normalizeImportance(importance):
            defaultNeuronSize = 7
            if importanceRange == 0:
                return neuronRadius
            return defaultNeuronSize + (importance - minImportance) / importanceRange * defaultNeuronSize

        # First pass: sort and prepare connections between layers.
        connections = []
        for i in range(numLayers - 1):
            currentLayerSize = layers[i]
            nextLayerSize = layers[i + 1]
            x = startWidth + i * layerSpacing
            nextX = startWidth + (i + 1) * layerSpacing

            currentNeuronSpacing = availableHeight / currentLayerSize
            nextNeuronSpacing = availableHeight / nextLayerSize

            currentTotalLayerHeight = (currentLayerSize - 1) * currentNeuronSpacing
            currentYStart = startHeight + (availableHeight - currentTotalLayerHeight) / 2

            nextTotalLayerHeight = (nextLayerSize - 1) * nextNeuronSpacing
            nextYStart = startHeight + (availableHeight - nextTotalLayerHeight) / 2

            # Cluster neurons in the current and next layers.
            # print("Shape of layersOutputs[i]", layersOutputs[i].shape)
            # print("layersOutputs[i]", layersOutputs[i])
            currentClusters = self.clusterNeurons(layersOutputs[i])
            nextClusters = self.clusterNeurons(layersOutputs[i + 1])

            # Normalize weights for the current layer.
            currentWeights = weights[i]
            maxWeight = np.max(np.abs(currentWeights))
            minWeight = np.min(np.abs(currentWeights))
            weightRange = maxWeight - minWeight if maxWeight != minWeight else 1

            for currentCluster in currentClusters:
                # Calculate mean output for the current cluster.
                currentClusterOutput = np.mean([layersOutputs[i][0][j] for j in currentCluster])
                # Normalize using the normalization range of the model.
                normalizedOutput = (currentClusterOutput - self.NORMALIZATION_RANGE[0]) / (
                    self.NORMALIZATION_RANGE[1] - self.NORMALIZATION_RANGE[0])
                color = self.interpolateColor(self.NEGATIVE_COLOR, self.POSITIVE_COLOR, normalizedOutput)
                y = currentYStart + np.mean([j * currentNeuronSpacing for j in currentCluster])
        
                for nextCluster in nextClusters:
                    nextY = nextYStart + np.mean([j * nextNeuronSpacing for j in nextCluster])
                    # print("shape of currentWeights", currentWeights.shape)
                    # print("currentWeights",currentWeights)
                    # print("currentCluster",currentCluster)
                    # print("nextCluster",nextCluster)
                    
                    weight = np.mean([currentWeights[k, j] for j in currentCluster for k in nextCluster])
                    # Adjust alpha based on weight magnitude.
                    alphaFactor = (np.abs(weight) - minWeight) / weightRange
                    blendedColor = self.interpolateColor(color, self.BACKGROUND_COLOR, 1 - alphaFactor)
        
                    connections.append((weight, (x, y, nextX, nextY, blendedColor)))


        self.clearVisualizationArea()
        
        self.screen.set_clip(self.visualisationArea)
        # Sort connections (thin lines first).
        connections.sort(key=lambda conn: abs(conn[0]))

        # Draw connections.
        for _, (x, y, nextX, nextY, blendedColor) in connections:
            pygame.draw.aaline(screen, blendedColor, (int(x), int(y)), (int(nextX), int(nextY)), blend=1)

        # Second pass: Draw neurons and their values.
        for i in range(numLayers):
            x = startWidth + i * layerSpacing
            clusters = self.clusterNeurons(layersOutputs[i])
            numClusters = len(clusters)
        
            neuronSpacing = availableHeight / numClusters
            totalLayerHeight = (numClusters - 1) * neuronSpacing
            yStart = startHeight + (availableHeight - totalLayerHeight) / 2

            blackCircleSize = 1.0
            edgeThickness = 1

            for clusterIndex, cluster in enumerate(clusters):
                clusterOutput = np.mean([layersOutputs[i][0][j] for j in cluster])
                
                if i == 0:
                    # Normalize input layer from [-1, 1] to [0, 1]
                    normalizedOutput = (clusterOutput + 1) / 2
                elif i == numLayers - 1 and outputLayerSize > 1:
                    # For output layer with multiple neurons.
                    normalizedOutput = 0.5 + clusterOutput / 2
                    print(f"Output layer normalization: {clusterOutput} -> {normalizedOutput}")
                else:
                    normalizedOutput = clusterOutput  # Already scaled to [0, 1]
                
                normalizedOutput = max(0, min(1, normalizedOutput))
                color = self.interpolateColor(self.NEGATIVE_COLOR, self.POSITIVE_COLOR, normalizedOutput)
                y = yStart + clusterIndex * neuronSpacing
                
                # Adjust neuron radius based on feature importance for input layer.
                adjustedRadius = neuronRadius
                if i == 0:
                    feature = list(self.mapping.keys())[cluster[0]]
                    importance = self.featuresImportance.get(feature, 1)
                    adjustedRadius = int(normalizeImportance(importance))
        
                # Draw the neuron with black fill and white edge.
                pygame.draw.circle(screen, (0, 0, 0), (int(x), int(y)), int(adjustedRadius * blackCircleSize))
                pygame.draw.circle(screen, (255, 255, 255), (int(x), int(y)), int(adjustedRadius * blackCircleSize) + edgeThickness, edgeThickness)
        
                # Determine inner circle size based on normalized output.
                if normalizedOutput <= 0.5:
                    innerRadius = int(adjustedRadius * (1 - 2 * normalizedOutput))
                else:
                    innerRadius = int(adjustedRadius * (2 * (normalizedOutput - 0.5)))
        
                pygame.draw.circle(screen, color, (int(x), int(y)), innerRadius)
        
                if self.displayValues:
                    valueText = f"{clusterOutput:.2f}"
                    textSurface = self.font.render(valueText, True, self.TEXT_COLOR)
                    textRect = textSurface.get_rect(
                        center=(int(x + 5 + adjustedRadius + (textSurface.get_width() / 2)), int(y)))
                    screen.blit(textSurface, textRect)

        self.screen.set_clip(None)
        print(f"Visualization took {time.time() - timeStart:.2f} seconds")

    def clearVisualizationArea(self):
        """
        Clear the visualization area.
        """
        self.visualisationArea = pygame.Rect(
            self.leftMargin-10,
            self.topMargin-10,
            self.screen.get_width() - self.leftMargin - self.rightMargin + 20,
            self.screen.get_height() - self.topMargin - self.bottomMargin + 20
        )
        self.screen.set_clip(self.visualisationArea)
        self.screen.fill(self.BACKGROUND_COLOR)
        self.screen.set_clip(None)

    def updateInputBoxes(self):
        """
        Update the input boxes based on the current window size.
        Creates them if they don't exist yet.
        """
        
        verticalSpacing = (self.screen.get_height() - self.INPUT_BOX_TOP_MARGIN - self.INPUT_BOX_BOTTOM_MARGIN - self.INPUT_BOX_HEIGHT) / len(self.mapping)
        inputBoxStartY = self.INPUT_BOX_TOP_MARGIN + self.INPUT_BOX_HEIGHT

        
        if not self.inputBoxes:
            for i, (feature, values) in enumerate(self.mapping.items()):
                inputBox = pygame.Rect(
                    10,
                    inputBoxStartY + int(verticalSpacing * i),
                    self.INPUT_BOX_WIDTH,
                    self.INPUT_BOX_HEIGHT
                )
                self.inputBoxes.append((feature, inputBox, values))
                self.inputValues[feature] = ""
        else:
            for i, (feature, inputBox, values) in enumerate(self.inputBoxes):
                newY = inputBoxStartY + int(verticalSpacing * i)
                self.inputBoxes[i] = (feature, pygame.Rect(10, newY, self.INPUT_BOX_WIDTH, self.INPUT_BOX_HEIGHT), values)


    def mainLoop(self, visualisation=True, displayValues=False):
        """
        Main loop for the application.
        """
        try:
            self.screen.fill(self.BACKGROUND_COLOR)

            self.displayValues = displayValues

            if not self.enableClustering:
                print("""
    =========================================================================================================================
    Clustering is disabled. This may lead to a large number of neurons being visualized, thus slowing down the visualization.
    Use clustering for models with a large number of neurons to improve performance.                                         
    =========================================================================================================================
    """)

            while self.running:
                self.screen.set_clip(
                    pygame.Rect(
                        0,
                        0,
                        self.screen.get_width(),
                        self.visualisationArea.top
                    )
                )
                self.screen.fill(self.BACKGROUND_COLOR)

                self.screen.set_clip(
                    pygame.Rect(
                        0,
                        self.visualisationArea.bottom,
                        self.screen.get_width(),
                        self.screen.get_height() - self.visualisationArea.bottom
                    )
                )
                self.screen.fill(self.BACKGROUND_COLOR)

                self.screen.set_clip(
                    pygame.Rect(
                        0,
                        0,
                        self.visualisationArea.left,
                        self.screen.get_height()
                    )
                )
                self.screen.fill(self.BACKGROUND_COLOR)

                self.screen.set_clip(
                    pygame.Rect(
                        self.visualisationArea.right,
                        0,
                        self.screen.get_width() - self.visualisationArea.right,
                        self.screen.get_height()
                    )
                )
                self.screen.fill(self.BACKGROUND_COLOR)

                self.screen.set_clip(None)

                self.cursorTimer += 1

                windowFocused = pygame.key.get_focused()
                if self.cursorTimer % (self.fps // 2) == 0 and windowFocused:
                    self.cursorVisible = not self.cursorVisible
                elif not windowFocused:
                    self.cursorVisible = False

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False

                    elif event.type == pygame.VIDEORESIZE:
                        newWidth = max(event.w, self.MIN_WIDTH)
                        newHeight = max(event.h, self.MIN_HEIGHT)
                        # self.screen = pygame.display.set_mode(
                        #     (newWidth, newHeight), pygame.RESIZABLE
                        # )
                        self.INPUT_TEXT_HORIZONTAL_SPACING = newHeight * 0.02
                        self.INPUT_TEXT_VERTICAL_SPACING = newHeight * 0.005

                        self.visualisationArea = pygame.Rect(
                            self.leftMargin-10,
                            self.topMargin-10,
                            self.screen.get_width() - self.leftMargin - self.rightMargin + 20,
                            self.screen.get_height() - self.topMargin - self.bottomMargin + 20
                        )

                        self.clearVisualizationArea()
                        self.lastInputValues = None
                        self.lastInputChangeTime = time.time() - self.waitTime

                        if self.modelFilePath:
                            self.updateInputBoxes()

                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        try:
                            if hasattr(self, 'selectModelButton') and self.selectModelButton.collidepoint(event.pos):
                                self.modelFilePath = self.selectModelFile()
                                if self.modelFilePath:
                                    self.newModel = True
                                    self.model = torch.load(self.modelFilePath, map_location=self.device, weights_only=False)
                                    self.loadModelParams()
                                    self.updateInputBoxes()
                            
                            if hasattr(self, 'toggleDisplayValuesButton') and self.toggleDisplayValuesButton.collidepoint(event.pos):
                                self.displayValues = not self.displayValues  # Toggle displayValues
                                self.lastInputValues = None
                                self.lastInputChangeTime = time.time() - self.waitTime
                        

                        except Exception as e:
                            print(f"Error selecting model: {e}")
                            self.clearVisualizationArea()
                            errorFont = pygame.font.Font(None, 20)
                            errorMessage = f"Error selecting model: {e}"
                            maxWidth = self.screen.get_width() - self.leftMargin - self.rightMargin
                            wrappedErrorText = self.wrapText(errorMessage, errorFont, maxWidth)
                            for lineIndex, line in enumerate(wrappedErrorText):
                                errorText = errorFont.render(line, True, self.NEGATIVE_COLOR)
                                self.screen.blit(errorText, (self.leftMargin, self.topMargin + lineIndex * (errorFont.get_height() + 5)))

                            self.modelFilePath = None

                        else:
                            dropdownClicked = False
                            if self.dropdownOpen:
                                for feature, inputBox, values in self.inputBoxes:
                                    if self.dropdownOpen == feature:
                                        options = self.mapping[feature]
                                        for i, option in enumerate(options):
                                            optionRect = pygame.Rect(inputBox.x, inputBox.y + self.INPUT_BOX_HEIGHT + i * self.INPUT_BOX_HEIGHT, self.INPUT_BOX_WIDTH, self.INPUT_BOX_HEIGHT)
                                            if optionRect.collidepoint(event.pos):
                                                self.inputValues[feature] = option
                                                self.dropdownOpen = None
                                                dropdownClicked = True
                                                break
                                    if dropdownClicked:
                                        break

                            if not dropdownClicked:
                                inputBoxClicked = False
                                for feature, inputBox, values in self.inputBoxes:
                                    if inputBox.collidepoint(event.pos):
                                        self.activeBox = inputBox
                                        self.activeFeature = feature
                                        self.cursorTimer = 0
                                        if isinstance(values[0], str):
                                            self.dropdownOpen = feature if self.dropdownOpen != feature else None
                                        else:
                                            self.dropdownOpen = None
                                        inputBoxClicked = True
                                        break
                                if not inputBoxClicked:
                                    self.activeBox = None
                                    self.dropdownOpen = None

                    elif event.type == pygame.KEYDOWN:
                        if self.activeBox and self.dropdownOpen is None:
                            if event.key == pygame.K_RETURN:
                                self.activeBox = None
                            elif event.key == pygame.K_BACKSPACE:
                                self.inputValues[self.activeFeature] = self.inputValues[self.activeFeature][:-1]
                            elif event.key == pygame.K_v and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                                clipboardText = pyperclip.paste()
                                self.inputValues[self.activeFeature] += clipboardText
                            elif event.key == pygame.K_c and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                                pyperclip.copy(self.inputValues[self.activeFeature])
                            elif event.key == pygame.K_UP:
                                currentIndex = next((i for i, (feature, inputBox, values) in enumerate(self.inputBoxes) if inputBox == self.activeBox), None)
                                if currentIndex is not None and currentIndex > 0:
                                    self.activeBox = self.inputBoxes[currentIndex - 1][1]
                                    self.activeFeature = self.inputBoxes[currentIndex - 1][0]
                                    self.cursorTimer = 0
                            elif event.key == pygame.K_DOWN:
                                currentIndex = next((i for i, (feature, inputBox, values) in enumerate(self.inputBoxes) if inputBox == self.activeBox), None)
                                if currentIndex is not None and currentIndex < len(self.inputBoxes) - 1:
                                    self.activeBox = self.inputBoxes[currentIndex + 1][1]
                                    self.activeFeature = self.inputBoxes[currentIndex + 1][0]
                                    self.cursorTimer = 0
                            else:
                                self.inputValues[self.activeFeature] += event.unicode

                if self.modelFilePath:
                    mousePos = pygame.mouse.get_pos()
                    for feature, inputBox, values in self.inputBoxes:
                        boxBackgroundColor = self.ACTIVE_COLOR if inputBox == self.activeBox and windowFocused else self.INPUT_BACKGROUND_COLOR

                        if inputBox.collidepoint(mousePos) and self.activeBox != inputBox and not self.dropdownOpen:
                            boxBackgroundColor = self.HOVER_COLOR

                        pygame.draw.rect(self.screen, boxBackgroundColor, inputBox, 0, border_radius=self.borderRadius)

                        textSurface = self.font.render(str(self.inputValues[feature]), True, self.TEXT_COLOR)
                        self.screen.blit(textSurface, (inputBox.x + 5, inputBox.y + 5))

                        if inputBox == self.activeBox and self.cursorVisible:
                            self.cursor.topleft = (inputBox.x + 5 + textSurface.get_width(), inputBox.y + 5)
                            pygame.draw.rect(self.screen, self.TEXT_COLOR, self.cursor, border_radius=self.borderRadius)

                        helpTextMiddleY = inputBox.y + self.INPUT_BOX_HEIGHT / 2
                        helpText = feature
                        if isinstance(values[0], (int, float)):
                            helpText += f" ({values[0]} - {values[1]})"

                        labelSurface = self.font.render(helpText, True, self.TEXT_COLOR)
                        labelPos = (inputBox.x + self.INPUT_BOX_WIDTH + self.INPUT_TEXT_HORIZONTAL_SPACING, helpTextMiddleY - self.FONT_SIZE / 2 + 4)
                        self.screen.blit(labelSurface, labelPos)

                        # Update left margin based on label width
                        self.leftMargin = labelPos[0] + labelSurface.get_width() + 50

                    if self.dropdownOpen:
                        for feature, inputBox, values in self.inputBoxes:
                            if self.dropdownOpen == feature:
                                self.renderDropdown(inputBox, values)


                    # Start prediction thread if not already running
                    with self.predictionLock:
                        if not self.predictionReady:
                            self.predictionThread = threading.Thread(target=self.getPredictionThread)
                            self.predictionThread.start()

                    prediction = None
                    with self.predictionLock or self.newModel:
                        if self.prediction is not None:
                            # self.clearVisualizationArea()
                            prediction = self.prediction
                            intermediateOutputs = self.intermediateOutputs
                            self.prediction = None
                            self.intermediateOutputs = None

                            if visualisation:
                                self.visualize1D(self.screen, self.model, intermediateOutputs)

                    if prediction is not None:
                        predictionFont = pygame.font.Font(None, 25)
                        if len(prediction[0]) == 1:
                            # [0, 1] -> [-1, 1]
                            predictionConfidence = prediction[0][0] * 2 - 1
                            predictionText = "YES" if predictionConfidence > 0 else "NO"
                            predictionColor = self.interpolateColor(self.POSITIVE_COLOR, self.NEGATIVE_COLOR, prediction[0][0])

                            # [-1, 0, 1] -> [100, 0, 100]
                            predictionConfidence = abs(predictionConfidence) * 100
                            predictionFont = pygame.font.Font(None, 25)
                            predictionSurface = predictionFont.render(f"{predictionText} ({predictionConfidence:.2f}%)", True, predictionColor)

                            predictionX = self.screen.get_width() - self.rightMargin - predictionSurface.get_width() - 10
                            predictionY = self.topMargin
                            self.screen.blit(predictionSurface, (predictionX, predictionY))

                        else:
                            for i, pred in enumerate(prediction[0]):
                                predictionText = f"Output {i + 1}: "

                                # Sorry, hard coded mapping for now
                                if i == 0:
                                    predictionText = "Stay"
                                    predictionColor = self.interpolateColor(self.TEXT_COLOR, self.POSITIVE_COLOR, pred)
                                elif i == 1:
                                    predictionText = "Quit"
                                    predictionColor = self.interpolateColor(self.TEXT_COLOR, self.NEGATIVE_COLOR, pred)

                                else:
                                    predictionColor = self.interpolateColor(self.NEGATIVE_COLOR, self.POSITIVE_COLOR, pred)
                                

                                predictionConfidence = abs(pred) * 100
                                predictionSurface = predictionFont.render(f"{predictionText} {predictionConfidence:.2f}%", True, predictionColor)

                                availabeHeight = self.screen.get_height() - self.topMargin - self.bottomMargin
                                outputSpacing = availabeHeight / (len(prediction[0])+1)
                                predictionY = outputSpacing + self.topMargin + i * outputSpacing
                                predictionX = self.screen.get_width() - self.rightMargin - predictionSurface.get_width() - 10
                                self.screen.blit(predictionSurface, (predictionX, predictionY))

                        if self.enableClustering:
                            visualisationText = self.font.render("Neurons are visualized using clustering", True, self.TEXT_COLOR)
                            visualisationTextPos = (self.leftMargin, self.topMargin - 5)
                            self.screen.blit(visualisationText, visualisationTextPos)

                # Display the select model button
                self.selectModelButton = pygame.Rect(10, 5, self.INPUT_BOX_WIDTH, self.INPUT_BOX_HEIGHT)
                color = self.HOVER_COLOR if self.selectModelButton.collidepoint(pygame.mouse.get_pos()) else self.TEXT_COLOR
                pygame.draw.rect(self.screen, color, self.selectModelButton, 2, border_radius=self.borderRadius)
                selectModelText = self.font.render("Select Model", True, color)
                selectModelTextRect = selectModelText.get_rect(center=self.selectModelButton.center)
                self.screen.blit(selectModelText, selectModelTextRect)

                # Display the toggle display values button
                self.toggleDisplayValuesButton = pygame.Rect(self.INPUT_BOX_WIDTH + 20 , 5, self.INPUT_BOX_WIDTH, self.INPUT_BOX_HEIGHT)

                # set color based on displayValues
                color = self.POSITIVE_COLOR if self.displayValues else self.NEGATIVE_COLOR
                # set color alpha to 0.2
                color.a = 50
                # hoveride the color if the mouse is over the button
                color = self.interpolateColor(color, self.HOVER_COLOR, 0.5) if self.toggleDisplayValuesButton.collidepoint(pygame.mouse.get_pos()) else color
                
                pygame.draw.rect(self.screen, color, self.toggleDisplayValuesButton, 2, border_radius=self.borderRadius)
                toggleDisplayValuesText = self.font.render("Toggle Values", True, color)
                toggleDisplayValuesTextRect = toggleDisplayValuesText.get_rect(center=self.toggleDisplayValuesButton.center)
                self.screen.blit(toggleDisplayValuesText, toggleDisplayValuesTextRect)

                pygame.display.flip()
                self.clock.tick(self.fps)
        
        except KeyboardInterrupt:
            print("Closing Neural Network Visualizer...")




def main():
    print("Loading Neural Network Visualizer...")
    neuralNetVis = NeuralNetApp()
    neuralNetVis.mainLoop(displayValues=False)

if __name__ == "__main__":
    main()