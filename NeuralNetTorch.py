from progressBar import getProgressBar
from math import floor
import pandas as pd
import numpy as np
import shap
import json
import os

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.optim as optim

from joblib import Parallel, delayed
import multiprocessing

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns



# Set environment variables to limit the number of threads used by OpenMP and other libraries
os.environ['OMP_NUM_THREADS'] = '11'
os.environ['OPENBLAS_NUM_THREADS'] = '11'
os.environ['MKL_NUM_THREADS'] = '11'
os.environ['VECLIB_MAXIMUM_THREADS'] = '11'
os.environ['NUMEXPR_NUM_THREADS'] = '11'

# Set a random seed for reproducibility
RANDOM_SEED = np.random.randint(0, 1000) # random seed per run of the program, but the same for all randoms functions for reproducibility after training
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


def saveCustomMapping(defaultMappingFilePath:str, updatedMappingFilePath:str, specifiedColumnsToDrop:list = []):
    """
    Load default mapping file and save the updated mapping file
    """
    # Load the default mapping file
    defaultMapping = pd.read_csv(defaultMappingFilePath)

    # Drop specified columns
    for column in specifiedColumnsToDrop:
        if column in defaultMapping.columns:
            defaultMapping = defaultMapping.drop(column, axis=1)

    # Save the updated mapping file
    if not updatedMappingFilePath.endswith('.csv'):
        updatedMappingFilePath += '.csv'
    
    defaultMapping.to_csv(updatedMappingFilePath, index=False)
    print(f"\nUpdated mapping file saved as '{updatedMappingFilePath}'\n")

def loadAndPreprocessData(filePath:str, specifiedColumnsToDrop:list = [], binaryToMultiple:bool = False):
    """Load and preprocess the dataset from a CSV file.

    This function reads data from the specified CSV file, performs preprocessing steps such as
    stripping whitespace from column names, dropping unnecessary columns, handling missing values,
    and encoding categorical variables. It then separates the features from the target variable.

    Args:
        filePath (str): The path to the CSV file containing the dataset.

    Returns:
        tuple: A tuple containing:
            - features (pd.DataFrame): The preprocessed feature columns.
            - target (pd.Series): The target variable column.

    Raises:
        FileNotFoundError: If the specified CSV file does not exist.
        pd.errors.EmptyDataError: If the CSV file is empty.
        KeyError: If the 'EmployeeID' or 'Attrition' columns are not found in the dataset.
    """
    # Load the dataset
    data = pd.read_csv(filePath)

    # Strip any leading or trailing spaces from column names
    data.columns = data.columns.str.strip()

    # Drop EmployeeID as it is not a feature
    data = data.drop('EmployeeID', axis=1)

    # Drop specified columns
    print("Dropping columns :")
    for column in specifiedColumnsToDrop:
        matchingColumns = [col for col in data.columns if col.lower() == column.lower()]
        for actualColumn in matchingColumns:
            data = data.drop(actualColumn, axis=1)
            print(f"\t- {actualColumn}")

    print()


    # Handle missing values by dropping rows with any NaN values
    data = data.dropna()

    # Encode categorical variables using LabelEncoder
    labelEncoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        labelEncoders[column] = LabelEncoder()
        data[column] = labelEncoders[column].fit_transform(data[column])

    if binaryToMultiple:
        # Create new target columns for "Stay" and "Quit"
        data['Stay'] = (data['Attrition'] == 0).astype(int)
        data['Quit'] = (data['Attrition'] == 1).astype(int)
        # Separate features and target variable
        features = data.drop(['Attrition', 'Stay', 'Quit'], axis=1)
        targets = data[['Stay', 'Quit']] 
    else:
        # Separate features and target variable
        features = data.drop('Attrition', axis=1)
        targets = data['Attrition']
    

    # Print column names for features and target
    print("Feature columns:", features.columns.tolist())
    if binaryToMultiple:
        print("Target columns:", targets.columns.tolist())
    else:
        print("Target column:", targets.name)

    return features, targets


class NeuralNetModel(nn.Module):
    def __init__(self, layers, input_activation, hidden_activation, outputActivation, dropout_rates):
        super(NeuralNetModel, self).__init__()
        layerList = []
        # Input layer
        layerList.append(nn.Linear(layers[0], layers[1]))
        layerList.append(nn.ReLU() if input_activation.lower()=='relu' else nn.Tanh())
        layerList.append(nn.BatchNorm1d(layers[1]))
        layerList.append(nn.Dropout(dropout_rates[0]))

        # Hidden layers
        prev = layers[1]
        for i, hidden in enumerate(layers[2:-1]):
            layerList.append(nn.Linear(prev, hidden))
            layerList.append(nn.ReLU() if hidden_activation.lower()=='relu' else nn.Tanh())
            layerList.append(nn.BatchNorm1d(hidden))
            dropoutIdx = min(i+1, len(dropout_rates)-1)
            layerList.append(nn.Dropout(dropout_rates[dropoutIdx]))
            prev = hidden

        # Output layer
        layerList.append(nn.Linear(prev, layers[-1]))
        if outputActivation.lower()=='sigmoid':
            layerList.append(nn.Sigmoid())
        elif outputActivation.lower()=='softmax':
            layerList.append(nn.Softmax(dim=1))

        self.net = nn.Sequential(*layerList)

    def forward(self, x):
        return self.net(x)

def buildNeuralNetModel(layers, inputActivation, hiddenActivation, outputActivation, metrics, loss, optimizer, dropoutRates, l2_reg=0.01):
    # Note: l2_reg will be passed to the optimizer via 'weight_decay'
    model = NeuralNetModel(layers, inputActivation, hiddenActivation, outputActivation, dropoutRates)
    return model

def trainNeuralNet(
    features,
    target,
    layers: list[int],
    epochs: int,
    batchSize: int,
    inputActivation: str = 'relu',
    hiddenActivation: str = 'relu',
    outputActivation: str = 'sigmoid',
    metrics: list = ['Accuracy'],
    loss: str = 'binary_crossentropy',
    optimizer_choice: str = 'adam',
    dropoutRates: list[float] = [0.5],
    trainingTestingSplit: float = 0.2,
    l2_reg: float = 0.01,
    verbose: int = 1
):
    # Split dataset (using same strategy as before)
    TrainingFeatures, TestFeatures, trainingLabels, testLabels = train_test_split(
        features, target, test_size=trainingTestingSplit, random_state=RANDOM_SEED, stratify=target
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = buildNeuralNetModel(layers, inputActivation, hiddenActivation, outputActivation, metrics, loss, optimizer_choice, dropoutRates, l2_reg)
    model.to(device)

    # Prepare optimizer (pass l2_reg as weight_decay)
    if optimizer_choice.lower()=='adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=l2_reg)
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=l2_reg)

    # Loss function
    criterion = nn.BCELoss() if loss=='binary_crossentropy' else nn.MSELoss()

    # Convert data to torch tensors
    trainX = torch.tensor(TrainingFeatures.values, dtype=torch.float32).to(device)
    trainy = torch.tensor(trainingLabels.values, dtype=torch.float32).unsqueeze(1).to(device)

    dataset = torch.utils.data.TensorDataset(trainX, trainy)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True)

    history = {'loss': [], 'Accuracy': []}
    for epoch in range(epochs):
        model.train()
        epochLoss = 0
        correct = 0
        total = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            lossVal = criterion(outputs, batch_y)
            lossVal.backward()
            optimizer.step()

            epochLoss += lossVal.item() * batch_X.size(0)
            predicted = (outputs > 0.5).float()
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        epochLoss /= len(dataset)
        accuracy = correct / total
        history['loss'].append(epochLoss)
        history['Accuracy'].append(accuracy)
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epochLoss:.4f} - Accuracy: {accuracy*100:.2f}%")

    # You can similarly evaluate on TestFeatures later.
    return model, history

def detectOverfitting(history, lossFunction):
    """Analyze the training history to detect potential overfitting.

    This function compares training and validation accuracy and loss to determine if the model is overfitting.

    Args:
        history (tf.keras.callbacks.History): The history object returned by the model's fit method.
        lossFunction (str): The loss function used during training.

    Returns:
        None

    Prints:
        - A warning message if overfitting is detected.
        - A confirmation message if the model is not overfitting.
    """
    print("----------------------------------")
    trainAcc = history.history['Accuracy']
    valAcc = history.history['val_Accuracy']
    trainLoss = history.history['loss']
    valLoss = history.history['val_loss']

    if lossFunction == 'squared_hinge':
        valLoss = np.array(valLoss) - 1

    accDiff = trainAcc[-1] - valAcc[-1]
    lossDiff = valLoss[-1] - trainLoss[-1]

    if (accDiff > 0.15) or (lossDiff > 0.15):
        print("Warning: The model is overfitting.")
    else:
        print("The model is not overfitting.")
    print("----------------------------------")


def findInputImportance(model, features, numSamples=50, shapSampleSize=100, numRuns=5, plotSavePath=None):
    """Calculate and plot SHAP values to determine feature importance.

    Args:
        model (tf.keras.Model): The trained neural network model.
        features (pd.DataFrame): The feature columns of the dataset.
        numSamples (int): The number of samples to use for summarizing the background data.
        shapSampleSize (int): The number of samples to use for calculating SHAP values.
        numRuns (int): The number of runs to average the SHAP values.
        plotSavePath (str): The directory path to save the plots.

    Returns:
        dict: A dictionary with features ordered from most to least important.

    Plots:
        - SHAP summary plot showing feature importance.
    """
    print("\nCalculating SHAP values for feature importance...")

    # Initialize a DataFrame to accumulate SHAP values
    accumulatedShapValues = pd.DataFrame(0, index=np.arange(shapSampleSize), columns=features.columns)

    for run in range(numRuns):
        print(getProgressBar(run / numRuns, run) + f" Run {run + 1}/{numRuns}")

        # Summarize the background data using shap.kmeans
        backgroundData = shap.kmeans(features, numSamples)

        # Create a SHAP explainer using KernelExplainer for more flexibility
        modelExplainer = shap.KernelExplainer(model.predict, backgroundData, seed=RANDOM_SEED)

        # Sample a subset of the dataset for SHAP value calculation
        shapSample = features.sample(shapSampleSize, random_state=RANDOM_SEED + run)

        # Calculate SHAP values for the sampled dataset
        shapValues = modelExplainer.shap_values(shapSample)

        # Ensure shapValues is 2D
        if isinstance(shapValues, list):
            shapValues = np.array(shapValues)
            if shapValues.ndim == 3:
                shapValues = shapValues[0]  # Assuming binary classification, take the first set of SHAP values

        # Reshape shapValues to 2D if necessary
        if shapValues.ndim == 3:
            shapValues = shapValues.reshape(shapValues.shape[0], -1)

        # Accumulate SHAP values
        accumulatedShapValues += pd.DataFrame(shapValues, columns=features.columns)

    print(getProgressBar(1, numRuns) + " SHAP calculation completed.", end='\r')

    # Average the accumulated SHAP values
    meanShapValues = accumulatedShapValues / numRuns

    # Calculate mean absolute SHAP values for each feature
    meanAbsShapValues = meanShapValues.abs().mean().sort_values(ascending=False)

    # Convert to dictionary
    featuresImportance = meanAbsShapValues.to_dict()

    # Plot the SHAP summary plot
    shap.summary_plot(meanShapValues.values, shapSample, show=False, max_display=10)
    if plotSavePath:
        plt.savefig(f"{plotSavePath}/shapSummaryPlot.png")

    plt.show()
    plt.clf()

    # Bar plot of feature importance
    shap.summary_plot(meanShapValues.values, shapSample, plot_type='bar', show=False, max_display=15)
    if plotSavePath:
        plt.savefig(f"{plotSavePath}/shapBarPlot.png")

    plt.show()
    plt.clf()

    return featuresImportance


def saveModel(model: nn.Module, filePath: str):
    """Save the trained neural network model to the specified file path.

    The model is saved in PyTorch format. If the provided file path does not end with '.pt',
    the '.pt' extension is appended.

    Args:
        model (nn.Module): The trained neural network model to be saved.
        filePath (str): The destination file path where the model will be saved.

    Returns:
        str: The file path where the model was saved.

    Raises:
        IOError: If the model cannot be saved to the specified path.
    """
    print(f"\nSaving model...")
    if not filePath.endswith('.pt'):
        filePath += '.pt'

    # Move model to CPU before saving
    modelCpu = model.cpu()
    torch.save(modelCpu.state_dict(), filePath)
    print(f"Model saved to '{filePath}'")
    return filePath


def plotLearningCurve(history, epochs, elapsedTime, lossFunction):
    """Plot the learning curves for accuracy and loss over epochs.

    This function creates an animated plot showing the progression of training and validation
    accuracy and loss over each epoch. The plot is styled with specified colors and saved as an animation.

    Args:
        history (tf.keras.callbacks.History): The history object returned by the model's fit method.
        epochs (int): Total number of epochs the model was trained for.
        elapsedTime (pd.Timedelta): The total time taken to train the model.
        lossFunction (str): The loss function used during training.

    Returns:
        matplotlib.animation.FuncAnimation: The animation object representing the learning curves.
    """
    # Define color variables using hex codes
    backgroundColor = '#222222'
    lineColorTrain = '#FD4F59'
    lineColorVal = '#5BAFFC'
    textColor = '#DDDDDD'
    gridColor = '#5B5B5B'

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))  # Adjusted height to 8
    fig.patch.set_facecolor(backgroundColor)
    ax1.set_facecolor(backgroundColor)
    ax2.set_facecolor(backgroundColor)

    # Adjust val_loss if the loss function is squared_hinge
    if lossFunction == 'squared_hinge':
        adjustedValLoss = np.array(history.history['val_loss']) - 1
    else:
        adjustedValLoss = np.array(history.history['val_loss'])

    def animate(i):
        ax1.clear()
        ax2.clear()

        if i < len(history.history['Accuracy']):
            ax1.plot(
                np.array(history.history['Accuracy'][:i]) * 100,
                color=lineColorTrain,
                label='Train Accuracy'
            )
            ax1.plot(
                np.array(history.history['val_Accuracy'][:i]) * 100,
                color=lineColorVal,
                label='Validation Accuracy'
            )
            ax2.plot(
                np.array(history.history['loss'][:i]),
                color=lineColorTrain,
                label='Train Loss'
            )
            ax2.plot(
                adjustedValLoss[:i],
                color=lineColorVal,
                label='Validation Loss'
            )
        else:
            ax1.plot(
                np.array(history.history['Accuracy']) * 100,
                color=lineColorTrain,
                label='Train Accuracy'
            )
            ax1.plot(
                np.array(history.history['val_Accuracy']) * 100,
                color=lineColorVal,
                label='Validation Accuracy'
            )
            ax2.plot(
                np.array(history.history['loss']),
                color=lineColorTrain,
                label='Train Loss'
            )
            ax2.plot(
                adjustedValLoss,
                color=lineColorVal,
                label='Validation Loss'
            )

        ax1.set_title(f'Model Accuracy (Elapsed Time: {elapsedTime})', color=textColor)
        ax1.set_ylabel('Accuracy (%)', color=textColor)
        ax1.set_xlabel('Epoch', color=textColor)
        ax1.legend(loc='upper left')
        ax1.set_ylim([0, 100])  # Fix the y-axis scale for Accuracy
        ax1.set_xlim([0, epochs])  # Fix the x-axis scale
        ax1.tick_params(axis='x', colors=textColor)
        ax1.tick_params(axis='y', colors=textColor)
        ax1.grid(True, color=gridColor, linestyle='--', linewidth=0.5)

        ax2.set_title('Model Loss', color=textColor)
        ax2.set_ylabel('Loss', color=textColor)
        ax2.set_xlabel('Epoch', color=textColor)
        ax2.legend(loc='upper left')

        ax2.set_ylim([0, 1])
        ax2.set_xlim([0, epochs])  # Fix the x-axis scale
        ax2.tick_params(axis='x', colors=textColor)
        ax2.tick_params(axis='y', colors=textColor)
        ax2.grid(True, color=gridColor, linestyle='--', linewidth=0.5)

    # Calculate the number of frames for the 5-second pause at the end
    pauseFrames = 5 * 30  # 5 seconds at 30 fps

    # Use the length of the history data for frames plus the pause frames
    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=len(history.history['Accuracy']) + pauseFrames,
        interval=50,
        repeat=False
    )

    # Adjust layout to prevent overlap
    plt.subplots_adjust(hspace=0.4, top=0.9, bottom=0.1)

    plt.show()

    return ani


def evaluateModel(model, TestFeatures, actualLabels, threshold=0.3, verbose=1):
    """Evaluate the trained model's performance on the test dataset.

    This function calculates the predicted probabilities, converts them to binary labels based on a threshold,
    and prints the classification report and confusion matrix.

    Args:
        model (tf.keras.Model): The trained neural network model.
        TestFeatures (pd.DataFrame): The feature columns of the test dataset.
        actualLabels (pd.Series): The true labels of the test dataset.
        threshold (float, optional): The probability threshold for converting predicted probabilities to binary labels. Defaults to 0.3.

    Returns:
        None

    Prints:
        - Classification report showing precision, recall, f1-score, and support.
        - Confusion matrix showing true positives, true negatives, false positives, and false negatives.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        test_X = torch.tensor(TestFeatures.values, dtype=torch.float32).to(device)
        predictedProb = model(test_X)
    
    predictedProb = predictedProb.cpu().numpy()

    predictedLabel = (predictedProb > threshold).astype("int32")
    if verbose: print("Classification Report:")
    classificationReport = classification_report(actualLabels, predictedLabel, zero_division=0)
    if verbose: print(classificationReport)
    if verbose: print("Confusion Matrix:")
    confusionMatrix = confusion_matrix(actualLabels, predictedLabel)
    if verbose: print(confusionMatrix)

    return classificationReport, confusionMatrix


def loadModel(modelPath: str, modelClass, modelArgs: tuple, verbose=1):
    """Load a trained PyTorch neural network model from the specified file path.

    Args:
        modelPath (str): The path to the saved state_dict.
        modelClass (class): The neural network class.
        modelArgs (tuple): The arguments needed to initialize the model.

    Returns:
        nn.Module: The loaded PyTorch neural network model.

    Raises:
        IOError: If the model cannot be loaded from the specified path.
    """
    if verbose:
        print(f"\nLoading model from '{modelPath}'...")
    
    model = modelClass(*modelArgs)
    stateDict = torch.load(modelPath)
    model.load_state_dict(stateDict)
    if verbose:
        print("Model loaded successfully")
    
    return model


def predictWithModel(model, inputData):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        inputTensor = torch.tensor(inputData, dtype=torch.float32).to(device)
        prediction = model(inputTensor)
    # Intermediate outputs would require a modified forward() or hooks.
    # For now, simply return the final prediction.
    return prediction.cpu().numpy(), None


def runGridSearch(features, target, paramGrid:dict, CPULimitation:float = 0.7):
    """Perform a grid search to find the best combination of hyperparameters.

    This function tests various combinations of neural network architectures and training parameters
    to identify the configuration that yields the highest validation accuracy.

    Args:
        features (pd.DataFrame): The preprocessed feature columns.
        target (pd.Series): The target variable column.

    Returns:
        dict: A dictionary containing the best hyperparameters found during grid search.

    Prints:
        - The best accuracy achieved.
        - The corresponding best hyperparameters.
    """
    searchStartTime = pd.Timestamp.now()
    
    grid = ParameterGrid(paramGrid)
    totalGrid = len(grid)
    
    cpuCount = multiprocessing.cpu_count()
    nJobs = max(1, floor(cpuCount * CPULimitation))

    print(f"\n\nRunning grid search on {nJobs} CPU cores (power equivalent) for {totalGrid} parameters combinations...")

    manager = multiprocessing.Manager()
    bestAccuracy = manager.Value('d', 0.0)
    bestParams = manager.dict()
    progressWheelIndex = manager.Value('i', 0)
    totalElapsedTime = manager.Value('d', 0.0)
    valAccuracyHistory = manager.list()  # List to store validation accuracy history
    bestValAccuracyHistory = manager.list()  # List to store best validation accuracy history

    print(getProgressBar(0, progressWheelIndex.value) + "Best Accuracy: --.--%  |  Time remaining: --h--min --s  |  est. Finish Time: --h--", end='\r')

    def evaluate(params, idx, bestAccuracy, bestParams, valAccuracyHistory):
        trainStartTime = pd.Timestamp.now()
        
        try:

            optimizerName = params['optimizer'].lower()
            if optimizerName == 'adam':
                optimizer = optim.Adam(model.parameters(), lr=params['learningRate'])
            elif optimizerName == 'rmsprop':
                optimizer = optim.RMSprop(model.parameters(), lr=params['learningRate'])
            elif optimizerName == 'sgd':
                optimizer = optim.SGD(model.parameters(), lr=params['learningRate'])
            else:
                raise ValueError(f"Unknown optimizer: {optimizer}")
            
            model, history = trainNeuralNet(
                features=features,
                target=target,
                layers=params['layers'],
                epochs=params['epochs'],
                batchSize=params['batchSize'],
                inputActivation=params['inputActivation'],
                hiddenActivation=params['hiddenActivation'],
                outputActivation=params['outputActivation'],
                metrics=params['metrics'],
                loss=params['loss'],
                optimizer=optimizer,
                dropoutRates=params['dropoutRate'],
                trainingTestingSplit=params['trainingTestingSplit'],
                l2_reg=params['l2_reg'],
                verbose=0
            )        
            valAccuracy = history.history.get('val_Accuracy', [0])[-1]
            
        except Exception as e:
            print(f"-- Error in iteration {idx}: {e} --")
            valAccuracy = 0.0

        valAccuracyHistory.append(valAccuracy)  # Append validation accuracy to history
        if valAccuracy > bestAccuracy.value:
            bestAccuracy.value = valAccuracy
            bestParams.update(params)

        bestValAccuracyHistory.append(bestAccuracy.value)  # Append best validation accuracy to history

        # Update index for progress wheel
        progressWheelIndex.value += 1

        trainEndTime = pd.Timestamp.now()
        elapsedTrainTime = trainEndTime - trainStartTime
        totalElapsedTime.value += elapsedTrainTime.total_seconds()

        averageTime = (totalElapsedTime.value / progressWheelIndex.value) / nJobs

        estTimeRemaining = averageTime * (totalGrid - progressWheelIndex.value)
        estHours = int(estTimeRemaining // 3600)
        estMinutes = int((estTimeRemaining % 3600) // 60)
        estSeconds = int(estTimeRemaining % 60)

        estDays = estHours // 24
        estHours = estHours % 24

        if estDays > 0:
            estTimeStr = f"{estDays}day{'s' if estDays > 1 else ''} {estHours:02}h{estMinutes:02}min {estSeconds:02}s"
        else:
            estTimeStr = f"{estHours:02}h{estMinutes:02}min {estSeconds:02}s"

        estFinishTime = pd.Timestamp.now() + pd.Timedelta(seconds=estTimeRemaining)
        if estTimeRemaining > 86400:  # More than 24 hours
            estFinishTimeStr = estFinishTime.strftime('%d %b %Y %H:%M')
        else:
            estFinishTimeStr = f"{estFinishTime.hour:02}h{estFinishTime.minute:02}"

        completion = progressWheelIndex.value / totalGrid
        print(getProgressBar(completion, progressWheelIndex.value) + f"Best Accuracy: {bestAccuracy.value * 100:.2f}%  |  Time remaining: {estTimeStr}  |  est. Finish Time: {estFinishTimeStr}", end='\r')

        return valAccuracy, params

    results = Parallel(n_jobs=nJobs)(
        delayed(evaluate)(params, idx, bestAccuracy, bestParams, valAccuracyHistory) for idx, params in enumerate(grid, 1)
    )

    print(getProgressBar(1, progressWheelIndex.value) + f"Best Accuracy: {bestAccuracy.value * 100:.2f}%", end='\r')
    searchEndTime = pd.Timestamp.now()

    elapsedTime = searchEndTime - searchStartTime
    print(f"\n\nGrid search completed in {int(elapsedTime.components.hours):02}h {int(elapsedTime.components.minutes):02}min {int(elapsedTime.components.seconds):02}s (average training time: {totalElapsedTime.value / totalGrid:.2f}s)")

    # add the random seed to the best parameters
    bestParams['randomSeed'] = RANDOM_SEED

    print(f"\nBest Accuracy: {bestAccuracy.value * 100:.2f}%")
    print("Best Parameters:")
    paramstr = "\n".join([f"\t{k}: {v}" for k, v in bestParams.items()])
    print(paramstr)

    # generate the model ID (check in the Models folder for the latest model, and increment the id)
    lastId = 0
    for file in os.listdir('Models'):
        if file.startswith('TrainedModel_'):
            try:
                lastId = max(lastId, int(file.split('_')[-1]))
            except:
                pass
    
    modelId = lastId + 1
    modelHash = str(hash(str(bestParams)))[:6] # hash the best parameters to get a unique id (6 characters long)
    modelDirectory = f'Models/TrainedModel_{modelHash}_{modelId}/'
    backgroundColor = '#222222'

    if not os.path.exists(modelDirectory):
        os.makedirs(modelDirectory)

    # Plot the validation accuracy history
    plt.figure(figsize=(10, 6), facecolor=backgroundColor)
    plt.gca().set_facecolor(backgroundColor)
    plt.plot(valAccuracyHistory, linestyle='-', color='#FD4F59', label='Validation Accuracy')
    plt.plot(bestValAccuracyHistory, linestyle='-', color='#5BAFFC', label='Best Validation Accuracy')
    plt.gca().tick_params(axis='y', colors='white')
    plt.gca().tick_params(axis='x', colors='white')
    plt.title('Grid Search Validation Accuracy History', color='white')
    plt.xlabel('Iteration', color='white')
    plt.ylabel('Validation Accuracy', color='white')
    plt.legend()
    plt.grid(True)
    
    # Save the plot before showing it
    plt.savefig(f'{modelDirectory}ValidationAccuracyHistory.png')
    plt.show()

    return dict(bestParams), modelDirectory


def runModelTraining():
    """Execute the entire model training pipeline, including preprocessing, hyperparameter optimization,
    training, evaluation, and saving of the final model.

    This function performs the following steps:
    1. Checks for available GPU and configures TensorFlow accordingly.
    2. Loads and preprocesses the dataset.
    3. Defines the neural network architecture and training parameters.
    4. Performs grid search to find the best hyperparameters.
    5. Trains the final model using the best hyperparameters.
    6. Evaluates the model for overfitting.
    7. Saves the trained model and the learning curve plot.

    Returns:
        str: The file path where the trained model is saved.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        Exception: If any step in the training pipeline fails.
    """
    print("Random seed set to:", RANDOM_SEED)
    print("NumPy version:", np.__version__)
    print("PyTorch version", torch.__version__)
    print()
    
    # Check if PyTorch is using GPU
    print("-" * 50)
    print("Torch parameters:")
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.cuda.current_device():", torch.cuda.current_device())
    print("torch.cuda.get_device_name(0):", torch.cuda.get_device_name(0))
    print("-" * 50)

    print()

    tableToDrop = ['AverageHoursWorked']

    # Load and preprocess the dataset
    datasetFilePath = r'GeneratedDataSet\ModelDataSet.csv'
    features, targets = loadAndPreprocessData(datasetFilePath, tableToDrop, binaryToMultiple=False)

    outputNeuronsNumber = targets.shape[1] if len(targets.shape) > 1 else 1

    # Define hyperparameter grid for grid search
    hyperparameterGrid = {
        'layers': [
            [features.shape[1], 512, 256, 128, 64, outputNeuronsNumber],
            # [features.shape[1], 1024, 512, 64, outputNeuronsNumber],
            
        ],
        'epochs': [150],
        'batchSize': [32, 20],
        'dropoutRate': [
            [0.2],
            [0.5],
        ], # better to use the same number as the number of hidden layers + input layer
        'l2_reg': [0.015, 0.01],
        'learningRate': [0.001, 0.0005],
        "metrics": [
            # ['Accuracy', 'Recall', 'Precision'],
            ['Accuracy', 'Precision'],
            ['Accuracy', 'Recall'],
            # ['Accuracy'],
        ],
        'trainingTestingSplit': [0.2],
        'inputActivation': ['relu', 'tanh'],
        'hiddenActivation': ['relu', 'tanh'],
        'outputActivation': ['sigmoid'],
        'loss': ['binary_crossentropy'],
        'optimizer': ['adam']
    }

    powerLimitation = 0.8

    # Start hyperparameter grid search
    print("\nStarting Grid Search for Hyperparameter Optimization...")
    bestParams, modelDirectory = runGridSearch(features, targets, hyperparameterGrid, powerLimitation)

    # Create the model directory after finding the best parameters
    os.makedirs(os.path.dirname(modelDirectory), exist_ok=True)

    defaultMappingFilePath = r'GeneratedDataSet\MappingValues.csv'
    updatedMappingFilePath = modelDirectory + 'MappingValues.csv'
    saveCustomMapping(defaultMappingFilePath, updatedMappingFilePath, tableToDrop)

    # Record the start time of training
    startTrainingTime = pd.Timestamp.now()

    # Use bestParams to train the final model
    optimizer = optim.Adam(model.parameters(), lr=bestParams['learningRate'])
    model, history = trainNeuralNet(
        features=features,
        target=targets,
        layers=bestParams['layers'],
        epochs=bestParams['epochs'],
        batchSize=bestParams['batchSize'],
        inputActivation=bestParams['inputActivation'],
        hiddenActivation=bestParams['hiddenActivation'],
        outputActivation=bestParams['outputActivation'],
        metrics=bestParams['metrics'],
        loss=bestParams['loss'],
        optimizer=optimizer,
        dropoutRates=bestParams['dropoutRate'],
        trainingTestingSplit=0.2,
        l2_reg=bestParams['l2_reg'],
        verbose=1
    )

    # Record the end time of training
    endTrainingTime = pd.Timestamp.now()
    elapsedTime = endTrainingTime - startTrainingTime

    print(f"Training time: {elapsedTime}")

    # Check for overfitting
    detectOverfitting(history, bestParams['loss'])

    trainAccuracy = history.history['Accuracy'][-1]
    validationAccuracy = history.history['val_Accuracy'][-1]

    modelName = modelDirectory + f"Model_{trainAccuracy:.2f}_{validationAccuracy:.2f}_{elapsedTime.total_seconds()}s"

    # Save the trained model
    savePath = saveModel(model, modelName)

    # Save the model parameters to a file
    paramsFilePath = modelName + ".params"
    with open(paramsFilePath, 'w') as paramsFile:
        json.dump(bestParams, paramsFile)
    print(f"Model parameters saved as '{paramsFilePath}'")

    # Plot and save the learning curve
    plot = plotLearningCurve(history, bestParams['epochs'], elapsedTime, bestParams['loss'])

    # Save the plot as a gif using PillowWriter
    print(f"Saving learning curve gif...")
    plot.save(f'{modelName}.gif', writer=animation.PillowWriter(fps=30))
    print(f"Learning curve saved as '{modelName}.gif'")


    findInputImportanceChoice = True
    findInputImportanceUserChoice = input("Do you want to find the feature importance using SHAP values? (Y/n): ")
    if 'n' in findInputImportanceUserChoice.lower():
        findInputImportanceChoice = False

    if findInputImportanceChoice:
        # Find feature importance using SHAP values 
        featuresImportance = findInputImportance(model, features, plotSavePath=modelDirectory)

        # Save the feature importance to a file
        importanceFilePath = modelDirectory + 'FeatureImportance.csv'
        
        featuresImportance = pd.DataFrame(featuresImportance.items(), columns=['Feature', 'Importance'])
        featuresImportance.to_csv(importanceFilePath, index=False)
        print(f"Feature importance saved as '{importanceFilePath}'")


    # Return the path where the model was saved
    return savePath


def loadAndEvaluateModel(modelPath, features:pd.DataFrame, target:pd.Series, verbose=1):
    """Load a trained model and generate the confusion matrix for the test dataset.

    Args:
        modelPath (str): The path to the saved Keras model file.
        features (pd.DataFrame): The preprocessed feature columns.
        target (pd.Series): The target variable column.

    Returns:
        None

    Prints:
        - Confusion matrix showing true positives, true negatives, false positives, and false negatives.
    """
    global RANDOM_SEED
    if isinstance(modelPath, str):
        if not os.path.exists(modelPath):
            raise FileNotFoundError(f"Model file not found: '{modelPath}'")
        model = loadModel(modelPath, verbose=verbose)

        modelDirectory = os.path.dirname(modelPath)
        modelName = os.path.basename(modelPath).rsplit('.', 1)[0]
        if os.path.exists(modelDirectory + f'/{modelName}.params'):
            if verbose: print(f"Loading model parameters from '{modelDirectory}/{modelName}.params'")
            with open(modelDirectory + f'/{modelName}.params', 'r') as paramsFile:
                modelParamters = json.load(paramsFile)
                RANDOM_SEED = modelParamters['randomSeed']
                if verbose: print(f"Random seed set to: {RANDOM_SEED}")
        
    else:
        model = modelPath

    if verbose: print("\nEvaluating model...")  

    modelEvaluation = evaluateModel(model, features, target, verbose=verbose)
    # calculate the score of the confusion matrix (sum of false positives and false negatives)
    modelMatrixScore = modelEvaluation[1][0][1] + modelEvaluation[1][1][0]
    
    return modelName, modelEvaluation, modelMatrixScore

def evaluateModelsFromDirectory(rootDir:str, dataset:tuple):
    """
    Evaluate all the models in the directory
    """
    evaluations = {}
    
    for root, dirs, files in os.walk(rootDir):
        for modelRoot, modelDir, modelFiles in os.walk(root):
            for file in modelFiles:
                if file.endswith('.keras'):
                    modelPath = os.path.join(modelRoot, file)
                    print(f"\n\nEvaluating model: {modelPath}")
                    try:
                        modelName, modelEvaluation, modelMatrixScore = loadAndEvaluateModel(modelPath, dataset[0], dataset[1], verbose=0)
                        # save the evals as .pngs
                        savePath = modelPath.rsplit('.', 1)[0]
                        savePath = savePath + f'_ConfusionMatrix_{modelMatrixScore}.png'
                        plt.figure(figsize=(8, 6))
                        sns.heatmap(modelEvaluation[1], annot=True, fmt='d', cmap='Blues', cbar=False)
                        plt.xlabel('Predicted Label')
                        plt.ylabel('True Label')
                        plt.title(f'Confusion Matrix for {modelName}')
                        plt.savefig(savePath)
                        print(f"Saved confusion matrix for {modelName} as '{savePath}'")
                        plt.close()

                        
                        evaluations[modelName] = (modelEvaluation, modelMatrixScore)
                    except Exception as e:
                        print(f"Error evaluating model: {e}")


    # sort the evaluations by matrix score (lower is better)
    sortedEvals = sorted(evaluations.items(), key=lambda x: x[1][1], reverse=False)

    for modelName, (modelEvaluation, modelMatrixScore) in sortedEvals:
        print(f"\n\nModel: {modelName}")
        print(modelEvaluation[0])
        print()
        print(modelEvaluation[1])
        print(f"Matrix Score: {modelMatrixScore}")

# if __name__ == '__main__':
    # modelPath = runModelTraining()