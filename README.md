# Heart Disease Prediction (Backend)

This is the backend of the Heart Disease Prediction project. This code is made to be run on a server and is responsible for handling the prediction requests from the frontend. The model itself is loaded from the file 'model.keras' from previous training. The model is a Neural Network model with the following hyperparameters:

- 3 tanh hidden layers with 6 neurons each
- 1 softmax output layer with 2 neurons
- Binary Crossentropy loss function
- Adam optimizer

The backend is made with Flask and is run using Gunicorn.

## Installation

First clone the repository:

```bash
git clone
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To run the backend, simply run the following command:

```bash
gunicorn -b 0.0.0.0:5000 app:app
```

If you want to run it for development purposes, you can run the following command:

```bash
python app.py
```
