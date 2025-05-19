# Implementation of Fraud Detection Techniques

A flexible and customizable approach to Fraud Detection techniques where the user can supply any dataset with a binary target. Overall, this framework offers structure that can be modified by the user and supports supervised, unsupervised learning and neural networks.

Table of Contents 
- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Usage](#usage)
- [Extensibility](#extensibility)
- [Acknowledgements](#acknowledgements)

---

## Overview

A flexible, modular and reusable approach for fraud detection setting. Overall the architecture supports: 
- Data pre-processing step that can be identified by the end user.
- Class imbalance handler to take the control over imbalanced datasets.
- Model training that supports classical machine learning, neural networks, autoencoders, convolutional neural networks and generative adversarial networks.
- Allows for hyperparameter tuning for classical machine learning algorithms.
- Allows researchers to compare their methods with state-of-the-art models.

## Dataset

Throughout the coding process, all of our experiments operate over **Credit Card Fraud Detection** dataset from Kaggle: 
- **Source**: [Credit Card Fraud Detection][https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data]
- **Records**: 284,807 transactions with 492 fraud, highly imbalanced dataset with 0.172%.
- **Features**: Scaling and Principal Component Analysis have been applied to features. However, in this dataset:
        - Time, V1 to V28, Amount, Class (0 and 1)

However, the end user can supply a different dataset to work with files.

## Requirements

Anaconda was used to create virtual environment. In addition, **environment.yml** includes Python 3.10 and all other dependencies. User can reach to dependencies via: 
```bash
conda env create -f environment.yml
conda activate env
```

## Usage 

The code snippet below shows an example usage of the offered framework: 

```python

#Initialization of data
data = DataGathering(file_path="creditcard.csv",
                     target="Class",
                     pipeline_numerical = pipe_num,
                     pipeline_categorical = pipe_cat)

#Class imbalance handler depending on the user's usage
imbalance_handler = ImbalanceHandler(method = "undersampling")

#Initation and training of Neural Network
neunet = NeuralN(input_dimension = inp_dim,
                 output_dimension = out_dim,
                 hidden_layers=hidden,
                 loss_method="BCEwLogit",
                 opt_method = "Adam",
                 lr = 0.001,
                 epochs=10)

model = FraudDetectionPipeline(data=data,
imbalance_handler=imbalance_handler, neural_networks=neunet)
FraudDetectionPipeline.run_neuralnetwork(model)

#Same data initialization can be used for supervised models
trainer = TrainModel(search_strategy = "grid",
                     param_grid = PARAM_GRID,
                     performance_measure = "average_precision",
                     cv = 3,
                     model_dictionary = MODEL_DICT,
                     threshold = 0.45,
                     calibration_needed = "yes")

model = FraudDetectionPipeline(data=data,
imbalance_handler=imbalance_handler, trainer=trainer)
FraudDetectionPipeline.run_supervisedmodels(model)

```

## Extensibility

The structure of the pipeline for fraud detection is designed to be modular, reusable and extendable. It can be extandable by custom models with custom loss functions, additional model and class imbalance techniques, new evaluation metrics to better understand and strong feature importance techniques to interpret the black box. 

## Acknowledgements 

We would like to express our deepest gratitude to our supervisor Prof. Dr. Wouter Verbeke and daily advisor Dr. Bruno Deprez. Full references and bibliography can be found in the thesis.
