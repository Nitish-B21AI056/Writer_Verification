# NCVPRIG Writer Verification
----------------------------------------------------------------------------------------------------------------------------------------------------------------------

This GitHub repository aims to identify whether a given pair of handwritten text samples was written by the same person or two different individuals. It provides a solution for verifying the authenticity of handwritten text samples, making it invaluable for real-world applications such as forensic analysis, document authentication, and signature verification systems.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Table of Contents

1. [Dataset](#dataset)
2. [Training Model](#training-model)
3. [Inference Model](#inference-model)
4. [Model Checkpoints](#model-checkpoints)
5. [Codebase](#codebase)

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Dataset <a name="dataset"></a>

The dataset used in this project was made available for the NCVPRIG Competition. However, due to the competition's terms and conditions, the actual dataset cannot be shared. We encourage you to use any other suitable dataset for your project.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Training Model <a name="training-model"></a>

If you want to train the model from scratch, follow these steps:

1. Add the path of your training and validation datasets in the `training_model.py` file.
2. Run the `training_model.py` script to initiate the training process.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Inference Model <a name="inference-model"></a>

If you want to use the pre-trained model directly, follow these steps:

1. Add the path of your test data in the `inference_model.py` file.
2. Run the `inference_model.py` script to perform inference using the pre-trained model.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Model Checkpoints <a name="model-checkpoints"></a>

Pre-trained models can be found in the following Google Drive folder:

[Trained models](https://drive.google.com/drive/folders/1GY2brp7-rYLxwLa6WBMvyC_SY3cjr1Cv?usp=sharing)

If you want to use the pre-trained model, download the models from the shared Google Drive folder and save them to your local location. Additionally, update the path location of the model in the code accordingly.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Codebase <a name="codebase"></a>

The codebase for this project consists of the following files:

- `training_model.py`: Script for training the model from scratch.
- `inference.py`: Script for performing inference using the pre-trained model.
- `Codebase.ipynb`: Jupyter Notebook containing the complete code of this project. We encourage you to go through it to understand the work done.

Feel free to explore and modify the codebase to suit your specific requirements.

