# Deep Learning Model for Predicting Biological Parameters

## Overview

This project explores the use of deep learning techniques to predict biological parameters based on a set of input features. Initially, multiple optimization methods were tested to predict any output from any input group. However, due to the limited amount of available data, achieving reliable results was challenging. As a result, the study was refined to focus on predicting a single parameter using simpler methods in another repository.

## Dataset

The dataset (dataset.csv) contains various biological features, including:

cell_migration

cell_invasion

cell_growth

wound_closure

protein_expression

colonization

average_tumor_volume

cell_proliferation_G0-G1phase

cell_proliferation_Sphase

cell_proliferation_G2-Mphase

apoptosis

mrna_expression_levels

The dataset undergoes preprocessing steps including handling missing values, splitting into training and testing sets, and standardization.

## Model Architecture

The neural network is designed as a feedforward model with regularization techniques:

Layers:

Dense layers with ReLU activation

L2 regularization to prevent overfitting

Batch Normalization for stable learning

Dropout layers (40% dropout rate) for better generalization

Loss Function: Mean Squared Error (MSE)

Optimizer: Adam (learning rate = 0.0001)

Early Stopping: Prevents overfitting by stopping training if validation loss does not improve for 10 consecutive epochs.

## Model Training

The model is trained using:

Training-Validation Split: 80% training, 20% validation

Epochs: 500 (with early stopping)

Batch Size: 32

Performance Metrics:

Training Loss vs. Validation Loss (plotted for monitoring overfitting)

Test Loss evaluation

Mean Absolute Error (MAE)

R-squared (R²) score for model performance

Flexible Prediction Interface

A flexible function predict_output allows users to:

Select a subset of input features

Choose a target feature to predict

Train a separate model on the selected inputs and predict the target feature

## Example usage:

selected_input_features = ['cell_migration', 'cell_invasion']
target_feature = 'wound_closure'
predict_fn = predict_output(selected_input_features, target_feature)
input_values = [0.15, 0.25]
predicted_value = predict_fn(input_values)
print(f'Predicted {target_feature} value: {predicted_value}')

## Model Evaluation

After training, the model is evaluated using:

Test Loss: To measure prediction accuracy

Mean Absolute Error (MAE):

mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error (MAE): {mae}')

R² Score:

r2 = r2_score(y_test, y_pred)
print(f'R-squared (R²): {r2}')

## Results and Findings

The model initially aimed to predict any output from any input group but faced challenges due to data limitations.

A refined approach focusing on a single output improved results.

The flexible prediction function allows customized model training on specific input-output relationships.

The model's generalization ability was evaluated by comparing training and test losses.
