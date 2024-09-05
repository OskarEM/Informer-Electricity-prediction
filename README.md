# Electricity Price Prediction Using Informer Model

This repository contains the implementation of an electricity price prediction system using the Informer model, a variation of the Transformer model, tailored for long-sequence time series forecasting. The model is designed to predict electricity prices in Norway, leveraging multivariate data and neural network techniques to achieve accurate forecasts.

## Project Overview

High electricity prices in Norway prompted the development of a predictive model that forecasts future prices using a multivariate, multi-step time series approach. The **Informer model** was selected due to its efficiency in handling long sequences compared to traditional recurrent neural networks.

This project aims to predict electricity prices six hours into the future using various data sources and to explore the performance of the model by measuring the **Root Mean Squared Error (RMSE)** and analyzing the results.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Future Work](#future-work)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Dataset

The dataset includes electricity prices from **Nordpool**, weather data from the **Norwegian Meteorological Institute**, and electricity consumption/production statistics from **Statnett**. The prediction target is the average electricity price (EUR/MWh), and the time series is collected in hourly steps. 

### Key Data Features:
- Electricity prices (low, high, average)
- Weather features (temperature, humidity, wind speed)
- Electricity consumption and production
- Data ranges from 2021 to 2023

## Methodology

1. **Data Preprocessing**:
   - Applied **Fourier Transform** and **sinusoidal/cosine transformations** to capture seasonal and cyclical patterns in time-series data.
   - Handled missing data using **linear interpolation** and **forward filling** techniques.
   - Created **lagged features**, **rolling mean**, and **standard deviation** columns to improve pattern recognition.

2. **Model Tuning**:
   - The Informer model parameters (e.g., sequence length, label length, prediction length) were optimized using **Bayesian optimization** with **Weights and Biases (WandB)**.
   - Different prediction lengths were tested: 1 hour, 6 hours, 12 hours, and 24 hours.

3. **Evaluation Metrics**:
   - **Root Mean Squared Error (RMSE)** was used to evaluate the prediction accuracy.
   - **Mean Squared Error (MSE)** was the primary loss function used during model training.

## Model Architecture

The **Informer model** is based on the Transformer architecture but optimized for long time-series data. It consists of:
- **Encoder**: Processes input data and reduces time dimension across different layers.
- **Decoder**: Generates predictions using attention mechanisms.
- **Attention Mechanisms**: Utilizes **ProbSparse attention** for efficient handling of long sequences.

## Results

The model was tested on multiple configurations, with the best results achieved on the 2023 dataset after excluding noisy data from earlier years.

### Key Results:
- **RMSE for 6-hour predictions**: 16.97 EUR/MWh
- **RMSE for 12-hour predictions**: 21.73 EUR/MWh
- **Best model configuration**: Forward-filling missing data, 6-hour prediction window, optimized with Bayesian sweeping.

## Future Work

Future enhancements could include:
- Expanding the dataset to include gas and oil prices to capture external economic factors.
- Improving data handling for missing values using more sophisticated techniques like LSTM-based predictions.
- Further optimizing parameters and exploring other normalization methods like **maximum absolute scaling**.
- Addressing overfitting by refining model validation and testing splits.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/electricity-prediction-informer.git
   cd electricity-prediction-informer
