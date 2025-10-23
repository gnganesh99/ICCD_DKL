# ICCD_DKL
Deep Kernel Learning on (Intensified Charge-Coupled Device) ICCD image sequence from Pulsed Laser Deposition (PLD) Experiments

## Overview

This project implements Deep Kernel Learning (DKL) methods for analyzing ICCD image sequences. The program combines neural networks with Gaussian processes to predict material properties from plasma deposition parameters and ICCD images.

The repository contains three Jupyter notebooks that demonstrate different approaches:

- **[DKL_with_customCNN.ipynb](DKL_with_customCNN.ipynb)**: Implements Deep Kernel Learning with a custom CNN tailored for ICCD image sequences to extract spatial features prior to Gaussian process modeling.

- **[DKL_with_Mixedmodels.ipynb](DKL_with_Mixedmodels.ipynb)**: Implements DKL with mixed neural network models that combine ICCD image analysis with plasma deposition parameters. Features two approaches: MixedICCDNet using (2+1)D CNN for image sequences concatenated with MLP for parameters, and RCNN-based mixed networks using LSTM for temporal analysis.

- **[VE_DKL_iccd.ipynb](VE_DKL_iccd.ipynb)**: Combines Variational Encoding (VE) with Deep Kernel Learning to segregate samples in the latent space. Uses Mixed-LSTMCNN as the neural network initializer for enhanced feature representation.

## Key Components
- **[custom_models.py](custom_models.py)**: Defines custom neural network architectures including DKL models that combine neural networks with Gaussian process layers for flexible feature extraction and uncertainty quantification.

- **[GP_functions.py](GP_functions.py)**: Contains Gaussian Process functions for training and prediction using GPyTorch and BoTorch libraries, providing the core GP functionality for the DKL framework.

- **[ICCD_Dataset.py](ICCD_Dataset.py)**: PyTorch Dataset class for loading and preprocessing ICCD image sequences along with growth parameters and Raman peak scores for model training.

- **[ICCDutils.py](ICCDutils.py)**: Utility functions for data loading and preprocessing, including functions to read JSON data files and normalize plasma deposition parameters.

- **[Plot_DKL_predictions.py](Plot_DKL_predictions.py)**: Visualization functions for plotting DKL predictions, training data series, parameter distributions, and 2D parameter space maps with uncertainty visualization.

## Credits
Experimental Data and initial analysis: Dr. Sumner Harris, Center for Nanophase Materials Sciences (CNMS), Oak Ridge National Laboratory (ORNL)

Supervision: Dr. Rama Vasudevan, CNMS, ORNL