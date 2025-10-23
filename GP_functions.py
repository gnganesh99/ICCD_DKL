"""
Gaussian Process functions for training and prediction using GPyTorch and BoTorch

Author: Ganesh Narasimha
"""
import matplotlib.pylab as plt
import numpy as np

import random
import torch
from torch import Tensor
from tqdm import tqdm


# Import GP and BoTorch functions
import gpytorch as gpt
import pandas as pd
from botorch.models import SingleTaskGP, ModelListGP

#from botorch.models import gpytorch
# from botorch.fit import fit_gpytorch_model
from botorch.models.gpytorch import GPyTorchModel
from botorch.utils import standardize
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel, PeriodicKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.constraints import GreaterThan
from gpytorch.models import ExactGP
from mpl_toolkits.axes_grid1 import make_axes_locatable
# from smt.sampling_methods import LHS
from torch.optim import SGD
from torch.optim import Adam
from scipy.stats import norm

from custom_models import DKL_Custom_nn, SimpleCNN
from torch.utils.data import DataLoader, Dataset

import torch.nn.functional as F

from Plot_DKL_predictions import plot_GP_mean


# Fitting of the GP model with the help of the base kernel

class SimpleGP(ExactGP, GPyTorchModel):
       # to inform GPyTorchModel API

    def __init__(self, train_X, train_Y, kernel = 'RBF', likelihood = GaussianLikelihood()):

      
        super().__init__(train_X, train_Y.squeeze(-1), likelihood=likelihood)  # squeeze output dim before passing train_Y to ExactGP

        # Mean Function
        self.mean_module = ConstantMean()
        #self.mean_module = LinearMean(train_X.shape[-1])


        # Covariance kernel
        if kernel == 'RBF':
            self.covar_module = ScaleKernel(
                base_kernel=RBFKernel(ard_num_dims=train_X.shape[-1]),
            )
        elif kernel == 'Matern':
            self.covar_module = ScaleKernel(
                base_kernel=MaternKernel(nu=2.5, ard_num_dims=train_X.shape[-1]),
            )
        elif kernel == 'Periodic':
            self.covar_module = ScaleKernel(
                base_kernel=PeriodicKernel(ard_num_dims=train_X.shape[-1]),
            )
        else:
            raise ValueError('Kernel should be either Periodic, Matern or RBF')

        # Register the model to the same device and precisioin as the data
        self.to(train_X)

    def forward(self, x):

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return MultivariateNormal(mean_x, covar_x)
    



def train_custom_nn_DKL(train_dataset, custom_nn, lr_custom_nn = 0.1, lr_gp = 0.1, num_epochs = 200, precision = 'double', device = None, plot_loss = False, n_batches = 3):
    
        if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
                device = device


        initialize_dataloader = DataLoader(train_dataset, batch_size = len(train_dataset), shuffle = False)
        
        train_batchsize = max(1, len(train_dataset)//n_batches)
        train_dataloader = DataLoader(train_dataset, batch_size = train_batchsize, shuffle = True)

        for train_X, _, train_Y in initialize_dataloader:
            # Load the data
            train_X, train_Y = train_X.to(device), train_Y.to(device)
            
      

        #Construct the joint model
          #Extract features from the custom_nn
        custom_nn.to(device)
        embeddings = custom_nn(train_X).detach() # No detach here

          # Define the GP model, Embed the features
        likelihood = GaussianLikelihood()
        gp = SimpleGP(embeddings, train_Y.squeeze(-1), likelihood = likelihood)
        gp.to(device)
       
        # Define the joint DKL model
        model = DKL_Custom_nn(custom_nn, gp)
        model.to(device)
    

        # Define the optimizer for the joint model
        optimizer = Adam([
            {'params': model.custom_nn.parameters(), 'lr': lr_custom_nn},
            {'params': model.gp.parameters(), 'lr': lr_gp}])


        #Register noise constraint (noise variance is always >= 0.1) to prevent degenerate solutions that may lead to overfitting
        model.gp.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-1))

        # Define the loss criterion
        mll = ExactMarginalLogLikelihood(likelihood, model.gp)
        # mll = mll.to(train_X)


        # set gp precision to double
        if precision == 'double':
            model.gp = model.gp.double()
        elif precision == 'single':
            model.gp = model.gp.float()
        else:
            raise ValueError('Precision should be either double or single')
        

        # Set the model and likelihood to training mode
        model.train()


        progress_bar = tqdm(total=num_epochs, desc="Training", leave=False)
        training_loss = []

        for epoch in range(num_epochs):
                
                epoch_loss = 0
                
                for train_X,_, train_Y in train_dataloader:
                     
                    train_X, train_Y = train_X.to(device), train_Y.to(device)
                

                    # clear gradients
                    optimizer.zero_grad()
                    
                    # forward pass through the custom_nn to obtain the embeddings
                    embeddings = model.custom_nn(train_X)

                    embeddings = embeddings.detach().to(device)

                    model.gp.set_train_data(inputs=embeddings, targets=train_Y.squeeze(-1), strict=False)

                    # forward pass through the model to obtain the output MultivariateNormal
                    output = model.gp(embeddings)

                    # calculate the loss
                    loss = -mll(output, train_Y.squeeze(-1))  # .train_targets is same as train_Y.squeeze(-1)
                    
                    epoch_loss += loss.item()/len(train_X)                  
                    
                    # back prop to compute gradients
                    loss.backward()         # use retain_graph=True if you want to reuse the computational graph during multiple loss functions

                    # update the parameters/weights
                    optimizer.step()

                            
                epoch_loss = epoch_loss/len(train_dataloader)
                training_loss.append(epoch_loss)


                # update progress bar
                progress_bar.update(1)
                progress_bar.set_postfix({'Loss': epoch_loss})

        progress_bar.close()
        
        model.eval()
        
        if plot_loss:
            plt.figure(figsize = (4,4))
            plt.plot(training_loss)
            plt.ylabel("Epoch loss")
            plt.xlabel("Epochs")
            plt.show()
         

        return model, training_loss


def train_mixed_nn_DKL(train_dataset, custom_nn, lr_custom_nn = 0.1, lr_gp = 0.1, num_epochs = 200, precision = 'double', device = None, plot_loss = False, n_batches = 3, weight_decay = 0):
    
        if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
                device = device

        initialize_dataloader = DataLoader(train_dataset, batch_size = len(train_dataset), shuffle = False)
        
        train_batchsize = max(1, len(train_dataset)//n_batches)
        train_dataloader = DataLoader(train_dataset, batch_size = train_batchsize, shuffle = True)

        for train_X, train_params, train_Y in initialize_dataloader:             
            # Load the data
            train_X, train_params, train_Y = train_X.to(device), train_params.to(device), train_Y.to(device)
            
      

        #Construct the joint model
          #Extract features from the custom_nn
        custom_nn.to(device)
        embeddings = custom_nn(train_X, train_params).to(device) 
          # Define the GP model, Embed the features
        likelihood = GaussianLikelihood()
        gp = SimpleGP(embeddings, train_Y.squeeze(-1), likelihood = likelihood) #.squeeze(-1) to match with gp.train_targets
        gp.to(device)
       
        # Define the joint DKL model
        model = DKL_Custom_nn(custom_nn, gp)
        model.to(device)
    

        # Define the optimizer for the joint model
        optimizer = Adam([
            {'params': model.custom_nn.parameters(), 'lr': lr_custom_nn, 'weight_decay': weight_decay},
            {'params': model.gp.parameters(), 'lr': lr_gp, 'weight_decay': weight_decay}])


        #Register noise constraint (noise variance is always >= 0.1) to prevent degenerate solutions that may lead to overfitting
        model.gp.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-1))

        # Define the loss criterion
        mll = ExactMarginalLogLikelihood(likelihood, model.gp)
        mll = mll.to(device)


        # set gp precision to double
        if precision == 'double':
            model.gp = model.gp.double()
        elif precision == 'single':
            model.gp = model.gp.float()
        else:
            raise ValueError('Precision should be either double or single')
        

        # Set the model and likelihood to training mode
        model.train()


        progress_bar = tqdm(total=num_epochs, desc="Training", leave=False)
        training_loss = []

        for epoch in range(num_epochs):
                
                epoch_loss = 0
                for train_X, train_params, train_Y in train_dataloader:
                        
                    train_X, train_params, train_Y = train_X.to(device), train_params.to(device), train_Y.to(device)

                    # clear gradients
                    optimizer.zero_grad()
                    
                    # forward pass through the custom_nn to obtain the embeddings
                    embeddings = model.custom_nn(train_X, train_params).to(device)
                    
                    #Reinitialize the model with the new embeddings. strict=False allows to set train data while retaining kernel hyperparameters
                    #train_Y.squeeze(-1), to match with gp output (or train_targets) which has the shape (n,)
                    model.gp.set_train_data(inputs=embeddings, targets=train_Y.squeeze(-1), strict=False)
                    
                    # forward pass through the model to obtain the output MultivariateNormal
                    output = model.gp(embeddings)
                    

                    # calculate the loss
                    loss = -mll(output, train_Y.squeeze(-1))  # model.gp.train_targets is same as train_Y.squeeze(-1)
                    epoch_loss += loss.item()/len(train_X)

                    
                    # back prop to compute gradients
                    loss.backward()         # use retain_graph=True if you want to reuse the computational graph during multiple loss functions

                    # update the parameters/weights
                    optimizer.step()
               
                
                epoch_loss = epoch_loss/len(train_dataloader)
                training_loss.append(epoch_loss)


                # update progress bar
                progress_bar.update(1)
                progress_bar.set_postfix({'Loss': epoch_loss})


        progress_bar.close()
        
        model.eval()
        
        if plot_loss:
            plt.figure(figsize = (4,4))
            plt.plot(training_loss)
            plt.ylabel("Epoch loss")
            plt.xlabel("Epochs")
            plt.show()
         

        return model, training_loss

def train_test_custom_nn_DKL(train_dataset, test_dataset, custom_nn, lr_custom_nn = 0.1, lr_gp = 0.1, num_epochs = 200, precision = 'double', device = None, plot_loss = False, n_batches = 3):
    
        if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
                device = device


        initialize_dataloader = DataLoader(train_dataset, batch_size = len(train_dataset), shuffle = False)
        
        
        train_batchsize = max(1, len(train_dataset)//n_batches)
        test_batchsize = max(1, len(test_dataset)//n_batches)
        
        train_dataloader = DataLoader(train_dataset, batch_size = train_batchsize, shuffle = True)
        test_dataloader = DataLoader(test_dataset, batch_size = test_batchsize, shuffle = True)
        
        for train_X, _, train_Y in initialize_dataloader:
    
            train_X, train_Y = train_X.to(device), train_Y.to(device)
      

        #Construct the joint model
          #Extract features from the custom_nn
        custom_nn.to(device)
        embeddings = custom_nn(train_X) # No detach here

          # Define the GP model, Embed the features
        likelihood = GaussianLikelihood()
        gp = SimpleGP(embeddings, train_Y.squeeze(-1), likelihood = likelihood)
        gp.to(device)
       
        # Define the joint DKL model
        model = DKL_Custom_nn(custom_nn, gp)
        model.to(device)
    

        # Define the optimizer for the joint model
        optimizer = Adam([
            {'params': model.custom_nn.parameters(), 'lr': lr_custom_nn},
            {'params': model.gp.parameters(), 'lr': lr_gp}])


        #Register noise constraint (noise variance is always >= 0.1) to prevent degenerate solutions that may lead to overfitting
        model.gp.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-1))

        # Define the loss criterion
        mll = ExactMarginalLogLikelihood(likelihood, model.gp)
        mll = mll.to(device)


        # set gp precision to double
        if precision == 'double':
            model.gp = model.gp.double()
        elif precision == 'single':
            model.gp = model.gp.float()
        else:
            raise ValueError('Precision should be either double or single')
        

        


        progress_bar = tqdm(total=num_epochs, desc="Training", leave=False)
        training_loss = []
        testing_loss = []

        for epoch in range(num_epochs):
                
                #Training the model
                epoch_loss = 0
                
                for train_X, _, train_Y in train_dataloader:

                    train_X, train_Y = train_X.to(device), train_Y.to(device)
                
                    # Set the model and likelihood to training mode
                    model.train()

                    # clear gradients
                    optimizer.zero_grad()
                    
                    # forward pass through the custom_nn to obtain the embeddings
                    embeddings = model.custom_nn(train_X)

                    embeddings = embeddings.to(device)

                    model.gp.set_train_data(inputs=embeddings, targets=train_Y.squeeze(-1), strict=False)

                    # forward pass through the model to obtain the output MultivariateNormal
                    output = model.gp(embeddings)

                    # calculate the loss. MLL loss is additive, so average it over the batch size
                    train_loss = -mll(output, train_Y.squeeze(-1)) # .train_targets is same as train_Y.squeeze(-1 )   
                    
                    epoch_loss += train_loss.item()/len(train_X)             
                    
                    # back prop to compute gradients
                    train_loss.backward()         # use retain_graph=True if you want to reuse the computational graph during multiple loss functions

                    # update the parameters/weights
                    optimizer.step()

                epoch_loss = epoch_loss/len(train_dataloader)
                training_loss.append(epoch_loss)


                #Testing the model
                test_loss = 0

                for test_X, _, test_Y in test_dataloader:

                    test_X, test_Y = test_X.to(device), test_Y.to(device)


                    #Testing the model
                    model.eval()

                    output = model.predict(test_X)

                    test_loss += -mll(output, test_Y.squeeze(-1)).item()/len(test_X)

                            
                test_loss = test_loss/len(test_dataloader)
                testing_loss.append(test_loss)

                # update progress bar
                progress_bar.update(1)
                progress_bar.set_postfix({'Train Loss': epoch_loss, 'Test Loss': test_loss})


        progress_bar.close()
        
        
        
        if plot_loss:
            plt.figure(figsize = (4,4))
            plt.plot(training_loss, label = 'Train Loss')
            plt.plot(testing_loss, label = 'Test Loss')
            plt.ylabel("Epoch loss")
            plt.xlabel("Epochs")
            plt.legend()
            plt.show()
         

        return model, training_loss, testing_loss


def train_test_mixed_nn_DKL(train_dataset, test_dataset, custom_nn, lr_custom_nn = 0.1, lr_gp = 0.1, num_epochs = 200, precision = 'double', device = None, plot_loss = False, n_batches = 3, weight_decay = 0):
    
        if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
                device = device

        
        initialize_dataloader = DataLoader(train_dataset, batch_size = len(train_dataset), shuffle = False)
        train_dataloader = DataLoader(train_dataset, batch_size = len(train_dataset)//n_batches, shuffle = True)
        test_dataloader = DataLoader(test_dataset, batch_size = len(test_dataset)//n_batches, shuffle = True)
        

        for train_X, train_params, train_Y in initialize_dataloader:
            # Load the data to the device
            train_X, train_params, train_Y = train_X.to(device), train_params.to(device), train_Y.to(device)
      


            
      

        #Construct the joint model
          #Extract features from the custom_nn
        custom_nn.to(device)
        embeddings = custom_nn(train_X, train_params).to(device) 
          # Define the GP model, Embed the features
        likelihood = GaussianLikelihood()
        gp = SimpleGP(embeddings, train_Y.squeeze(-1), likelihood = likelihood) #.squeeze(-1) to match with gp.train_targets
        gp.to(device)
       
        # Define the joint DKL model
        model = DKL_Custom_nn(custom_nn, gp)
        model.to(device)
    

        # Define the optimizer for the joint model
        optimizer = Adam([
            {'params': model.custom_nn.parameters(), 'lr': lr_custom_nn, 'weight_decay':weight_decay},
            {'params': model.gp.parameters(), 'lr': lr_gp, 'weight_decay':weight_decay}])


        #Register noise constraint (noise variance is always >= 0.1) to prevent degenerate solutions that may lead to overfitting
        model.gp.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-1))

        # Define the loss criterion
        mll = ExactMarginalLogLikelihood(likelihood, model.gp)
        mll = mll.to(device)


        # set gp precision to double
        if precision == 'double':
            model.gp = model.gp.double()
        elif precision == 'single':
            model.gp = model.gp.float()
        else:
            raise ValueError('Precision should be either double or single')
        
       


        progress_bar = tqdm(total=num_epochs, desc="Training", leave=False)
        training_loss = []
        testing_loss = []

        for epoch in range(num_epochs):
                
                #Training the model
                epoch_loss = 0
                for train_X, train_params, train_Y in train_dataloader:
                     
                    train_X, train_params, train_Y = train_X.to(device), train_params.to(device), train_Y.to(device)

                
                    # Set the model to training mode
                    model.train()

                    # clear gradients
                    optimizer.zero_grad()
                    
                    # forward pass through the custom_nn to obtain the embeddings
                    embeddings = model.custom_nn(train_X, train_params) #.to(device)
                    
                    #Reinitialize the model with the new embeddings. strict=False allows to set train data while retaining kernel hyperparameters
                    #train_Y.squeeze(-1), to match with gp output (or train_targets) which has the shape (n,)
                    model.gp.set_train_data(inputs=embeddings, targets=train_Y.squeeze(-1), strict=False)
                    
                    # forward pass through the model to obtain the output MultivariateNormal
                    output = model.gp(embeddings)
                    

                    # calculate the loss
                    train_loss = -mll(output, train_Y.squeeze(-1))  # model.gp.train_targets is same as train_Y.squeeze(-1)
                    
                    epoch_loss += train_loss.item()/len(train_X)

                    

                    
                    # back prop to compute gradients
                    train_loss.backward()         # use retain_graph=True if you want to reuse the computational graph during multiple loss functions

                    # update the parameters/weights
                    optimizer.step()

                epoch_loss = epoch_loss/len(train_dataloader)
                training_loss.append(epoch_loss)



                #Testing the model
                test_loss = 0


                for test_X, test_params, test_Y in test_dataloader:
                     
                    test_X, test_params, test_Y = test_X.to(device), test_params.to(device), test_Y.to(device)

                    #progress_bar.close()
            
                    model.eval()
                    
                    output = model.predict(test_X, test_params)
                    test_loss += -mll(output, test_Y.squeeze(-1)).item()/len(test_X)

                test_loss = test_loss/len(test_dataloader)
                testing_loss.append(test_loss)


                # update progress bar
                progress_bar.update(1)
                progress_bar.set_postfix({'Training Loss': epoch_loss, 'Test Loss': test_loss})

        progress_bar.close()



        
        
        if plot_loss:
            plt.figure(figsize = (4,4))
            plt.plot(training_loss, label = 'Train Loss')
            plt.plot(testing_loss, label = 'Test Loss')
            plt.ylabel("Epoch loss")
            plt.xlabel("Epochs")
            plt.legend()
            plt.show()

        return model, training_loss, testing_loss


def train_GPR(train_X, train_Y, precision = 'double', learning_rate = 0.1, num_epochs = 200, device = None, 
              model = None, plot_loss = False):

    # Define the device for training
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = device

    # Move the training data to the same device
    train_X, train_Y = train_X.to(device), train_Y.to(device)

    # Define the GP model
    if model is None:
        gp_model = SimpleGP(train_X, train_Y)
    else:
        gp_model = model(train_X, train_Y)

    # Define Precision
    if precision == 'single':
        gp_model = gp_model.float()    
    elif precision == 'double':
        gp_model = gp_model.double() 
    else:
        raise ValueError('Precision should be either single or double')
    

    #Registers a constraint (noise variance is always >= 0.1) to prevent degenerate solutions that may lead to overfitting
    gp_model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-1))

    # Define the Maximum Likelihood Estimation Loss Criterion. here likihood is Gaussian
    mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)

    # Move the MLL criterion to the same device and precision as the data
    mll = mll.to(train_X)

    # Set the model and likelihood into training mode
    gp_model.train()
    gp_model.likelihood.train()


    # Define the Adam optimizer
    optimizer = Adam([{'params': gp_model.parameters()}], lr = learning_rate) 

    progress_bar = tqdm(total=num_epochs, desc="GP Training", leave=False)
    training_loss = []

    # Training the GP model
    for epoch in range(num_epochs):

        # clear gradients
        optimizer.zero_grad()
        
        # forward pass through the model to obtain the output MultivariateNormal
        output = gp_model(train_X)
        
        # Compute MLL Loss
        loss = - mll(output, gp_model.train_targets)  # .train_targets is same as train_Y.squeeze(-1)
        
        training_loss.append(loss.item()/len(train_X))
 
        # back prop to compute gradients
        loss.backward()         # use retain_graph=True if you want to reuse the computational graph during multiple loss functions
 
        # update the parameters/weights
        optimizer.step()

        # update progress bar
        progress_bar.update(1)
        progress_bar.set_postfix({'Loss': loss.item()})

    progress_bar.close()

    # Set the model and likelihood into evaluation mode
    gp_model.eval()
    gp_model.likelihood.eval()

    training_loss = np.asarray(training_loss)
    
    if plot_loss:
            plt.figure(figsize = (4,4))
            plt.plot(training_loss, label = 'Train Loss')
            plt.ylabel("Epoch loss")
            plt.xlabel("Epochs")
            plt.legend()
            plt.show()

    return gp_model, training_loss


def parameter_mapping(train_params, train_y, orig_parameters, param_divs = [10, 10, 10, 10], plot_GP = False, num_epochs = 10):

    device = "cuda" if torch.cuda.is_available() else 'cpu'

    params, y = train_params.to(device), train_y.to(device)

    model, training_loss = train_GPR(params, y, num_epochs = num_epochs, device= device)
    print("training_loss", training_loss[-1])

    # Create parmater grid space

    F1 = np.linspace(0, 1, param_divs[0])
    F2 = np.linspace(0, 1, param_divs[1])
    P = np.linspace(0, 1, param_divs[2])
    T = np.linspace(0, 1, param_divs[3])

    param_grid = np.asarray(np.meshgrid(F1, F2, P, T))

    param_grid = param_grid .T .reshape(-1, 4)

    X = torch.tensor(param_grid, dtype = torch.float32).to(device)

    y_means, y_vars = GP_posterior(model, X)

    if plot_GP:

        plot_GP_mean(y_means, orig_parameters, param_grid, train_params, train_y)

    return y_means, y_vars, param_grid


def vae_loss_mse(output, train_spectra, beta_elbo = 1e-3):
    
    pred_spectra, mu, logvar = output  
    
    # Reconstruction Loss (Mean Squared Error)
    recon_loss = F.mse_loss(pred_spectra, train_spectra, reduction='mean')

    # KL Divergence Loss (Regularization)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta_elbo*kl_loss


def DKL_posterior(model, X_space, params = None, num_tasks = 1):

    """
    Calculate the posterior mean and variance of the DKL model
    To use posterior function, the gp_model should be inherited from GPyTorchModel

    params is the additional parameters that are required for the mixed_nn models
    """
    
    
    # move the model and the data to the device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    
    model.eval()
    
    X_space = X_space.to(device)

    if params is not None:
        params = params.to(device)

    
    # Initialize the shape of the predictions
    y_pred_means = torch.empty(len(X_space), num_tasks)
    
    y_pred_vars = torch.empty(len(X_space), num_tasks)
    
    t_X = torch.empty_like((X_space[0]))
    
    t_X = t_X.unsqueeze(0)
    
    
    for t in range(0, len(X_space)):
    
        with torch.no_grad(), gpt.settings.max_lanczos_quadrature_iterations(32), \
            gpt.settings.fast_computations(covar_root_decomposition=False, log_prob=False,
                                                      solves=True), \
            gpt.settings.max_cg_iterations(100), \
            gpt.settings.max_preconditioner_size(80), \
            gpt.settings.num_trace_samples(128):

                t_X = X_space[t:t+1]
                t_X.to(device)

                if params is not None:
                    t_params = params[t:t+1]
                    t_params = t_params.to(device)
                    predict_embeddings = model.custom_nn(t_X, t_params)
                
                else:
                    predict_embeddings = model.custom_nn(t_X)
                
                y_predictions = model.gp.posterior(predict_embeddings)

                y_pred_means[t] = y_predictions.mean
                y_pred_vars[t] = y_predictions.variance

    return y_pred_means, y_pred_vars


def veDKL_posterior(model, X_space, params = None):

    """
    Calculate the posterior mean and variance of the DKL model
    To use posterior function, the gp_model should be inherited from GPyTorchModel

    params is the additional parameters that are required for the mixed_nn models
    """
    
    # move the model and the data to the device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    
    model.eval()
    
    X_space = X_space.to(device)

    if params is not None:
        params = params.to(device)

    
    # Initialize the shape of the predictions
    y_pred_means = torch.empty(len(X_space), 1)
    
    y_pred_vars = torch.empty(len(X_space), 1)
    
    t_X = torch.empty_like((X_space[0]))
    
    t_X = t_X.unsqueeze(0)
    
    latent_embeddings_list = []
    
    
    for t in range(0, len(X_space)):
    
        with torch.no_grad(), gpt.settings.max_lanczos_quadrature_iterations(32), \
            gpt.settings.fast_computations(covar_root_decomposition=False, log_prob=False,
                                                      solves=True), \
            gpt.settings.max_cg_iterations(100), \
            gpt.settings.max_preconditioner_size(80), \
            gpt.settings.num_trace_samples(128):

                t_X = X_space[t:t+1]
                t_X.to(device)

                if params is not None:
                    t_params = params[t:t+1]
                    t_params = t_params.to(device)
                    predict_embeddings, _, _ = model.custom_nn(t_X, t_params)
                
                else:
                    predict_embeddings, _, _ = model.custom_nn(t_X)
                
                latent_embeddings_list.append(predict_embeddings)
                y_predictions = model.gp.posterior(predict_embeddings)
                y_pred_means[t, 0] = y_predictions.mean
                y_pred_vars[t, 0] = y_predictions.variance
               
            
    latent_embeddings = torch.cat(latent_embeddings_list, dim = 0)        
                    
    return y_pred_means, y_pred_vars, latent_embeddings




def GP_posterior(model, X_space):

    """
    Calculate the posterior mean and variance of the GP model
    To use posterior function, the gp_model should be inherited from GPyTorchModel

    """
    # move the model and the data to the device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    
    model.eval()
    
    X_space = X_space.to(device)



    y_pred_means = torch.empty(len(X_space), 1)
    
    y_pred_vars = torch.empty(len(X_space), 1)

    t_X = torch.empty_like((X_space[0]))
    
    t_X = t_X.unsqueeze(0)
    
    
    
    
    
    for t in range(0, len(X_space)):
    
        with torch.no_grad(), gpt.settings.max_lanczos_quadrature_iterations(32), \
            gpt.settings.fast_computations(covar_root_decomposition=False, log_prob=False,
                                                      solves=True), \
            gpt.settings.max_cg_iterations(100), \
            gpt.settings.max_preconditioner_size(80), \
            gpt.settings.num_trace_samples(128):

                t_X = X_space[t:t+1]
                            
                
                y_predictions = model.posterior(t_X)
                y_pred_means[t, 0] = y_predictions.mean
                y_pred_vars[t, 0] = y_predictions.variance

    return y_pred_means, y_pred_vars





def best_mean_estimate (train_X, train_Y, X, y_pred_means):
    #Best solution among the evaluated data

    ind = torch.argmax(train_Y)
    X_best_train = torch.empty((1,2))
    X_best_train[0, 0] = train_X[ind, 0]
    X_best_train[0, 1] = train_X[ind, 1]

    # Best estimated solution from GP model considering the non-evaluated solution

    ind = torch.argmax(y_pred_means)
    X_best_pred = torch.empty((1,2))
    X_best_pred[0, 0] = X[ind, 0]
    X_best_pred[0, 1] = X[ind, 1]

    return X_best_train, X_best_pred






def point_average(array, points = 5):
  p_average = []

  for i in range(len(array)):
    if i+1 < points:
      k = i+1
    else:
      k = points

    sum = 0
    for j in range(k):
      sum += array[i-j]

    p_average.append(sum/k)

  return p_average





def norm_0to1(arr):

    arr = np.asarray(arr)
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))



def acq_fn_EI(y_means, y_vars, train_Y_norm, eta = 0.01, index_exclude = [], sample_next_points = 1):

    y_means = y_means.squeeze().detach().numpy()
    y_vars = y_vars.squeeze().detach().numpy()
    y_std_dev = np.sqrt(y_vars)

    # Get the best value from the training data    
    best_value = train_Y_norm.max()
    best_value = best_value.detach().numpy()

    
    # Intialize the acquisition function values and Z values
    Acq_vals = np.zeros(len(y_means))
    Z_vals = np.zeros(len(y_means))


    # Currently not considering normalizing since the training data is already normalized
    # # Normalize the y_means
    # y_means_norm = norm_0to1(y_means)

    # #scale the best value accordingly
    # best_value_norm = (best_value - np.min(y_means)) / (np.max(y_means) - np.min(y_means))

    # Calculate the acquisition function values
    for i in range(0, len(y_std_dev)):

        if (y_std_dev[i] <= 0):
            
            Acq_vals[i] = 0
        
        else:
            Z_vals[i] =  (y_means[i] - best_value - eta)/y_std_dev[i]

            Acq_vals[i] = (y_means[i]- best_value - eta)*norm.cdf(Z_vals[i]) + y_std_dev[i]*norm.pdf(Z_vals[i])


    # Eliminate evaluated samples from consideration to avoid repeatation in future sampling
    Acq_vals[index_exclude] = -1          # setting it to -1 would render the acq value very low.

    
    # Get the index of the maximum value of the acquisition function
    acq_ind = np.argsort(Acq_vals)[::-1][:sample_next_points]
    
    # Get the maximum value of the acquisition function
    acq_val_max = Acq_vals[acq_ind]




    return acq_ind, acq_val_max, Acq_vals


def acq_fn_PI(y_means, y_vars, train_Y_norm, eta = 0.01, index_exclude = [], sample_next_points = 1):

    y_means = y_means.squeeze().detach().numpy()
    y_vars = y_vars.squeeze().detach().numpy()
    y_std_dev = np.sqrt(y_vars)

    # Get the best value from the training data    
    best_value = train_Y_norm.max()
    best_value = best_value.detach().numpy()

    
    # Intialize the acquisition function values and Z values
    Acq_vals = np.zeros(len(y_means))
    Z_vals = np.zeros(len(y_means))

    # Not considering normalizing since the training data is already normalized
    # # Normalize the y_means
    # y_means_norm = norm_0to1(y_means)

    # #scale the best value accordingly
    # best_value_norm = (best_value - np.min(y_means)) / (np.max(y_means) - np.min(y_means))

    # Calculate the acquisition function values
    for i in range(0, len(y_std_dev)):

        if (y_std_dev[i] <= 0):
            
            Acq_vals[i] = 0
        
        else:
            Z_vals[i] =  (y_means[i] - best_value - eta)/y_std_dev[i]

            # Probability of Improvement (PI) acquisition function
            Acq_vals[i] = norm.cdf(Z_vals[i])


    # Eliminate evaluated samples from consideration to avoid repeatation in future sampling
    Acq_vals[index_exclude] = -1          # setting it to -1 would render the acq value very low.

    
    
    # Get the index of the maximum value of the acquisition function
    acq_ind = np.argsort(Acq_vals)[::-1][:sample_next_points]
    
    # Get the maximum value of the acquisition function
    acq_val_max = Acq_vals[acq_ind]
    

    return acq_ind, acq_val_max, Acq_vals







def acq_fn_UCB(y_means, y_vars, beta = 1, index_exclude = [], sample_next_points = 1):

    y_means = y_means.squeeze().detach().numpy()
    y_vars = y_vars.squeeze().detach().numpy()
    y_std_dev = np.sqrt(y_vars)

    
    # Intialize the acquisition function values and Z values
    Acq_vals = np.zeros(len(y_means))


    # Calculate the acquisition function values
    for i in range(0, len(y_std_dev)):

        if (y_std_dev[i] <= 0):
            
            Acq_vals[i] = 0
        
        else:
            
            # Probability of Improvement (PI) acquisition function
            Acq_vals[i] = y_means[i] + beta*y_std_dev[i]


    # Eliminate evaluated samples from consideration to avoid repeatation in future sampling
    Acq_vals[index_exclude] = -1          # setting it to -1 would render the acq value very low.

    
    # Get the index of the maximum value of the acquisition function
    acq_ind = np.argsort(Acq_vals)[::-1][:sample_next_points]
    
    # Get the maximum value of the acquisition function
    acq_val_max = Acq_vals[acq_ind]
    
    return acq_ind, acq_val_max, Acq_vals


