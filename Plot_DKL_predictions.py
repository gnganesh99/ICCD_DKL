import os
import numpy as np
import torch
import matplotlib.pyplot as plt



def plot_train_series(train_indices, orig_params, score):

    if torch.is_tensor(orig_params):
        orig_params = orig_params.detach().numpy()

    F1 = orig_params[train_indices,0]
    F2 = orig_params[train_indices,1]
    P = orig_params[train_indices,2]
    T = orig_params[train_indices,3]

    score = score.detach().numpy()[train_indices,0]


    ## Plot the results
    fig, ax = plt.subplots(1, 5, figsize=(22, 5))

    ax[0].plot(F1,'o-', label='F1')
    ax[0].set_title('Fluence 1')

    ax[1].plot(F2,'o-', label='F2')
    ax[1].set_title('Fluence 2')

    ax[2].plot(P,'o-', label='P')
    ax[2].set_title('Pressure')

    ax[3].plot(T,'o-', label='T')
    ax[3].set_title('Temperature')

    ax[4].plot(score,'o-', label='Score')
    ax[4].set_title('Score')


    plt.show()



def plot_train_hist(train_indices, orig_params):

    if torch.is_tensor(orig_params):
        orig_params = orig_params.detach().numpy()

    F1 = orig_params[train_indices,0]
    F2 = orig_params[train_indices,1]
    P = orig_params[train_indices,2]
    T = orig_params[train_indices,3]
    
    # score = score.detach().numpy()[train_indices,0]

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))

    ax[0].hist(F1, bins=20, alpha=0.8)
    ax[0].set_title('Fluence 1')

    ax[1].hist(F2, bins=20, alpha=0.8)
    ax[1].set_title('Fluence 2')

    ax[2].hist(P, bins=20, alpha=0.8)
    ax[2].set_title('Pressure')

    ax[3].hist(T, bins=20, alpha=0.8)
    ax[3].set_title('Temperature')
    
    # ax[4].hist(score, bins=20, alpha=0.8)
    # ax[4].set_title('Score')

    plt.show()



def plot_mean_map(y_means, train_indices, orig_params, score):

    F1 = orig_params[:,0]
    F2 = orig_params[:,1]
    P = orig_params[:,2]
    T = orig_params[:,3]

    score = score.detach().numpy().squeeze()

    y = score


    P.shape, T.shape, F1.shape, F2.shape, y_means.shape

    fig, ax = plt.subplots(2, 3, figsize=(20, 10))

    ax[0,0].scatter(T, P, c=y_means, cmap='viridis', alpha=0.5)
    ax[0,0].scatter(T[train_indices], P[train_indices], c=y[train_indices], cmap = 'jet', s=100)
    ax[0,0].set_ylabel('Pressure')
    ax[0,0].set_xlabel('Temperature')

    ax[0,1].scatter(F1, P, c=y_means, cmap='viridis')
    ax[0,1].scatter(F1[train_indices], P[train_indices], c=y[train_indices], cmap = 'jet', s=100)
    ax[0,1].set_ylabel('Pressure')
    ax[0,1].set_xlabel('Fluence 1')

    ax[0,2].scatter(F2, P, c=y_means, cmap='viridis')
    ax[0,2].scatter(F2[train_indices], P[train_indices], c=y[train_indices], cmap = 'jet', s=100)
    ax[0,2].set_ylabel('Pressure')
    ax[0,2].set_xlabel('Fluence 2')

    ax[1,0].scatter(F1, T, c=y_means, cmap='viridis')
    ax[1,0].scatter(F1[train_indices], T[train_indices], c=y[train_indices], cmap = 'jet', s=100)
    ax[1,0].set_ylabel('Temperature')
    ax[1,0].set_xlabel('Fluence 1')


    ax[1,1].scatter(F2, T, c=y_means, cmap='viridis')
    ax[1,1].scatter(F2[train_indices], T[train_indices], c=y[train_indices], cmap = 'jet', s=100)
    ax[1,1].set_ylabel('Temperature')
    ax[1,1].set_xlabel('Fluence 2')

    ax[1,2].scatter(F2, F1, c=y_means, cmap='viridis')
    ax[1,2].scatter(F2[train_indices], F1[train_indices], c=y[train_indices], cmap = 'jet', s=100)
    ax[1,2].set_ylabel('Fluence 1')
    ax[1,2].set_xlabel('Fluence 2')

    plt.show()


def plot_GP_mean(y_means, orig_params, params_grid, train_params, y_train):

    if torch.is_tensor(orig_params):
        orig_params = orig_params.detach().numpy()

    if torch.is_tensor(params_grid):
        params_grid = params_grid.detach().numpy()

    F1_orig = orig_params[:,0]
    F2_orig = orig_params[:,1]
    P_orig = orig_params[:,2]
    T_orig = orig_params[:,3]

    F1_orig_max = np.max(F1_orig)
    F2_orig_max = np.max(F2_orig)
    P_orig_max = np.max(P_orig)
    T_orig_max = np.max(T_orig)

    F1_orig_min = np.min(F1_orig)
    F2_orig_min = np.min(F2_orig)
    P_orig_min = np.min(P_orig)
    T_orig_min = np.min(T_orig)
    

    F1 = params_grid[:,0]* (F1_orig_max - F1_orig_min) + F1_orig_min
    F2 = params_grid[:,1]* (F2_orig_max - F2_orig_min) + F2_orig_min
    P = params_grid[:,2]* (P_orig_max - P_orig_min) + P_orig_min
    T = params_grid[:,3]* (T_orig_max - T_orig_min) + T_orig_min

    F1_train = train_params[:,0]* (F1_orig_max - F1_orig_min) + F1_orig_min
    F2_train = train_params[:,1]* (F2_orig_max - F2_orig_min) + F2_orig_min
    P_train = train_params[:,2]* (P_orig_max - P_orig_min) + P_orig_min
    T_train = train_params[:,3]* (T_orig_max - T_orig_min) + T_orig_min


    y_train = y_train.detach().numpy().squeeze()
    y_means = y_means.squeeze().numpy().squeeze()


    P.shape, T.shape, F1.shape, F2.shape, y_means.shape

    fig, ax = plt.subplots(2, 3, figsize=(20, 10))

    ax[0,0].scatter(T, P, c=y_means, cmap='viridis', alpha=0.5)
    ax[0,0].scatter(T_train, P_train, c=y_train, cmap = 'jet', s=100)
    ax[0,0].set_ylabel('Pressure')
    ax[0,0].set_xlabel('Temperature')

    ax[0,1].scatter(F1, P, c=y_means, cmap='viridis')
    ax[0,1].scatter(F1_train, P_train, c=y_train, cmap = 'jet', s=100)
    ax[0,1].set_ylabel('Pressure')
    ax[0,1].set_xlabel('Fluence 1')

    ax[0,2].scatter(F2, P, c=y_means, cmap='viridis')
    ax[0,2].scatter(F2_train, P_train, c=y_train, cmap = 'jet', s=100)
    ax[0,2].set_ylabel('Pressure')
    ax[0,2].set_xlabel('Fluence 2')

    ax[1,0].scatter(F1, T, c=y_means, cmap='viridis')
    ax[1,0].scatter(F1_train, T_train, c=y_train, cmap = 'jet', s=100)
    ax[1,0].set_ylabel('Temperature')
    ax[1,0].set_xlabel('Fluence 1')


    ax[1,1].scatter(F2, T, c=y_means, cmap='viridis')
    ax[1,1].scatter(F2_train, T_train, c=y_train, cmap = 'jet', s=100)
    ax[1,1].set_ylabel('Temperature')
    ax[1,1].set_xlabel('Fluence 2')

    ax[1,2].scatter(F2, F1, c=y_means, cmap='viridis')
    ax[1,2].scatter(F2_train, F1_train, c=y_train, cmap = 'jet', s=100)
    ax[1,2].set_ylabel('Fluence 1')
    ax[1,2].set_xlabel('Fluence 2')

    plt.show()
