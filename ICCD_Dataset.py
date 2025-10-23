import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split, Dataset, ConcatDataset
from torchvision import transforms

import numpy as np

from ICCDutils import load_df


class ICCDDataset(Dataset):

    """
    Loads the ICCD dataset and returns the ICCD images, growth parameters and Raman peak scores

    Input:

        datafile: str, path to the data file (pandas dataframe)
        transform: torchvision.transforms, optional, default: None
        params_noise: function, optional, default: None

    Output:
    
        iccd: torch.tensor, shape (n, 1, 50, 40, 40)
        params: normalized parameters, torch.tensor of shape (n, 4) [F1, F2, P, T]
        score: Raman score, torch.tensor, shape (n, 1)
    
    """

    
    def __init__(self, datafile, transform = None, params_noise = None, image_for_rcnn = False, norm = False):
        
        df = load_df(datafile)
        iccd_3Dimages, params, score = extract_data(df, norm = norm)

        self.iccd_3Dimages = iccd_3Dimages
        self.params = params
        self.score = score

        # Transformations to apply to the images
        self.transform = transform

        # Add noise to the params
        self.params_noise = params_noise
        
        # Transform to normalize the images in range [0, 1]
        self.normalize_transform = transforms.Compose([
                        transforms.Normalize(mean=[0.0], std=[1.0]) ])

        self.image_for_rcnn = image_for_rcnn


    def __len__(self):
        return len(self.iccd_3Dimages)

    def __getitem__(self, idx):

        # if idx is a tensor, convert it to a list (optional)
        if torch.is_tensor(idx):
            idx = idx.tolist()

        iccd_images = self.iccd_3Dimages[idx]
        iccd_images = norm_0to1(iccd_images) # we are normalizing here, not above!
        params = self.params[idx]
        score = self.score[idx]

        
        iccd_images = torch.tensor(iccd_images).float()

        if self.image_for_rcnn == True:
            iccd_images = iccd_images.squeeze(0)
            iccd_images = torch.stack([image.unsqueeze(0) for image in iccd_images])

        # Normalize the images in range [0, 1]
        iccd_images = self.normalize_transform(iccd_images)


        # Apply same transformation to all the images in the stack
        if self.transform is not None:
            iccd_images = self.transform(iccd_images)


        params = torch.tensor(params).float()

        # Add noise to the params
        if self.params_noise is not None:
            params = self.params_noise(params)

        score = torch.tensor(score).float()
        
        # Add a dimension to the score tensor to match [n, 1]
        score = score.unsqueeze(-1)


        return iccd_images, params, score




def extract_data(df, norm = False):
    
    """
    Extracts data from the dataframe and returns the ICCD images, growth parameters and Raman peak scores

    Parameters:
        df: pandas dataframe
        norm: normalizes the params and score in the range [0, 1] (default: False)
    
    Output:
        iccd_3Dimages: numpy array of shape (n, 50, 40, 40)
        params: numpy array of shape (n, 4) [F1, F2, P, T]
        score: numpy array of shape (n, 1)
    """



    #Convert dataframe to dictionary
    df_dict = df.to_dict()
    
    # ICCD_images

    iccd_images = df_dict["ICCD"]
    iccd_3Dimages = []

    for key, vals in iccd_images.items():
        iccd_3Dimages.append(np.array(vals))
        #print(key, len(vals))

    iccd_3Dimages = np.array(iccd_3Dimages)


    # ICCD_labels

    F1 = []
    F2 = []
    P = []
    T = []
    score = []  

    E1 = df_dict["E1"]          # Energy 1
    E2 = df_dict["E2"]          # Energy 2
    pressure = df_dict["P"]     # Pressure
    Temp = df_dict["T"]          # Temperature
    score_ = df_dict["score"]     # Raman peak Score

    for key, vals in E1.items():
        F1.append(np.array(vals))
        F2.append(np.array(E2[key]))
        P.append(np.array(pressure[key]))
        T.append(np.array(Temp[key]))
        score.append(np.array(score_[key]))

    F1 = np.array(F1)   # Fluence 1
    F2 = np.array(F2)   # Fluence 2
    P = np.array(P)    # Pressure
    T = np.array(T)   # Temperature
    score = np.array(score) # Raman peak Score

    if norm:    
        F1 = norm_0to1(F1)
        F2 = norm_0to1(F2)
        P = norm_0to1(P)
        T = norm_0to1(T)
        score = norm_0to1(score)

    params = np.dstack((F1, F2, P, T))[0]

    return iccd_3Dimages, params, score



def get_growth_params(datafile, norm = False):
    """
    Extracts growth parameters and returns them as a numpy array: [s0, s1, J]
    """
    df = load_df(datafile)

    df_dict = df.to_dict()
    growth_params = []
    s0 = df_dict["s0"]
    s1 = df_dict["s1"]
    J = df_dict["J"]

    for key, vals in s0.items():
        growth_params.append(np.array([vals, s1[key], J[key]]))

    growth_params = np.array(growth_params)

    if norm:
        for i in range(growth_params.shape[1]):
            growth_params[:, i] = norm_0to1(growth_params[:, i]) 
        growth_params = np.array(growth_params)

    growth_params = torch.tensor(growth_params).float()
        
    return growth_params



class AddGaussianNoise():
    """
    Adds Gaussian noise to the input tensor

    Input:
        noise_factor: float, optional, default: 0.1
        mean: float, optional, default: 0.0
        std: float, optional, default: 1.0

    Output:
        tensor: tensor with added noise
    """    
    
    def __init__(self, noise_factor=0.1, mean=0.0, std=1.0):

        self.noise_factor = noise_factor
        self.mean = mean
        self.std = std
        

    def __call__(self, tensor):

        return tensor + self.noise_factor * torch.normal(mean=self.mean, std=self.std, size=tensor.size())
    



def norm_0to1(arr):
    arr = np.asarray(arr)
    return (arr - arr.min())/(arr.max() - arr.min())



class TrainDataset(Dataset):

    """
    Creates the ICCD dataset and returns the ICCD images, growth parameters and Raman peak scores

    Input:

        iccd_3Dimages: numpy array of shape (n, 50, 40, 40)
        params: numpy array of shape (n, 4) [F1, F2, P, T]
        score: numpy array of shape (n, 1)
        transform: torchvision.transforms, optional, default: None
        params_noise: function, optional, default: None

    Output:
    
        iccd: torch.tensor, shape (n, 1, 50, 40, 40)
        params: normalized parameters, torch.tensor of shape (n, 4) [F1, F2, P, T]
        score: Raman score, torch.tensor, shape (n, 1)
    
    """

    
    def __init__(self, iccd_3Dimages, params, score, transform = None, params_noise = None):
        
        
        self.iccd_3Dimages = iccd_3Dimages
        self.params = params
        self.score = score

        # Transformations to apply to the images
        self.transform = transform

        # Add noise to the params
        self.params_noise = params_noise
        
        # Transform to normalize the images in range [0, 1]
        self.normalize_transform = transforms.Compose([
                        transforms.Normalize(mean=[0.0], std=[1.0]) ])

    


    def __len__(self):
        return len(self.iccd_3Dimages)

    def __getitem__(self, idx):

        # if idx is a tensor, convert it to a list (optional)
        if torch.is_tensor(idx):
            idx = idx.tolist()

        iccd_images = self.iccd_3Dimages[idx]
        params = self.params[idx]
        score = self.score[idx]

        
        # Normalize the images in range [0, 1]
        iccd_images = self.normalize_transform(iccd_images)


        # Apply same transformation to all the images in the stack
        if self.transform is not None:
            iccd_images = self.transform(iccd_images)


        return iccd_images, params, score