import torch
import torch.nn as nn
import torch.nn.functional as F



class DKL_Custom_nn(nn.Module):
    """
    Combines a custom neural network with a Gaussian process layer.
    The neural network can be eiher a CNN or a MLP
    The output of the neural network is considered as the embeding into the GP layer

    Input: 
    - custom_nn: neural network model
    - gp: Gaussian process model (ex: SimpleGP layer)
 
    """

    def __init__(self, custom_nn, gp):
        super().__init__()

        self.custom_nn = custom_nn

        self.gp = gp

    def forward(self, x, y = None):

        # y are the additional params for mixed models
        if y is not None:
            features = self.custom_nn(x, y)
        else:
            features = self.custom_nn(x)

        return self.gp(features)
    
    def predict(self, x, y = None):
        
        self.gp.eval()
        self.gp.likelihood.eval()
        self.custom_nn.eval()

        with torch.no_grad():
            prediction = self.forward(x, y)        
            
        return prediction
    
    def posterior(self, x, y = None):
            
        self.gp.eval()
        self.gp.likelihood.eval()
        self.custom_nn.eval()

        with torch.no_grad():
            prediction = self.predict(x, y)

            mean = prediction.mean
            var = prediction.variance  
        
        mean.unsqueeze_(-1)
        var.unsqueeze_(-1)

        return mean, var
   
    def train(self):

        self.gp.train()
        self.gp.likelihood.train() 
        self.custom_nn.train()

    def eval(self):

        self.gp.eval()
        self.gp.likelihood.eval()
        self.custom_nn.eval()

    def to(self, device):
        self.gp.to(device)
        self.gp.likelihood.to(device)
        self.custom_nn.to(device)

        return self




class SimpleCNN(nn.Module):

    """
    Simple CNN model for image conolution

    Input: Images of size (N, 1, 40, 40)
    Output: 3 coordinates (N, 3)
    """

    def __init__(self, output_dim = 3):
        super().__init__()

        # input shape = (N, 1, 40, 40)
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size=3, stride = 1) # shape = (N, 6, 38, 38)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 12, kernel_size=3, stride =2) # shape = (N, 12, 18, 18)

        self.fc1 = nn.Linear(12*18*18, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)    

    def forward(self, x):
        x= F.relu(self.conv1(x))
        x= F.relu(self. conv2(x))

        N = x.size(0)

        #faltern the 2D image. view to yeild the total number of elements in the second dimension
        x = x.view(N, -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x



class CNN_FeatureExtractor(nn.Module):

    """
    CNN feature extractor
    """

    def __init__(self):
        super().__init__()

        # input shape = (N, 1, 40, 40)
        self.cnn =  nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size=3, stride = 1), # shape = (N, 16, 38, 38)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride = 2), # shape = (N, 16, 19, 19)
            
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=3, stride = 1), # shape = (N, 32, 17, 17)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride = 2), # shape = (N, 32, 8 , 8)
        )
        
        self.Flatten = nn.Flatten() # shape = (N, 32*8*8)

    def forward(self, x):

        x = self.cnn(x)
        
        x = self.Flatten(x) # this is same as x = x.view(x.size(0), -1) 

        return x
    

class RCNN_FeatureExtractor(nn.Module):

    """
    Recurrent CNN feature extractor

    """

    def __init__(self, cnn_feature_size = 32*8*8, output_dim = 3):
        super().__init__()

        # input shape = (N, 1, 40, 40)
        self.cnn = CNN_FeatureExtractor()
        self.rnn = nn.RNN(input_size = cnn_feature_size, hidden_size = 64, num_layers = 1, batch_first = True)
        self.fc = nn.Linear(64, output_dim)


    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        
        # Extract features for each image in the sequence
        cnn_features = []
        for t in range(seq_len):
            cnn_out = self.cnn(x[:, t, :, :, :])  # Process each frame independently
            cnn_features.append(cnn_out)
        
        # Stack CNN features into a sequence
        cnn_features = torch.stack(cnn_features, dim=1)  # Shape: (batch_size, seq_len, feature_dim)
        
        # Pass the sequence of features into the RNN
        rnn_out, final_hidden_state = self.rnn(cnn_features)  # Shape: (batch_size, seq_len, hidden_dim)
        
        # Use the output from the last time step
        output = self.fc(rnn_out[:, -1, :])  # Shape: (batch_size, 1)
        return output
        
    

class Mixed_RCNN_FeatureExtractor(nn.Module):

    """
    Recurrent CNN feature extractor

    """

    def __init__(self, cnn_feature_size = 32*8*8, output_dim = 3):
        super().__init__()

        # input shape = (N, 1, 40, 40)
        self.cnn = CNN_FeatureExtractor()
        self.rnn = nn.RNN(input_size = cnn_feature_size, hidden_size = 64, num_layers = 1, batch_first = True)
        self.fc = nn.Linear(64, 64)

        self.param_fc1 = nn.Linear(4, 32)
        self.param_fc2 = nn.Linear(32, 64)

        self.join_fc1 = nn.Linear(128, 64)
        self.join_fc2 = nn.Linear(64, 32)
        self.join_fc3 = nn.Linear(32, output_dim)

    def forward(self, x, params):
        batch_size, seq_len, c, h, w = x.size()
        
        # Extract features for each image in the sequence
        cnn_features = []
        for t in range(seq_len):
            cnn_out = self.cnn(x[:, t, :, :, :])  # Process each frame independently
            cnn_features.append(cnn_out)
        
        # Stack CNN features into a sequence
        cnn_features = torch.stack(cnn_features, dim=1)  # Shape: (batch_size, seq_len, feature_dim)
        
        # Pass the sequence of features into the RNN
        rnn_out, final_hidden_state = self.rnn(cnn_features)  # Shape: (batch_size, seq_len, hidden_dim)
        
        # Use the output from the last time step
        rcnn_out = self.fc(rnn_out[:, -1, :])  # Shape: (batch_size, output_dim)
        rcnn_out = F.relu(rcnn_out)
        rcnn_out = rcnn_out.view(batch_size, -1)

        # Process the parameters
        y = F.relu(self.param_fc1(params))
        y = F.dropout(y, p=0.1, training=self.training)
        y = F.relu(self.param_fc2(y))
        y = y.view(batch_size, -1)


        # Concatenate the RCNN and parameter features
        joint = torch.cat([rcnn_out, y], dim=1)        
        joint = F.leaky_relu(self.join_fc1(joint), 0.2, inplace=True)
        joint = F.leaky_relu(self.join_fc2(joint), 0.1, inplace=True)
        joint_output = self.join_fc3(joint)


        return joint_output
        


class ICCDNet(nn.Module):
    def __init__(self,l1=64,l2=32, output_dim = 3):
        super(ICCDNet, self).__init__()
        # ICCD imaging feature inputs, the full image size is BATCH,C,frames,H,W where it is N,50,40,40
        self.ICCD_features_ = nn.Sequential(
            #Spatial convolution
            nn.Conv3d(1,64, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),
            nn.BatchNorm3d(64),
            #Temportal convolution
            nn.Conv3d(64,64,kernel_size=(3,1,1), stride=(1,1,1), padding=(1,0,0)),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(inplace=True),

            #Downsample
            nn.AvgPool3d(kernel_size=(2,2,2),stride=(2,2,2)),

            #Spatial convolution
            nn.Conv3d(64,128, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),
            nn.BatchNorm3d(128),
            #Temportal convolution
            nn.Conv3d(128,128,kernel_size=(3,1,1), stride=(1,1,1), padding=(1,0,0)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(inplace=True),

            #Downsample
            nn.AvgPool3d(kernel_size=(2,2,2),stride=(2,2,2)),

            #Spatial convolution
            nn.Conv3d(128,256, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            #Temportal convolution
            nn.Conv3d(256,256,kernel_size=(3,1,1), stride=(1,1,1), padding=(1,0,0)),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(inplace=True),

            #Downsample
            nn.AvgPool3d(kernel_size=(2,2,2),stride=(2,2,2)),

            nn.Flatten(start_dim=1),
            nn.Linear(256*6*5*5,l1),
            nn.LeakyReLU(inplace=True),
            nn.Linear(l1,l2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(l2, output_dim)
        )

    def forward(self,x1):
        #ICCD features
        x1=self.ICCD_features_(x1)
        return x1
 
    


class MixedICCDNet(nn.Module):
    def __init__(self, output_dim = 3, l1=64,l2=32,param_l1=48,param_out=32,c1=16,c2=24,c3=32):

        super(MixedICCDNet, self).__init__()
        # ICCD imaging feature inputs, the full image size is BATCH,C,frames,H,W where it is N,50,40,40
        
        self.ICCD_features_ = nn.Sequential(
            #Spatial convolution
            nn.Conv3d(1,64, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),
            nn.BatchNorm3d(64),
            #Temportal convolution
            nn.Conv3d(64,64,kernel_size=(3,1,1), stride=(1,1,1), padding=(1,0,0)),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(inplace=True),

            #Downsample
            nn.AvgPool3d(kernel_size=(2,2,2),stride=(2,2,2)),

            #Spatial convolution
            nn.Conv3d(64,128, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),
            nn.BatchNorm3d(128),
            #Temportal convolution
            nn.Conv3d(128,128,kernel_size=(3,1,1), stride=(1,1,1), padding=(1,0,0)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(inplace=True),

            #Downsample
            nn.AvgPool3d(kernel_size=(2,2,2),stride=(2,2,2)),

            #Spatial convolution
            nn.Conv3d(128,256, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            #Temportal convolution
            nn.Conv3d(256,256,kernel_size=(3,1,1), stride=(1,1,1), padding=(1,0,0)),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(inplace=True),

            #Downsample
            nn.AvgPool3d(kernel_size=(2,2,2),stride=(2,2,2)),

            nn.Flatten(start_dim=1),
            nn.Linear(256*6*5*5,l1),
            nn.LeakyReLU(inplace=True),
            nn.Linear(l1,l2),
            nn.LeakyReLU(inplace=True),
            #nn.Linear(l2,3)
        )
        
        self.parameter_features = nn.Sequential(
            nn.Linear(4,param_l1),
            nn.LeakyReLU(inplace=True),
            nn.Linear(param_l1,param_out),
            nn.LeakyReLU(inplace=True),
            #nn.Linear(param_out,3)
        )
        
        self.combined_features_ = nn.Sequential(
            nn.Linear(l2+param_out,c1),
            nn.LeakyReLU(inplace=True),
            nn.Linear(c1,c2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(c2,c3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(c3, output_dim),
        )

    def forward(self,x,y):
        
        x=self.ICCD_features_(x)
        y=self.parameter_features(y)
        
        x=x.view(x.shape[0],-1)
        
        y=y.view(y.shape[0],-1)        
        
        # Concatenate the features in second dimension (dim=1). First dim is the batch size
        z = torch.cat((x,y),1)

        z = self.combined_features_(z)    

        return z
    





class MLP(nn.Module):

    """
    Simple MLP model for feature extraction

    Inputs:
    - input_dim: int, input dimension (default = 128)
    - l1: int, number of units in the first layer(default = 64)
    - l2: int, number of units in the second layer(default = 32)
    - output_dim: int, output dimension (default = 3)
    
    """

    def __init__(self, input_dim = 128, l1=64, l2=32, output_dim = 3):

        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim,l1),

            # Leaky ReLU activation, inplace modifies the input directly, saving memory
            nn.LeakyReLU(inplace=True),
            
            nn.Linear(l1,l2),
            
            nn.LeakyReLU(inplace=True),
            
            nn.Linear(l2,output_dim)
        )

    def forward(self, x):

        x = self.feature_extractor(x)
        
        return x
    




