import os.path
import pickle
from dataclasses import dataclass
from time import time
from typing import Type
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import monai
import mlflow

from src.utils.load_data import nifti_df_from_local, dataframe_to_tensor
from src.predict.explainability import *


class VeroResNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        self.res = monai.networks.nets.resnet18(spatial_dims = 3, n_input_channels=1, num_classes = 3, pretrained=True, feed_forward = False, bias_downsample = True, shortcut_type="A")
        self.fc1 = nn.Linear(in_features=512, out_features=512)
        self.dropout = nn.Dropout(p=0.15)

        self.fc2 = nn.Linear(in_features=512, out_features=256)

        self.out_layer = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, x: torch.Tensor):
        x = self.res(x)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)
        x =self.out_layer(x)
        return x


    def freeze_layers(self, freeze_level):
        """
        Freezes the layers of the network based on the freeze_level specified in the init phase

        Prints
        ------
        Reminder of the freeze level
        """

        if freeze_level == 2:
            print("Freeze level 2: only dense layers and output layer are trained")
            for param in self.res.parameters():
                param.requires_grad = False
        else:
            print("Unfreezed")

@dataclass
class CNNFitterInput:
    model: nn.Module
    loss_function: nn.Module
    optimizer: Type[optim.Optimizer]
    learning_rate: float = 1e-3
    epochs: int = 100
    mb_size:int = 10
    log:bool = True
    freeze_level:int = 2
    logdir:str = "logs"


class NeuralNetworkFitter:
    def __init__(self, inp_list:CNNFitterInput):
        self.inp_list = inp_list
        self.criterion = inp_list.loss_function(reduction="none")

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = inp_list.model.to(self.device)
        self.optimizer = inp_list.optimizer(params=self.model.parameters(), lr=self.inp_list.learning_rate)


    def fit(self, data:pd.DataFrame, labels: pd.DataFrame, sample_weight:pd.Series): #Datframe, with weighted loss

        self.model.freeze_layers(freeze_level=self.inp_list.freeze_level)
#        with mlflow():
        labels = torch.tensor(labels.values, dtype=torch.int64) - 1 #perchè cross entropy loss vuole da 0
        labels = labels.to(self.device)
        weights_tensor = torch.tensor(sample_weight.values, dtype=torch.float32).unsqueeze(1)
        weights_tensor = weights_tensor.to(self.device) / torch.mean(weights_tensor)
        weights_tensor = weights_tensor.squeeze(-1)
        losses = []
        for epoch in range(self.inp_list.epochs):
            for mb in range(0, len(data), self.inp_list.mb_size):
                self.optimizer.zero_grad()
                x = data[mb: mb+self.inp_list.mb_size]
                y_mb = labels[mb: mb+self.inp_list.mb_size]
                weights_mb = weights_tensor[mb:mb+self.inp_list.mb_size]
                x_df = nifti_df_from_local(paths_df=x)
                x_mb = dataframe_to_tensor(x_df)
                x_mb = x_mb.to(self.device)
                outputs = self.model.forward(x_mb)
                if len(outputs.shape)<2:
                    outputs = outputs.unsqueeze(0)
                loss = self.criterion(outputs, y_mb)
                loss = (weights_mb * loss).mean()
                losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
            print(f"Epoca {epoch + 1}/{self.inp_list.epochs}, Loss: {loss.item()}")
            mlflow.log_metric(f"NeuralNetworkFitter_epoch_loss",loss.item(),step=epoch)
            if np.mean(losses[-len(data):]) <= 1e-4:
                break


    def predict(self, data: pd.DataFrame, mb_size = 2):
        out = self.predict_proba(data, mb_size)
        print(out)
        predicted_classes = torch.argmax(out, dim=-1) + 1

        return predicted_classes


    def predict_proba(self, data: pd.DataFrame, mb_size=2,is_Exp=False):
        self.model.eval()

        list_cam = []

        with torch.no_grad():
            predicts = torch.zeros([1,self.model.num_classes]).to(self.device)
            for mb in range(0, len(data), mb_size):
                x = data[mb: mb+mb_size]
                x_df = nifti_df_from_local(paths_df=x)
                x_mb = dataframe_to_tensor(x_df)
                x_mb = x_mb.to(self.device)
                if is_Exp:
                    with torch.enable_grad():
                        x_mb.requires_grad_(True)
                        # Se il modello è una VeroResNet con attributo "res", usa il layer specifico, altrimenti cerca l'ultimo layer conv
                        if hasattr(self.model, "res"):
                            target_layer = self.model.res.layer4[-1]
                        else:
                            target_layer = self.find_last_conv(self.model)
                    
                        vis =get_grad_cam_3d(
                            model=self.model,
                            input_tensor=x_mb,
                            device=self.device,
                            target_layer=target_layer,
                            target_category=None,
                            return_raw=True
                        )
                        #print(f"Shape della Grad-CAM per minibatch {mb}: {vis.shape}")
                        originals = x_mb.cpu().detach().numpy() 
                        # Fai una copia esplicita per evitare riferimenti allo stesso oggetto
                        list_cam.append((np.copy(originals),np.copy(vis)))
            
                        # Crea la figura e il relativo asse
                        #fig, ax = plt.subplots(figsize=(8, 8))
                        #ax.imshow(vis)
                        #ax.axis('off')  # Nasconde gli assi
                        #ax.set_title(f"Grad-CAM Visualization{mb}")

                        # Registra la figura come artefatto in MLflow
                        #mlflow.log_figure(fig, f"grad_cam_images/n_model{n_model}/gradcam_visualization{mb}.png")
                        #plt.close(fig)
                outputs=self.model.forward(x_mb)
                predicts = torch.cat([predicts,torch.softmax(outputs, dim=-1)], dim=0)
         # Per debug: mostra la lunghezza totale della lista
        print(f"Numero totale di grad-cam raccolte: {len(list_cam)}")
        if is_Exp:
            return predicts[1:],list_cam
        return predicts[1:]

@dataclass()
class ImageRFFitterInput:
    model: RandomForestClassifier
    num_classes: int = 3
    temp_file:str = "image_embeddings.pkl"


class ImageRFFitter:
    def __init__(self, inp_list:ImageRFFitterInput):
        self.inp_list = inp_list
        self.rf = inp_list.model
        self.res = monai.networks.nets.resnet18(spatial_dims = 3, n_input_channels=1, num_classes = 3, pretrained=True, feed_forward = False, bias_downsample = True, shortcut_type="A")
        self.res.eval()
        self.num_classes = inp_list.num_classes
        self.temp_file = inp_list.temp_file

    def embed(self, data):
        if os.path.exists(self.temp_file):
            ret_obj = pickle.load(open(self.temp_file, "rb"))
            return ret_obj
        else:
            with torch.no_grad():
                predicts = torch.zeros([1, 512])
                for mb in range(0, len(data)):
                    x = data[mb: mb + 1]
                    x_df = nifti_df_from_local(paths_df=x)
                    x_mb = dataframe_to_tensor(x_df)
                    outputs = self.res.forward(x_mb)
                    predicts = torch.cat([predicts, outputs], dim=0)
            out = predicts[1:].numpy()
            pickle.dump(out, open(self.temp_file, "wb"))
            return out

    def fit(self, data:pd.DataFrame, labels: pd.Series, sample_weight:pd.Series): #Datframe, with weighted loss
        #Trasformare tutte quante le immagini in features

        data = self.embed(data)

        # Addestrare un rf su quelle features
        self.rf.fit(data, labels, sample_weight=sample_weight)


    def predict(self, data: pd.DataFrame):
        out = self.predict_proba(data)
        out = torch.tensor(out)
        predicted_classes = torch.argmax(out, dim=-1) + 1

        return predicted_classes


    def predict_proba(self, data: pd.DataFrame):
        data = self.embed(data)

        return self.rf.predict_proba(data)

# Define the autoencoder architecture
# class Autoencoder(nn.Module):
#     def __init__(self, input_dim, hidden_dim1, hidden_dim2, bottleneck_dim, sigmoid: bool = False, noise_factor = 0):
#         super(Autoencoder, self).__init__()
#
#         self.noise_factor = noise_factor # noise factor for a Denoising AE
#
#         # Encoder part
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim1),
#             nn.ReLU(),
#             nn.Linear(hidden_dim1, hidden_dim2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim2, bottleneck_dim),  # Bottleneck layer
#             nn.ReLU()
#         )
#
#         # Decoder part
#         layers = [
#             nn.Linear(bottleneck_dim, hidden_dim2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim2, hidden_dim1),
#             nn.ReLU(),
#             nn.Linear(hidden_dim1, input_dim),
#         ]
#
#         # Add either Sigmoid or ReLU based on the flag
#         if sigmoid:
#             layers.append(nn.Sigmoid())
#
#         # Define the decoder as a Sequential model
#         self.decoder = nn.Sequential(*layers)
#
#
#     def forward(self, x):
#         if self.noise_factor != 0: # Introduce noise
#             x = self._add_noise(x)
#         # Encode input to latent space
#         encoded = self.encoder(x)
#         # Decode latent space back to input space
#         decoded = self.decoder(encoded)
#         return decoded
#
#
#     def _add_noise(self, x):
#         noisy_data = x.clone()
#         for i in range(x.shape[1]):
#             range_col = x[:, i].max() - x[:, i].min()
#             noise = torch.normal(0, self.noise_factor * range_col, size=x[:, i].size())
#             noisy_data[:, i] += noise
#         return noisy_data
#
#
#     def fit(self,data: pd.DataFrame,epochs=100):
#         self.train()
#
#         criterion = nn.MSELoss()
#         optimizer = optim.Adam(self.parameters(),lr=1e-2)
#
#         x = self._dataframe_to_tensor(data)
#
#         for epoch in range(epochs):
#             optimizer.zero_grad()
#
#             outputs = self.forward(x)
#             loss = criterion(outputs,x)
#
#             loss.backward()
#             optimizer.step()
#
#             print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
#
#
#     def _dataframe_to_tensor(self, data: pd.DataFrame):
#         # Assuming you want to convert all columns to a tensor
#         # Convert the dataframe to a numpy array and then to a tensor
#         data_array = data.to_numpy()
#         return torch.tensor(data_array, dtype=torch.float32)
#
#
# class EncoderClassifier(nn.Module):
#     def __init__(self,autoencoder,fc_hidden_dims,num_classes, encoder_freeze = True):
#         super(EncoderClassifier,self).__init__()
#
#         # Extract encoder from instance of autoencoder
#         self.encoder = autoencoder.encoder
#         bottleneck_dim = self.encoder[-2].out_features # access bottleneck from autoencoder
#
#         for param in self.encoder.parameters():
#             param.requires_grad = not encoder_freeze
#
#         # dense layers
#         layers = []
#         previous_dim = bottleneck_dim
#
#         for hidden_dim in fc_hidden_dims:
#             layers.append(nn.Linear(previous_dim,hidden_dim))
#             layers.append(nn.ReLU())
#             previous_dim=hidden_dim
#
#         layers.append(nn.Linear(previous_dim,num_classes))
#         self.classifier = nn.Sequential(*layers)
#
#
#     def forward(self,x):
#         return self.classifier(self.encoder(x))
#
#
#     def fit(self,data: pd.DataFrame,labels, sample_weight = pd.Series(1.), epochs=10, mini_batch_size = 25):
#         self.train()
#
#         criterion = nn.CrossEntropyLoss(reduction='none')
#         optimizer = optim.Adam(self.parameters(),lr=1e-4)
#
#         x = self._dataframe_to_tensor(data)
#         labels = torch.tensor(labels.values,dtype=torch.long)-1
#         weights_tensor = torch.tensor(sample_weight.values,dtype=torch.float32).unsqueeze(1)
#
#         for epoch in range(epochs):
#             #Modifica di Peppino
#             cumulated_outputs = torch.tensor([0])
#
#             for mb in range(0, len(x), mini_batch_size):
#                 #self.train()
#                 optimizer.zero_grad()
#                 x_mb = x[mb: mb+mini_batch_size]
#                 y_mb = labels[mb: mb+mini_batch_size]
#                 weights_mb = weights_tensor[mb:mb+mini_batch_size]
#                 outputs = self.forward(x_mb)
#                 if len(outputs.shape)<2:
#                     outputs = outputs.unsqueeze(0)
#
#                 loss = criterion(outputs, y_mb)
#                 loss = (loss * weights_mb).mean()
#                 loss.backward()
#                 optimizer.step()
#
#                 #Modifica di Peppino
#                 cumulated_outputs = torch.cat([cumulated_outputs, torch.argmax(outputs, -1)], 0)
#
#             cumulated_outputs = cumulated_outputs[1:]
#
#             # Calculate accuracy
# #            _, predicted = torch.max(outputs, 1)
#             total = labels.size(0)
#             correct = (labels == cumulated_outputs).sum().item()
#             accuracy = correct / total
#             print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')
#
#
#     def predict(self,data: pd.DataFrame):
#         self.eval()
#         with torch.no_grad():
#             x = self._dataframe_to_tensor(data)
#             outputs = self.forward(x)
#             predicted_classes = torch.argmax(outputs,dim=1)+1 #TODO check the labelling problem of having positive natural labels while training with cross entropy which wants also 0 labels
#             return predicted_classes
#
#
#     def predict_proba(self, data: pd.DataFrame):
#         self.eval()
#         with torch.no_grad():
#             x = self._dataframe_to_tensor(data)
#             outputs = self.forward(x)
#             return torch.softmax(outputs, dim=1)
#
#
#     def accuracy_score(self,data: pd.DataFrame,labels):
#         # Calculates accuracy on data, which is a pandas DataFrame
#         predicted_classes = self.predict(data)
#         labels = torch.tensor(labels.values,dtype=torch.long)
#         correct = (predicted_classes == labels).sum().item()
#         accuracy = correct / labels.size(0)
#         return accuracy
#
#
#     # TODO this method is repeated over all models, it seems. Is it the best practice?
#     def _dataframe_to_tensor(self, data: pd.DataFrame):
#         # Assuming you want to convert all columns to a tensor
#         # Convert the dataframe to a numpy array and then to a tensor
#         data_array = data.to_numpy()
#         return torch.tensor(data_array, dtype=torch.float32)
