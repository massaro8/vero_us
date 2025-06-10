from sklearn.ensemble import RandomForestClassifier
from src.models.custom_base_estimators import VeroResNet, NeuralNetworkFitter, CNNFitterInput
from src.models.lutech_models import IRBoostSH
import torch
import numpy as np

NUM_CLASSES= 3
EPOCHS = 100
MB_SIZE_TRAIN = 10
N_ITERATION = 10

def training_boosting(X_mods,y_train):
    #Clinical model
    rf = RandomForestClassifier(10, min_samples_split=60)

    #Images
    net = VeroResNet(num_classes=NUM_CLASSES)

    cnn_fitter_input = CNNFitterInput(
        model=net,
        loss_function= torch.nn.CrossEntropyLoss,
        optimizer= torch.optim.Adam,
        learning_rate= 1e-3,
        epochs= EPOCHS,
        mb_size= MB_SIZE_TRAIN,
        #log = True,
        freeze_level = 3,
        #logdir= r"C:\Users\giuseppe.lamanna\Desktop\progetti\fair-lab-local\fair-lab\Veronet_algorithm\experiments"
        )

    nn_trainer = NeuralNetworkFitter(cnn_fitter_input)

    base_estimators={'clinical': rf, 'images': nn_trainer}

    ir_boost = IRBoostSH(base_estimators=base_estimators, n_iter=N_ITERATION, learning_rate=1.)
    ir_boost.fit(X_mods, y_train)

    return ir_boost
