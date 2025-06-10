import numpy as np
import pandas as pd
import os
import copy
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_predict
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from src.predict.explainability import visualize_gradcam_pairs
import sys

class BoostSH(BaseEstimator, ClassifierMixin):

    def __init__(self, base_estimators:dict, n_iter=10, learning_rate=1.):
        """
            Boost SH : Boosting classification algorithm for multimodal with shared weights
            Greedy approach in which each modality is tested to evaluate the one with larger
            edge

            Arguments:
                base_estimators: dict, {modality:model}
                n_iter {int} -- Number of boosting iterations
                learning_rate {float} --  Learning rate for boosting (default: 1)
        """
        #super(BoostSH, self).__init__()
        super().__init__()
        self.base_estimators = base_estimators 
        self.modalities = {}

        self.models = []
        self.classes = []
        self.alphas = []
        self.modalities_selected = []
        self.weights = []
        self.eps = 10 ** (-6)

        self.n_iter = n_iter
        self.learning_rate = learning_rate

        # # Flag to track if the CNN has been fine-tuned
        # self.cnn_finetuned = False

    def fit(self, X, y, forecast_cv=None, sample_weights=None):
        """
            Fit the model by adding models in an adaboost fashion

            Arguments:
                X {Dict of pd Dataframe} -- Modalities to use for the task
                y {pd Dataframe} -- Labels - Index has to be contained in modality union
                forecast_cv {int} -- Number of fold used to estimate the edge
                    (default: None - Performance are computed on training set)
        """
        self.check_input(X, y)

        self.modalities = copy.deepcopy(X)
        self.classes = np.unique(y)

        index = self.__index_union__(self.modalities)
        y = y.reindex(index)

        # Initialize distribution over weights
        self.initialize_weights(sample_weights, index)
        self.weights = pd.Series(self.weights)

        for t in range(self.n_iter):
            if self.weights.sum() == 0:
                break

            self.weights /= self.weights.sum()

            selected = {'max_edge': -np.inf}
            for m in self.modalities:
                mask = self.modalities[m].index.tolist()
                weak_forecast = self.__compute_weak_forecast__(m, self.modalities[m], y[mask], self.weights[mask], forecast_cv, True)

                tmp = 2 * ((weak_forecast['forecast'] == y[mask].values) - .5)
                edge = (self.weights[mask].values * tmp).sum()

                if edge > selected['max_edge']:
                    selected = {'max_edge': edge, 'modality': m, 'forecast': weak_forecast['forecast'], 'mask': mask,
                                'model': weak_forecast['model'], 'tmp': tmp}

            if (1 - selected['max_edge']) < self.eps:
                alpha = self.learning_rate * .5 * 10.
            else:
                alpha = self.learning_rate * .5 * np.log((1 + selected['max_edge']) / (1 - selected['max_edge']))

            # Update weights
            self.weights[selected['mask']] *= np.exp(- alpha * selected['tmp'])

            self.models.append(selected['model'])
            self.alphas.append(alpha)
            self.modalities_selected.append(selected['modality'])

            print('Boost.SH iteration ', t)
            print('Winning modality ', selected['modality'])
            print('Edge ', selected['max_edge'])
            print('')

        return
    
    def initialize_weights(self, sample_weights=None, index=None):
        if sample_weights is None:
            # Initialize uniform distribution over weights
            self.weights = pd.Series(1, index=index)

        else:
            # Assign pre-defined distribution over weights
            self.weights = pd.Series(sample_weights, index=index)
    
    
    def __compute_weak_forecast__(self, m:str, data, labels, weights, forecast_cv=None, return_model=True):
        weak_forecast = dict()
        model = universal_clone(self.base_estimators[m])
        print(f'Model architecture: {model.__class__.__name__}.')
        # data, labels, weights = shuffle(data, labels, weights)

        if forecast_cv is None:
            print(f'START Training {m} modality...')
            model.fit(data, labels, sample_weight=weights) #fit a definito modalità per modalità
            print(f'END Training {m} modality...')
            forecast = model.predict(data)

            # print accuracy for random forest (da cancellare in un secondo momento forse)
            if model.__class__.__name__ == 'RandomForestClassifier':
                accuracy = accuracy_score(labels, forecast,sample_weight=weights)
                print(f'RandomForest Training accuracy: {accuracy:.2f}')

        else:
            forecast = cross_val_predict(model, data.values, labels.values, cv=forecast_cv,
                                         fit_params={'sample_weight': weights.values})

        weak_forecast['forecast'] = forecast
        if return_model:
            weak_forecast['model'] = model
        print(f"Forecast: {forecast}")
        return weak_forecast
    
    def predict_proba(self, X,is_Exp=False):

        index = self.__index_union__(X)
        predictions = pd.DataFrame(0., index=index, columns=self.classes)
        for t in range(len(self.models)):    
            if self.modalities_selected[t] in X.keys():
                X_test = X[self.modalities_selected[t]]
                # print(f"model{t} dataset index:{X_test.index}")
                #int_index = [list(index).index(i) for i in X_test.index]
                test_index = X_test.index
                if self.models[t].__class__.__name__ == 'RandomForestClassifier':
                    probas = self.models[t].predict_proba(X_test)
                else:
                    if is_Exp:
                        probas,list_cam = self.models[t].predict_proba(X_test,is_Exp=is_Exp)
                        visualize_gradcam_pairs(list_cam)
                    else:
                        probas = self.models[t].predict_proba(X_test,is_Exp=is_Exp)
                probas = probas.detach().cpu().numpy() if not isinstance(probas, np.ndarray) else probas
                
                for i, idx in enumerate(test_index):
                    self.alphas[t] = float(self.alphas[t])
                    predictions.loc[idx] += self.alphas[t] * probas[i]
                    
        
        # Normalizzare le previsioni dividendo per la somma su ogni riga
        predictions_normalized = pd.DataFrame(0, columns = predictions.columns, index = predictions.index, dtype="float")
        for i in range(0, len(predictions)):
            if predictions.iloc[i,:].sum() == 0:
                continue
            else:
                predictions_normalized.iloc[i,:] =predictions.iloc[i,:]/predictions.iloc[i,:].sum() 
        return predictions_normalized
        #return predictions.div(predictions.sum(axis=1), axis=0)


    def predict(self, X,is_Exp=False):
        self.check_X(X)
        assert len(self.models) > 0, 'Model not trained'
        pp = self.predict_proba(X,is_Exp).idxmax(axis=1)
        return pp

    def check_input(self, X, y):
        self.check_X(X)
        self.check_y(y)

    def check_X(self, X):
        assert isinstance(X, dict), "Not right format for X"
        for key in X.keys():
            assert isinstance(X[key], pd.DataFrame)
            assert not X[key].empty, "Empty dataframe"

    def check_y(self, y):
        assert isinstance(y, pd.Series), "Not right format for y"
        # assert len(y.unique()) > 1, "One class in data"
        assert not y.empty, "Empty dataframe"

    def __index_union__(self, modalities):
        self.check_X(modalities)
        index = set([])
        for mod in modalities:
            index = index.union(set(modalities[mod].index))

        return list(index)

 


class IRBoostSH(BoostSH):

    def __init__(self, base_estimators:dict, n_iter=10, learning_rate=1., sigma=0.15, gamma=0.3):
        """
            rBoost SH : Boosting classification for multimodal with shared weights.
            Multi-arm bandit approach in which a modality is selected at each iteration

            Arguments:
                base_estimator {sklearn model} -- Base classifier to use on each modality
                n_iter {int} -- Number of boosting iterations
                learning_rate {float} -- Learning rate for boosting (default: 1)
        """
        super().__init__(base_estimators, n_iter=n_iter, learning_rate=learning_rate)
        self.sigma = sigma
        self.gamma = gamma

    def fit(self, X, y, forecast_cv=None, sample_weights=None):
        """
            Fit the model by adding models in a boosting fashion

            Arguments:
                X {Dict of pandas Dataframes/ torch tensors} -- Modalities to use for the task
                y {pandas Series} -- Labels - Index has to be contained in modality union
                forecast_cv {int} -- Number of fold used to estimate the edge
                    (default: None - Performance are computed on training set)
        """
        self.check_input(X, y)
        self.modalities = copy.deepcopy(X)
        self.classes = np.unique(y)
        K = len(self.modalities) 
        possible_modalities = list(self.modalities.keys())

        index = self.__index_union__(self.modalities)
        # print(f"index:{index}")
        # Reorder labels
        y = y.reindex(index)

        # Initialize distribution over weights
        self.initialize_weights(sample_weights, index)
        self.weights = pd.Series(self.weights)

        p_mods = pd.Series(np.exp(self.sigma * self.gamma / 3 * np.sqrt(self.n_iter / K)), index=possible_modalities)
       
        for t in range(self.n_iter):
            
            print('')
            print(f'irBoost.SH training: Iteration {t+1}/{self.n_iter}.')
            print(f'Sample weights: {self.weights}')

            if self.weights.sum() == 0:
                break

            self.weights /= self.weights.sum()

            # Bandit selection of best modality
            q_mods = (1 - self.gamma) * p_mods / p_mods.sum() + self.gamma / K
            print(f"q_mods: {q_mods.to_string()}") ##

            selected_mod = np.random.choice(possible_modalities, p=q_mods)
            #selected_mod = 'images'
            print(f'Winning modality: {selected_mod}.')
        
            mask = self.modalities[selected_mod].index.tolist()
            weak_forecast = self.__compute_weak_forecast__(selected_mod, self.modalities[selected_mod], y[mask], self.weights[mask])

            # Calculate edge 
            
            if not isinstance(weak_forecast['forecast'], (np.ndarray)):
                forecast_np = weak_forecast['forecast'].detach().cpu().numpy()
            else:
                forecast_np = weak_forecast['forecast']
            target_np = y[mask].values

            # Use this with python version > 3.11
            if (sys.version_info.major, sys.version_info.minor) >= (3, 12):
                tmp = 2 * (forecast_np == target_np).astype(int) - 0.5
            else:
                tmp = 2 * (forecast_np == target_np) - 0.5
            
            ## Use this for python version <= 3.11
            # tmp = 2 * ((weak_forecast['forecast'] == y[mask].values) - .5)

            edge = (self.weights[mask].values * tmp).sum()

            if (1 - edge) < self.eps:
                alpha = self.learning_rate * .5 * 10.
            elif edge <= 0:
                alpha = 0
            else:
                alpha = self.learning_rate * .5 * np.log((1 + edge) / (1 - edge))

            # Update weights
            self.weights[mask] *= np.exp(- alpha * tmp)

            # Update arm probability
            r_mods = pd.Series(0., index=possible_modalities)
            square = np.sqrt(1 - edge ** 2) if edge < 1 else 0
            r_mods[selected_mod] = (1 - square) / q_mods[selected_mod]
            p_mods *= np.exp(self.gamma / (3 * K) * (r_mods + self.sigma / (q_mods * np.sqrt(self.n_iter * K))))

            self.models.append(weak_forecast['model'])
            self.alphas.append(alpha)
            self.modalities_selected.append(selected_mod)

            # print('')
            # print('Iteration ', t)
            # print('Winning modality ', selected_mod)
            print(f'Edge {edge}')

        return


    def modality_weights(self):
        """
            Return relative importance of the different modality in the final decision
        """
        assert len(self.models) > 0, 'Model not trained'
        mod_weights = pd.DataFrame({"modality": self.modalities_selected, "alpha": np.abs(self.alphas)})
        return (mod_weights.groupby('modality').sum() / np.sum(np.abs(self.alphas))).sort_values('alpha')


    def save_modalities(self, path):
        df = pd.DataFrame(self.modalities_selected)
        if path.endswith('xlsx'):
            file = path
        else:
            file = os.path.join(path, 'winning modalities.xlsx')
        df.to_excel(file)

        
    # TODO: If needed, add other classification metrics (recall, f1 score, ...) using the same logic as follows.
    def accuracy_score(self, X: dict, y: pd.Series) -> float:
        """
            Calculates the accuracy score of the irBoost.SH model comparing predictions against the true labels.
            This method predicts labels based on the input data `X`, reorders the predictions to match the index of the true labels `y`, and removes any NaN values. It then compares the predictions with the true labels to determine the accuracy score.

            Parameters:
            X (dict): A dictionary of pandas DataFrames, where each key represents a modality/view and the value is the corresponding DataFrame.
            y (pd.Series): A pandas Series containing the true labels.

            Returns:
            float: The accuracy score of the irBoost.SH model predictions.
        """
        # Predict labels based on the input X
        predictions = self.predict(X) 
        # Reorder predictions to match the index of the true labels and remove any NaN values
        predictions_ordered = predictions.reindex(y.index).dropna()
        # Compare the predictions with the true labels to get a boolean series indicating correct predictions
        results_comparison = (predictions_ordered == y[predictions_ordered.index])
        # Calculate the accuracy by summing the correct predictions and dividing by the total number of predictions
        ir_boost_accuracy = sum(results_comparison) / len(results_comparison)
        return ir_boost_accuracy
    

def universal_clone(obj):
    if isinstance(obj, BaseEstimator):
        # Clone per oggetti scikit-learn
        return clone(obj)
    elif isinstance(obj, torch.nn.Module):
        # Deepcopy per oggetti torch
        return copy.deepcopy(obj)
    else:
        # Prova un deepcopy generico
        return copy.deepcopy(obj)