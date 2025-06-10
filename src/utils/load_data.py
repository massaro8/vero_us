"""
Handles data genearation or loading and preparation for training
"""

from io import BytesIO
from typing import List

import numpy as np
import pandas as pd
import sklearn.preprocessing
import torch
from Crypto.SelfTest.Cipher.test_SIV import transform
from minio import Minio
import nibabel as nib
from pathlib import Path
import os
import pickle

from minio.datatypes import Object

pd.set_option('future.no_silent_downcasting', True)

class ImagePreprocessor:
    def __init__(self, client:Minio, bucket:str, img_input_size:tuple[int, int ,int, int], legend:dict, local_path:Path = os.path.join(Path(__file__).parent.parent,"data")):
        self.client = client
        self.bucket = bucket
        self.img_input_size = img_input_size
        self.legend = legend
        self.paths_df = None
        self.local_path = local_path

    def get_images_paths_df(self,  prefix:str) -> pd.DataFrame:
        objects = self.client.list_objects(self.bucket, prefix=prefix, recursive=True)
        temp_dict = self.process_images(objects)
        out = pd.DataFrame(temp_dict)
        self.paths_df = out
        return out

    def process_images(self, images_path_list:List[Object]) -> dict:
        out_dict = {"image_path": [], "Patient ID": [], "VISCODE": []}
        for obj in images_path_list:
            this_name = obj.object_name
            this_path = str(os.path.join(self.local_path, this_name.split("/")[-1].split(".")[0] + ".pkl"))
            if not os.path.exists(this_path):
                img_tensor = self.read_image_from_minio(this_name)
                img_tensor_rs = self.reshape_img(img_tensor, self.img_input_size)
                img_tensror_mm = self.min_max_scale_image(img_tensor_rs)
                with open(this_path, "wb") as lp:
                    # noinspection PyTypeChecker
                    pickle.dump(img_tensror_mm, lp)
            out_dict["image_path"].append(this_path)
            out_dict["Patient ID"].append(self.legend["Subject"][self.legend["Image Data ID"].index(this_name.split(".")[0].split("/")[-2])])
            out_dict["VISCODE"].append(self.legend["Visit"][self.legend["Image Data ID"].index(this_name.split(".")[0].split("/")[-2])])
        return out_dict

    def read_image_from_minio(self, this_name:str) -> torch.tensor:
        obj = self.client.get_object(self.bucket, this_name)
        data = BytesIO(obj.read())
        fh = nib.FileHolder(fileobj=data)
        img = nib.Nifti1Image.from_file_map({'header': fh, 'image': fh})
        out_tensor = torch.tensor(np.array(img.dataobj)).unsqueeze(0)
        return out_tensor

    @staticmethod
    def reshape_img(img: torch.tensor, desired_shape: [int, int, int, int]) -> torch.tensor:
        x_start = int(list(img.shape)[1] / 2) - int(desired_shape[1] / 2)
        x_end = x_start + desired_shape[1]
        y_start = int(list(img.shape)[2] / 2) - int(desired_shape[2] / 2)
        y_end = y_start + desired_shape[2]
        z_start = int(list(img.shape)[3] / 2) - int(desired_shape[3] / 2)
        z_end = z_start + desired_shape[3]
        rs = img[:, x_start:x_end, y_start:y_end, z_start:z_end]
        return rs

    @staticmethod
    def min_max_scale_image(image:torch.tensor) -> torch.tensor:
        eps = torch.max(torch.tensor([1e-8]), torch.max(image) - torch.min(image))
        pp = (image - torch.min(image)) / eps
        return pp


def dataframe_to_tensor(data: pd.DataFrame):
    x_list = []
    for index in data.index:
        x_list.append(data.images[index])

    return torch.stack(x_list, dim=0)

def merge_modalities(different_modalities_dfs: list, labels_df: pd.DataFrame):
    """
    Merges data from different modalities in the format required by veronet
    inputs:
        different_modalities_dfs: list of pandas DataFrames containing the data for each of the modalities, each one must have a 
        "Patient ID" and a "VISCODE" (optional) column to do the merging
        labels_df: a pandas DataFrame containing the labels and a column called "Patient ID" to do the merging
    returns:
        out_dataframe: a pandas DataFrame containing all the data
    """
    out_dataframe = labels_df
    for modality in different_modalities_dfs:
        if "VISCODE" in modality.columns:
            out_dataframe = pd.merge(out_dataframe, modality, left_on=['Patient ID','VISCODE'], right_on=['Patient ID','VISCODE'], how='left')
        else:
            out_dataframe = pd.merge(out_dataframe, modality, left_on=['Patient ID'], right_on=['Patient ID'], how='left')
    return out_dataframe.drop_duplicates()


def nifti_df_from_local(paths_df: pd.DataFrame):
    imgs_list = []
    for obj in paths_df["image_path"]:
        with open(obj, "rb") as fh:
            pp = pickle.load(fh)
            imgs_list.append(pp)
    out_df = pd.DataFrame({"images": imgs_list})
    return out_df


def retrieve_object_from_minio(client: Minio, bucket:str, url:str):
    response = client.get_object(bucket, url)
    out = BytesIO(response.read())
    return out


def treat_labels(raw_labels_file:BytesIO):
    complete_df = pd.read_csv(raw_labels_file, on_bad_lines='skip',sep=',')
    labels_df = complete_df[["PTID", "DX", "VISCODE"]]
    labels_df = labels_df.replace("Dementia", "AD")
    labels_df = labels_df.rename({"PTID": "Patient ID", "DX": "Label"}, axis=1)
    labels_df = labels_df.replace({"CN": 1, "MCI": 2, "AD": 3}).infer_objects(copy=False)
    labels_df = labels_df.dropna(axis=0)
    return labels_df


def treat_clinical_data(raw_labels_file: BytesIO, unwanted_cols : list[str], threshold=0.8):
    complete_df = pd.read_csv(raw_labels_file, on_bad_lines='skip',sep=',')
    clinical_df = complete_df.drop("DX", axis=1)
    ptid = clinical_df["PTID"]
    viscode = clinical_df["VISCODE"]
    clinical_df = clinical_df.drop("VISCODE", axis=1)
    clinical_df = clinical_df.drop("PTID", axis=1)
    #select only numerical types
    clinical_df = get_rid_of_unwanted_cols(clinical_df, unwanted_cols)
    clinical_df = treat_missing_clinical(clinical_df, threshold)

    df_numeric = clinical_df.select_dtypes(include = 'number')
    df_categorical = clinical_df.select_dtypes(exclude = 'number')
    ohe = sklearn.preprocessing.OneHotEncoder(sparse_output=False).set_output(transform="pandas")
    df_ohe = ohe.fit_transform(df_categorical)
    df_total = pd.concat([df_numeric, df_ohe], axis=1)
    df_total['Patient ID'] = ptid
    df_total["VISCODE"] = viscode
    return df_total

    
def get_rid_of_unwanted_cols(clinical_df, unwanted_cols):
    """
    Since many columns are not compliant in the adnimerge file, some of them must be discarded
    """
    for i in unwanted_cols:
        if i in clinical_df.columns:
            clinical_df = clinical_df.drop(i, axis=1)
    for i in clinical_df.columns:
        if "_bl" in i or "_BL" in i:
            clinical_df = clinical_df.drop(i, axis = 1)
        
    return clinical_df
    


def treat_missing_clinical(df, alpha):
    """
    Filter the colu,ns based on percentage of null values.
    :param df: input dataframe full of NaNs
    :param alpha: Threshold of percentage of missing values to exclude (eg. 0.9 for 90%)
    :return: filtered DataFrame
    """
    # Soglia assoluta di valori nulli consentiti per colonna
    threshold = alpha * len(df)
    # Filtra le colonne con valori nulli minori della soglia
    filtered_df = df.loc[:, df.isnull().sum() < threshold]
    return filtered_df
