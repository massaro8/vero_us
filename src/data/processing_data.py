from src.utils.load_data import ImagePreprocessor, retrieve_object_from_minio, treat_labels, treat_clinical_data, merge_modalities
from sklearn.model_selection import train_test_split
from src.repositories.object_storage import minio_client
import pandas as pd

BUCKET = "fair"
IMG_PREFIX = "multimodal_fullsample/processed_MRI/"
LOCAL_IMAGES_PATH = "src/data/images"
IMG_INPUT_SIZE = (1,128,128,50)
LABELS_PATH = "multimodal_fullsample/ADNIMERGE_29Nov2024.csv"
ADNI_WASTE = ["ABETA", "update_stamp","TAU", "PTAU","FSVERSION","RID","CDRSB", "EcoPtMem", "EXAMDATE", "SITE", "COLPROT", "ORIGPROT", "PTETHCAT", 'PTRACCAT', "APOE4", "FDG", "PIB", "AV45", "Hippocampus", "WholeBrain", "Enthorhinal", "Fusiform", "MidTemp", "ICV", "M"]
TEST_PERC = 0.2


def processing_features():
    # Download and pre-treat labels
    labels_file = retrieve_object_from_minio(minio_client, BUCKET, LABELS_PATH)
    df_labels = treat_labels(labels_file)


    # Dati clinici: prima modalità
    labels_file = retrieve_object_from_minio(minio_client, BUCKET, LABELS_PATH)
    df_clinical = treat_clinical_data(labels_file, unwanted_cols = ADNI_WASTE, threshold=0.9)

    #La legenda è un prerequisito per le immagini,quindi è importante che venga data prima.
    img_legend = retrieve_object_from_minio(minio_client, BUCKET, "multimodal_fullsample/ADNI_T1_MPRAGE_12_17_2024.csv")
    legend = pd.read_csv(img_legend, on_bad_lines='skip',sep=',').to_dict(orient = "list")

    #inizializzo un ImagePreprocessor che gestisce le immagini da minio e non.
    img_preprocessor = ImagePreprocessor(
        minio_client,
        bucket=BUCKET,
        img_input_size=IMG_INPUT_SIZE,
        legend=legend,
        local_path=LOCAL_IMAGES_PATH
    )

    #Costruisco il dataframe dei path alle immagini
    images_paths = img_preprocessor.get_images_paths_df(IMG_PREFIX)


    #Mergio le modalità in un unico dataset
    # Merge modalities
    df_merge = merge_modalities([df_clinical,images_paths],df_labels)
    df_merge = df_merge.drop(columns=['Patient ID'])
    df_merge = df_merge.drop(columns=['VISCODE'])
    df_merge = df_merge.reset_index(drop=True)

    #train-test-split
    x_train, x_test = train_test_split(df_merge, test_size=TEST_PERC, random_state=42)

    clinical_cols = df_clinical.columns.tolist()
    clinical_cols.remove('Patient ID')
    clinical_cols.remove('VISCODE')

    X_mod1_train = x_train[clinical_cols]
    X_mod3_train = x_train[["image_path"]]

    X_mod1_test = x_test[clinical_cols]
    X_mod3_test = x_test[["image_path"]]

    

    X_mods = {'clinical': X_mod1_train, 'images': X_mod3_train.dropna()}

    X_test = {'clinical': X_mod1_test,  'images': X_mod3_test.dropna()}
    y_train = x_train["Label"]
    y_test = x_test["Label"]

    return X_mods,X_test,y_train,y_test