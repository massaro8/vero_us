import shap
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image


shap.initjs()

def explain_rf(rf_model, X_clinical, num_top_features=10, 
                        sample_indices=[0], 
                        log_to_mlflow=True):
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_clinical)

    #Bar plot globale
    if isinstance(shap_values, list):
        # Multi-class: creiamo un array di shap values mediati (approccio semplificato)
        shap_values_mc = sum(shap_values) / len(shap_values)  # media su tutte le classi
    else:
        # Binaria / regressione
        shap_values_mc = shap_values

    
    #if len(shap_values_mc.shape) == 2:
    #    for col_idx in range(shap_values_mc.shape[1]):
    #        col_j = shap_values_mc[:, col_idx]
    #        min_j = col_j.min()
    #        max_j = col_j.max()
    #        range_j = max_j - min_j
    #        if range_j == 0:
    #            # Evito la divisione per 0 assegnando un valore costante (es. 0.5) 
    #            shap_values_mc[:, col_idx] = 0.5
    #        else:
    #            shap_values_mc[:, col_idx] = (col_j - min_j) / range_j
    
    # Normalizzazione tra 0 e 1
    #flat_shap = shap_values_mc.flatten()
    #min_val = flat_shap.min()
    #max_val = flat_shap.max()
    #range_val = max_val - min_val if max_val != min_val else 1.0

    #shap_values_mc = (shap_values_mc - min_val) / range_val

    plt.figure()
    shap.summary_plot(
        shap_values_mc, 
        X_clinical, 
        plot_type="bar",  # bar plot
        max_display=num_top_features, 
        show=False
    )
    plt.title("Feature importance (SHAP) – Bar plot")
    plt.tight_layout()
    plt.savefig("shap_bar_plot.png", dpi=120)
    if log_to_mlflow:
        mlflow.log_artifact("shap_bar_plot.png")
    plt.close()


    #Plot individuale su alcuni esempi specifici
    for idx in sample_indices:
        if idx < 0 or idx >= len(X_clinical):
            continue

        # Prendiamo i shap values di quell’esempio
        single_shap = (shap_values_mc[idx] 
                       if shap_values_mc.ndim == 2 
                       else shap_values_mc[idx, :])  
        # Forse potrebbe essere shap_values_mc[:, idx], 
        # dipende dalla forma. Verifica in base a come esce shap_values.

        sample_input = X_clinical.iloc[idx,:]
        
        # Generate force plot (testualmente e graficamente)
        force_plot = shap.plots.force(explainer.expected_value[0], single_shap[:, 0], sample_input, feature_names=sample_input.index
            )

        # Salviamo come .html
        # (force_plot è interattivo, in HTML: per visualizzarlo serve un browser o un iframe)
        out_html = f"shap_force_{idx}.html"
        shap.save_html(out_html, force_plot)
        if log_to_mlflow:
            mlflow.log_artifact(out_html)

        plt.figure()
        shap.plots.decision(
            base_value=explainer.expected_value[0],
            shap_values=single_shap[:,0],
            features=sample_input,
            feature_names=sample_input.index.tolist(),
            feature_order='importance'
        )
        plt.title(f"Decision plot - record {idx}")
        plt.tight_layout()
        out_decision_png = f"shap_decision_{idx}.png"
        plt.savefig(out_decision_png, dpi=120)
        if log_to_mlflow:
            mlflow.log_artifact(out_decision_png)
        plt.close()

        # Generazione di una mini-spiegazione testuale
    top_features = (
        abs(shap_values_mc).mean(axis=0) if shap_values_mc.ndim==2 
        else abs(shap_values_mc).mean(0)
    )
    # Ordino per importanza media globale
    feature_importance = sorted(
        list(zip(X_clinical.columns, top_features)), 
        key=lambda x: x[1].mean(), 
        reverse=True
    )
    # Prendo i primi 3
    top3 = feature_importance[:3]
    text_explanation = (
        "In base all'analisi SHAP, le feature più rilevanti sono: \n"
        + "\n".join([f" - {nm} (contributo medio {val.mean():.4f})" for nm, val in top3]) 
        + "\nQueste variabili sembrano avere il maggior impatto sulle predizioni del modello."
    )
    # Stampa e log su MLflow come file di testo
    print(text_explanation)
    with open("shap_text_explanation.txt", "w") as f:
        f.write(text_explanation)
    if log_to_mlflow:
        mlflow.log_artifact("shap_text_explanation.txt")

    # STAMPA DI TUTTE LE FEATURE ORDINATE PER IMPORTANZA
    all_feature_importance = sorted(
        list(zip(X_clinical.columns, top_features)),
        key=lambda x: x[1].mean(), 
        reverse=True
    )
    print("\nTutte le feature ordinate per importanza (contributo medio):")
    for feature, value in all_feature_importance:
        print(f" - {feature}: {value.mean():.4f}")


def get_grad_cam_2d(model, input_tensor, device, target_layer, target_category):
    targets = [ClassifierOutputTarget(2)]
    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        input_tensor.requires_grad_(True)
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]  # ora ha forma (H, W, D)
        
        # Ottieni il primo campione
        sample = input_tensor[0].cpu().detach().numpy()  # forma: (1, 128, 128, 50)
        # Estrai il canale e la slice desiderata (indice 25)
        img_slice = sample[0, :, :, 25]  # forma: (128, 128)
        # Replica il canale per ottenere un'immagine RGB
        rgb_img = np.stack([img_slice, img_slice, img_slice], axis=-1)  # forma: (128, 128, 3)
        
        # Estrai la slice della mappa Grad-CAM
        cam_slice = grayscale_cam[:, :, 25]  # ora correttamente (128, 128)
        
        visualization = show_cam_on_image(rgb_img, cam_slice, use_rgb=True)
        model_outputs = cam.outputs
        return visualization


def get_grad_cam_3d(model, input_tensor, device, target_layer, target_category, return_raw=True):
    targets = [ClassifierOutputTarget(2)]
    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        #input_tensor = input_tensor.unsqueeze(1)
        input_tensor.requires_grad_(True)
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

        if return_raw:

            return grayscale_cam
        else:

            grayscale_cam = grayscale_cam[0, :]  # ora ha forma (H, W, D)
            
            # Ottieni il primo campione
            sample = input_tensor[0].cpu().detach().numpy()  # forma: (1, 128, 128, 50)
            # Estrai il canale e la slice desiderata (indice 25)
            img_slice = sample[0, :, :, 25]  # forma: (128, 128)
            # Replica il canale per ottenere un'immagine RGB
            rgb_img = np.stack([img_slice, img_slice, img_slice], axis=-1)  # forma: (128, 128, 3)
            
            # Estrai la slice della mappa Grad-CAM
            cam_slice = grayscale_cam[:, :, 25]  # ora correttamente (128, 128)
            
            visualization = show_cam_on_image(rgb_img, cam_slice, use_rgb=True)

            model_outputs = cam.outputs

            return visualization





def visualize_gradcam_pairs(gradcam_pairs, slice_index=None):
    """
    Per ciascuna coppia (original_image, gradcam_map) estrae una slice (default quella centrale)
    e crea l'overlay usando show_cam_on_image, salvando poi la figura tramite mlflow.
    
    Parameters:
      - gradcam_pairs: lista di tuple, dove ogni tupla è (original_image, gradcam_map).
        Si assume che ogni immagine abbia shape (H, W, D) (in scala di grigi) e che la gradcam_map abbia lo stesso formato.
      - slice_index: indice della slice da utilizzare; se None verrà usata la slice centrale.
    """
    for idx, (orig, cam) in enumerate(gradcam_pairs):

        grayscale_cam = cam[0, :]  # ora ha forma (H, W, D)
            
            # Ottieni il primo campione
        sample = orig[0]#.cpu().detach().numpy()  # forma: (1, 128, 128, 50)
            # Estrai il canale e la slice desiderata (indice 25)
        img_slice = sample[0, :, :, 25]  # forma: (128, 128)
            # Replica il canale per ottenere un'immagine RGB
        rgb_img = np.stack([img_slice, img_slice, img_slice], axis=-1)  # forma: (128, 128, 3)
            
            # Estrai la slice della mappa Grad-CAM
        cam_slice = grayscale_cam[:, :, 25]  # ora correttamente (128, 128)
            
        visualization = show_cam_on_image(rgb_img, cam_slice, use_rgb=True)
        

        # Visualizza l'overlay
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(visualization)
        ax.axis('off')
        ax.set_title(f"Grad-CAM Visualization {idx}")

        # Salva la figura tramite mlflow
        mlflow.log_figure(fig, f"grad_cam_images/gradcam_visualization{idx}.png")
        plt.close(fig)



