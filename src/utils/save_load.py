import tempfile
from src.models.boost_wrapper import IrBoostWrapper
import mlflow
import os
import mlflow.pyfunc
import pickle

def save_model(ir_boost):

    # Creiamo un file temporaneo e salviamo il modello pickle in esso
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        pickle.dump(ir_boost, tmp)
        tmp.flush()
        tmp_path = tmp.name  # Salviamo il percorso del file temporaneo

    # Logghiamo il modello su MLflow utilizzando il nostro wrapper e il file temporaneo come artifact
    mlflow.pyfunc.log_model(
        artifact_path="models",
        python_model=IrBoostWrapper(),
        artifacts={"model_pickle": tmp_path}
    )

    
    os.remove(tmp_path)

    return None

def load_model():

    model_uri= 'models:/ir_boost_fair/Production'

    ir_boost = mlflow.pyfunc.load_model(model_uri)

    return ir_boost