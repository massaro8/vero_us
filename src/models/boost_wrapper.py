import mlflow
import mlflow.pyfunc
import pickle
import tempfile



# Definiamo un wrapper che estende mlflow.pyfunc.PythonModel
class IrBoostWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Carichiamo il modello dal file pickle salvato come artifact
        with open(context.artifacts["model_pickle"], "rb") as f:
            self.model = pickle.load(f)
    
    def predict(self, context, model_input,is_Exp=True):
        # Definiamo il metodo predict che utilizza il modello caricato
        return self.model.predict(model_input,is_Exp)


