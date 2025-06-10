from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow

def evaluate(ir_boost,X_mods,X_test,y_train,y_test,is_Exp=False):


    train_preds = ir_boost.predict(X_mods,is_Exp)
    train_accuracy = accuracy_score(y_train.sort_index(), train_preds.sort_index())
    train_precision = precision_score(y_train.sort_index(), train_preds.sort_index(), average='macro', zero_division=0)
    train_recall = recall_score(y_train.sort_index(), train_preds.sort_index(), average='macro', zero_division=0)
    train_f1 = f1_score(y_train.sort_index(), train_preds.sort_index(), average='macro', zero_division=0)

    

    mlflow.log_metric("Train Accuracy", train_accuracy)
    mlflow.log_metric("Train Precision", train_precision)
    mlflow.log_metric("Train Recall", train_recall)
    mlflow.log_metric("Train F1 Score", train_f1)


    preds = ir_boost.predict(X_test,is_Exp)
    test_accuracy = accuracy_score(y_test.sort_index(), preds.sort_index())
    test_precision = precision_score(y_test.sort_index(), preds.sort_index(), average='macro', zero_division=0)
    test_recall = recall_score(y_test.sort_index(), preds.sort_index(), average='macro', zero_division=0)
    test_f1 = f1_score(y_test.sort_index(), preds.sort_index(), average='macro', zero_division=0)

    

    mlflow.log_metric("Test Accuracy", test_accuracy)
    mlflow.log_metric("Test Precision", test_precision)
    mlflow.log_metric("Test Recall", test_recall)
    mlflow.log_metric("Test F1 Score", test_f1)

    return None