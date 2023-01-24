from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, accuracy_score

def is_in_top_x(neighbour_predictions, real_label, x):
    return real_label in neighbour_predictions[:x]

def top_k_accuracy(predictions, x):
    pass

def compute_prediction_metrics(y_true, y_pred)-> dict:
    #f"top-{str(x)}-accuracy": top_x_accuracy(predictions, x),
    #    "mAP": mean_average_precision(predctions),
    print("Recall", recall_score(y_true, y_pred,average='weighted'))
    print("Precision", recall_score(y_true, y_pred,average='weighted'))
    print("F1Score", f1_score(y_true, y_pred, average='weighted'))
    print("Accuracy", accuracy_score(y_true, y_pred))
    print("Confusion,matrix", confusion_matrix(y_true, y_pred))
    return {   
        "recall": recall_score(y_true, y_pred,average='micro'),
        "precision": precision_score(y_true, y_pred,average='micro'),
        "f1Score": f1_score(y_true, y_pred, average='micro'),
        "accuracy": accuracy_score(y_true, y_pred),
        "consusion_matrix": confusion_matrix(y_true, y_pred),
        "recall_per_class": recall_score(y_true, y_pred,average=None),
        "precision_per_class": precision_score(y_true, y_pred,average=None),
        "f1Score_per_class": f1_score(y_true, y_pred, average=None),
    }