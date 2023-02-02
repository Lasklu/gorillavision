from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, accuracy_score, classification_report, top_k_accuracy_score

def compute_prediction_metrics(y_true, y_pred, y_score, all_labels)-> dict:
    metrics = {   
        "recall": recall_score(y_true, y_pred,average='micro'),
        "precision": precision_score(y_true, y_pred,average='micro'),
        "f1Score": f1_score(y_true, y_pred, average='micro'),
        "accuracy": accuracy_score(y_true, y_pred),
        #"top_3_accuracy": top_k_accuracy_score(y_true, y_score, labels=all_labels, k=3),
        #"top_5_accuracy": top_k_accuracy_score(y_true, y_score, labels=all_labels, k=5),
        #"top_10_accuracy": top_k_accuracy_score(y_true, y_score, labels=all_labels, k=10),
        "consusion_matrix": confusion_matrix(y_true, y_pred),
        "recall_per_class": recall_score(y_true, y_pred,average=None),
        "precision_per_class": precision_score(y_true, y_pred,average=None),
        "f1Score_per_class": f1_score(y_true, y_pred, average=None),
        "individual_results": classification_report(y_true, y_pred, output_dict=True)
    }
    
    print(metrics)
    return metrics