from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, accuracy_score, classification_report, top_k_accuracy_score

def compute_prediction_metrics(y_true, y_pred, y_score, all_labels)-> dict:
    if len(y_score[0]) == 2:
        y_score = [all_labels[score.index(min(score))] for score in y_score]

    """y_true_lab = list(set(y_true))
    y_pred_lab = list(set(y_pred.tolist()))
    mat_labels = y_true_lab.append(y_pred_lab)
    mat_labels.sort()


    mat = confusion_matrix(y_true, y_pred)

    new_mat = []
    new_labels = []

    for c in eval_classes:
    c_idx_mat = mat_labels.index(c)
    pred_old = 0
    preds = []
    for i in range(0, len(mat[0])):
        if mat_labels[c_idx_mat][i] in train_classes:
        pred_old += mat_labelsmat_labels[c_idx_mat][i]
        else:
        preds.append(mat_labels[c_idx_mat][i])
    preds.append(pred_old)
    new_mat.append(preds)
    new_labels.append(c)

    new_mat.append([0 for i in range(0, len(eval_classes+1))])
    new_labels.append("old_classes")

    print(new_mat)
    print(new_labels)"""

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

    return metrics