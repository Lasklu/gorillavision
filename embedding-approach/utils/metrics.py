def is_in_top_x(neighbour_predictions, real_label, x):
    return real_label in neighbour_predictions[:x]

def top_x_accuracy(predictions, x):
    pass

def mean_average_precision(predictions):
    pass

def compute_prediction_metrics(predictions, x):
    return {
        f"top-{str(x)}-accuracy": top_x_accuracy(predictions, x),
        "mAP": mean_average_precision(predctions)
    }