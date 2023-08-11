import evaluate
import numpy as np

def compute_metrics(eval_pred):

    accuracy = evaluate.load("accuracy")

    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

def compute_metrics2(eval_pred):

    f1 = evaluate.load("f1")

    return f1.compute(predictions=eval_pred.predictions,references=eval_pred.label_ids)
