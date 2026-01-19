import torch
import random
import CNN
from sklearn.metrics import f1_score, accuracy_score
random.seed(13)
def run_baseline(baseline, type,size=85):
    """
    Runs baseline model on given model and on given data type. Computes f1 and accuracy scores for train, dev and test sets.
    
    :param baseline: Baseline model to run
    :param type: feature set to use, either "all_features", "all_eye_features", "all_pupil_features", "all_audio_features" or "all_audio_pupil_features"
    :param size: Amount of features in the featureset

    :return: None, but prints the f1 and accuracy scores for train, dev and test sets.
    """
    train_data, train, dev, test, y_train, y_dev, y_test = CNN.get_all_data(type,corrected=True, size=size)
    output = baseline(train)
    output_dev = baseline(dev)
    output_test = baseline(test)
    f1_train = f1_score(y_train, output)
    f1_dev = f1_score(y_dev, output_dev)
    f1_test = f1_score(y_test, output_test)
    accuracy_train = accuracy_score(y_train, output)
    accuracy_dev = accuracy_score(y_dev, output_dev)
    accuracy_test = accuracy_score(y_test, output_test)
    print(f"f1_test: {f1_test}, accuracy_test: {accuracy_test}")
    print(f"f1_train: {f1_train}, accuracy_train: {accuracy_train}")
    print(f"f1_dev: {f1_dev}, accuracy_dev: {accuracy_dev}")

    
def baseline_category_1(x:torch.Tensor) -> list:
    """
    Baseline model. This model predicts 1 for all inputs except those where the category feature is 1, in which case it predicts category 0.
    So, this model predicts that the word is in the vocabulary unless the category feature indicates that the word is difficult.
    
    :param x: Data to feed to the model
    :return: List of predictions
    """
    output = []
    for row in x:
        if row[-3][0] == 1:
            output.append(0)
        elif row[-3][0]== 0:
            output.append(1)
        else:
            output.append(1)
    print(output)
    return output

def baseline_category_0(x:torch.Tensor) -> list:
    """
    Baseline model. This model predicts 0 for all inputs except those where the category feature is 0, in which case it predicts 1.
    So, this model predicts that the word is not in the vocabulary unless the category feature indicates that the word is easy.

    :param x: Data to feed to the model
    :return: List of predictions
    """
    output = []
    for row in x:
        if row[-3][0] == 1:
            output.append(0)
        elif row[-3][0]== 0:
            output.append(1)
        else:
            output.append(0)
    print(output)
    return output
            
            
