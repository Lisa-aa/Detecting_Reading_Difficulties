import torch
import random
import CNN
import load_data
from sklearn.metrics import f1_score, accuracy_score
random.seed(13)
# For all functions in this file, normalization needs to be removed from the load_data functions
def run_baseline(baseline):
    """
    Runs baseline model on given model and on "all_pupil_features". Computes f1 and accuracy scores for train, dev and test sets.
    
    :param baseline: Baseline model to run

    :return: None, but prints the f1 and accuracy scores for train, dev and test sets.
    """
    train_data, train, dev, test, y_train, y_dev, y_test = load_data.get_all_data("all_pupil_features",corrected=True, size=6, model="SVM")
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

def run_baseline_per_cat(baseline, cat, place):
    """
    Runs baseline model on given model and on "all_pupil_features". Computes f1 and accuracy scores for train, dev and test sets.
    
    :param baseline: Baseline model to run
    :param cat: category to use
    :param place: place where to find the category

    :return: None, but prints the f1 and accuracy scores for train, dev and test sets for this category.
    """
    train, dev, test, y_train, y_dev, y_test = load_data.get_data_per_cat("all_pupil_features",cat, place,corrected=True, size=6, model="SVM")
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

def run_baseline_pp(baseline):
    """
    Runs baseline model on given model and on "all_pupil_features" and per participant. Computes f1 and accuracy scores for train, dev and test sets for each participant and averages it.
    
    :param baseline: Baseline model to run

    :return: None, but prints the f1 and accuracy scores for train, dev and test sets on average for all participants.
    """
    acc_train_av = []
    acc_dev_av = []
    acc_test_av = []
    f1_train_av = []
    f1_dev_av = []
    f1_test_av = []
    pos_train = []
    pos_dev = []
    pos_test = []
    for p in [1,11,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30]:
        train_data, train, dev, test, y_train, y_dev, y_test = load_data.get_data(f"Participant_data\\Participant_{p}\\all_pupil_features_{p}", f"Participant_data\\Participant_{p}\\corrected_labels_{p}.csv", size=6, model="SVM")
        output = baseline(train)
        output_dev = baseline(dev)
        output_test = baseline(test)
        f1_train_av.append(f1_score(y_train, output))
        f1_dev_av.append(f1_score(y_dev, output_dev))
        f1_test_av.append(f1_score(y_test, output_test))
        acc_train_av.append(accuracy_score(y_train, output))
        acc_dev_av.append(accuracy_score(y_dev, output_dev))
        acc_test_av.append(accuracy_score(y_test, output_test))
        pos_train.append(sum(y_train))
        pos_dev.append(sum(y_dev))
        pos_test.append(sum(y_test))
    print(f"f1_test: {sum(f1_test_av)/len(f1_test_av)}, accuracy_test: {sum(acc_test_av)/len(acc_test_av)}, pos percentage test: {sum(pos_test)/(20*6)}")
    print(f"f1_train: {sum(f1_train_av)/len(f1_train_av)}, accuracy_train: {sum(acc_train_av)/len(acc_train_av)}, pos percentage train: {sum(pos_train)/(20*26)}")
    print(f"f1_dev: {sum(f1_dev_av)/len(f1_dev_av)}, accuracy_dev: {sum(acc_dev_av)/len(acc_dev_av)}, pos percentage dev: {sum(pos_dev)/(20*6)}")

def run_baseline_per_cat_pp(baseline, cat, place):
    """
    Runs baseline model on given model and on "all_pupil_features" and per participant. Computes f1 and accuracy scores for train, dev and test sets for each participant and averages it.
    
    :param baseline: Baseline model to run
    :param cat: Category to use for the baseline model
    :param place: Place in the feature vector to check for the category

    :return: None, but prints the f1 and accuracy scores for train, dev and test sets on average for all participants.
    """
    acc_train_av = []
    acc_dev_av = []
    acc_test_av = []
    f1_train_av = []
    f1_dev_av = []
    f1_test_av = []
    pos_train = []
    pos_dev = []
    pos_test = []
    for p in [1,11,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30]:
        train, dev,test, y_train, y_dev, y_test = load_data.get_data_per_cat_pp(f"Participant_data\\Participant_{p}\\all_pupil_features_{p}", f"Participant_data\\Participant_{p}\\corrected_labels_{p}.csv", cat, place, size=6, model="SVM")
        output = baseline(train)
        output_dev = baseline(dev)
        output_test = baseline(test)
        f1_train_av.append(f1_score(y_train, output))
        f1_dev_av.append(f1_score(y_dev, output_dev))
        f1_test_av.append(f1_score(y_test, output_test))
        acc_train_av.append(accuracy_score(y_train, output))
        acc_dev_av.append(accuracy_score(y_dev, output_dev))
        acc_test_av.append(accuracy_score(y_test, output_test))
        pos_train.append(sum(y_train))
        pos_dev.append(sum(y_dev))
        pos_test.append(sum(y_test))
    acc_test_av =  [x for x in acc_test_av if str(x) != 'nan']
    acc_dev_av =  [x for x in acc_dev_av if str(x) != 'nan']
    print(f"f1_test: {sum(f1_test_av)/len(f1_test_av)}, accuracy_test: {sum(acc_test_av)/len(acc_test_av)}, pos percentage test: {sum(pos_test)/(20*6)}")
    print(f"f1_train: {sum(f1_train_av)/len(f1_train_av)}, accuracy_train: {sum(acc_train_av)/len(acc_train_av)}, pos percentage train: {sum(pos_train)/(20*26)}")
    print(f"f1_dev: {sum(f1_dev_av)/len(f1_dev_av)}, accuracy_dev: {sum(acc_dev_av)/len(acc_dev_av)}, pos percentage dev: {sum(pos_dev)/(20*6)}")


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
        if 1 in row[-3]:
            check = 1
        elif -1 in row[-3]:
            check = -1
        else:
            check = 0
        if check == 1:
            output.append(0)
        elif check == 0:
            output.append(1)
        else:
            output.append(0)
    return output
            
            
