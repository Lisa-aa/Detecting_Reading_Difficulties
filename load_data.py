from sklearn.feature_selection import f_classif, SelectKBest
import pandas as pd
import os
from torch.nn.utils.rnn import pad_sequence
import torch
from torch.utils.data import DataLoader, TensorDataset
import math


def feature_select_resample(train, dev, test, y_train, y_dev, y_test,model="Res", shap=False):
    """ 
        Feature selection based on ANOVA F-value between label/feature for each timestep per channel.
        Selects the most informative timesteps per channel and creates a new X with only those timesteps.
        
        :param train: Training data
        :param dev: Development data
        :param test: Test data
        :param y_train: Training labels
        :param y_dev: Development labels
        :param y_test: Test labels
        :param model: Model type, used to determine minimum amount of timesteps to select. AlexNet requires a higher minimum.
        7 for ResNet, 63 for AlexNet
        :param shap: Boolean, whether to return timesteps for SHAP analysis.

        :return: train, dev, test, y_train, y_dev, y_test (and timesteps if shap=True)
    """
    # Determine minimum amount of timesteps to select
    if model == "Res":
        min = 7
    else:
        min = 63
    # Get most informative timesteps per channel
    timesteps = []
    amount_of_channels = train.shape[1]
    max = math.floor((16*39/10) / amount_of_channels) # using the rule: for every feature you need 100 examples. With 20 participants and 39 examples
    if max <1:
        max =1
    if max <min:
      max = min
    k = 1
    try_timestemps = []
    while (len(timesteps)<min and len(try_timestemps)<max):
        try_timestemps =[]
        for ch in range(train.shape[1]):
            X_time = train[:, ch, :].numpy()
            sel = SelectKBest(f_classif, k=k)
            x_new = sel.fit_transform(X_time, y_train)
            try_timestemps.append(sel.get_support(indices=True))
        try_timestemps = list(set([item for sublist in try_timestemps for item in sublist]))
        if len(try_timestemps) < max or len(timesteps) < min:
            timesteps = try_timestemps
        k +=1
    if timesteps == []:
      timesteps = try_timestemps
    # Create a new X with only the most informative timesteps per channel
    new_x = [torch.tensor([]) for _ in train]
    for i, elem in enumerate(train):
        lijstje = []
        for feature in elem:
            lijstje.append(feature[timesteps])
        new_x[i] = torch.stack(lijstje)
    train = torch.stack(new_x)

    # Now dev
    new_x = [torch.tensor([]) for _ in dev]
    for i, elem in enumerate(dev):
        lijstje = []
        for feature in elem:
            lijstje.append(feature[timesteps])
        new_x[i] = torch.stack(lijstje)
    dev = torch.stack(new_x)

    # Now test
    new_x = [torch.tensor([]) for _ in test]
    for i, elem in enumerate(test):
        lijstje = []
        for feature in elem:
            lijstje.append(feature[timesteps])
        new_x[i] = torch.stack(lijstje)
    test = torch.stack(new_x)

    # If shap analysis, return timesteps as well
    if shap:
        return  train, dev, test, y_train, y_dev, y_test,timesteps
    return train, dev, test, y_train, y_dev, y_test


def get_data_per_word(type, word, corrected=False,size=85,model="Res"):
    """
        Creates a dataset containing only samples of a specific stimulus.
        
        :param type: The featuretype to collect. Either 'all_eye_features', 'all_audio_features', 'all_pupil_features', 'all_audio_pupil_features' or 'all_features'.
        :param word: The stimulus to collect data for.
        :param corrected: Whether to use corrected labels or not. Labels are corrected for guessing.
        :param size: Amount of features in the data to collect.
        :param model: Collect for ResNet50 or AlexNet, because the require a different minimum of timesteps.

        :return: train, dev, test, y_train, y_dev, y_test
    """
    # Determine which data should be test, dev and train based on participant number
    X_train = []
    train_p = [1,11,12,13,14,16,17,20,21,22,23,24,25,26,28,29]
    dev_p = [15,18]
    test_p = [30,27]
    counter = 0
    # Go through all data and determine positions for the specified stimulus for train
    for folder in os.listdir("Participant_data"):
            participant = folder.split("_")[-1]
            if int(participant) in dev_p or int(participant) in test_p:
                continue
            for file in os.listdir(f"participant_data\\{folder}\\{type}_{participant}"):
                if word in file:
                    X_train.append(counter)
                counter += 1
    # Now for dev
    X_dev = []
    counter = 0
    for folder in os.listdir("Participant_data"):
        participant = folder.split("_")[-1]
        if int(participant) in train_p or int(participant) in test_p:
            continue
        for file in os.listdir(f"participant_data\\{folder}\\{type}_{participant}"):
            if word in file:
                X_dev.append(counter)
            counter += 1
    # Now for test
    X_test = []
    counter = 0
    for folder in os.listdir("Participant_data"):
        participant = folder.split("_")[-1]
        if int(participant) in train_p or int(participant) in dev_p:
            continue
        for file in os.listdir(f"participant_data\\{folder}\\{type}_{participant}"):
            if word in file:
                X_test.append(counter)
            counter += 1
    # Get all data
    _, train, dev, test, y_train, y_dev, y_test = get_all_data(type, corrected,size=size, model=model)
    # Select positions for the specified stimulus and return
    return train[X_train], dev[X_dev], test[X_test], y_train[X_train], y_dev[X_dev], y_test[X_test]


def get_data_per_cat(type, cat, place,corrected=False,size=85,model="Res",cat2=False, place2=False):
    """
    Creates a dataset containing only samples of a specific category.

    :param type: The featuretype to collect. Either 'all_eye_features', 'all_audio_features', 'all_pupil_features', 'all_audio_pupil_features' or 'all_features'.
    :param cat: The category to collect data for.
    :param place: The index of the category in the data.
    :param corrected: Whether to use corrected labels or not. Labels are corrected for guessing.
    :param size: Amount of features in the data to collect.
    :param model: Collect for ResNet50 or AlexNet, because the require a different minimum of timesteps.
    :param cat2: Optional, second category to filter on.
    :param place2: Optional, index of the second category in the data.

    :return: train, dev, test, y_train, y_dev, y_test

    Can also be used to get data per another feature, does not have to be category. Just give the value of the feature and the place to cat and place.
    """
    # Determine which data should be test, dev and train based on participant number
    X_train = []
    train_p = [1,11,12,13,14,16,17,20,21,22,23,24,25,26,28,29]
    dev_p = [15,18]
    test_p = [30,27]
    counter = 0
    # Go through all data and determine positions for the specified category for train
    for folder in os.listdir("Participant_data"):
            participant = folder.split("_")[-1]
            if int(participant) in dev_p or int(participant) in test_p:
                continue
            for file in os.listdir(f"participant_data\\{folder}\\{type}_{participant}"):
                with open(f"participant_data\\{folder}\\{type}_{participant}\\{file}") as f:
                    first_line = f.readline()
                    second_line = f.readline()
                    this_cat = second_line.split(",")[place].strip()
                if cat == this_cat:
                    if not cat2 == False:
                        this_cat2 = second_line.split(",")[place2].strip()
                        if cat2 == this_cat2:
                            X_train.append(counter)
                    else:
                        X_train.append(counter)
                counter += 1
    # Now for dev
    X_dev = []
    counter = 0
    for folder in os.listdir("Participant_data"):
        participant = folder.split("_")[-1]
        if int(participant) in train_p or int(participant) in test_p:
            continue
        for file in os.listdir(f"participant_data\\{folder}\\{type}_{participant}"):
            with open(f"participant_data\\{folder}\\{type}_{participant}\\{file}") as f:
                    first_line = f.readline()
                    second_line = f.readline()
                    this_cat = second_line.split(",")[place].strip()
            if cat == this_cat:
                if not cat2 == False:
                    this_cat2 = second_line.split(",")[place2].strip()
                    if cat2 == this_cat2:
                        X_dev.append(counter)
                else:
                    X_dev.append(counter)
            counter += 1
    # Now for test
    X_test = []
    counter = 0
    for folder in os.listdir("Participant_data"):
        participant = folder.split("_")[-1]
        if int(participant) in train_p or int(participant) in dev_p:
            continue
        for file in os.listdir(f"participant_data\\{folder}\\{type}_{participant}"):
            with open(f"participant_data\\{folder}\\{type}_{participant}\\{file}") as f:
                    first_line = f.readline()
                    second_line = f.readline()
                    this_cat = second_line.split(",")[place].strip()
            if cat == this_cat:
                if not cat2 == False:
                    this_cat2 = second_line.split(",")[place2].strip()
                    if cat2 == this_cat2:
                        X_test.append(counter)
                else:
                    X_test.append(counter)
            counter += 1
    # Get all data
    _, train, dev, test, y_train, y_dev, y_test = get_all_data(type, corrected,size=size, model=model)
    # Select positions for the specified category and return
    return train[X_train], dev[X_dev], test[X_test], y_train[X_train], y_dev[X_dev], y_test[X_test]


def get_data(x_folder, y_file, size, model="Res", shap=False):
    """
    Creates a dataset containing only data from one participant for one feature type.
    
    :param x_folder: The folder where the data of this participant for for the feature type is stored.
    :param y_file: Folder containing labels for all datapoints.
    :param size: Amount of features in the data to collect.
    :param model: Collect for ResNet50 or AlexNet, because the require a different minimum of timesteps.
    :param shap: Boolean, whether to return timesteps for SHAP analysis.

    :return: train_data (data loader), train, dev, test, y_train, y_dev, y_test (and timesteps if shap=True)
    """
    X_train = []
    train_stimuli = []
    for file in os.listdir(f"{x_folder}_train"):
        path = os.path.join(f"{x_folder}_train", file)
        data = pd.read_csv(path)
        train_stimuli.append(file.split("_")[1])
        data = data.fillna(0)
        loose = 0
        if len(data.columns) >size:
            loose = len(data.columns) -size
        data = data.values.T
        data = torch.tensor(data, dtype=torch.float32)[loose:]
        X_train.append(data) 
    X_dev = []
    dev_stimuli = []
    for file in os.listdir(f"{x_folder}_dev"):
        path = os.path.join(f"{x_folder}_dev", file)
        data = pd.read_csv(path)
        dev_stimuli.append(file.split("_")[1])
        data = data.fillna(0)
        loose = 0
        if len(data.columns) >size:
            loose = len(data.columns) -size
        data = data.values.T
        data = torch.tensor(data, dtype=torch.float32)[loose:]
        X_dev.append(data)
    X_test = []
    test_stimuli = []
    for file in os.listdir(f"{x_folder}_test"):
        path = os.path.join(f"{x_folder}_test", file)
        data = pd.read_csv(path)
        test_stimuli.append(file.split("_")[1])
        data = data.fillna(0)
        loose=0
        if len(data.columns) >size:
            loose = len(data.columns) -size
        data = data.values.T
        data = torch.tensor(data, dtype=torch.float32)[loose:]
        X_test.append(data)
    X = X_train + X_dev + X_test
    X = pad_sequence([S.T for S in X], batch_first=True)
    X = X.permute(0, 2, 1)
    X = torch.nn.functional.normalize(X, dim=2) # Make sure values of different source are in same range
    X = torch.nn.functional.normalize(X) # Normalize all values in same column
    X.numpy
    train = torch.tensor(X[:len(X_train)])
    dev = torch.tensor(X[len(X_train):(len(X_dev) + len(X_train))])
    test = torch.tensor(X[(len(X_dev) + len(X_train)):])
    y = pd.read_csv(y_file)
    if "dorsale" in train_stimuli or "dorsale" in dev_stimuli or "dorsale" in test_stimuli:
      y.loc[y["Stimuli"] == "dorsaal","Stimuli"] = "dorsale"
    y_train = [y[y["Stimuli"] ==i]["correct"].values[0] for i in train_stimuli]
    y_dev = [y[y["Stimuli"] ==i]["correct"].values[0] for i in dev_stimuli]
    y_test = [y[y["Stimuli"] ==i]["correct"].values[0] for i in test_stimuli]
    y_train = torch.tensor(y_train)
    y_dev = torch.tensor(y_dev)
    y_test = torch.tensor(y_test)
    if not model == "SVM":
        if shap:
            train, dev, test, y_train, y_dev, y_test,timesteps = feature_select_resample(train, dev, test, y_train, y_dev, y_test, model=model, shap=shap)
        else:
            train, dev, test, y_train, y_dev, y_test = feature_select_resample(train, dev, test, y_train, y_dev, y_test, model=model, shap=shap)

    train_data = DataLoader(TensorDataset(train, y_train), batch_size=5, shuffle=True)
    if shap:
        return train_data, train, dev, test, y_train, y_dev, y_test, timesteps
    return train_data, train, dev, test, y_train, y_dev, y_test

def get_data_per_task(type, task, corrected= False,size=85,model ="Res"):
    """
    Creates a dataset containing only samples of a specific task.   

    :param type: The featuretype to collect. Either 'all_eye_features', 'all_audio_features', 'all_pupil_features', 'all_audio_pupil_features' or 'all_features'.
    :param task: The task to collect data for.
    :param corrected: Whether to use corrected labels or not. Labels are corrected for guessing.
    :param size: Amount of features in the data to collect.
    :param model: Collect for ResNet50 or AlexNet, because the require a different minimum of timesteps.
    
    :return: train_data (data loader), train, dev, test, y_train, y_dev, y_test
    """
    if type == "all_eye_features":
        if task == 2:
            cols = list(range(0,2)) +[4,5,6]
        elif task == 3:
            cols = [2,3] + list(range(4,7))
        else:
            cols = list(range(4,7))
    if type == "all_audio_Features":
        if task == 1:
            cols = list(range(0,25)) + [75,76,77]
        elif task ==2:
            cols = list(range(25,50)) + [75,76,77]
        else:
             cols = list(range(50,78))
    if type == "all_features":
        if task == 1:
            cols = list(range(0,25)) + [75,76,77,80,83,84]
        elif task ==2:
            cols = list(range(25,50)) + [78,79,80,83,84]
        else:
             cols = list(range(50,75)) + [80,81,82,83,84]
    if type == "all_pupil_features":
        if task == 1:
            cols =list(range(0,6))
        else:
            cols = [3,4,5]
    if type == "all_audio_pupil_features":
        if task == 1:
            cols = list(range(0,25)) + [75,76,77,78,79,80]
        if task == 2:
            cols = list(range(25,50)) + [78,79,80]
        else:
            cols = list(range(50,75)) + [78,79,80]

    X_train = []
    y_train = []
    train_p = [1,11,12,13,14,16,17,20,21,22,23,24,25,26,28,29]
    dev_p = [15,18]
    test_p = [30,27]
    for folder in os.listdir("Participant_data"):
            participant = folder.split("_")[-1]
            if int(participant) in dev_p or int(participant) in test_p:
                continue
            for file in os.listdir(f"participant_data\\{folder}\\{type}_{participant}"):
                path = os.path.join(f"participant_data\\{folder}\{type}_{participant}", file)
                data = pd.read_csv(path)
                data = data.fillna(0)
                loose=0
                if len(data.columns) >size:
                    loose = len(data.columns) -size
                data = data.values.T
                data = torch.tensor(data, dtype=torch.float32)[loose:]
                X_train.append(data[cols]) 
            if corrected:
                label_path = os.path.join(f"participant_data\\{folder}\\corrected_labels_{participant}.csv")
            else:
                label_path = os.path.join(f"participant_data\\{folder}\\labels_{participant}.csv")
            labels = pd.read_csv(label_path)
            labels = labels["correct"].values
            y_train.extend(labels)
    X_dev = []
    y_dev = []
    for folder in os.listdir("Participant_data"):
        participant = folder.split("_")[-1]
        if int(participant) in train_p or int(participant) in test_p:
            continue
        for file in os.listdir(f"participant_data\\{folder}\\{type}_{participant}"):
            path = os.path.join(f"participant_data\\{folder}\\{type}_{participant}", file)
            data = pd.read_csv(path)
            data = data.fillna(0)
            loose=0
            if len(data.columns) >size:
                loose = len(data.columns) -size
            data = data.values.T
            data = torch.tensor(data, dtype=torch.float32)[loose:]
            X_dev.append(data[cols])
        if corrected:
            label_path = os.path.join(f"participant_data\\{folder}\\corrected_labels_{participant}.csv")
        else:
            label_path = os.path.join(f"participant_data\\{folder}\\labels_{participant}.csv")
        labels = pd.read_csv(label_path)
        labels = labels["correct"].values
        y_dev.extend(labels)
    X_test = []
    y_test = []
    for folder in os.listdir("Participant_data"):
        participant = folder.split("_")[-1]
        if int(participant) in train_p or int(participant) in dev_p:
            continue
        for file in os.listdir(f"participant_data\\{folder}\\{type}_{participant}"):
            path = os.path.join(f"participant_data\\{folder}\\{type}_{participant}", file)
            data = pd.read_csv(path)
            data = data.fillna(0)
            loose=0
            if len(data.columns) >size:
                loose = len(data.columns) -size
            data = data.values.T
            data = torch.tensor(data, dtype=torch.float32)[loose:]
            X_test.append(data[cols])
        if corrected:
            label_path = os.path.join(f"participant_data\\{folder}\\corrected_labels_{participant}.csv")
        else:
            label_path = os.path.join(f"participant_data\\{folder}\\labels_{participant}.csv")
        labels = pd.read_csv(label_path)
        labels = labels["correct"].values
        y_test.extend(labels)
    X = X_train + X_dev + X_test
    X = pad_sequence([S.T for S in X], batch_first=True)
    X = X.permute(0, 2, 1)
    X = torch.nn.functional.normalize(X, dim=2) # Make sure values of different source are in same range
    X = torch.nn.functional.normalize(X)
    X.numpy
    train = torch.tensor(X[:len(X_train)])
    dev = torch.tensor(X[len(X_train):(len(X_dev) + len(X_train))])
    test = torch.tensor(X[(len(X_dev) + len(X_train)):])
    y_train = torch.tensor(y_train)
    y_dev = torch.tensor(y_dev)
    y_test = torch.tensor(y_test)
    train, dev, test, y_train, y_dev, y_test = feature_select_resample(train, dev, test, y_train, y_dev, y_test, model=model)
    train_data = DataLoader(TensorDataset(train, y_train), batch_size=50, shuffle=True)
    return train_data, train, dev, test, y_train, y_dev, y_test

def get_all_data(type, corrected = False, size=85, model="Res", shap=False):
    """
    Creates a dataset containing all samples for a specific feature type.
    
    :param type: The featuretype to collect. Either 'all_eye_features', 'all_audio_features', 'all_pupil_features' or 'all_features'.
    :param corrected: Whether to use corrected labels or not. Labels are corrected for guessing.
    :param size: Amount of features in the data to collect.
    :param model: Collect for ResNet50 or AlexNet, because the require a different minimum of timesteps.
    :param shap: Boolean, whether to return timesteps for SHAP analysis.

    :return: train_data (data loader), train, dev, test, y_train, y_dev, y_test (and timesteps if shap=True)
    """
    X_train = []
    y_train = []
    train_p = [1,11,12,13,14,16,17,20,21,22,23,24,25,26,28,29]
    dev_p = [15,18]
    test_p = [30,27]
    for folder in os.listdir("Participant_data"):
            participant = folder.split("_")[-1]
            if int(participant) in dev_p or int(participant) in test_p:
                continue
            for file in os.listdir(f"participant_data\\{folder}\\{type}_{participant}"):
                path = os.path.join(f"participant_data\\{folder}\{type}_{participant}", file)
                data = pd.read_csv(path)
                data = data.fillna(0)
                loose=0
                if len(data.columns) >size:
                    loose = len(data.columns) -size
                data = data.values.T
                data = torch.tensor(data, dtype=torch.float32)[loose:]
                X_train.append(data) 
            if corrected:
                label_path = os.path.join(f"participant_data\\{folder}\\corrected_labels_{participant}.csv")
            else:
                label_path = os.path.join(f"participant_data\\{folder}\\labels_{participant}.csv")
            labels = pd.read_csv(label_path)
            labels = labels["correct"].values
            y_train.extend(labels)
    X_dev = []
    y_dev = []
    for folder in os.listdir("Participant_data"):
        participant = folder.split("_")[-1]
        if int(participant) in train_p or int(participant) in test_p:
            continue
        for file in os.listdir(f"participant_data\\{folder}\\{type}_{participant}"):
            path = os.path.join(f"participant_data\\{folder}\\{type}_{participant}", file)
            data = pd.read_csv(path)
            data = data.fillna(0)
            loose=0
            if len(data.columns) >size:
                loose = len(data.columns) -size
            data = data.values.T
            data = torch.tensor(data, dtype=torch.float32)[loose:]
            X_dev.append(data)
        if corrected:
            label_path = os.path.join(f"participant_data\\{folder}\\corrected_labels_{participant}.csv")
        else:
            label_path = os.path.join(f"participant_data\\{folder}\\labels_{participant}.csv")
        labels = pd.read_csv(label_path)
        labels = labels["correct"].values
        y_dev.extend(labels)
    X_test = []
    y_test = []
    for folder in os.listdir("Participant_data"):
        participant = folder.split("_")[-1]
        if int(participant) in train_p or int(participant) in dev_p:
            continue
        for file in os.listdir(f"participant_data\\{folder}\\{type}_{participant}"):
            path = os.path.join(f"participant_data\\{folder}\\{type}_{participant}", file)
            data = pd.read_csv(path)
            data = data.fillna(0)
            loose=0
            if len(data.columns) >size:
                loose = len(data.columns) -size
            data = data.values.T
            data = torch.tensor(data, dtype=torch.float32)[loose:]
            X_test.append(data)
        if corrected:
            label_path = os.path.join(f"participant_data\\{folder}\\corrected_labels_{participant}.csv")
        else:
            label_path = os.path.join(f"participant_data\\{folder}\\labels_{participant}.csv")
        labels = pd.read_csv(label_path)
        labels = labels["correct"].values
        y_test.extend(labels)
    X = X_train + X_dev + X_test
    X = pad_sequence([S.T for S in X], batch_first=True)
    X = X.permute(0, 2, 1)
    X = torch.nn.functional.normalize(X, dim=2) # Make sure values of different source are in same range
    X = torch.nn.functional.normalize(X)
    X.numpy
    train = torch.tensor(X[:len(X_train)])
    dev = torch.tensor(X[len(X_train):(len(X_dev) + len(X_train))])
    test = torch.tensor(X[(len(X_dev) + len(X_train)):])
    y_train = torch.tensor(y_train)
    y_dev = torch.tensor(y_dev)
    y_test = torch.tensor(y_test)
    if not model == "SVM":
        if shap:
            train, dev, test, y_train, y_dev, y_test, timestep = feature_select_resample(train, dev, test, y_train, y_dev, y_test, model=model, shap=True)
            train_data = DataLoader(TensorDataset(train, y_train), batch_size=50, shuffle=True)
            return train_data, train, dev, test, y_train, y_dev, y_test, timestep
        else:
            train, dev, test, y_train, y_dev, y_test = feature_select_resample(train, dev, test, y_train, y_dev, y_test, model=model)
            train_data = DataLoader(TensorDataset(train, y_train), batch_size=50, shuffle=True)
            return train_data, train, dev, test, y_train, y_dev, y_test