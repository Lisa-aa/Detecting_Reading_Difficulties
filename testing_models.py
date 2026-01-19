from load_data import get_data_per_task, get_all_data, get_data
from CNN import ResNet50_1D, AlexNet1D
import torch
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import pickle as pkl
from sklearn.feature_selection import SelectKBest

def test_cnn_model_per_task(model,type, model_location, per_task=1):
    """
    Test a CNN model on a specific type of features and task. 
    
    :param model: CNN model to be tested, either ResNet50_1D or AlexNet1D
    :param type: type of features used for testing, either "all_eye_features", "all_audio_pupil_features", "all_pupil_features", "all_audio_Features", or "all_features"
    :param model_location: path to the saved model
    :param per_task: task number to be tested

    :return: f1 score and accuracy of the model on the test data
    """
    # determine the number of channels based on the type of features
    if type == "all_eye_features":
        if not per_task == 1:
            channels = 5
        else:
            channels = 3
        size = 7
    elif type == "all_audio_pupil_Features":
        if per_task == 1:
            channels = 31
        else:
            channels = 28
        size = 81
    elif type == "all_pupil_features":
        if per_task == 1:
            channels = 6
        else:
            channels = 3
        size = 6
    elif type == "all_audio_Features":
        channels = 28
        size = 78
    elif type == "all_features":
        if per_task == 1:
            channels = 31
        else:
            channels = 30
        size = 85
    # Determine the model name for loading data
    if model == ResNet50_1D:
        model_name = "ResNet50_1D"
    else:
        model_name = "AlexNet1D"
    # Load the models of the participant
    with torch.no_grad():
        model_eval = model(num_classes=2, in_channels=channels)  
        state_dict = torch.load(model_location, map_location=torch.device('cpu'))
        model_eval.load_state_dict(state_dict)
        model_eval.eval()

    # Get the data
    if per_task != 0:
        _, _,_,test, _,_,y = get_data_per_task(type, per_task, True, size=size,model=model_name[:3])
    else:
        _, _,_,test, _,_,y = get_all_data(type, True, channels,model=model_name[:3])

    # Get the outputs of the different models
    outputs_model = model_eval(test)
    f1 = f1_score(y.numpy(), torch.argmax(outputs_model, dim=1).numpy())
    acc = accuracy_score(y.numpy(), torch.argmax(outputs_model, dim=1).numpy())
    # Return the f1 score and accuracy
    print(f"F1 score {model_name} on {type} with task {per_task} and features: {f1}")
    print(f"Accuracy {model_name} on {type} with task {per_task} and features: {acc}")
    return f1, acc

def test_cnn_model_on_all(model,type, model_location):
    """
    Test a CNN model on a specific type of features. 
    
    :param model: CNN model to be tested, either ResNet50_1D or AlexNet1D
    :param type: type of features used for testing, either "all_eye_features", "all_audio_pupil_features", "all_pupil_features", "all_audio_Features", or "all_features"
    :param model_location: path to the saved model

    :return: f1 score and accuracy of the model on the test data
    """
    # determine the number of channels based on the type of features
    if type == "all_eye_features":
        channels = 7
    elif type == "all_audio_pupil_Features":
        channels = 81
    elif type == "all_pupil_features":
        channels = 6
    elif type == "all_audio_Features":
        channels = 78
    elif type == "all_features":
        channels = 85
    # Determine the model name for loading data
    if model == ResNet50_1D:
        model_name = "ResNet50_1D"
    else:
        model_name = "AlexNet1D"
    # Load the models of the participant
    with torch.no_grad():
        model_eval = model(num_classes=2, in_channels=channels)  
        state_dict = torch.load(model_location, map_location=torch.device('cpu'))
        model_eval.load_state_dict(state_dict)
        model_eval.eval()


    _, _,_,test, _,_,y = get_all_data(type, True, channels,model=model_name[:3])

    # Get the outputs of the different models
    outputs_model = model_eval(test)
    f1 = f1_score(y.numpy(), torch.argmax(outputs_model, dim=1).numpy())
    acc = accuracy_score(y.numpy(), torch.argmax(outputs_model, dim=1).numpy())
    print(f"F1 score {model_name} on {type}: {f1}")
    print(f"Accuracy {model_name} on {type}: {acc}")
    return f1, acc

def test_all_models():
    """
    Test all CNN models on all types of features and tasks.

    :return: None, but prints the f1 score and accuracy of each model on each type of features and task.
    """
    features = ["all_features", "all_eye_features","all_audio_Features" ,"all_audio_pupil_Features", "all_pupil_features"]
    models = [ResNet50_1D, AlexNet1D]
    # Loop through all models, features, and tasks
    for model in models:
        if model == ResNet50_1D:
            model_name = "ResNet50_1D"
        else:
            model_name = "AlexNet1D"
        for feature in features:
            if feature == "all_features":
                loc = "Model_all"
            elif feature == "all_eye_features":
                loc = "Model_eye"
            elif feature == "all_audio_pupil_Features":
                loc = "Model_audio_pupil"
            elif feature == "all_audio_Features":
                loc = "Model_audio"
            elif feature == "all_pupil_features":
                loc = "Model_pupil"

            for task in [0,1,2,3,4]:
                if task ==0:
                    model_location = f"{loc}\\{str(model_name)}_final_epoch_50.pth"
                    test_cnn_model_on_all(model, feature, model_location, per_task=task)
                else:
                    model_location =f"{loc}\\{str(model_name)}_final_epoch_50_task_{task}.pth"
                    test_cnn_model_per_task(model, feature, model_location, per_task=task)

def test_models_pp():
    """
    Test all CNN models on all types of features and tasks per participant.

    :return: None, but prints the f1 score and accuracy of each model on each type of features and task.
    """
    models = [ResNet50_1D, AlexNet1D]
    # Loop through all models, features, and participants
    for model in models:
        if model == ResNet50_1D:
            model_name = "ResNet50_1D"
        else:
            model_name = "AlexNet1D"
        for feature in ["all_features", "all_eye_features","all_audio_Features" ,"all_audio_pupil_Features", "all_pupil_features"]:
            f1_scores = []
            accuracies = []
            if feature == "all_features":
                loc = "Model_all"
            elif feature == "all_eye_features":
                loc = "Model_eye"
            elif feature == "all_audio_pupil_Features":
                loc = "Model_audio_pupil"
            elif feature == "all_audio_Features":
                loc = "Model_audio"
            elif feature == "all_pupil_features":
                loc = "Model_pupil"
            if feature == "all_eye_features":
                channels = 7
            elif feature == "all_audio_pupil_Features":
                channels = 81
            elif feature == "all_pupil_features":
                channels = 6
            elif feature == "all_audio_Features":
                channels = 78
            elif feature == "all_features":
                channels = 85
            for participant in [1,11,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30]:
                model_location =f"Participant_data\\Participant_{participant}\\{loc}\\{model_name}_epoch_50.pth"
                # Load the models of the participant
                with torch.no_grad():
                    # Load model trained on eye features
                    model_eval = model(num_classes=2, in_channels=channels)  
                    state_dict = torch.load(model_location, map_location=torch.device('cpu'))
                    model_eval.load_state_dict(state_dict)
                    model_eval.eval()
                
                    _, _,_,test, _,_,y = get_data(f"Participant_data\\Participant_{participant}\\{feature}_{participant}",f"Participant_data\\Participant_{participant}\\corrected_labels_{participant}.csv", channels,model=model_name[:3])

                    # Get the outputs of the different models
                    outputs_model = model_eval(test)
                    f1 = f1_score(y.numpy(), torch.argmax(outputs_model, dim=1).numpy())
                    acc = accuracy_score(y.numpy(), torch.argmax(outputs_model, dim=1).numpy())
                    f1_scores.append(f1)
                    accuracies.append(acc)
            print(f"F1 score {model_name} on {feature} and features: {np.mean(f1_scores)}")
            print(f"Accuracy {model_name} on {feature} and features: {np.mean(accuracies)}")
    return np.mean(f1_scores), np.mean(accuracies)

def test_svm_all():
    """
    Test all SVM models on all types of features.

    :return: None, but prints the f1 score and accuracy of each model on each type of features.
    """
    # Iterate over all feature types
    for feature in ["all_features", "all_eye_features","all_audio_Features" ,"all_audio_pupil_Features", "all_pupil_features"]:
        f1_scores = []
        accuracies = []
        if feature == "all_features":
            loc = "Model_all"
        elif feature == "all_eye_features":
            loc = "Model_eye"
        elif feature == "all_audio_pupil_Features":
            loc = "Model_audio_pupil"
        elif feature == "all_audio_Features":
            loc = "Model_audio"
        elif feature == "all_pupil_features":
            loc = "Model_pupil"
        if feature == "all_eye_features":
            channels = 7
        elif feature == "all_audio_pupil_Features":
            channels = 81
        elif feature == "all_pupil_features":
            channels = 6
        elif feature == "all_audio_Features":
            channels = 78
        elif feature == "all_features":
            channels = 85
        model_location =f"{loc}\svm_{feature}.pkl"
        model = pkl.load(open(model_location,"rb"))
        # Get the data to test on
        _, train,_,test, y_train,_,y = get_all_data(feature, True, channels, model="SVM")
        examples, channels, timesteps = test.shape
        test = test.reshape(examples,channels * timesteps)
        examples, channels, timesteps = train.shape
        train = train.reshape(examples,channels * timesteps)
        # Get the outputs of the different models
        feature_selector = SelectKBest(k=62)
        x_new = feature_selector.fit_transform(train, y_train)
        xtest_new = feature_selector.transform(test)
        outputs_model = model.predict(xtest_new)
        f1 = f1_score(torch.Tensor(y),outputs_model)
        acc = accuracy_score(torch.Tensor(y),outputs_model)
        f1_scores.append(f1)
        accuracies.append(acc)
        # Print the results
        print(f"F1 score svm on {feature}: {np.mean(f1_scores)}")
        print(f"Accuracy svm on {feature}: {np.mean(accuracies)}")
    return np.mean(f1_scores), np.mean(accuracies)

def test_svm_pp():
    """
    Test all SVM models on all types of features per participant.

    :return: None, but prints the f1 score and accuracy of each model on each type of features.
    """
    # Iterate over all feature types
    for feature in ["all_features", "all_eye_features","all_audio_Features" ,"all_audio_pupil_Features", "all_pupil_features"]:
        f1_scores = []
        accuracies = []
        if feature == "all_features":
            loc = "Model_all"
        elif feature == "all_eye_features":
            loc = "Model_eye"
        elif feature == "all_audio_pupil_Features":
            loc = "Model_audio_pupil"
        elif feature == "all_audio_Features":
            loc = "Model_audio"
        elif feature == "all_pupil_features":
            loc = "Model_pupil"
        if feature == "all_eye_features":
            channels = 7
        elif feature == "all_audio_pupil_Features":
            channels = 81
        elif feature == "all_pupil_features":
            channels = 6
        elif feature == "all_audio_Features":
            channels = 78
        elif feature == "all_features":
            channels = 85
        # Iterate over all participants
        for participant in [1,11,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30]:
            model_location =f"Participant_data\Participant_{participant}\svm_{feature}.pkl"
            model = pkl.load(open(model_location,"rb"))
            # Get data to test on
            _, train,_,test, y_train,_,y = get_data(f"Participant_data\\Participant_{participant}\\{feature}_{participant}",f"Participant_data\\Participant_{participant}\\corrected_labels_{participant}.csv", channels,  model="SVM")
            examples, channels, timesteps = test.shape
            test = test.reshape(examples,channels * timesteps)
            examples, channels, timesteps = train.shape
            train = train.reshape(examples,channels * timesteps)
            # Get the outputs of the different models
            feature_selector = SelectKBest(k=62)
            x_new = feature_selector.fit_transform(train, y_train)
            xtest_new = feature_selector.transform(test)
            outputs_model = model.predict(xtest_new)
            f1 = f1_score(torch.Tensor(y),outputs_model)
            acc = accuracy_score(torch.Tensor(y),outputs_model)
            f1_scores.append(f1)
            accuracies.append(acc)
        print(f"F1 score svm on {feature}: {np.mean(f1_scores)}")
        print(f"Accuracy svm on {feature}: {np.mean(accuracies)}")
    return np.mean(f1_scores), np.mean(accuracies)

def test_cnn_pp_ens():
    """
    Test all per participant ensemble models on all featuresets on both CNNs.

    :return: None, but prints the f1 score and accuracy of each ensemble model.
    """
    # iterate over the different CNNS
    for model in [ResNet50_1D, AlexNet1D]:
        f1_scores = []
        accuracies = []
        if model == ResNet50_1D:
            model_name = "ResNet50_1D"
        else:
            model_name = "AlexNet1D"
        # Iterate over all participants
        for participant in [1,11,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30]:
            model_eye = model(num_classes=2, in_channels=7)  
            state_dict = torch.load(f"Participant_data\\Participant_{participant}\\Model_eye\\{model_name}_epoch_50.pth", map_location=torch.device('cpu'))
            model_eye.load_state_dict(state_dict)
            model_eye.eval()

            model_pupil = model(num_classes=2, in_channels=6)
            state_dict = torch.load(f"Participant_data\\Participant_{participant}\\Model_pupil\\{model_name}_epoch_50.pth", map_location=torch.device('cpu'))
            model_pupil.load_state_dict(state_dict)
            model_pupil.eval()

            model_audio = model(num_classes=2, in_channels=78)
            state_dict = torch.load(f"Participant_data\\Participant_{participant}\\Model_audio\\{model_name}_epoch_50.pth", map_location=torch.device('cpu'))
            model_audio.load_state_dict(state_dict)
            model_audio.eval()

            model_location = f"Participant_data\Participant_{participant}\ensemble_model_{model_name}.pkl"
            model_ens = pkl.load(open(model_location,"rb"))

            
            # Get the different kinds of data of participants
            _,train_eye, _,test_eye, y_train, _,y_test = get_data(f"Participant_data\\Participant_{participant}\\all_eye_features_{participant}", f"Participant_data\\Participant_{participant}\\corrected_labels_{participant}.csv",size=7,model=model_name[:3])
            _,train_pupil, _, test_pupil, _,_,_ = get_data(f"Participant_data\\Participant_{participant}\\all_pupil_features_{participant}", f"Participant_data\\Participant_{participant}\\corrected_labels_{participant}.csv", size=6,model=model_name[:3])
            _,train_audio, _,test_audio,_ ,_, _ = get_data(f"Participant_data\\Participant_{participant}\\all_audio_Features_{participant}", f"Participant_data\\Participant_{participant}\\corrected_labels_{participant}.csv",size=78, model=model_name[:3])

            output_eye = model_eye(test_eye)
            output_pupil = model_pupil(test_pupil)
            output_audio = model_audio(test_audio)
            xtest_new = torch.stack((output_eye, output_pupil, output_audio),dim=1)
            xtest_new = torch.nan_to_num(xtest_new, nan=0)
            outputs_model = model_ens.predict(xtest_new.reshape(xtest_new.shape[0], -1).detach().numpy())
            f1 = f1_score(torch.Tensor(y_test),outputs_model)
            acc = accuracy_score(torch.Tensor(y_test),outputs_model)
            f1_scores.append(f1)
            accuracies.append(acc)
        # print mean results for the per participant models per feature set and CNN type
        print(f"F1 score {model_name} on ensemble: {np.mean(f1_scores)}")
        print(f"Accuracy {model_name} on ensemble: {np.mean(accuracies)}")
    return np.mean(f1_scores), np.mean(accuracies)

def test_cnn_ens():
    """
    Test all ensemble models on all featuresets on both CNNs.
    :return: None, but prints the f1 score and accuracy of each ensemble model.
    """
    for model in [AlexNet1D, ResNet50_1D]: 
        f1_scores = []
        accuracies = []
        if model == ResNet50_1D:
            model_name = "ResNet50_1D"
        else:
            model_name = "AlexNet1D"
        # Load eye model
        model_eye = model(num_classes=2, in_channels=7)  
        state_dict = torch.load(f"Model_eye\\{model_name}_final_epoch_50.pth", map_location=torch.device('cpu'))
        model_eye.load_state_dict(state_dict)
        model_eye.eval()
        # load pupil model
        model_pupil = model(num_classes=2, in_channels=6)
        state_dict = torch.load(f"Model_pupil\\{model_name}_final_epoch_50.pth", map_location=torch.device('cpu'))
        model_pupil.load_state_dict(state_dict)
        model_pupil.eval()
        # load audio model
        model_audio = model(num_classes=2, in_channels=78)
        state_dict = torch.load(f"Model_audio\\{model_name}_final_epoch_50.pth", map_location=torch.device('cpu'))
        model_audio.load_state_dict(state_dict)
        model_audio.eval()
        # Load ensemble model
        model_location = f"ensemble\ensemble_model_{model_name}.pkl"
        model_ens = pkl.load(open(model_location,"rb"))
        print(model_ens)
        
        # Get the different kinds of data of participants
        _,train_eye, _,test_eye, y_train, _,y_test = get_all_data("all_eye_features", True, 7, model=model_name[:3])
        _,train_pupil, _, test_pupil, _,_,_ = get_all_data("all_pupil_features", True, 6, model=model_name[:3])
        _,train_audio, _,test_audio,_ ,_, _ = get_all_data("all_audio_Features", True,78, model=model_name[:3])
        # run all models on the test data.
        output_eye = model_eye(test_eye)
        output_pupil = model_pupil(test_pupil)
        output_audio = model_audio(test_audio)
        xtest_new = torch.stack((output_eye, output_pupil, output_audio),dim=1)
        xtest_new = torch.nan_to_num(xtest_new, nan=0)
        outputs_model = model_ens.predict(xtest_new.reshape(xtest_new.shape[0], -1).detach().numpy())
        f1 = f1_score(torch.Tensor(y_test),outputs_model)
        acc = accuracy_score(torch.Tensor(y_test),outputs_model)
        f1_scores.append(f1)
        accuracies.append(acc)
        # Print results for the per participant models per feature set and CNN type
        print(f"F1 score {model_name} on ensemble: {np.mean(f1_scores)}")
        print(f"Accuracy {model_name} on ensemble: {np.mean(accuracies)}")
    return np.mean(f1_scores), np.mean(accuracies)

def test_svm_pp_ens():
    """
    Test all ensemble models on all featuresets on the svms.
    :return: None, but prints the f1 score and accuracy of each ensemble model.
    """
    f1_scores = []
    accuracies = []
    # Iterate over all participants
    for participant in [1,11,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30]:
        # Get all models for the participant
        model_eye = pkl.load(open(f"Participant_data\\Participant_{participant}\\svm_all_eye_features.pkl", "rb"))
        model_pupil = pkl.load(open(f"Participant_data\\Participant_{participant}\\svm_all_pupil_features.pkl", "rb"))
        model_audio = pkl.load(open(f"Participant_data\\Participant_{participant}\\svm_all_audio_Features.pkl", "rb"))
        model_location =f"Participant_data\Participant_{participant}\ensemble_model_SVM.pkl"
        model = pkl.load(open(model_location,"rb"))
        
        # Get the different kinds of data of participants
        _,train_eye, _,test_eye, y_train, _,y_test = get_data(f"Participant_data\\Participant_{participant}\\all_eye_features_{participant}", f"Participant_data\\Participant_{participant}\\corrected_labels_{participant}.csv",size=7, model="SVM")
        examples, channels, timesteps = train_eye.shape
        train_eye = train_eye.reshape(examples,channels * timesteps)
        examples, channels, timesteps = test_eye.shape
        test_eye = test_eye.reshape(examples,channels * timesteps)

        _,train_pupil, _, test_pupil, _,_,_ = get_data(f"Participant_data\\Participant_{participant}\\all_pupil_features_{participant}", f"Participant_data\\Participant_{participant}\\corrected_labels_{participant}.csv", size=6, model="SVM")
        examples, channels, timesteps = train_pupil.shape
        train_pupil = train_pupil.reshape(examples,channels * timesteps)
        examples, channels, timesteps = test_pupil.shape
        test_pupil = test_pupil.reshape(examples,channels * timesteps)
        
        _,train_audio, _,test_audio,_ ,_, _ = get_data(f"Participant_data\\Participant_{participant}\\all_audio_Features_{participant}", f"Participant_data\\Participant_{participant}\\corrected_labels_{participant}.csv",size=78 , model="SVM")
        examples, channels, timesteps = train_audio.shape
        train_audio = train_audio.reshape(examples,channels * timesteps)
        examples, channels, timesteps = test_audio.shape
        test_audio = test_audio.reshape(examples,channels * timesteps)
        # Select the best features as done during training
        feature_selector = SelectKBest(k=62)
        train_eye = feature_selector.fit_transform(train_eye, y_train)
        test_eye = feature_selector.transform(test_eye)
        feature_selector = SelectKBest(k=62)
        train_pupil = feature_selector.fit_transform(train_pupil, y_train)
        test_pupil = feature_selector.transform(test_pupil)
        feature_selector = SelectKBest(k=62)
        train_audio = feature_selector.fit_transform(train_audio, y_train)
        test_audio = feature_selector.transform(test_audio)
        # Get the outputs of the different models on the test set
        output_eye = model_eye.predict(test_eye)
        output_pupil = model_pupil.predict(test_pupil)
        output_audio = model_audio.predict(test_audio)
        xtest_new = np.column_stack((output_eye, output_pupil, output_audio))
        outputs_model = model.predict(xtest_new)
        f1 = f1_score(torch.Tensor(y_test),outputs_model)
        acc = accuracy_score(torch.Tensor(y_test),outputs_model)
        f1_scores.append(f1)
        accuracies.append(acc)
    # Print mean results for the per participant models per feature set and CNN type
    print(f"F1 score svm on ensemble: {np.mean(f1_scores)}")
    print(f"Accuracy svm on ensemble: {np.mean(accuracies)}")
    return np.mean(f1_scores), np.mean(accuracies)

def test_svm_ens():
    """
    Test all ensemble models on all featuresets on the svms.
    :return: None, but prints the f1 score and accuracy of each ensemble model.
    """
    f1_scores = []
    accuracies = []
    # Get all models
    model_eye = pkl.load(open(f"Model_eye\\svm_all_eye_features.pkl", "rb"))
    model_pupil = pkl.load(open(f"Model_pupil\\svm_all_pupil_features.pkl", "rb"))
    model_audio = pkl.load(open(f"Model_audio\\svm_all_audio_Features.pkl", "rb"))
    model_location =f"ensemble\ensemble_model_SVM.pkl"
    model = pkl.load(open(model_location,"rb"))
    
    # Get the different kinds of data of participants
    _,train_eye, _,test_eye, y_train, _,y_test = get_all_data("all_eye_features", True, 7, model="SVM")
    examples, channels, timesteps = train_eye.shape
    train_eye = train_eye.reshape(examples,channels * timesteps)
    examples, channels, timesteps = test_eye.shape
    test_eye = test_eye.reshape(examples,channels * timesteps)

    _,train_pupil, _, test_pupil, _,_,_ = get_all_data("all_pupil_features", True, 6, model="SVM")
    examples, channels, timesteps = train_pupil.shape
    train_pupil = train_pupil.reshape(examples,channels * timesteps)
    examples, channels, timesteps = test_pupil.shape
    test_pupil = test_pupil.reshape(examples,channels * timesteps)
    
    _,train_audio, _,test_audio,_ ,_, _ = get_all_data("all_audio_Features", True,78, model="SVM")
    examples, channels, timesteps = train_audio.shape
    train_audio = train_audio.reshape(examples,channels * timesteps)
    examples, channels, timesteps = test_audio.shape
    test_audio = test_audio.reshape(examples,channels * timesteps)
    # Select the best features as done during training
    feature_selector = SelectKBest(k=62)
    train_eye = feature_selector.fit_transform(train_eye, y_train)
    test_eye = feature_selector.transform(test_eye)
    feature_selector = SelectKBest(k=62)
    train_pupil = feature_selector.fit_transform(train_pupil, y_train)
    test_pupil = feature_selector.transform(test_pupil)
    feature_selector = SelectKBest(k=62)
    train_audio = feature_selector.fit_transform(train_audio, y_train)
    test_audio = feature_selector.transform(test_audio)
    # Get the outputs of the different models on the test set
    output_eye = model_eye.predict(test_eye)
    output_pupil = model_pupil.predict(test_pupil)
    output_audio = model_audio.predict(test_audio)
    xtest_new = np.column_stack((output_eye, output_pupil, output_audio))
    outputs_model = model.predict(xtest_new)
    f1 = f1_score(torch.Tensor(y_test),outputs_model)
    acc = accuracy_score(torch.Tensor(y_test),outputs_model)
    f1_scores.append(f1)
    accuracies.append(acc)
    # Return mean results for the ensemble model
    print(f"F1 score svm on ensemble: {np.mean(f1_scores)}")
    print(f"Accuracy svm on ensemble: {np.mean(accuracies)}")
    return np.mean(f1_scores), np.mean(accuracies)
