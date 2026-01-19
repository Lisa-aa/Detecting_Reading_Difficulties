import torch
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, log_loss
import pickle
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score
from load_data import get_data, get_data_per_task, get_all_data
import math

def train_best_svm(x_data=None, y_data=None, all_data=False, feature=None,size=85,task=0):
    """
    This function collects data, does a gridsearch for best hyperparameters and then trains the model with the best.
    
    :param x_data: if the model should be trained on all data, set to none, otherwise path of folder containing data to train on.
    :param y_data: if the model should be trained on all data, set to none, otherwise path of file containing labels to train on.
    :param all_data: Whether model should be trained on all participants.
    :param feature: which featureset to train in, either "all_features", "all_eye_features", "all_pupil_features", "all_audio_features" or "all_audio_pupil_features"
    :param size: amount of columns in the featureset
    :param task: when training on all tasks, set to zero, otherwise the number of the task to train on.

    :return: None, but saves the trained model and results to files.
    """
    # Initialize model and hyperparameters to test.
    first_svm = svm.SVC() 
    parameters = {'svm__C':np.arange(0.0, 1.0, 0.01),'svm__kernel':('linear', 'rbf', 'poly')}
    # Collect correct data
    if not task==0:
        _,train, dev,_, y_train, y_dev,_ = get_data_per_task(feature,task=task, corrected=True,size=size)
    elif all_data:
        _,train, dev,_, y_train, y_dev,_ = get_all_data(feature,corrected=True,size=size, model="SVM")
    else:
        _,train, dev,_, y_train, y_dev,_ = get_data(x_data, y_data,size, model="SVM")
        train, dev, y_train, y_dev = train.numpy(), dev.numpy(), y_train.numpy(), y_dev.numpy()
    print("Data collected")
    # Reshape data for SVM. It needs to be 2D
    examples, channels, timesteps = train.shape
    feature_selector = SelectKBest(k=(math.floor(39*16/10)))
    print(timesteps*channels)
    train = train.reshape(examples,channels * timesteps)
    examples, channels, timesteps = dev.shape
    dev = dev.reshape(examples,channels * timesteps)
    print("Start fitting model")
    # Gridsearch and fit the model.
    pipeline = Pipeline([('features',feature_selector),('svm',first_svm)])
    best_svm = GridSearchCV(pipeline, parameters)
    x_new = feature_selector.fit_transform(train, y_train)
    xdev_new = feature_selector.transform(dev)
    best_svm.fit(x_new, y_train)
    print("Model fitted")
    # Evaluate results of trained model
    train_result = best_svm.predict(x_new)
    acc = accuracy_score(y_train, train_result)
    loss = log_loss(y_train, train_result)
    f1 = f1_score(y_train, train_result)
    result = best_svm.predict(xdev_new)
    dev_acc = accuracy_score(y_dev, result)
    dev_loss = log_loss(y_dev, result)
    dev_f1 = f1_score(y_dev, result)
    info = best_svm.best_params_
    print(best_svm.best_params_)
    # Save model and its performance
    if not all_data:
        path = "/".join(str(x_data).split("/")[:-1])
        with open(f"{path}/svm_{feature}.txt", 'w') as f:
            f.write("Train accuracy:"+ str(acc)+ "\n Train loss:" + str(loss) + "\n Train f1:" + str(f1) +"\n" + "Dev accuracy:"+ str(dev_acc)+ "\n Dev loss:" + str(dev_loss) +"\n Dev f1:" + str(dev_f1))
        with open(f"{path}/svm_{feature}.pkl",'wb') as f:
            pickle.dump(best_svm,f)
    elif not task ==0:
        with open(f"svm_{feature}_task{task}.txt", 'w') as f:
            f.write("Train accuracy:"+ str(acc)+ "\n Train loss:" + str(loss) + "\n Train f1:" + str(f1) +"\n" + "Dev accuracy:"+ str(dev_acc)+ "\n Dev loss:" + str(dev_loss) +"\n Dev f1:" + str(dev_f1) +"\n" + str(info))
        with open(f"svm_{feature}_task{task}.pkl",'wb') as f:
            pickle.dump(best_svm,f)
    else:
        with open(f"svm_{feature}.txt", 'w') as f:
            f.write("Train accuracy:"+ str(acc)+ "\n Train loss:" + str(loss) + "\n Train f1:" + str(f1) +"\n" + "Dev accuracy:"+ str(dev_acc)+ "\n Dev loss:" + str(dev_loss) +"\n Dev f1:" + str(dev_f1) +"\n" + str(info))
        with open(f"svm_{feature}.pkl",'wb') as f:
            pickle.dump(best_svm,f)

    
def ensemble_model_on_all():
    """
       trains an ensemble model on top of the models trained on pupil, audio and eye for data from all participants.

       :return: None, but saves the trained ensemble model and results to files.
    """

    # Load model trained on eye features
    model_pupil =  pickle.load(open("Model_pupil\\svm_all_pupil_features.pkl","rb"))
    # Load model trained on pupil features
    model_audio = pickle.load(open("Model_audio\\svm_all_audio_features.pkl", "rb")) 
    # Load model trained on audio features
    model_eye = pickle.load(open("Model_eye\\svm_all_eye_features.pkl", "rb"))

    # Get the different kinds of data of participants
    _,train_eye, dev_eye,test_eye, y_train, y_dev,_ = get_all_data("all_eye_features", True, 7, model="SVM")
    examples, channels, timesteps = train_eye.shape
    train_eye = train_eye.reshape(examples,channels * timesteps)
    examples, channels, timesteps = dev_eye.shape
    dev_eye = dev_eye.reshape(examples,channels * timesteps)

    _,train_pupil, dev_pupil, test_pupil, _,_,_ = get_all_data("all_pupil_features", True, 6, model="SVM")
    examples, channels, timesteps = train_pupil.shape
    train_pupil = train_pupil.reshape(examples,channels * timesteps)
    examples, channels, timesteps = dev_pupil.shape
    dev_pupil = dev_pupil.reshape(examples,channels * timesteps)
    
    _,train_audio, dev_audio,test_audio,_ ,_, _ = get_all_data("all_audio_Features", True,78,model="SVM")
    examples, channels, timesteps = train_audio.shape
    train_audio = train_audio.reshape(examples,channels * timesteps)
    examples, channels, timesteps = dev_audio.shape
    dev_audio = dev_audio.reshape(examples,channels * timesteps)
    
    # Select best features and transform data
    feature_selector = SelectKBest(k=62)
    train_eye = feature_selector.fit_transform(train_eye, y_train)
    dev_eye = feature_selector.transform(dev_eye)
    feature_selector = SelectKBest(k=62)
    train_pupil = feature_selector.fit_transform(train_pupil, y_train)
    dev_pupil = feature_selector.transform(dev_pupil)
    feature_selector = SelectKBest(k=62)
    train_audio = feature_selector.fit_transform(train_audio, y_train)
    dev_audio = feature_selector.transform(dev_audio)

    # Get the outputs of the different models
    outputs_eye = torch.tensor(model_eye.predict(train_eye))
    outputs_pupil = torch.tensor(model_pupil.predict(train_pupil))
    outputs_audio = torch.tensor(model_audio.predict(train_audio))
    # Also on dev set for validation
    outputs_eye_dev = torch.tensor(model_eye.predict(dev_eye))
    outputs_pupil_dev = torch.tensor(model_pupil.predict(dev_pupil))
    outputs_audio_dev = torch.tensor(model_audio.predict(dev_audio))
    # Combine the outputs of the different models
    combined_dev = torch.stack((outputs_eye_dev, outputs_pupil_dev, outputs_audio_dev), dim=1)
    combined_outputs = torch.stack((outputs_eye, outputs_pupil, outputs_audio), dim=1)

    # Train MLP classifier on the combined outputs
    parameter_space = {'hidden_layer_sizes': [(50,20,2),(20,2)], 'activation': ['tanh', 'relu'],'solver': ['lbfgs','sgd', 'adam'],'alpha': [0.00001, 0.0001, 0.01],'learning_rate': ['constant','adaptive']}
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(5, 2), random_state=1)
    best_classifier = GridSearchCV(clf, parameter_space)
    combined_outputs = combined_outputs.reshape(combined_outputs.shape[0], -1).detach().numpy()
    combined_dev = combined_dev.reshape(combined_dev.shape[0], -1).detach().numpy()
    best_classifier.fit(combined_outputs, y_train)
    predicted_train = best_classifier.predict(combined_outputs)
    predicted_dev = best_classifier.predict(combined_dev)
    # Save the model and its performance
    pickle.dump(best_classifier, open(f"ensemble\\ensemble_model_SVM.pkl", 'wb'))
    with open(f"ensemble\\ensemble_model_SVM_performance.txt", "w") as f:
        f.write(f"Training accuracy: {accuracy_score(y_train, predicted_train)}\n")
        f.write(f"Dev accuracy: {accuracy_score(y_dev, predicted_dev)}\n")
        f.write(f"f1 score on train: {f1_score(y_train, predicted_train)}\n")
        f.write(f"f1 score on dev: {f1_score(y_dev, predicted_dev)}\n")
    print("Ensemble model trained and saved.")
    

    
def ensemble_model_per_participant(participant):
    """
        trains an ensemble model on top of the models trained on pupil, audio and eye for data from a single participant.

        :param participant: number of the participant to train the ensemble model for.

        :return: None, but saves the trained ensemble model and results to files.
    """
    # Load model trained on eye features
    model_pupil =  pickle.load(open(f"Participant_data\\Participant_{participant}\\svm_all_pupil_features.pkl","rb"))
    # Load model trained on pupil features
    model_audio = pickle.load(open(f"Participant_data\\Participant_{participant}\\svm_all_audio_Features.pkl", "rb")) 
    # Load model trained on audio features
    model_eye = pickle.load(open(f"Participant_data\\Participant_{participant}\\svm_all_eye_features.pkl", "rb"))

    # Get the different kinds of data of participants
    _,train_eye, dev_eye,_, y_train, y_dev,_ = get_data(f"Participant_data\\Participant_{participant}\\all_eye_features_{participant}", f"Participant_data\\Participant_{participant}\\corrected_labels_{participant}.csv",size=7, model="SVM")
    examples, channels, timesteps = train_eye.shape
    train_eye = train_eye.reshape(examples,channels * timesteps)
    examples, channels, timesteps = dev_eye.shape
    dev_eye = dev_eye.reshape(examples,channels * timesteps)
    
    _,train_pupil, dev_pupil, _, _,_,_ = get_data(f"Participant_data\\Participant_{participant}\\all_pupil_features_{participant}", f"Participant_data\\Participant_{participant}\\corrected_labels_{participant}.csv", size=6,model="SVM")
    examples, channels, timesteps = train_pupil.shape
    train_pupil = train_pupil.reshape(examples,channels * timesteps)
    examples, channels, timesteps = dev_pupil.shape
    dev_pupil = dev_pupil.reshape(examples,channels * timesteps)
    
    _,train_audio, dev_audio,_, _, _ ,_ = get_data(f"Participant_data\\Participant_{participant}\\all_audio_Features_{participant}", f"Participant_data\\Participant_{participant}\\corrected_labels_{participant}.csv",size=78,model="SVM")
    examples, channels, timesteps = train_audio.shape
    train_audio = train_audio.reshape(examples,channels * timesteps)
    examples, channels, timesteps = dev_audio.shape
    dev_audio = dev_audio.reshape(examples,channels * timesteps)

    # Select best features and transform data
    feature_selector = SelectKBest(k=62)
    train_eye = feature_selector.fit_transform(train_eye, y_train)
    dev_eye = feature_selector.transform(dev_eye)
    feature_selector = SelectKBest(k=62)
    train_pupil = feature_selector.fit_transform(train_pupil, y_train)
    dev_pupil = feature_selector.transform(dev_pupil)
    feature_selector = SelectKBest(k=62)
    train_audio = feature_selector.fit_transform(train_audio, y_train)
    dev_audio = feature_selector.transform(dev_audio)


    # Get the outputs of the different models
    outputs_eye = torch.tensor(model_eye.predict(train_eye))
    outputs_pupil = torch.tensor(model_pupil.predict(train_pupil))
    outputs_audio = torch.tensor(model_audio.predict(train_audio))
    # Also on dev set for validation
    outputs_eye_dev = torch.tensor(model_eye.predict(dev_eye))
    outputs_pupil_dev = torch.tensor(model_pupil.predict(dev_pupil))
    outputs_audio_dev = torch.tensor(model_audio.predict(dev_audio))
    # Combine the outputs of the different models
    combined_dev = torch.stack((outputs_eye_dev, outputs_pupil_dev, outputs_audio_dev), dim=1)
    combined_outputs = torch.stack((outputs_eye, outputs_pupil, outputs_audio), dim=1)

    # Train MLP classifier on the combined outputs
    parameter_space = {'hidden_layer_sizes': [(50,20,2),(20,2)], 'activation': ['tanh', 'relu'],'solver': ['lbfgs','sgd', 'adam'],'alpha': [0.00001, 0.0001, 0.01],'learning_rate': ['constant','adaptive']}
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(5, 2), random_state=1)
    best_classifier = GridSearchCV(clf, parameter_space)
    best_classifier.fit(combined_outputs, y_train)
    combined_outputs = combined_outputs.reshape(combined_outputs.shape[0], -1).detach().numpy()
    combined_dev = combined_dev.reshape(combined_dev.shape[0], -1).detach().numpy()
    best_classifier.fit(combined_outputs, y_train)
    predicted_train = best_classifier.predict(combined_outputs)
    predicted_dev = best_classifier.predict(combined_dev)
    # Save the model and its performance
    pickle.dump(best_classifier, open(f"Participant_data\\Participant_{participant}\\ensemble_model_SVM.pkl", 'wb'))
    with open(f"Participant_data\\Participant_{participant}\\ensemble_model_SVM_performance.txt", "w") as f:
        f.write(f"Training accuracy: {accuracy_score(y_train, predicted_train)}\n")
        f.write(f"Dev accuracy: {accuracy_score(y_dev, predicted_dev)}\n")
        f.write(f"f1 score on train: {f1_score(y_train, predicted_train)}\n")
        f.write(f"f1 score on dev: {f1_score(y_dev, predicted_dev)}\n")
    print("Ensemble model trained and saved.")