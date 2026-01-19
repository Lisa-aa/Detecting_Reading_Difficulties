import CNN
from CNN import ResNet50_1D, AlexNet1D
import SVM
import os
import numpy as np

def run_on_all_participants(model):
    """
        Train for all participants and all featureset a model of the given type and an ensemble model.
        Then, read results of CNN model on all participants seperately and compute average performance.
        This is done per featureset (eye, pupil, audio, all, audio_pupil) and also for the ensemble model.

        :param model: CNN model to be used (ResNet50_1D or AlexNet1D)

        :return: None, but writes average performance to a text file.
    """
    # Determine which model to use.
    if model is ResNet50_1D:
        model_name = "ResNet50_1D"
    else:
        model_name = "AlexNet1D"
    # All participants in our dataset
    participants = [1,11,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,28,29,30]
    # Train models for all participants
    for elem in participants:
        run_cnn_per_participant(elem, model)
    # Open file to write average results
    with open(f"average_participants_{model_name}.txt", "w") as f:
        # Get average performance per featureset
        for modality in ["eye", "pupil", "audio", "all", "audio_pupil"]:
            train_acc = []
            dev_acc = []
            train_f1 = []
            dev_f1 = []
            for elem in participants:
                with open(f"Participant_data\\Participant_{elem}\\Model_{modality}\\performance_{model_name}_epoch_50.txt") as data:
                    tekst = data.read()
                    tekst = tekst.split("\n")
                    accuracy_train = float(tekst[2].split(",")[-1][:-1])
                    accuracy_dev = float(tekst[5].split(",")[-1][:-1])
                    f1_train = float(tekst[3].split(":")[-1])
                    f1_dev = float(tekst[6].split(":")[-1])
                    print(elem, accuracy_train, accuracy_dev, f1_train, f1_dev)
                    train_acc.append(accuracy_train)
                    dev_acc.append(accuracy_dev)
                    train_f1.append(f1_train)
                    dev_f1.append(f1_dev)
            f.writelines(["\n", str(modality), " \n", "accuracy train ", str(np.mean(train_acc)), " accuracy dev ", str(np.mean(dev_acc)),"\nf1 train ", str(np.mean(train_f1)), "f1 dev ", str(np.mean(dev_f1))])
        ensemble_train_acc =[]
        ensemble_dev_acc =[]
        ensemble_train_f1 =[]
        ensemble_dev_f1 =[]
        # Get average performance of ensemble model
        for elem in participants:
            with open(f"Participant_data\\Participant_{elem}\\ensemble_{model_name}_performance.txt") as data:
                tekst = data.read()
                tekst = tekst.split("\n")
                accuracy_train = float(tekst[0].split(":")[-1])
                accuracy_dev = float(tekst[1].split(":")[-1])
                f1_train = float(tekst[2].split(":")[-1])
                f1_dev = float(tekst[3].split(":")[-1])
                ensemble_train_acc.append(accuracy_train)
                ensemble_dev_acc.append(accuracy_dev)
                ensemble_train_f1.append(f1_train)
                ensemble_dev_f1.append(f1_dev)
        f.writelines(["\n ensemble","\n accuracy train ", str(np.mean(ensemble_train_acc)), " accuracy dev ", str(np.mean(ensemble_dev_acc)),"\n f1 train ", str(np.mean(ensemble_train_f1)), " f1 dev", str(np.mean(ensemble_dev_f1))])
    
def run_cnn_per_participant(participant, model):
    """
    train model of given type for the given participant on all featuresets and create ensemble model.
    
    :param participant: The participant to train a model on all featuresets for.
    :param model: The typer of CNN model to use (ResNet50_1D or AlexNet1D)

    :return: None, but writes model performance to text files in the specially created participant folders.
    """
    # Determine which model to use.
    if model == ResNet50_1D:
        model_name = "ResNet50_1D"
    else:
        model_name = "AlexNet1D"
    # Train ResNet on all features
    os.makedirs(f"Participant_data\\Participant_{participant}\\Model_all", exist_ok=True)
    if not os.path.exists(f"Participant_data\\Participant_{participant}\\Model_all\\performance_{model_name}_epoch_50.txt"):
        print(f"train participant: {participant} all")
        CNN.search_and_train_pp(f"Participant_data\\Participant_{participant}\\Model_all", participant, model, "all_features",85)
    # Train ResNet on audio and pupil features
    os.makedirs(f"Participant_data\\Participant_{participant}\\Model_audio_pupil", exist_ok=True)
    if not os.path.exists(f"Participant_data\\Participant_{participant}\\Model_audio_pupil\\performance_{model_name}_epoch_50.txt"):    
        print(f"train participant: {participant} audio/pupil")
        CNN.search_and_train_pp(f"Participant_data\\Participant_{participant}\\Model_audio_pupil", participant, model, "all_audio_pupil_features",81 )    # Train ResNet on eye movement features
    # Train ResNet on eye movement features
    os.makedirs(f"Participant_data\\Participant_{participant}\\Model_eye", exist_ok=True)
    if not os.path.exists(f"Participant_data\\Participant_{participant}\\Model_eye\\performance_{model_name}_epoch_50.txt"):
        print(f"train participant: {participant} eye")
        CNN.search_and_train_pp(f"Participant_data\\Participant_{participant}\\Model_eye", participant, model, "all_eye_features",7)    # Train ResNet on pupil features
    # Train ResNet on pupil features
    os.makedirs(f"Participant_data\\Participant_{participant}\\Model_pupil", exist_ok=True)
    if not os.path.exists(f"Participant_data\\Participant_{participant}\\Model_pupil\\performance_{model_name}_epoch_50.txt"):
        print(f"train participant: {participant} pupil")
        CNN.search_and_train_pp(f"Participant_data\\Participant_{participant}\\Model_pupil", participant, model, "all_pupil_features",6)    # Train ResNet on audio features
    # Train ResNet on audio features
    os.makedirs(f"Participant_data\\Participant_{participant}\\Model_audio", exist_ok=True)
    if not os.path.exists(f"Participant_data\\Participant_{participant}\\Model_audio\\performance_{model_name}_epoch_50.txt"):
        print(f"train participant: {participant} audio")
        CNN.search_and_train_pp(f"Participant_data/Participant_{participant}/Model_audio", participant, model, "all_audio_Features",78)
    # Train ensemble model
    if not os.path.exists(f"Participant_data/Participant_{participant}/ensemble_{model_name}_performance.txt"):
        CNN.ensemble_model(participant, model)
    
def run_cnn_on_all_variants(model):
    """
    Train CNN models on all participants for all featuresets and also per task and create ensemble model.
    To run this function, folders have to be created for all path names given to the search_and_train_all function.
    :param model: The type of CNN model to use (ResNet50_1D or AlexNet1D)

    :return: None, but writes model performance to text files in the specified folders.
    """
    # All full models
    CNN.search_and_train_all("model_all", model, "all_features",85)
    CNN.search_and_train_all("model_audio", model, "all_audio_Features",78)
    CNN.search_and_train_all("model_pupil", model, "all_pupil_features",6)
    CNN.search_and_train_all("model_eye", model, "all_eye_features",7)
    CNN.search_and_train_all("model_audio_pupil", model, "all_audio_pupil_features",81)
    print("done complete models")
    # Per task models
    CNN.search_and_train_all("model_all", model, "all_features",85, task=1)
    CNN.search_and_train_all("model_audio", model, "all_audio_Features",78, task=1)
    CNN.search_and_train_all("model_pupil", model, "all_pupil_features",6, task=1)
    CNN.search_and_train_all("model_eye", model, "all_eye_features",7, task=1)
    CNN.search_and_train_all("model_audio_pupil", model, "all_audio_pupil_features",81, task=1)

    CNN.search_and_train_all("model_all", model, "all_features",85, task=2)
    CNN.search_and_train_all("model_audio", model, "all_audio_Features",75, task=2)
    CNN.search_and_train_all("model_pupil", model, "all_pupil_features",6, task=2)
    CNN.search_and_train_all("model_eye", model, "all_eye_features",7, task=2)
    CNN.search_and_train_all("model_audio_pupil", model, "all_audio_pupil_features",81, task=2)

    CNN.search_and_train_all("model_all", model, "all_features",85, task=3)
    CNN.search_and_train_all("model_audio", model, "all_audio_Features",75, task=3)
    CNN.search_and_train_all("model_pupil", model, "all_pupil_features",6, task=3)
    CNN.search_and_train_all("model_eye", model, "all_eye_features",7, task=3)
    CNN.search_and_train_all("model_audio_pupil", model, "all_audio_pupil_features",81, task=3)
    print("done per task models")
    CNN.ensemble_model_on_all(model)
    print("done")


def run_svm_on_all():
    """
    Train svm models on all participants seperately for all featuresets and ensemble model

    :return: None, but created models and the performances are saved in the folders corresponding to the features and participants. 
    """
    participants = [1,11,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30]
    # Train the model for all participants and all featuresets
    for p in participants:
        for feature in ["all_features", "all_audio_pupil_features", "all_eye_features", "all_audio_Features", "all_pupil_features"]:
            # Get data
            x_data = f"Participant_data/Participant_{p}/{feature}_{p}"
            y_data = f"Participant_data/Participant_{p}/corrected_labels_{p}.csv"
            # determine feature size
            if feature == "all_features":
                feature_size = 85
            elif feature == "all_eye_features":
                feature_size = 7
            elif feature == "all_pupil_features":
                feature_size = 6
            elif feature == "all_audio_features":
                feature_size = 78
            elif feature == "all_audio_pupil_features":
                feature_size = 81
            SVM.train_best_svm(x_data, y_data, feature=feature, size=feature_size)
        SVM.ensemble_model_per_participant(p)
    # Open all the files with performance per participant and calculate average performance
    with open(f"average_participants_svm.txt", "w") as f:
        for modality in ["all_features","all_eye_features", "all_pupil_features", "all_audio_Features", "all_features", "all_audio_pupil_features"]:
            train_acc = []
            dev_acc = []
            train_f1 = []
            dev_f1 = []
            for elem in participants:
                with open(f"Participant_data\\Participant_{elem}\\svm_{modality}.txt") as data:
                    tekst = data.read()
                    tekst = tekst.split("\n")
                    accuracy_train = float(tekst[0].split(":")[-1])
                    accuracy_dev = float(tekst[3].split(":")[-1])
                    f1_train = float(tekst[2].split(":")[-1])
                    f1_dev = float(tekst[5].split(":")[-1])
                    print(elem, accuracy_train, accuracy_dev, f1_train, f1_dev)
                    train_acc.append(accuracy_train)
                    dev_acc.append(accuracy_dev)
                    train_f1.append(f1_train)
                    dev_f1.append(f1_dev)
            # Write average performance per featureset to file
            f.writelines(["\n", str(modality), " \n", "accuracy train ", str(np.mean(train_acc)), " accuracy dev ", str(np.mean(dev_acc)),"\nf1 train ", str(np.mean(train_f1)), "f1 dev ", str(np.mean(dev_f1))])

def run_svm_on_all_data():
    """
        Train svm models on all participants for all featuresets and ensemble model.

        :return: None, but created models and the performances are saved in the folders corresponding to the features.
    """
    # Iterate over all featuresets
    for feature in [ "all_features","all_audio_pupil_features", "all_eye_features", "all_audio_Features", "all_pupil_features",]:
        # determine featureset size
        if feature == "all_features":
            feature_size = 85
        elif feature == "all_eye_features":
            feature_size = 7
        elif feature == "all_pupil_features":
            feature_size = 6
        elif feature == "all_audio_features":
            feature_size = 78
        elif feature == "all_audio_pupil_features":
            feature_size = 81
        # Train SVM on all data
        SVM.train_best_svm(all_data=True, feature=feature,size=feature_size)
    SVM.ensemble_model_on_all()

def run_svm_per_task():
    """
    Train SVM models on all participants for all featuresets per task.
    """
    # Iterate over all featuresets
    for feature in ["all_eye_features","all_audio_Features" ,"all_audio_pupil_Features", "all_pupil_features"]:
        # Iterate over all tasks
        for task in [1,2,3]:
            # Determine featureset size
            if feature == "all_features":
                feature_size = 85
            elif feature == "all_eye_features":
                feature_size = 7
            elif feature == "all_pupil_features":
                feature_size = 6
            elif feature == "all_audio_features":
                feature_size = 78
            elif feature == "all_audio_pupil_features":
                feature_size = 81
            # Train SVM on all data for the given task
            SVM.train_best_svm(all_data=True, feature=feature,size=feature_size, task=task)

def result_esemble_svm():
    """
    Get average performance of ensemble SVM model over all participants.

    :return: None, but writes average performance to a text file.
    """
    participants = [1,11,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,28,29,30]#
    # Get average performance
    with open(f"average_participants_svm_ensemble.txt", "w") as f:
        ensemble_train_acc =[]
        ensemble_dev_acc =[]
        ensemble_train_f1 =[]
        ensemble_dev_f1 =[]
        for elem in participants:
            with open(f"Participant_data\\Participant_{elem}\\ensemble_model_SVM_performance.txt") as data:
                tekst = data.read()
                tekst = tekst.split("\n")
                accuracy_train = float(tekst[0].split(":")[-1])
                accuracy_dev = float(tekst[1].split(":")[-1])
                f1_train = float(tekst[2].split(":")[-1])
                f1_dev = float(tekst[3].split(":")[-1])
                ensemble_train_acc.append(accuracy_train)
                ensemble_dev_acc.append(accuracy_dev)
                ensemble_train_f1.append(f1_train)
                ensemble_dev_f1.append(f1_dev)
        # Write average performance to file
        f.writelines(["\n ensemble","\n accuracy train ", str(np.mean(ensemble_train_acc)), " accuracy dev ", str(np.mean(ensemble_dev_acc)),"\n f1 train ", str(np.mean(ensemble_train_f1)), " f1 dev", str(np.mean(ensemble_dev_f1))])


