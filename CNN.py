import statistics 
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from sklearn.neural_network import MLPClassifier
import pickle as pkl
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from load_data import get_data, get_all_data, get_data_per_task, get_data_per_word

# determine where the models will run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def search_and_train_pp(path, participant, model, features, feature_number):
    """
    Search for the best hyperparameters for a participant and featureset and train the model with these hyperparameters.
    
    :param path: Where the trained model should be saved
    :param participant: Participant number
    :param model: model to train, either ResNet50_1D or AlexNet1D
    :param features: featureset to train on, either "all_features", "all_eye_features", "all_pupil_features", "all_audio_features" or "all_audio_pupil_features"
    :param feature_number: number of features in the featureset

    :return: None, but the trained model is saved in the given path.
    """
    # Get training data and development data
    _, train, dev, _, ytrain, ydev,_ = get_data(f"Participant_data\\Participant_{participant}\\all_features_{participant}", f"Participant_data\\Participant_{participant}\\corrected_labels_{participant}.csv", size=feature_number)
    # Concatenate training and development data for hyperparameter search
    x = torch.concat((train, dev))
    x = x.to(device)
    label = torch.concat((ytrain, ydev))
    label = label.to(device)
    if model == ResNet50_1D:
        model_name = "Res"
    else:
        model_name = "Alex"
    # Hyperparameter search using k-fold cross validation
    best = hyperparameter_search(model_name, feature_number, x, label, 50)
    optimizer, lr_rate, decay = best[1]
    # Train the model with the best hyperparameters on train and development data
    train_model(path, model(in_channels=feature_number), 50, f"Participant_data\\Participant_{participant}\\all_features_{participant}", f"Participant_data\\Participant_{participant}\\corrected_labels_{participant}.csv", optimizer, lr_rate, decay)

def search_and_train_all(path, model, features,channels, task =0):
    """
    Search for the best hyperparameters for all participants and a specific featureset and train the model with these hyperparameters.    

    :param path: Where the trained model should be saved
    :param model: model to train, either ResNet50_1D or AlexNet1D
    :param features: featureset to train on, either "all_features", "all_eye_features", "all_pupil_features", "all_audio_features" or "all_audio_pupil_features"
    :param channels: number of channels in the featureset
    :param task: task to train on, 0 for all tasks
    """
    # Get training data and development data
    if not task ==0:
        _, train, dev, _, ytrain, ydev,_  = get_data_per_task(features, task, True)
    else:
        _, train, dev, _, ytrain, ydev,_ = get_all_data(features, True,channels)
    # Concatenate training and development data for hyperparameter search
    x = torch.concat((train, dev))
    label = torch.concat((ytrain, ydev))
    if model == ResNet50_1D:
        model_name = "Res"
    else:
        model_name = "Alex"
    # Hyperparameter search using k-fold cross validation
    best = hyperparameter_search(model_name, channels, x, label, 50)
    optimizer, lr_rate, decay = best[1]
    # Train the model with the best hyperparameters on train and development data
    try:
        train_model_on_all(path, model(in_channels = channels), 50, features, optimizer, lr_rate, decay, True, channels=channels, per_task=task)
    except:
        # In case the hyperparameter search was succesful but training fails, best hyperparameters are printed.
        print(best)

def training_model(path, model, data, epoch, name, optimizer, train_data, train_labels, dev_data, dev_labels, lr_decrease=False, task=0):
    """
    Train a model on the given data for a number of epochs with given hyperparameters.

    :param path: Where the trained model should be saved
    :param model: The model to train
    :param data: The training data in batches
    :param epoch: The number of epochs to train the model
    :param name: The name under which the performance file will be saved
    :param optimizer: The optimizer to use during training
    :param train_data: The full training data for evaluation after each epoch
    :param train_labels: The labels for the training data
    :param dev_data: The full development data for evaluation after each epoch
    :param dev_labels: The labels for the development data
    :param lr_decrease: Whether to decrease the learning rate during training
    :param task: The task number, 0 if all tasks

    :return: None, but the performance of the model during training is saved in a text file and it creates a trained model.
    """
    # Move model to device and initialize performance lists
    model = model.to(device)
    losses_train = []
    losses_dev = []
    accuracies_train = []
    accuracies_dev = []
    f1_train = []
    f1_dev = []
    # Define loss function
    loss_function = nn.CrossEntropyLoss()
    # training loop
    for e in range(epoch):        
        model.train()
        # Loop over batches
        for batch in data:
            x = batch[0].to(device)
            scores = batch[1].to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_function(outputs, scores)
            loss.backward()
            optimizer.step()
        
        model.eval()
        # After all batches are finished, evaluate performance on full train and dev set
        with torch.no_grad():
            # Testing performance on train
            outputs = model(train_data)
            predicted = torch.argmax(outputs, axis=1)
            losses_train.append(loss_function(outputs, train_labels).item())
            accuracies_train.append((predicted == train_labels).sum().item() / len(train_labels))
            f1_train.append(f1_score(train_labels, predicted))
            # Testing performance on dev
            outputs_dev = model(dev_data)
            predicted_dev = torch.argmax(outputs_dev, axis=1)
            losses_dev.append(loss_function(outputs_dev, dev_labels).item())
            accuracies_dev.append((predicted_dev == dev_labels).sum().item() / len(dev_labels))
            f1_dev.append(f1_score(dev_labels, predicted_dev))
            # Early stopping if overfitting is detected
            try:
                if losses_dev[-1] > losses_dev[-2] and losses_dev[-2] >losses_dev[-3] and losses_train[-1] < losses_train[-2] and losses_train[-2] < losses_train[-3]:
                    print(accuracies_dev)
                    break   
            except:
                pass 
        # Learning rate decrease schedule if lr_decrease is True
        if lr_decrease:
            if (e + 1) % 5 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
            if e + 1 == 50:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.1
        print(f"Epoch {e+1}/{epoch}")
        print(f"Training - Loss: {losses_train[-1]:.4f}, Accuracy: {accuracies_train[-1]:.4f},  f1: {f1_train[-1]:.4f}")
        print(f"Dev - Loss: {losses_dev[-1]:.4f}, Accuracy: {accuracies_dev[-1]:.4f}, f1: {f1_dev[-1]:.4f}")

    # Save final performance to a text file
    dev_f1 = f1_score(dev_labels, predicted_dev)
    train_f1 = f1_score(train_labels, predicted)
    if task==0:
        f = open("{}\\performance_final_{}.txt".format(path, name),"w") 
    else: 
        f = open(f"{path}\\performance_final_{name}_task_{task}.txt","w") 
    f.write("Loss train:" + str(losses_train) + "\n" + "Accuracy train:" + str(accuracies_train)+ "\n" + "F1 train:" + str(train_f1) + "\n"+ "Loss dev:" + str(losses_dev) + "\n" + "Accuracy dev:" + str(accuracies_dev) + "\n" + "F1 dev:" + str(dev_f1))
    f.close()

# Adaptation of AlexNet to 1D convolutions
class AlexNet1D(nn.Module):
    def __init__(self, num_classes=2, in_channels=10):
        super(AlexNet1D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=False),
            nn.MaxPool1d(kernel_size=3, stride=2),
            
            nn.Conv1d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=False),
            nn.MaxPool1d(kernel_size=3, stride=2),
            
            nn.Conv1d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv1d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool1d(6)  # maakt vaste outputlengte
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6, 4096),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=False),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# This is a help class for ResNet50_1D    
class Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels * self.expansion, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out
    

## Model based on ResNet50, adapted to 1D convolutions made with help from this paper and ChatGPT.
##  https://medium.com/@nitishkundu1993/exploring-resnet50-an-in-depth-look-at-the-model-architecture-and-code-implementation-d8d8fa67e46f
    
class ResNet50_1D(nn.Module):
    def __init__(self, num_classes=2, in_channels=10):
        block = Bottleneck1D
        block_sizes = [3, 4, 6, 3]
        super(ResNet50_1D, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, block_sizes[0])
        self.layer2 = self._make_layer(block, 128, block_sizes[1], stride=2)
        self.layer3 = self._make_layer(block, 256, block_sizes[2], stride=2)
        self.layer4 = self._make_layer(block, 512, block_sizes[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)


    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        # Projection shortcut for when dimensions increase with kernel 1x1 stride 2
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm1d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        layer1 = self.conv1(x) # 7x7 with 64 out channels sride 2
        layer1 = self.bn1(layer1)
        layer1 = self.relu(layer1)
        layer1 = self.maxpool(layer1)

        block1 = self.layer1(layer1) # 3 blocks with 64 out channels
        block2 = self.layer2(block1) # 4 blocks with 128 out channels
        block3 = self.layer3(block2) # 6 blocks with 256 out channels
        block4 = self.layer4(block3) # 3 blocks with 512 out channels

        result = self.avgpool(block4)
        result = torch.flatten(result, 1)
        result = self.fc(result)

        return result


def ensemble_model(participant, model):
    """
    Train an ensemble model for a specific participant using pre-trained models on eye, pupil and audio features.
    
    :param participant: The participant to train the model for
    :param model: type of model to use, either ResNet50_1D or AlexNet1D

    :return: None, but the trained ensemble model is saved for the participant.
    """
    if model == ResNet50_1D:
        model_name = "ResNet50_1D"
    else:
        model_name = "AlexNet1D"
    # Load the models of the participant
    with torch.no_grad():
        # Load model trained on eye features
        model_eye = model(num_classes=2, in_channels=7)  
        state_dict = torch.load(f"Participant_data\Participant_{participant}\Model_eye\{model_name}_epoch_50.pth", map_location=torch.device('cpu'))
        model_eye.load_state_dict(state_dict)
        model_eye.eval()
        # Load model trained on pupil features
        model_pupil = model(num_classes=2, in_channels=6)
        state_dict = torch.load(f"Participant_data\Participant_{participant}\Model_pupil\{model_name}_epoch_50.pth",map_location=torch.device('cpu'))
        model_pupil.load_state_dict(state_dict)
        model_pupil.eval()
        # Load model trained on audio features
        model_audio = model(num_classes=2, in_channels=78)
        state_dict = torch.load(f"Participant_data\\Participant_{participant}\\Model_audio\\{model_name}_epoch_50.pth",map_location=torch.device('cpu'))
        model_audio.load_state_dict(state_dict) 
        model_audio.eval()

    # Get the different kinds of data of participants
    _, train_eye, dev_eye,_, y_train, y_dev,_ = get_data(f"Participant_data\\Participant_{participant}\\all_eye_features_{participant}", f"Participant_data\Participant_{participant}\labels_{participant}.csv", 7,model=model_name[:3])
    _, train_pupil, dev_pupil,_,_, _, _ = get_data(f"Participant_data\\Participant_{participant}\\all_pupil_features_{participant}", f"Participant_data\Participant_{participant}\labels_{participant}.csv",6,model=model_name[:3])
    _, train_audio, dev_audio,_,_, _, _  = get_data(f"Participant_data\\Participant_{participant}\\all_audio_features_{participant}", f"Participant_data\Participant_{participant}\labels_{participant}.csv",78,model=model_name[:3])

    # Get the outputs of the different models
    outputs_eye = model_eye(train_eye)
    outputs_pupil = model_pupil(train_pupil)
    outputs_audio = model_audio(train_audio)
    # Also on dev set for validation
    outputs_eye_dev = model_eye(dev_eye)
    outputs_pupil_dev = model_pupil(dev_pupil)
    outputs_audio_dev = model_audio(dev_audio)
    # Combine the outputs of the different models
    combined_dev = torch.stack((outputs_eye_dev, outputs_pupil_dev, outputs_audio_dev), dim=1)
    combined_outputs = torch.stack((outputs_eye, outputs_pupil, outputs_audio), dim=1)
    combined_outputs = torch.nan_to_num(combined_outputs, nan=0)
    combined_dev = torch.nan_to_num(combined_dev, nan=0)
    # Train MLP classifier on the combined outputs
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(5, 2), random_state=13)
    # Get best classifier
    parameter_space = {'hidden_layer_sizes': [(100,50,2),(50,20,2),(20,2)], 'activation': ['tanh', 'relu'],'solver': ['lbfgs','sgd', 'adam'],'alpha': [0.00001, 0.0001, 0.01],'learning_rate': ['constant','adaptive']}
    combined_outputs = combined_outputs.reshape(combined_outputs.shape[0], -1).detach().numpy()
    combined_dev = combined_dev.reshape(combined_dev.shape[0], -1).detach().numpy()
    best_classifier = GridSearchCV(clf, parameter_space)
    best_classifier.fit(combined_outputs, y_train)
    predicted_train = best_classifier.predict(combined_outputs)
    predicted_dev = best_classifier.predict(combined_dev)
    print("estimator", best_classifier.best_estimator_)
    # Save the model and its performance
    pkl.dump(best_classifier, open(f"Participant_data\\Participant_{participant}\\ensemble_model_{model_name}.pkl", 'wb'))
    with open(f"Participant_data\\Participant_{participant}\\ensemble_{model_name}_performance.txt", "w") as f:
        f.write(f"Training accuracy: {accuracy_score(y_train, predicted_train)}\n")
        f.write(f"Dev accuracy: {accuracy_score(y_dev, predicted_dev)}\n")
        f.write(f"f1 score on train: {f1_score(y_train, predicted_train)}\n")
        f.write(f"f1 score on dev: {f1_score(y_dev, predicted_dev)}\n")
    print("Ensemble model trained and saved.")

def ensemble_model_on_all(model):
    """
    train an ensemble model for all participants using pre-trained models on eye, pupil and audio features.
    
    :param model: Type of model to use, either ResNet50_1D or AlexNet1D

    :train: None, but the trained ensemble model is saved.
    """
    # Load the models of the participant
    with torch.no_grad():
        # Load model trained on eye features
        if model == ResNet50_1D:
            model_name = "ResNet50_1D"
        else:
            model_name = "AlexNet1D"

        # load model trained on eye features
        model_eye = model(num_classes=2, in_channels=7)  
        state_dict = torch.load(f"Model_eye\{model_name}_final_epoch_50.pth",map_location=torch.device('cpu') )
        model_eye.load_state_dict(state_dict)
        model_eye.eval()
        # Load model trained on pupil features
        model_pupil = model(num_classes=2, in_channels=6)
        state_dict = torch.load(f"Model_pupil\{model_name}_final_epoch_50.pth",map_location=torch.device('cpu') )
        model_pupil.load_state_dict(state_dict)
        model_pupil.eval()
        # Load model trained on audio features
        model_audio = model(num_classes=2, in_channels=78)
        state_dict = torch.load(f"Model_audio\\{model_name}_final_epoch_50.pth",map_location=torch.device('cpu') )
        model_audio.load_state_dict(state_dict) 
        model_audio.eval()

    # Get the different kinds of data of participants
    _, train_eye, dev_eye, _, y_train, y_dev, _ = get_all_data("all_eye_features", True,size=7 ,model=model_name[:3])
    _, train_pupil, dev_pupil, _, _, _, _ = get_all_data("all_pupil_features",True,size=6 ,model=model_name[:3])
    _, train_audio, dev_audio, _, _, _ ,_  = get_all_data("all_audio_features", True,size=78,model=model_name[:3])
    # Get the outputs of the different models
    outputs_eye = model_eye(train_eye)
    outputs_pupil = model_pupil(train_pupil)
    outputs_audio = model_audio(train_audio)
    # Also on dev set for validation
    outputs_eye_dev = model_eye(dev_eye)
    outputs_pupil_dev = model_pupil(dev_pupil)
    outputs_audio_dev = model_audio(dev_audio)
    # Combine the outputs of the different models
    combined_dev = torch.stack((outputs_eye_dev, outputs_pupil_dev, outputs_audio_dev), dim=1)
    combined_outputs = torch.stack((outputs_eye, outputs_pupil, outputs_audio), dim=1)

    # Train MLP classifier on the combined outputs
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(5, 2), random_state=1)
    parameter_space = {'hidden_layer_sizes': [(20,2)], 'activation': ['tanh', 'relu'],'solver': ['lbfgs','sgd', 'adam'],'alpha': [0.00001, 0.0001, 0.01],'learning_rate': ['constant','adaptive']}
    combined_outputs = combined_outputs.reshape(combined_outputs.shape[0], -1).detach().numpy()
    combined_dev = combined_dev.reshape(combined_dev.shape[0], -1).detach().numpy()
    best_classifier = GridSearchCV(clf, parameter_space)
    best_classifier.fit(combined_outputs, y_train)
    predicted_train = best_classifier.predict(combined_outputs)
    predicted_dev = best_classifier.predict(combined_dev)
    print("estimator", best_classifier.best_estimator_)
    predicted_train = best_classifier.predict(combined_outputs)
    predicted_dev = best_classifier.predict(combined_dev)
    # Save the model and its performance
    pkl.dump(best_classifier, open(f"ensemble_model_{model_name}.pkl", 'wb'))
    with open(f"ensemble_model_{model_name}_performance.txt", "w") as f:
        f.write(f"Training accuracy: {accuracy_score(y_train, predicted_train)}\n")
        f.write(f"Dev accuracy: {accuracy_score(y_dev, predicted_dev)}\n")
        f.write(f"f1 score on train: {f1_score(y_train, predicted_train)}\n")
        f.write(f"f1 score on dev: {f1_score(y_dev, predicted_dev)}\n")
    print("Ensemble model trained and saved.")


def train_model_on_all(path, model, epoch, features, optimizer, lr_rate, decay, corrected= False,channels=85, lr_decrease=False, per_task=0):
    """
    Collect data and train model given chosen hyperparameters.

    :param path: where to save the trained model.
    :param model: the model to train.
    :param epoch: amount of epochs to train the model for.
    :param features: featureset to train on, either "all_features", "all_eye_features", "all_pupil_features", "all_audio_features" or "all_audio_pupil_features"
    :param optimizer: the chosen optimizer, a hyperparameter.
    :param lr_rate: the chosen learning rate, a hyperparameter.
    :param decay: the chosen decay, a hyperparameter.
    :param corrected: whether to use labels corrected for guessing or not.
    :param learning_decrease: whether to use a learning rate schedule or not.

    :return: none, but it saves the model in the provided path.
    """
    name_model = str(model).split("(")[0]
    name_model_data = name_model[:3]
    # Get data
    if per_task > 0:
        train_data, train, dev, _, y_train, y_dev,_ = get_data_per_task(features,per_task, corrected)
    else:
        train_data, train, dev, _, y_train, y_dev,_ = get_all_data(features, corrected,channels,model="Akex" )
    optimizer = optimizer(model.parameters(), lr=lr_rate, weight_decay=decay)
    # Train the model
    training_model(path, model, train_data, epoch, f"{name_model}_epoch_{epoch}", optimizer, train, y_train, dev, y_dev, lr_decrease, per_task)
    # Save the trained model
    if per_task == 0:
        torch.save(model.state_dict(), f'{path}\\{name_model}_final_epoch_{epoch}.pth')
    else:
        torch.save(model.state_dict(), f'{path}\\{name_model}_final_epoch_{epoch}_task_{per_task}.pth')

def train_model(path, model, epoch, x_folder, y_file, optimizer, lr_rate, decay, corrected = False, lr_decrease=False):
    """
    Collect data and train model given chosen hyperparameters.

    :param path: where to save the trained model.
    :param model: the model to train.
    :param epoch: amount of epochs to train the model for.
    :param x_folder: folder to load data from.
    :param y_file: file to load the labels from for the data.
    :param optimizer: the chosen optimizer, a hyperparameter.
    :param lr_rate: the chosen learning rate, a hyperparameter.
    :param decay: the chosen decay, a hyperparameter.
    :param corrected: whether to use labels corrected for guessing or not.
    :param learning_decrease: whether to use a learning rate schedule or not.

    :return: none, but it saves the model in the provided path.
    """
    # Get data
    train_data, train, dev,_, y_train, y_dev,_ = get_data(x_folder, y_file)
    name_model = str(model).split("(")[0]
    optimizer = optimizer(model.parameters(), lr=lr_rate, weight_decay=decay)
    # Train the model
    training_model(path, model, train_data, epoch, f"{name_model}_epoch_{epoch}", optimizer, train.to(device), y_train.to(device), dev.to(device), y_dev.to(device), lr_decrease)
    # Save the trained model
    torch.save(model.state_dict(), f'{path}\\{name_model}_epoch_{epoch}.pth')

def hyperparameter_search(model_class, channels, train_data, train_labels, epoch, lr_decrease=False, k=5):
    """
    Training a model with different hyperparameters and testing there performance with k-fold cross validation

    :param model_class: The type of model to train.
    :param channels: amount of channels in the chosen featureset to train on.
    :param train_data: the data to train on.
    :param train_labels: the labels corresponding to the data.
    :param epochs: the number of epochs to train the model for.
    :param lr_decrease: whether to use a learning rate schedule or not.
    :param k: the number of folds to use for the k-fold corssvalidation.

    :return: None, but the average performance of every hyperparameters over the five folds is printed.
    """
    best = [0,[]]
    # Hyperparameters to evaluate
    optimizers = [torch.optim.Adam,torch.optim.Adadelta,torch.optim.SGD ]
    lr = [0.001, 0.01, 0.1]
    weight_decay = [1.e-2, 1.e-4]
    # Iterate over every combination of hyperparammeters
    for opt in optimizers:
        for rate in lr:
            for decay in weight_decay:
                # Create batches
                data = TensorDataset(train_data, train_labels)
                kf = KFold(n_splits=5, shuffle=True)
                kf.get_n_splits(data)
                fold_results = {
                    "train_loss": [],
                    "train_acc":[],
                    "val_loss": [],
                    "val_acc": [],
                    "train_f1": [],
                    "val_f1": []
                }  # Store performance of each 
                
                # Iterate over all folds
                for idx, (train_fold, dev_fold) in enumerate(kf.split(data)):     
                    # Initialize the correct model and optimizer
                    if model_class == "Alex":
                        model = AlexNet1D(in_channels=channels).to(device)
                    elif model_class == "Res":
                        model = ResNet50_1D(in_channels=channels).to(device)
                    else:
                        print("error")
                    optimizer = opt(model.parameters(), lr=rate,weight_decay=decay)
                    exists = False
                    loss_function = nn.CrossEntropyLoss()
                    # Get data from folds 
                    dev_fold = [(train_data[i], train_labels[i]) for i in dev_fold]
                    train_fold = [(train_data[i], train_labels[i]) for i in train_fold]
                    # Split data in x and y
                    train_x = torch.stack([item[0] for item in train_fold])
                    train_scores = torch.stack([item[1] for item in train_fold])
                    dev_x = torch.stack([item[0] for item in dev_fold])
                    dev_scores = torch.stack([item[1] for item in dev_fold])
                    # Create batches
                    batches_kfold = DataLoader(train_fold, batch_size=50, shuffle=True)
                    # Initialize lists to store performance
                    fold_losses_train = []
                    fold_losses_dev = []
                    fold_accuracies_train = []
                    fold_accuracies_dev = []
                    
                    # Do multiple epochs
                    for e in range(epoch):    
                        model.train()
                        # Iterate over batches
                        for batch in batches_kfold:  
                            x = batch[0]
                            y = batch[1]
                            optimizer.zero_grad()
                            outputs = model(x)
                            loss = loss_function(outputs, y)
                            loss.backward()
                            optimizer.step()
                        model.eval()
                        with torch.no_grad():
                            # Calculate accuracy and performance on train and dev set
                            outputs = model(train_x)
                            predicted_train = torch.argmax(outputs, axis=1)
                            train_loss = loss_function(outputs, train_scores).item()
                            train_acc = (predicted_train == train_scores).sum().item() / len(train_scores)
                            train_f1 = f1_score(train_scores.to("cpu"), predicted_train.to("cpu"))

                            outputs_val = model(dev_x)
                            predicted_val = torch.argmax(outputs_val, axis=1)
                            val_loss = loss_function(outputs_val, dev_scores).item()
                            val_acc = (predicted_val == dev_scores).sum().item() / len(dev_scores)
                            val_f1 = f1_score(dev_scores.to("cpu"), predicted_val.to("cpu"))
                            fold_losses_train.append(train_loss)
                            fold_accuracies_train.append(train_acc)
                            fold_losses_dev.append(val_loss)
                            fold_accuracies_dev.append(val_acc)
                        try:
                            if fold_losses_dev[-1] > fold_losses_dev[-2] and fold_losses_dev[-2] > fold_losses_dev[-3] and fold_losses_train[-1] < fold_losses_train[-2] and fold_losses_train[-2] < fold_losses_train[-3]:
                                print(fold_accuracies_dev)
                                break   
                        except:
                            pass 
                        if lr_decrease:
                            if (e + 1) % 5 == 0:
                                for param_group in optimizer.param_groups:
                                    param_group['lr'] *= 0.5
                        # print(f"Epoch {e+1}/{epoch}")
                        # print(f"Training - Loss: {fold_losses_train[-1]:.4f}, Accuracy: {fold_accuracies_train[-1]:.4f}")
                        # print(f"Dev - Loss: {fold_losses_dev[-1]:.4f}, Accuracy: {fold_accuracies_dev[-1]:.4f}")
        
                    # Store and print final accuracy and loss of a fold
                    fold_results["train_loss"].append(fold_losses_train[-1])
                    fold_results["train_acc"].append(fold_accuracies_train[-1])     
                    fold_results["val_loss"].append(fold_losses_dev[-1])
                    fold_results["val_acc"].append(fold_accuracies_dev[-1])
                    fold_results["train_f1"].append(train_f1)
                    fold_results["val_f1"].append(val_f1)
                # Print the average performance over the folds
                print("hyperparameters: lr:{}, opt:{}, decay{}".format(rate, opt, decay) +"Train loss, train accuracy, validation loss, validation accuracy") 
                print(statistics.mean(fold_results["train_loss"]), statistics.mean(fold_results["train_acc"]), statistics.mean(fold_results["val_loss"]), statistics.mean(fold_results["val_acc"]))
                average_f1 = statistics.mean(fold_results["val_f1"])
                if average_f1 > best[0]:
                    best[0] = average_f1
                    best[1] = [opt, rate, decay]
    print("hyperparamter search done.")
    return best