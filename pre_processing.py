import opensmile
from pydub import AudioSegment
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import whisper
import scipy.signal as signal
import soundfile as sf 
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import shutil

class Data:
    # Function to pre-process and extract features from audio and eye-tracking data
    def __init__(self, participant_number, audio_folder, eye_folder, stimulus_folder):
        """
        Docstring for __init__
        
        :param self: Data object
        :param participant_number: The participant to process data for
        :param audio_folder: where the audio files are located
        :param eye_folder: where the eye-tracking files are located from tobi 
        :param stimulus_folder: where the excel file with data from psychopy can be found.
        """
        # Define all stimuli for later use
        self.available_stimuli = ["hydrant","lieta","geres","vuilnis","varen","dorsale","wetten","nautisch","kuimop","revers","conisch","trompet","noelie","feusut","handschoen","trekken","drinken","velours","gourmand","timpaan","springen","slopen","dukdalf","lubuf","danes","joefeum","vliegtuig","lopen","ijlen","emmer","wosel","ranom","hiewam","schildpad","weugof","hoornvlies","convex","trommel", "sieboe"]
        self.difficult_stimuli = ["hydrant", "dorsaal", "dorsale", "wetten", "nautisch", "revers", "conisch", "velours", "gourmand", "timpaan", "dukdalf", "ijlen", "hoornvlies", "convex"]
        self.easy_stimuli = ["vuilnis", "varen", "trompet", "handschoen", "trekken", "drinken", "springen", "slopen", "vliegtuig", "lopen", "emmer", "schildpad", "trommel"]
        # Initialize variables to store extracted and processed features
        self.all_audio_features = None
        self.all_pupil_features = None
        self.all_eye_features = None
        self.all_audio_pupil_features = None
        self.participant_number = participant_number
        labelaar = Label(self.participant_number)
        # Get labels for the data from this participant. Stored in file but also in self.label.
        if not os.path.exists(f"Participant_data\\Participant_{participant_number}\\labels_{self.participant_number}.csv"):
            print(f"C:\\Users\\lisaa\\Documents\\GitHub\\Models_thesis\\Models_thesis\\Participant_data\\labels_{self.participant_number}.csv")
            label = labelaar.get_labels(audio_folder,stimulus_folder)
            self.label = label.reset_index(drop=True)
        else:
            self.label = labelaar.read_labels(f"Participant_data\\Participant_{participant_number}\\labels_{self.participant_number}.csv")
        # Store folder paths
        self.audio_folder = audio_folder
        self.stimulus_folder = stimulus_folder
        self.eye_folder = eye_folder


    def get_audio_features(self, audio_path,stimulus_path, task1 =False):
        """
        Extract audio features from files in the specified directory using the opensmile library.
        :param audio_path: Path to the directory containing audio files.
        :param stimulus_path: Path to the stimulus file.
        :param task1: Boolean indicating if this is task 1. If True, padding is added to the audio features because audio starts later than pupil features.

        :return: DataFrame containing extracted audio features for each audio file.
        """
        # Get all stimuli
        try:
            stimuli = pd.read_csv(stimulus_path, delimiter=",")["Stimuli"][7:].reset_index(drop=True)
        except:
            stimuli = self.sentence_to_stimulus(stimulus_path)
        # initialize DataFrame to store audio features
        audio_features = pd.DataFrame(columns = ["stimuli", "values"] )
        counter = 0
        # Process all audio files corresponding to a stimulus
        for dirpath, _, filenames in os.walk(audio_path):
            for idx, filename in enumerate(filenames[4:]):
                filepath = os.path.join(dirpath, filename)
                # Opensmile extracts the features from the audiofile
                smile = opensmile.Smile(
                    feature_set=opensmile.FeatureSet.eGeMAPSv02,
                    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
                )
                data = smile.process_file(filepath)
                data.reset_index(inplace=True)
                data = data.drop(columns=["file", "end", "start"])
                # add padding if necessary
                if task1:
                    padding = pd.DataFrame(0, index=range(312), columns= data.columns)
                    data = pd.concat([padding, data], ignore_index=True)
                if stimuli[counter] in self.difficult_stimuli:
                    cat = 1
                elif stimuli[counter] in self.easy_stimuli:
                    cat = 0
                else:
                    cat = -1
                length, _ = data.shape
                # Add category feature to the audio featureset
                cat = pd.DataFrame({"category": [cat]* length})
                data = pd.concat([data, cat], axis = 1)
                audio_features.loc[counter] = [stimuli[counter], data]
                counter +=1
        # Sort the features so all collected features are alfhabetically ordered based on stimulus name.
        sorted_audio_features= audio_features.sort_values(by=["stimuli"]).reset_index(drop=True)
        return sorted_audio_features

    def relevant_part_audio(self, audio_path):
        """
        For task 1, only the relevant part of the audio is used. The first 5200 milliseconds are cut off because the participant starts speaking after that.
                
        :param self: Data object
        :param audio_path: folder containing the audio files

        :return: location where the relevant audio files are stored
        """
        os.makedirs(f"Participant_data\\Participant_{self.participant_number}\\audio_files_{self.participant_number}", exist_ok = True)
        save_loc = f"Participant_data\\Participant_{self.participant_number}\\audio_files_{self.participant_number}"
        # Iterate over all audiofiles and cut off the first 5200 milliseconds
        for dirpath, _, filenames in os.walk(audio_path):
            for idx, filename in enumerate(filenames):
                filepath = os.path.join(dirpath, filename)
                audio = AudioSegment.from_file(file=filepath)
                audio = audio[5200:]
                audio.export(out_f = f"{save_loc}\\{filename}")
        return save_loc
    
    def get_all_audio_features(self):
        """
        Combine audio features from task 1 and task 2 and 3 into a single DataFrame.
        Each row corresponds to a stimulus, with all features from both tasks concatenated.

        :param self: Data object

        :return: DataFrame containing combined audio features for all stimuli.
        """
        # Initialize lists to store data and audio paths for each task
        data_per_task = [None, None, None]
        audio_per_task = [None, None, None]

        # Get audio files for each task
        for dirpath, _, _ in os.walk(self.audio_folder):
            if dirpath == self.audio_folder:
                continue
            if "1_recorded" in os.fspath(dirpath):
                audio_per_task[0] = dirpath
            elif "2_recorded" in os.fspath(dirpath):
                audio_per_task[1] = dirpath
            elif "3_recorded" in os.fspath(dirpath):
                audio_per_task[2] = dirpath

        # Get stimulus files for each task and extract audio features
        for dirpath, _, filenames in os.walk(self.stimulus_folder):
            for file in filenames:
                file = os.path.join(dirpath, file)
                if "3_Experiment" in os.fspath(file):
                    data_per_task[2] = self.get_audio_features(audio_per_task[2], file)
                elif "2_Experiment" in os.fspath(file):
                    data_per_task[1] = self.get_audio_features(audio_per_task[1], file)
                else:
                    if not os.path.isdir(f"Participant_data\\Participant_{self.participant_number}\\audio_files_{self.participant_number}"):
                        loc = self.relevant_part_audio(audio_per_task[0])
                    else:
                        loc = f"Participant_data\\Participant_{self.participant_number}\\audio_files_{self.participant_number}"
                    data_per_task[0] = self.get_audio_features(loc, file, task1=True)
        print("data per task klaar")
        # Merge task1 & task2 by stimulus
        merged_features = pd.DataFrame(columns=["stimuli", "values"])
        for idx,stim in enumerate(data_per_task[0]["stimuli"]):
            # find the corresponding row in both dataframes
            task1_row = data_per_task[0].loc[data_per_task[0]["stimuli"] == stim, "values"].values[0].copy()
            task2_row = data_per_task[1].loc[data_per_task[1]["stimuli"] == stim, "values"].values[0].copy()
            print(task2_row)
            if stim in data_per_task[2]["stimuli"].values:
                task3_row = data_per_task[2].loc[data_per_task[2]["stimuli"] == stim, "values"].values[0].copy()
            else:
                task3_row =  pd.DataFrame(0, index=range(len(task1_row["Loudness_sma3"])), columns= task1_row.columns)
            
            task1_row = task1_row.reset_index(drop=True)
            task2_row = task2_row.reset_index(drop=True)
            task3_row = task3_row.reset_index(drop=True)

            # rename columns so we know which task they belong to
            task1_row = task1_row.add_suffix("_task1")
            task2_row = task2_row.add_suffix("_task2")
            task3_row = task3_row.add_suffix("_task3")

            # concatenate horizontally
            combined = pd.concat([task1_row, task2_row], axis=1)
            combined = pd.concat([combined.reset_index(drop=True), task3_row], axis=1).reset_index(drop=True)
            combined = combined.drop(columns=["category_task1", "category_task2"])
            merged_features.loc[idx] = [stim, combined]

        # Build final dataframe
        combined = pd.concat([merged_features, self.label], axis=1)
        self.all_audio_features = combined
        # Save the audio features for each stimulus separately
        os.makedirs(f"Participant_data\\Participant_{self.participant_number}\\all_audio_Features_{self.participant_number}", exist_ok = True)
        for idx, stimuli in self.all_audio_features.iterrows():
            stim = stimuli['stimuli']
            self.all_audio_features.loc[idx,"values"].to_csv(f"Participant_data\\Participant_{self.participant_number}\\all_audio_Features_{self.participant_number}\\{self.participant_number}_{stim}_audio_feautres.csv", index=False)
        return self.all_audio_features

    def sentence_to_stimulus(self, stimulus_file):
        """
        Convert sentences in a stimulus file to a list of stimuli. In task 3 data is collected based on sentences containing the stimuli, instead of only the stimuli.
        
        :param object self: Data object
        :param stimulus_file: Path to the stimulus file.

        :return: List of stimuli extracted from the file.
        """
        stimuli = []
        sentences = pd.read_csv(stimulus_file, delimiter=",")["Sentences"][7:]
        # Iterate over all sentences
        for sentence in sentences:
            # Iterate over all words in the sentence and find the stimulus
            for word in self.available_stimuli:
                if word in sentence:
                    stimuli.append(word)
                    break
        return pd.Series(stimuli)
    

    def get_eye_features(self, eye_path, stimulus_path):
        """
        Extract pupil data corresponding to specific stimuli based on timestamps from the file from psychopy and the eye-tracking data.
        
        :param pupil_path: Path to the file containing pupil data.
        :param stimulus_path: Path to the file containing stimulus timing information.

        :return: List of DataFrames, each containing pupil data for a specific stimulus.
        """
        # Get data from files
        eye_data = pd.read_csv(eye_path, delimiter="\t")  
        eye_data = eye_data[["Recording timestamp", "Gaze point X","Gaze point Y"]]
        stimuli = pd.read_csv(stimulus_path, delimiter=",")[7:].reset_index(drop=True)  
        if "Stimuli" not in stimuli.columns:
            stimuli_form_sentence = self.sentence_to_stimulus(stimulus_path)
            stimuli = pd.concat([pd.DataFrame({"Stimuli": stimuli_form_sentence}), stimuli], axis=1).reset_index(drop=True)
        stimulus_time = stimuli[["Stimuli","stimulus.started", "stimulus.stopped"]]
        counter = 0
        datapoints = pd.DataFrame(columns = ["stimuli", "values"] )
        total = len(stimuli)
        # Iterate over all stimuli and extract corresponding eye-tracking data
        for rij in range(0,total):
            word = pd.DataFrame(columns=eye_data.columns)
            for index, row in eye_data.iterrows():
                    # Check if the timestamp falls within the stimulus time window
                    if (row["Recording timestamp"] /1000000) >= stimulus_time["stimulus.started"][rij] and  (row["Recording timestamp"] /1000000) <= stimulus_time["stimulus.stopped"][rij]:                    
                        word.loc[counter] = row
                        counter += 1
                    elif (row["Recording timestamp"]/1000000) > stimulus_time["stimulus.stopped"][rij]:
                        break
            for elem in ["Recording timestamp", "Gaze point X","Gaze point Y"]:
                word[elem] = word[elem].apply(self.string_to_float)
            # Add category feature
            if stimulus_time["Stimuli"][rij] in self.difficult_stimuli:
                cat = 1
            elif stimulus_time["Stimuli"][rij] in self.easy_stimuli:
                cat = 0
            else:
                cat = -1
            length, _ = word.shape
            cat = pd.DataFrame({"category": [cat]* length}).reset_index(drop=True)
            word = pd.concat([word.reset_index(drop=True), cat], axis = 1).reset_index(drop=True)
            datapoints.loc[rij] = [stimuli["Stimuli"][rij], word]
        # Sort the result based on stimulus name
        sorted_result = datapoints.sort_values(by=["stimuli"]).reset_index(drop=True)
        return sorted_result
    
    def string_to_float(self, s):
        """
        Convert a string representation of a float with a comma as decimal separator to a float.
        Example: "3,14" -> 3.14

        :param self: data object
        :param s: String representation of a float.

        :return : Float value.
        """
        if isinstance(s, str):
            return float(s.replace(',', '.'))
        return s

    def get_all_eye_features(self):
        """
        Create a list of all eye-tracking features from task two and three as no eye-tracking data is collected during task 1. The function takes the folders containing right files and finds these files based on their name.
        
        :param self: Data object

        :return: A list of DataFrames, each containing eye-tracking data for a specific stimulus. Only from task 2 and 3 as no eye-tracking data is collected during task 1.
        """
        # Initialize lists to store data and eye paths for each task
        data_per_task = [None,None]
        eye_per_task = [None,None]
        # Get eye-tracking files for each task
        for dirpath, _, filenames in os.walk(self.eye_folder):
            for filename in filenames:
                file = os.path.join(dirpath, filename)
                if "(1)" in os.fspath(file):
                    eye_per_task[0] = file
                elif "(2)" in os.fspath(file):
                    eye_per_task[1] = file
        # Get stimulus files for each task and extract eye-tracking features
        for dirpath, _, filenames in os.walk(self.stimulus_folder):
            for file in filenames:
                file = os.path.join(dirpath, file)
                if "1_Experiment" in os.fspath(file):
                    continue
                elif "2_Experiment" in os.fspath(file):
                    data_per_task[0] = self.get_eye_features(eye_per_task[0], file).drop(columns=["stimuli"])
                elif "3_Experiment" in os.fspath(file):
                    task_3 = self.get_eye_features(eye_per_task[1], file)
                    size = len(task_3.loc[0, "values"])
                    columns = task_3.loc[0, "values"].columns
                    extra_rows = pd.DataFrame(columns = columns, data = [[None]*len(columns)]*size)
                    nonwords = pd.DataFrame({"stimuli": ["lieta","geres","kuimop","noelie","feusut","lubuf","danes","joefeum","wosel","ranom","hiewam","weugof","sieboe"], "values": 13* [extra_rows]})
                    task_3 = pd.concat([task_3, nonwords]).sort_values(by=["stimuli"]).reset_index(drop=True)
                    data_per_task[1] = task_3["values"]
        all_eye_features = pd.DataFrame(columns=["eye_data"])
        # Data collected for task two needs an original name before data of two tasks are combined
        for idx, elem in enumerate(data_per_task[0].loc[:, "values"]):
            elem.rename(columns=lambda x: x + '_task2', inplace=True)
            # Remove the timestamp columns since they are not useful and category for task 3 because otherwise there are two category columns
            all_eye_features.loc[idx] = [pd.concat([elem.drop(columns=["Recording timestamp_task2"]).reset_index(drop=True) , data_per_task[1].loc[idx].drop(columns=["Recording timestamp", "category"]).reset_index(drop=True)], axis=1)]
        # Combine data from the two tasks collecting eye data
        combined = pd.concat([all_eye_features, self.label], axis=1).reset_index(drop=True)
        stimuli = pd.DataFrame({"stimuli": self.available_stimuli}).sort_values(by = "stimuli").reset_index(drop=True)
        combined = pd.concat([stimuli, combined], axis=1)
        self.all_eye_features = combined
        # Save the eye features for each stimulus separately
        os.makedirs(f"Participant_data\\Participant_{self.participant_number}\\all_eye_features_{self.participant_number}", exist_ok = True)
        for idx, stimuli in self.all_eye_features.iterrows():
            stim = stimuli['stimuli']
            self.all_eye_features.loc[idx,"eye_data"].to_csv(f"Participant_data\\Participant_{self.participant_number}\\all_eye_Features_{self.participant_number}\\{self.participant_number}_{stim}_eye_feautres.csv", index=False)
        return self.all_eye_features
    
    
    def get_pupil_features(self, pupil_path, stimulus_path):
        """
        Extract pupil data corresponding to specific stimuli based on timestamps. This is only done for task 1.
        
        :param pupil_path: Path to the file containing pupil data.
        :param stimulus_path: Path to the file containing stimulus timing information.

        :return: List of DataFrames, each containing pupil data for a specific stimulus.
        """
        # Get data from files
        eye_data = pd.read_csv(pupil_path, delimiter="\t")  
        eye_data = eye_data[["Recording timestamp","Pupil diameter left","Pupil diameter right","Pupil diameter filtered"]]
        stimuli = pd.read_csv(stimulus_path, delimiter=",")  
        stimulus_time = stimuli[["Stimuli","audio.started","repetition.start","repetition.stopped", "images.started", "images.stopped"]][7:].reset_index(drop=True)
        counter = 0
        result = pd.DataFrame(columns= ["stimuli", "values"])
        # Iterate over all stimuli and extract corresponding pupil data based on timestamps
        for rij in range(0,39):
            word = pd.DataFrame(columns=["Pupil diameter left","Pupil diameter right","Pupil diameter filtered"])
            for index, row in eye_data.iterrows():
                    # Check if the timestamp falls within the stimulus time window
                    if (row["Recording timestamp"] /1000000) >= stimulus_time["audio.started"][rij] and  (row["Recording timestamp"] /1000000) <= stimulus_time["repetition.stopped"][rij] :
                        word.loc[counter] = row.drop("Recording timestamp")
                        counter += 1
                    elif (row["Recording timestamp"]/1000000) > stimulus_time["repetition.stopped"][rij]:
                        break
            for elem in ["Pupil diameter filtered", "Pupil diameter right", "Pupil diameter left"]:
                word[elem] = word[elem].apply(self.string_to_float)
            # Add category feature
            if stimulus_time["Stimuli"][rij] in self.difficult_stimuli:
                cat = 1
            elif stimulus_time["Stimuli"][rij] in self.easy_stimuli:
                cat = 0
            else:
                cat = -1
            length, _ = word.shape
            cat = pd.DataFrame({"category": [cat]* length})
            word = pd.concat([word.reset_index(drop=True), cat.reset_index(drop=True)], axis = 1)
            result.loc[rij] = [stimulus_time["Stimuli"][rij]] +  [word]

        return result
    
    def get_all_pupil_features(self):
        """
        Create a list of all pupil features from task one as no pupil data is collected during task 2 and 3. The function takes the folders containing right files and finds these files based on their name.
    
        :param pupil_folder: Path to the folder containing pupil data files for all tasks.
        :param stimulus_folder: Path to the folder containing stimulus files for all tasks.

        :return:A list of DataFrames, each containing pupil data for a specific stimulus. Only from task 1 as no pupil data is collected during task 2 and 3.
        """
        data_per_task = None
        pupil_per_task = None
        # Get pupil file for task one
        for dirpath, _, filenames in os.walk(self.eye_folder):
            for filename in filenames:
                file = os.path.join(dirpath, filename)
                if "(2)" in os.fspath(file) or "(1)" in os.fspath(file):
                    continue
                else:
                    pupil_per_task = file
        # Get stimulus files for each task and extract pupil features
        for dirpath, _, filenames in os.walk(self.stimulus_folder):
            for file in filenames:
                file = os.path.join(dirpath, file)
                if "3_Experiment" in os.fspath(file) or "2_Experiment" in os.fspath(file):
                    continue
                else:
                    data_per_task = self.get_pupil_features(pupil_per_task, file)
        combined = pd.concat([data_per_task, self.label], axis=1).sort_values(by="stimuli")
        self.all_pupil_features = combined
        # Save the pupil features for each stimulus separately
        os.makedirs(f"Participant_data\\Participant_{self.participant_number}\\all_pupil_features_{self.participant_number}", exist_ok = True)
        for idx, stimuli in self.all_pupil_features.iterrows():
            stim = stimuli['stimuli']
            self.all_pupil_features.loc[idx,"values"].to_csv(f"Participant_data\\Participant_{self.participant_number}\\all_pupil_Features_{self.participant_number}\\{self.participant_number}_{stim}_pupil_feautres.csv", index=False)
        return self.all_pupil_features
        
    def get_audio_and_pupil_features(self):
        """
        Get all audio and pupil features for the participant and combine them into one featureset
        
        :param audio_folder: Path to the folder containing audio files for all tasks.
        :param pupil_folder: Path to the folder containing pupil data files for all tasks.
        :param stimulus_folder: Path to the folder containing stimulus files for all tasks.
        
        :return: DataFrame of all audio features and all pupil features.
        """
        # If audio or pupil features are not yet extracted, extract them
        if self.all_audio_features is None:
            self.get_all_audio_features()
        if self.all_pupil_features is None:
            self.get_all_pupil_features()
        self.all_audio_pupil_features = pd.DataFrame(columns=["pupil_audio_data"])
        merged_features = pd.DataFrame(columns=["stimuli", "values"])
        # Merge audio and pupil features by stimulus
        for idx, stim in enumerate(self.all_pupil_features["stimuli"]):
            # find the corresponding row in both dataframes
            audio_row = self.all_audio_features.loc[self.all_audio_features["stimuli"] == stim, "values"].values[0].copy()
            pupil_row = self.all_pupil_features.loc[self.all_pupil_features["stimuli"] == stim, "values"].values[0].copy()
            
            audio_row = audio_row.reset_index(drop=True)
            pupil_row = pupil_row.reset_index(drop=True)

            # concatenate horizontally
            combined = pd.concat([audio_row, pupil_row], axis=1)
            # drop category column from pupil data to avoid duplicates
            combined = combined.drop(columns =["category_task3"])
            merged_features.loc[idx] = [stim, combined]

        
        combined = pd.concat([merged_features, self.label], axis=1)
        self.all_audio_pupil_features = combined
        # Save the audio and pupil features for each stimulus separately
        os.makedirs(f"Participant_data\\Participant_{self.participant_number}\\all_audio_pupil_features_{self.participant_number}", exist_ok = True)
        for idx, stimuli in self.all_audio_pupil_features.iterrows():
            stim = stimuli['stimuli']
            self.all_audio_pupil_features.loc[idx,"values"].to_csv(f"Participant_data\\Participant_{self.participant_number}\\all_audio_pupil_Features_{self.participant_number}\\{self.participant_number}_{stim}_pupil_feautres.csv", index=False)
        return self.all_audio_pupil_features
    
    def get_all_features(self):
        """
        create a featureset containing all features: audio, pupil and eye-tracking features. 
        This is done by combining if they already exist otherwise extracting them first.
        
        :param self: Data object
        :return: DataFrame of all features: audio, pupil and eye-tracking features.
        """
        # Extract features if not done yet
        if self.all_audio_pupil_features is None:
            self.get_audio_and_pupil_features()
        if self.all_eye_features is None:
            self.get_all_eye_features()
        self.all_features = pd.DataFrame(columns=["all_data"])
        merged_features = pd.DataFrame(columns=["stimuli", "values"])
        # merge audio+pupil features with eye-tracking features by stimulus
        for idx, stim in enumerate(self.all_pupil_features["stimuli"]):
            # find the corresponding row in both dataframes
            audio_pupil_row = self.all_audio_pupil_features.loc[self.all_audio_pupil_features["stimuli"] == stim, "values"].values[0].copy()
            # "dorsaal" is used as "dorsale" in sentences of task 3.
            if stim == "dorsaal":
                stim = "dorsale"
            eye_row = self.all_eye_features.loc[self.all_eye_features["stimuli"] == stim, "eye_data"]

            # reset index zodat concat werkt
            audio_pupil_row = audio_pupil_row.reset_index(drop=True)
            eye_row = eye_row.iloc[0].reset_index(drop=True)
            # drop category column from audio+pupil data to avoid duplicates
            audio_pupil_row = audio_pupil_row.drop(columns = "category")
            new_frame = pd.concat([audio_pupil_row, eye_row], axis=1)
            merged_features.loc[idx] = [stim, new_frame]

        combined = pd.concat([merged_features, self.label], axis=1)
        self.all_features = combined
        # Save the all features for each stimulus separately
        os.makedirs(f"Participant_data\\Participant_{self.participant_number}\\all_features_{self.participant_number}", exist_ok = True)
        for idx, stimuli in self.all_features.iterrows():
            stim = stimuli['stimuli']
            self.all_features.loc[idx,"values"].to_csv(f"Participant_data\\Participant_{self.participant_number}\\all_Features_{self.participant_number}\\{self.participant_number}_{stim}_pupil_feautres.csv", index=False)
        return self.all_features


class Label:
    def __init__(self, participant_number):
        """
        This class is for generating labels for the participant based on their audio responses in task1b.
        
        :param self: Label object
        :param participant_number: Participant to extract labels for.
        """
        self.participant_number = participant_number
        self.labels = None
        self.difficult_stimuli = ["hydrant","dorsaal", "dorsale","wetten","nautisch","revers","conisch","velours","gourmand","timpaan","dukdalf","ijlen","hoornvlies","convex"]



    def get_speech(self, audio_path, stimulus_path):
        """
        Get the transcriptions of the participant's audio responses using the Whisper model.
        
        :param audio_path: Path to the directory containing audio files.
        :param stimulus_path: Path to the file containing stimulus information.

        :return: DataFrame containing the transcriptions for each stimulus.
        """
        # load model and data
        model = whisper.load_model("turbo")
        stimuli = pd.read_csv(stimulus_path, delimiter=",")  
        stimuli = stimuli["Stimuli"].drop(index=[0,1,6]).reset_index(drop=True)
        data = pd.DataFrame(columns=["Stimuli", "output"])
        # Iterate over all 39 features and transcribe the given response on the task
        with tqdm(total=39, desc="Transcribing Files") as pbar:
            for dirpath, dirnames, filenames in os.walk(audio_path):
                for idx, filename in enumerate(filenames):
                    if filename.endswith(".wav"):
                        filepath = os.path.join(dirpath, filename)
                        result = model.transcribe(filepath, language="Dutch", fp16=False)
                        transcription = result['text']
                        data.loc[idx] = [stimuli.loc[idx]] + [transcription]
                        pbar.update(1)
        return data


    def get_labels(self, audio_path, stimulus_path):
        """
        Get the labels for the participant based on their audio responses.
         
        :param audio_path: Path to the directory containing audio files.
        :param stimulus_path: Path to the file containing stimulus information.

        :return: DataFrame containing the labels for each stimulus.
        """
        # Get audio files for each task
        for dirpath, filenames,_ in os.walk(audio_path):
            for filename in filenames:
                file = os.path.join(dirpath, filename)
                if "1_2_recorded" in os.fspath(file):
                    audio = file
        # Get stimulus files for each task and extract eye-tracking features
        for dirpath, _, filenames in os.walk(stimulus_path):
            for file in filenames:
                file = os.path.join(dirpath, file)
                if "1_Experiment" in os.fspath(file):
                    stimulus = file
        # Get response for each stimulus
        data = self.get_speech(audio, stimulus)
        x = data.copy()
        # Correct answers for each stimulus
        correct = [1,3,0,1,3,2,4,0,0,3,2,0,4,2,3,0,0,0,2,0,4,0,0,2,1,0,3,3,3,3,3,4,1,4,1,3,1,0,0]
        x= x[4:].reset_index(drop=True)
        # Convert transcriptions to numerical labels, also when Whisper heart something incorrect but similar.
        for idx, (stimuli, output) in enumerate(x.values):
            # fear and fire sound like vier. When Whisper hears this, the participant always meant 4.
            if "fear" in output.lower() or "vier" in output.lower() or "fire" in output.lower() or "4" in output.lower():
                x.loc[idx, 'output'] = 4
            elif "een" in output.lower() or "in" in output.lower() or "1" in output.lower():
                x.loc[idx, 'output'] = 1
            elif "twee" in output.lower() or "2" in output.lower():
                x.loc[idx, 'output'] = 2
            elif "drie" in output.lower()  or "3" in output.lower():
                x.loc[idx, 'output'] = 3
            else :
                x.loc[idx, 'output'] = 0
        x = x.sort_values(by=['Stimuli']).reset_index(drop=True)
        x = pd.concat([x, pd.DataFrame(data = {"correct" : correct})], axis=1)
        # Determine if the answer is correct or not en save in the new column "correct"
        for idx, (_, output, correct) in enumerate(x.values):
            if output != correct or correct ==0:
                x.loc[idx, "correct"] = 0
            else:
                x.loc[idx,"correct"] = 1
        # save the labels
        data.to_csv(f"Participant_data\\Participant_{self.participant_number}\\speech_labels_{self.participant_number}.csv")
        x.drop(columns=["output", "Stimuli"], inplace=True)
        self.labels = x
        return x
    
    def read_labels(self, file_name):
        """
        Read existing labels from a CSV file.  

        :param self: Label object
        :param file_name: the file containing the labels.
        """
        self.labels = pd.read_csv(file_name)
        return self.labels
    
    def labels_corrected(self):
        """
        Correct labels for difficult stimuli to 0 (incorrect) because these were most likely guessed
        
        :param self: Data Object

        :return: none, but saves corrected labels to a new csv file.
        """
        data = pd.read_csv(f"Participant_data\\Participant_{self.participant_number}\\labels_{self.participant_number}.csv").reset_index(drop=True)
        for i,elem in data.iterrows():
            if elem["Stimuli"] in self.difficult_stimuli:
                data["correct"].loc[i] = 0
        data.to_csv(f"Participant_data\\Participant_{self.participant_number}\\corrected_labels_{self.participant_number}.csv")



class all_data:
    def split_data(participant_number):
        list = [f"all_audio_Features_{participant_number}", f"all_eye_features_{participant_number}", f"all_pupil_features_{participant_number}", f"all_features_{participant_number}", f"all_audio_pupil_features_{participant_number}"]
        elements = [i for i in range(39)]
        y = pd.read_csv(f"Participant_data\\Participant_{participant_number}\\labels_{participant_number}.csv")
        y = y["correct"].values
        train, dev, _, ydev = train_test_split(elements, y, test_size=0.3, random_state=42, stratify=y)
        dev, test, ydev, _ = train_test_split(dev, ydev, test_size=0.5, random_state=42, stratify=ydev)
        for folder in list:
            dir = f"Participant_data\\Participant_{participant_number}"
            print(dir)
            if folder in os.fspath(f"{dir}\\{folder}"):
                if not os.path.isdir(f"{dir}\\{folder}_train"):
                    os.makedirs(f"{dir}\\{folder}_train", exist_ok = True)
                if not os.path.isdir(f"{dir}\\{folder}_dev"):
                    os.makedirs(f"{dir}\\{folder}_dev", exist_ok = True)
                if not os.path.isdir(f"{dir}\\{folder}_test"):
                    os.makedirs(f"{dir}\\{folder}_test", exist_ok = True)
                for i, file in enumerate(os.listdir(f"{dir}\\{folder}")):
                    if i in train:
                        shutil.copyfile(f"{dir}\\{folder}\\{file}", f"{dir}\\{folder}_train\\{file}")
                    elif i in dev:
                        shutil.copyfile(f"{dir}\\{folder}\\{file}", f"{dir}\\{folder}_dev\\{file}")
                    else:
                        shutil.copyfile(f"{dir}\\{folder}\\{file}", f"{dir}\\{folder}_test\\{file}")
                file = pd.read_csv(f"{dir}\\labels_{participant_number}.csv")
                file_train = file.loc[train]
                file_train.to_csv(f"{dir}\\labels_{participant_number}_train.csv", index=False)
                file_test = file.loc[test]
                file_test.to_csv(f"{dir}\\labels_{participant_number}_test.csv", index=False)
                file_dev = file.loc[dev]
                file_dev.to_csv(f"{dir}\\labels_{participant_number}_dev.csv", index=False)

    def add_age(participant_number,age):
        for cat in ["all_features", "all_eye_features", "all_pupil_features","all_audio_features", "all_audio_pupil_features"]:
            for dirpath, _,filenames in os.walk(f"Participant_data\\Participant_{participant_number}\\{cat}_{participant_number}"):
                for filename in filenames:
                    file = os.path.join(dirpath, filename)
                    feature_file = pd.read_csv(file)
                    shape,_ = feature_file.shape
                    feature_file["age"] = [age]*shape
                    feature_file.to_csv(file)

    def add_nt(participant_number, level):
         for cat in ["all_features", "all_eye_features", "all_pupil_features","all_audio_features", "all_audio_pupil_features"]:
            for dirpath, _,filenames in os.walk(f"Participant_data\\Participant_{participant_number}\\{cat}_{participant_number}"):
                for filename in filenames:
                    file = os.path.join(dirpath, filename)
                    feature_file = pd.read_csv(file)
                    shape,_ = feature_file.shape
                    feature_file["nt"] = [level]*shape
                    feature_file.to_csv(file)


class data_analyzis:
    def analyze_repeat(audio_path, stimulus_path, participant_number):
        """
        Get the transcriptions of the participant's audio responses when repeating stimulus in task 1a using the Whisper model.
        
        :param audio_path: Path to the directory containing audio files.
        :param stimulus_path: Path to the file containing stimulus information.
        :param participant_number: Participant to analyze.

        :return: DataFrame containing the transcriptions for each stimulus.
        """
        # get data from task one, when repeating stimulus
        for dirpath, filenames,_ in os.walk(audio_path):
            for filename in filenames:
                file = os.path.join(dirpath, filename)
                if "1_recorded" in os.fspath(file):
                    audio_path = file
                    break
        # Get stimulus files for each task and extract eye-tracking features
        for dirpath, _, filenames in os.walk(stimulus_path):
            for file in filenames:
                file = os.path.join(dirpath, file)
                if "1_Experiment" in os.fspath(file):
                    stimulus_path = file
                    break
        model = whisper.load_model("turbo")
        stimuli = pd.read_csv(stimulus_path, delimiter=",")  
        stimuli = stimuli["Stimuli"].drop(index=[0,1,6]).reset_index(drop=True)
        data = pd.DataFrame(columns=["Stimuli", "output"])
        # Iterate over all 39 features and transcribe the given response on the task
        with tqdm(total=39, desc="Transcribing Files") as pbar:
            for dirpath, dirnames, filenames in os.walk(audio_path):
                for idx, filename in enumerate(filenames):
                    if filename.endswith(".wav"):
                        filepath = os.path.join(dirpath, filename)
                        result = model.transcribe(filepath, language="Dutch", fp16=False)
                        transcription = result['text']
                        data.loc[idx] = [stimuli.loc[idx]] + [transcription]
                        pbar.update(1)
        # Determine if the transcription contains the stimulus word
        check = []
        for elem in data.iterrows():
            if elem[1][0] in elem[1][1] or elem[1][0] in elem[1][1].lower():
                check.append(1)
            else:
                check.append(0)
        data["correct"] = check
        # Save the results
        data.to_csv(f"Participant_data\\Participant_{participant_number}\\repetition_{participant_number}.csv")
        return data
    
    def heatmap_labels(participant_number):
        """
        Create a heatmap for a participant for the eye-tracking data when viewing images in task 1b, same tasks as used to determine labels.
        
        :param participant_number: The participant to create heatmaps for.

        :return: None, but saves heatmaps as images.
        """
        # Get eye-tracking data
        try:
            path = f"Participant_data\\Participant_{participant_number}\\eye\\Experiment_eye_tracking_low_literate image_viewing.tsv"
            data = pd.read_csv(path, delimiter="\t")
        except:
            path = f"Participant_data\\Participant_{participant_number}\\eye\\Experiment_eye_tracking_low_literate image_viewing.csv"
            data = pd.read_csv(path, delimiter="\t")
        data = data[["Recording timestamp", "Gaze point X","Gaze point Y"]]
        # Get information about when images are shown
        for dirpath,dirnames, filenames in os.walk(f"Participant_data\\Participant_{participant_number}\\stimuli"):
            for file in filenames:
                stimulus_file = os.path.join(dirpath, file)
                if "1_Experiment" in os.fspath(file):
                    break
        stimuli = pd.read_csv(stimulus_file, delimiter=",")[7:].reset_index(drop=True)
        stimuli= stimuli[["Stimuli","images.started", "images.stopped"]]
        # Create heatmaps for each stimulus
        for rij in range(len(stimuli)):
            counter = 0
            new_data = pd.DataFrame(columns=["Recording timestamp", "Gaze point X","Gaze point Y"])
            for _, row in data.iterrows():
                if (row["Recording timestamp"] /1000000) >= stimuli["images.started"][rij] and  (row["Recording timestamp"] /1000000) <= stimuli["images.stopped"][rij]:                    
                    new_data.loc[counter] = row
                    counter += 1
                elif (row["Recording timestamp"]/1000000) > stimuli["images.stopped"][rij]:
                    break
            new_data = new_data.dropna()
            # plot the heatmap and save the image
            plt.hist2d(new_data["Gaze point X"], new_data["Gaze point Y"],range=([0,1920],[0,1080]),bins=[50,25])
            ax = plt.gca().invert_yaxis()
            plt.xlabel('Gaze point X')
            plt.ylabel('Gaze point Y')
            os.makedirs(f"Participant_data\\Participant_{participant_number}\\eye_heatmaps", exist_ok=True)
            stimulus =stimuli["Stimuli"].loc[rij]
            plt.savefig(f"Participant_data\\Participant_{participant_number}\\eye_heatmaps\\heatmap_{stimulus}.png")
        print("All heatmaps saved.")
                                        
    def rotate_gaze(data,number):
        """
        Rotate the gaze points based on the orientation of the image shown. We want to transform the data so that the correct response is always on the bottom right.

        :param data: The gaze data to rotate.
        :param number: The orientation of the image. 1 is top left, 2 is top right, 3 is bottom left, 4 is bottom right.
        """
        # Get data
        x = data["Gaze point X"]
        y = data["Gaze point Y"]
        # Rotate based on orientation
        if number == 1:
            x = 1920-x
            y = 1080 - y
        elif number == 2:
            y = 1080 - y
        elif number ==3:
            x = 1920 - x
        data["Gaze point X"] = x
        data["Gaze point Y"] = y
        return data
    
    def heatmap_per_cat():
        """
        Create heatmaps per category (difficult, easy, non-word) for all participants combined based on their eye-tracking data when viewing images in task 1b.

        :return: None, but shows heatmaps as images.
        """
        # initialize the dataframes
        difficult_data = pd.DataFrame(columns=["Recording timestamp", "Gaze point X","Gaze point Y"])
        easy_data = pd.DataFrame(columns=["Recording timestamp", "Gaze point X","Gaze point Y"])
        non_data = pd.DataFrame(columns=["Recording timestamp", "Gaze point X","Gaze point Y"])
        difficult_data_wrong = pd.DataFrame(columns=["Recording timestamp", "Gaze point X","Gaze point Y"])
        easy_data_wrong = pd.DataFrame(columns=["Recording timestamp", "Gaze point X","Gaze point Y"])
        non_data_wrong = pd.DataFrame(columns=["Recording timestamp", "Gaze point X","Gaze point Y"])
        counter_non = 0
        counter_difficult = 0
        counter_easy = 0
        counter_non_wrong = 0
        counter_difficult_wrong = 0
        counter_easy_wrong = 0
        # Iterate over all participants
        for participant_number in [1,11,12,13,14,15,17,18,20,21,22,23,24,25,26,27,28,29,30]:
            # Get all data
            difficult_stimuli = ["hydrant","dorsaal","wetten","nautisch","revers","conisch","velours","gourmand","timpaan","dukdalf","ijlen","hoornvlies","convex"]
            easy_stimuli = ["vuilnis","varen","trompet","handschoen","trekken","drinken","springen","slopen","vliegtuig","lopen","emmer","schildpad","trommel"]
            all_stimuli = pd.Series(["hydrant","lieta","geres","vuilnis","varen","dorsaal","wetten","nautisch","kuimop","revers","conisch","trompet","noelie","feusut","handschoen","trekken","drinken","velours","gourmand","timpaan","springen","slopen","dukdalf","lubuf","danes","joefeum","vliegtuig","lopen","ijlen","emmer","wosel","ranom","hiewam","schildpad","weugof","hoornvlies","convex","trommel","sieboe"]).sort_values()
            correct = [1,3,0,1,3,2,4,0,0,3,2,0,4,2,3,0,0,0,2,0,4,0,0,2,1,0,3,3,3,3,3,4,1,4,1,3,1,0,0]
            labels= pd.read_csv(f"Participant_data\Participant_{participant_number}\labels_{participant_number}.csv")
            images = dict()
            label_correct = dict()
            for i,stimuli in enumerate(all_stimuli):
                images[stimuli] = correct[i]
                label_correct[stimuli] = labels["correct"].loc[i]
            try:
                path = f"Participant_data\\Participant_{participant_number}\\eye\\Experiment_eye_tracking_low_literate image_viewing.tsv"
                data = pd.read_csv(path, delimiter="\t")
            except:
                path = f"Participant_data\\Participant_{participant_number}\\eye\\Experiment_eye_tracking_low_literate image_viewing.csv"
                data = pd.read_csv(path, delimiter="\t")
            data = data[["Recording timestamp", "Gaze point X","Gaze point Y"]]
            # Get information about when images are shown
            for dirpath,dirnames, filenames in os.walk(f"Participant_data\\Participant_{participant_number}\\stimuli"):
                for file in filenames:
                    stimulus_file = os.path.join(dirpath, file)
                    if "1_Experiment" in os.fspath(file):
                        break
            stimuli = pd.read_csv(stimulus_file, delimiter=",")[7:].reset_index(drop=True)
            stimuli= stimuli[["Stimuli","images.started", "images.stopped"]]
            # Collect the data
            for rij in range(len(stimuli)):
                for _, row in data.iterrows():
                    if (row["Recording timestamp"] /1000000) >= stimuli["images.started"][rij] and  (row["Recording timestamp"] /1000000) <= stimuli["images.stopped"][rij]:     
                        if stimuli["Stimuli"].loc[rij] in difficult_stimuli:
                            number  = images[stimuli["Stimuli"].loc[rij]]
                            if label_correct[stimuli["Stimuli"].loc[rij]] == 1:
                                difficult_data.loc[counter_difficult] = data_analyzis.rotate_gaze(row,number)
                                counter_difficult +=1
                            else:
                                difficult_data_wrong.loc[counter_difficult_wrong] = data_analyzis.rotate_gaze(row,number)
                                counter_difficult_wrong +=1
                        elif stimuli["Stimuli"].loc[rij] in easy_stimuli:
                            number  = images[stimuli["Stimuli"].loc[rij]]
                            if label_correct[stimuli["Stimuli"].loc[rij]] == 1:
                                easy_data.loc[counter_easy] = data_analyzis.rotate_gaze(row,number)
                                counter_easy +=1
                            else:
                                easy_data_wrong.loc[counter_easy_wrong] = data_analyzis.rotate_gaze(row,number)
                                counter_easy_wrong +=1
                        else:
                            number  = images[stimuli["Stimuli"].loc[rij]]
                            if label_correct[stimuli["Stimuli"].loc[rij]] == 1:
                                non_data.loc[counter_non] = data_analyzis.rotate_gaze(row,number)
                                counter_non += 1
                            else:
                                non_data_wrong.loc[counter_non_wrong] = data_analyzis.rotate_gaze(row,number)
                                counter_non_wrong += 1
                    elif (row["Recording timestamp"]/1000000) > stimuli["images.stopped"][rij]:
                        break
        # Creating the heatmaps
        # Plot difficult
        difficult_data = difficult_data.dropna()
        plt.hist2d(difficult_data["Gaze point X"], difficult_data["Gaze point Y"],range=([0,1920],[0,1080]),bins=[50,25])
        ax = plt.gca().invert_yaxis()
        plt.xlabel('Gaze point X')
        plt.ylabel('Gaze point Y')
        plt.title("difficult")
        plt.show()

        # Plot easy
        easy_data = easy_data.dropna()
        plt.hist2d(easy_data["Gaze point X"], easy_data["Gaze point Y"],range=([0,1920],[0,1080]),bins=[50,25])
        ax = plt.gca().invert_yaxis()
        plt.xlabel('Gaze point X')
        plt.ylabel('Gaze point Y')
        plt.title("easy")
        plt.show()

        # Plot easy
        non_data = non_data.dropna()
        plt.hist2d(non_data["Gaze point X"], non_data["Gaze point Y"],range=([0,1920],[0,1080]),bins=[50,25])
        ax = plt.gca().invert_yaxis()
        plt.xlabel('Gaze point X')
        plt.ylabel('Gaze point Y')
        plt.title("non")
        plt.show()

        # Plot difficult wrong
        difficult_data_wrong = difficult_data_wrong.dropna()
        plt.hist2d(difficult_data_wrong["Gaze point X"], difficult_data_wrong["Gaze point Y"],range=([0,1920],[0,1080]),bins=[50,25])
        ax = plt.gca().invert_yaxis()
        plt.xlabel('Gaze point X')
        plt.ylabel('Gaze point Y')
        plt.title("difficult wrong")
        plt.show()

        # Plot easy wrong
        easy_data_wrong = easy_data_wrong.dropna()
        plt.hist2d(easy_data_wrong["Gaze point X"], easy_data_wrong["Gaze point Y"],range=([0,1920],[0,1080]),bins=[50,25])
        ax = plt.gca().invert_yaxis()
        plt.xlabel('Gaze point X')
        plt.ylabel('Gaze point Y')
        plt.title("easy wrong")
        plt.show()

        # Plot non wrong
        non_data_wrong = non_data_wrong.dropna()
        plt.hist2d(non_data_wrong["Gaze point X"], non_data_wrong["Gaze point Y"],range=([0,1920],[0,1080]),bins=[50,25])
        ax = plt.gca().invert_yaxis()
        plt.xlabel('Gaze point X')
        plt.ylabel('Gaze point Y')
        plt.title("non wrong")
        plt.show()
      
    def string_to_float(s):
        """
        Convert a string representation of a float with a comma as decimal separator to a float.
        Example: "3,14" -> 3.14

        :param self: data object
        :param s: String representation of a float.

        :return : Float value.
        """
        if isinstance(s, str):
            return float(s.replace(',', '.'))
        return s
    
    def analyze_sentence_reading(audio_path, stimulus_path, participant_number):
        """
        Get the transcriptions of the participant's audio responses using the Whisper model.

        :param audio_path: Path to the directory containing audio files.
        :param stimulus_path: Path to the file containing stimulus information.
        :param participant_number: Number of the participant.

        :return: DataFrame containing the transcriptions for each stimulus.
        """
        # get data from task three, sentence reading
        for dirpath, filenames,_ in os.walk(audio_path):
            for filename in filenames:
                file = os.path.join(dirpath, filename)
                if "3_recorded" in os.fspath(file):
                    audio_path = file
                    break
        # Get stimulus files for each task and extract eye-tracking features
        for dirpath, _, filenames in os.walk(stimulus_path):
            for file in filenames:
                file = os.path.join(dirpath, file)
                if "3_Experiment" in os.fspath(file):
                    stimulus_path = file
                    break
        model = whisper.load_model("turbo")
        stimuli = pd.read_csv(stimulus_path, delimiter=",")  
        stimuli = stimuli["Sentences"].drop(index=[0,1,6]).reset_index(drop=True)
        data = pd.DataFrame(columns=["Stimuli", "output"])
        # Iterate over all 39 features and transcribe the given response on the task
        with tqdm(total=39, desc="Transcribing Files") as pbar:
            for dirpath, dirnames, filenames in os.walk(audio_path):
                for idx, filename in enumerate(filenames):
                    if filename.endswith(".wav"):
                        filepath = os.path.join(dirpath, filename)
                        result = model.transcribe(filepath, language="Dutch", fp16=False)
                        transcription = result['text']
                        data.loc[idx] = [stimuli.loc[idx]] + [transcription]
                        pbar.update(1)
        correct = []
        # Determine if the transcription matches the stimulus sentence
        for i,row in data.iterrows():
            if row["Stimuli"] == row["output"][1:]:
                correct.append(1)
            else:
                correct.append(0)
        data["correct"] = correct
        # Save the results
        data.to_csv(f"Participant_data\\Participant_{participant_number}\\sentence_reading{participant_number}.csv")
        return data