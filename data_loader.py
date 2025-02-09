import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

def load_participants_info(dataset_path):
    '''Load participants information'''
    
    participants_file = os.path.join(dataset_path, "participants.tsv")
    if os.path.exists(participants_file):
        df_participants = pd.read_csv(participants_file, sep='\t')
        # print("Participants Information:")
        # print(df_participants.head())
        return df_participants
    else:
        print("Participants file not found.")
        return None

def load_behavioral_data(dataset_path, participant_id):
    '''Load behavioral and pupil data from a participant's functional folder'''
    
    participant_folder = os.path.join(dataset_path, f"sub-{participant_id}", "func")
    
    if not os.path.exists(participant_folder):
        # print(f"Functional folder not found for participant {participant_id}")
        return None

    data_files = [f for f in os.listdir(participant_folder) if f.endswith(".tsv")]
    
    df_list = []
    for file in data_files:
        file_path = os.path.join(participant_folder, file)
        df = pd.read_csv(file_path, sep='\t')
        df["subject"] = participant_id
        df["task"] = file.split("_")[1]  # Extract task name
        df_list.append(df)
    
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    else:
        print(f"No .tsv files found for participant {participant_id}")
        return None

def load_event_descriptions(dataset_path):
    '''Load event descriptions from JSON file'''
    
    json_file = os.path.join(dataset_path, "task-Overlap_events.json")
    
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            event_descriptions = json.load(f)
            # print("Event Description:")
            # print(json.dumps(event_descriptions, indent=4))
            return event_descriptions
    else:
        print("Event description JSON not found.")
        return None