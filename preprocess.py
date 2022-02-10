import os
import music21 as m21
from pip import main
from music21 import environment
import json
import tensorflow.keras as keras
import numpy as np

import subprocess
subprocess.run("D:\musescore\bin",shell=True)


us = environment.UserSettings()
us["musescoreDirectPNGPath"] = r"D:\musescore\bin\MuseScore3.exe"
us['musicxmlPath'] = r"D:\musescore\bin\MuseScore3.exe"

KERN_DATASET_PATH = r"C:\Users\MYPC\Desktop\computer vision\melody generation\deutschl\essen\europa\deutschl\erk"
ACCEPTABLE_DURATIONS = [
    0.25, 0.5, 0.75, 1.0, 1.5, 2, 3, 4
]
SAVE_DIR = r"C:\Users\MYPC\Desktop\computer vision\melody generation\dataset"

SINGLE_FILE_DATASET = r"C:\Users\MYPC\Desktop\computer vision\melody generation\file_dataset"

SEQUENCE_LENGTH = 64

MAPPING_PATH = r"C:\Users\MYPC\Desktop\computer vision\melody generation\mapping.json"



def load_songs_in_kern(dataset_path):

    songs = []
    # go through all the files in dataset and load them wiht music21
    for path, subdirs, files in os.walk(dataset_path):
        for file in files:
            if file[-3:]=="krn":
                song = m21.converter.parse(os.path.join(path,file))
                songs.append(song)
                
    return songs

def has_acceptable_durations(song,acceptable_durations):
    for note in song.flat.notesAndRests:
        if  note.duration.quarterLength not in acceptable_durations:
            return False
    return True

def transpose(song):
    # get key from the score
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]

    # estimate key using music21
    if not isinstance(key,m21.key.Key):
        key = song.analyze("key")

    print(key)

    # get interval for transposition Eg. Bmaj -> Cmaj
    if key.mode=="major":
        interval = m21.interval.Interval(key.tonic,m21.pitch.Pitch("C"))
    elif key.mode=="minor":
        interval = m21.interval.Interval(key.tonic,m21.pitch.Pitch("A"))

    # transpose song by calculated interval
    transposed_song = song.transpose(interval)
    return transposed_song

def encode_song(song,time_step=0.25):
    encoded_song = []
    for event in song.flat.notesAndRests:
        # handle notes
        if isinstance(event,m21.note.Note):
            symbol = event.pitch.midi

        # handle rests
        elif isinstance(event,m21.note.Rest):
            symbol = 'r'

        # convert the note/rest into time series notation
        steps = int(event.duration.quarterLength/time_step)

        for step in range(steps):
            if step==0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    # cast encoded song to string

    encoded_song = " ".join(map(str,encoded_song))
    return encoded_song


def preprocess(dataset_path):

    #load the folk songs
    print("Loading Songs...")
    songs = load_songs_in_kern(KERN_DATASET_PATH)
    print(f"Loaded {len(songs)} songs.")

    for i,song in enumerate(songs):
        #filter out songs that have non-acceptable durations
        if not has_acceptable_durations(song,ACCEPTABLE_DURATIONS):
            continue
        # transpose songs to Cmaj/Amin
        song = transpose(song)

        # encode songs with music time series representation
        encoded_song = encode_song(song)
        # save songs to text file

        save_path = os.path.join(SAVE_DIR,str(i))

        with open(save_path,"w") as fp:
            fp.write(encoded_song)

def load(file_path):
    with open(file_path,"r") as fp:
        song = fp.read()
    return song



def create_single_file_dataset(dataset_path, file_dataset_path,sequence_length):
    new_song_delimiter = "/ " *sequence_length
    songs = ""

    # load encoded songs and add delimiters
    for path,_,files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path,file)
            song = load(file_path)
            songs = songs+song+" "+new_song_delimiter

    songs = songs[:-1]

    # save string that contain all dataset
    with open(file_dataset_path,"w") as fp:
        fp.write(songs)

    return songs


def create_mapping(songs,mappin_path):
    mappings = {}

    # identify the vocabulary
    songs = songs.split()
    vocabulary = list(set(songs))

    #create mappings
    for i,symbol in enumerate(vocabulary):
        mappings[symbol] = i

    # save vocabulary to json file
    with open(mappin_path,"w") as fp:
        json.dump(mappings,fp,indent=4)


def convert_songs_to_int(songs):
    int_songs = []
    # load mappings
    with open(MAPPING_PATH,"r") as fp:
        mappings =json.load(fp)

    # cast song strings to list
    songs = songs.split()

    # map songs to int
    for symbol in songs:
        int_songs.append(mappings[symbol])

    return int_songs

def generating_training_sequences(sequence_length):

    # load songs and map them to int
    songs = load(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)

    # generate training_sequences

    inputs = []
    targets = []

    num_sequences = len(int_songs)-sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])

    # one-hot encode the sequences
    vocabulary_size = len(set(int_songs))
    inputs = keras.utils.to_categorical(inputs,num_classes=vocabulary_size)
    targets = np.array(targets)

    return inputs,targets



def main():
    preprocess(KERN_DATASET_PATH)
    songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET,SEQUENCE_LENGTH)
    create_mapping(songs,MAPPING_PATH)
    inputs,targets = generating_training_sequences(SEQUENCE_LENGTH) 



