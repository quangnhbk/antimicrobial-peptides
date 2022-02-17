# Short Antimicrobial Peptides Prediction
# 
# @ PhD. Quang H. Nguyen
# School of Information and Communication Technology
# Hanoi University of Science and Technology
# Hanoi, September 09 2021

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np
from sklearn.metrics import roc_curve, auc

from tensorflow.keras import Input
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import average_precision_score, precision_recall_curve

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pickle
import math

import h5py

from params.blosum import *

from params.amino_acid_properties import *
from params.chou_fasman import *
from params.amino_acid_alphabet import *
from params.pmbec import *
from params.residue_contact_energies import *

###### Reproducible Result
seed_value= 0

import os
os.environ['PYTHONHASHSEED']=str(seed_value)
import random
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Run on CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ALPHABET = "AVLIFWMPGSTCYNQHKRDE-"

###### HYPER-PARAMETERS #####
MY_RANDOM_STATE = 70
#EPOCHS = 2
EPOCHS = 100
BATCH_SIZE = 32
MAX_LENGTH = 30

NUMBER_OF_PARAMS = 183

###### FILES #####
DIR_DATASET = "./ShortAntimicrobialPeptidesPrediction/Dataset/"
TRAIN_POSITIVE_FILE = DIR_DATASET + "train_po.fasta"
TRAIN_NEGATIVE_FILE = DIR_DATASET + "train_ne.fasta"

TEST_POSITIVE_FILE = DIR_DATASET + "test_po.fasta"
TEST_NEGATIVE_FILE = DIR_DATASET + "test_ne.fasta"

FILE_MODEL = "weights.best.hdf5"
TRAIN_DIR = "./train1/"


# Format FASTA files:
# >unip30_cdh10_stdif_dpAmpepTr30_iamp2l_ampScan_ampepTr_cdh8_sample94_1
# NYIYSGHNYHQ
# >unip30_cdh10_stdif_dpAmpepTr30_iamp2l_ampScan_ampepTr_cdh8_sample94_2
# DPNATIIMLGTGTGIAPFR

def read_file(filename):
    text_file = open(filename)
    lines = text_file.readlines()
    text_file.close()
    
    lst_seq = []
    for line in lines[1::2]:
        seq = line.strip()
        lst_seq.append(seq)    
    return lst_seq

def coded(sequences):
    """char to int"""
    AA = {'A':0,'V':1,'L':2,'I':3,'F':4,'W':5,'M':6,'P':7,'G':8,'S':9,'T':10,'C':11,'Y':12,'N':13,'Q':14,'H':15,'K':16,'R':17,'D':18,'E':19, '-':20}  
    int_seq = []
    for index in range(len(sequences)):
        int_seq.append([])
        for i in range(len(sequences[index])):
            if sequences[index][i] in AA: int_seq[index].append(AA[sequences[index][i]])
            else: int_seq[index].append(20)          
    return int_seq

def prepare_data():
    train_positive_seqs = read_file(TRAIN_POSITIVE_FILE)
    print("train_positive_seqs: ", len(train_positive_seqs))
    
    train_negative_seqs = read_file(TRAIN_NEGATIVE_FILE)
    print("train_negative_seqs: ", len(train_negative_seqs))
    
    train_seqs = train_positive_seqs + train_negative_seqs
    print("train_seqs: ", len(train_seqs))
    
    train_positive_labels = np.ones((len(train_positive_seqs),1))
    train_negative_labels = np.zeros((len(train_negative_seqs), 1))
    train_labels = np.concatenate((train_positive_labels, train_negative_labels))    
    
    test_positive_seqs = read_file(TEST_POSITIVE_FILE)
    print("test_positive_seqs: ", len(test_positive_seqs))
    
    test_negative_seqs = read_file(TEST_NEGATIVE_FILE)
    print("test_negative_seqs: ", len(test_negative_seqs))
    
    test_seqs = test_positive_seqs + test_negative_seqs
    print("test_seqs: ", len(test_seqs))
    
    test_positive_labels = np.ones((len(test_positive_seqs),1))
    test_negative_labels = np.zeros((len(test_negative_seqs), 1))
    test_labels = np.concatenate((test_positive_labels, test_negative_labels))
    print("test_labels: ", len(test_labels))
        
    return train_seqs, train_labels, test_seqs, test_labels

def sequence_coded_v1(sequence):
    """char to int"""
    AA = {'A':20,'V':1,'L':2,'I':3,'F':4,'W':5,'M':6,'P':7,'G':8,'S':9,'T':10,'C':11,'Y':12,'N':13,'Q':14,'H':15,'K':16,'R':17,'D':18,'E':19, '-':0}  
    #int_seq = []
    int_seq = np.zeros(MAX_LENGTH)
    for i in range(len(sequence)):
        if sequence[i] in AA:  int_seq[i] = AA[sequence[i]]
        else: int_seq[i] = 0
    return int_seq

def array_seq_coded_v1(arr_sequences):    
    sequences_coded = np.zeros((len(arr_sequences), MAX_LENGTH))
    for index in range(len(arr_sequences)):
        seq_coded = sequence_coded_v1(arr_sequences[index])        
        sequences_coded[index] = seq_coded
    return sequences_coded

########## BLOSUM62 Parameters #################
# ALPHABET = ["AVLIFWMPGSTCYNQHKRDE-"]
def get_blosum62_char(ch):
    ch = ch.upper()
    if ch in blosum62_dict.keys(): ch_blosum62_dict = blosum62_dict[ch]
    else: ch_blosum62_dict = blosum62_dict["*"]
    #print(ch_blosum62_dict)
    ch_blosum62 = []
    for c in ALPHABET:
        if c in ch_blosum62_dict.keys(): ch_blosum62.append(ch_blosum62_dict[c] / 4.0)
        else: ch_blosum62.append(ch_blosum62_dict["*"] / 4.0)        
    ch_blosum62 = np.array(ch_blosum62)    
    return ch_blosum62

########## BLOSUM50 Parameters #################
# ALPHABET = ["AVLIFWMPGSTCYNQHKRDE-"]
def get_blosum50_char(ch):
    ch = ch.upper()
    if ch in blosum50_dict.keys(): ch_blosum50_dict = blosum50_dict[ch]
    #else: ch_blosum50_dict = blosum50_dict["*"]
    #print(ch_blosum50_dict)
    ch_blosum50 = []
    for c in ALPHABET:
        if c in ch_blosum50_dict.keys(): ch_blosum50.append(ch_blosum50_dict[c] / 4.0)
        else: ch_blosum50.append(0.0)
        
    ch_blosum50 = np.array(ch_blosum50)    
    return ch_blosum50

########## BLOSUM50 Parameters #################
# ALPHABET = ["AVLIFWMPGSTCYNQHKRDE-"]
def get_blosum30_char(ch):
    ch = ch.upper()
    if ch in blosum30_dict.keys(): ch_blosum30_dict = blosum30_dict[ch]
    else: ch_blosum30_dict = blosum50_dict["*"]
    #print(ch_blosum30_dict)
    ch_blosum30 = []
    for c in ALPHABET:
        if c in ch_blosum30_dict.keys(): ch_blosum30.append(ch_blosum30_dict[c] / 4.0)
        else: 
            ch_blosum30.append(ch_blosum30_dict["*"] / 4.0)           
        
    ch_blosum30 = np.array(ch_blosum30)    
    return ch_blosum30
    
##### pmbec ####
def get_from_dict_char(ch, code_dict):
    ch = ch.upper()
    if ch in code_dict.keys(): ch_dict = code_dict[ch]
    else: ch_dict = {}
    
    ch_coded = []
    for c in ALPHABET:
        if c in ch_dict.keys(): ch_coded.append(ch_dict[c])
        else: ch_coded.append(0.0)
        
    ch_coded = np.array(ch_coded)    
    return ch_coded
    
# hydropathy: {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}
def get_amino_acid_properties_char(ch):
    ch = ch.upper()
    ch_properties = []
    if ch in hydropathy_norm.keys(): ch_properties.append(hydropathy_norm[ch])
    else: ch_properties.append(0) 
    
    if ch in volume_norm.keys(): ch_properties.append(volume_norm[ch])
    else: ch_properties.append(0)
    
    if ch in polarity_norm.keys(): ch_properties.append(polarity_norm[ch])
    else: ch_properties.append(0)
    
    if ch in pK_side_chain_norm.keys(): ch_properties.append(pK_side_chain_norm[ch])
    else: ch_properties.append(0)
    
    if ch in prct_exposed_residues_norm.keys(): 
        ch_properties.append(prct_exposed_residues_norm [ch])
    else: ch_properties.append(0)
    
    if ch in hydrophilicity_norm.keys(): ch_properties.append(hydrophilicity_norm[ch])
    else: ch_properties.append(0)
    
    if ch in accessible_surface_area_norm.keys(): 
        ch_properties.append(accessible_surface_area_norm[ch])
    else: ch_properties.append(0)
    
    if ch in local_flexibility_norm.keys(): ch_properties.append(local_flexibility_norm[ch])
    else: ch_properties.append(0)
    
    if ch in accessible_surface_area_folded_norm.keys(): 
        ch_properties.append(accessible_surface_area_folded_norm[ch])
    else: ch_properties.append(0)
    
    if ch in refractivity_norm.keys(): ch_properties.append(refractivity_norm[ch])
    else: ch_properties.append(0)
    
    if ch in mass_norm.keys(): ch_properties.append(mass_norm[ch])
    else: ch_properties.append(0)
    
    if ch in solvent_exposed_area_norm.keys(): 
        ch_properties.append(solvent_exposed_area_norm[ch])
    else: ch_properties.append(0)    
    
    if ch in turn_score_norm.keys(): 
        ch_properties.append(turn_score_norm[ch])
        #print("turn_score_norm: ", turn_score_norm[ch])
    else: 
        ch_properties.append(0)
        #print("turn_score_norm: 0")
    
    if ch in alpha_helix_score_norm.keys(): 
        ch_properties.append(alpha_helix_score_norm[ch])
    else: ch_properties.append(0)
    
    if ch in beta_sheet_score_norm.keys(): 
        ch_properties.append(beta_sheet_score_norm[ch])
    else: ch_properties.append(0)
    
    ch_properties = np.array(ch_properties)
    return ch_properties

def get_seq_coded(seq):
    AA = {'A':0,'V':1,'L':2,'I':3,'F':4,'W':5,'M':6,'P':7,'G':8,'S':9,'T':10,'C':11,'Y':12,'N':13,'Q':14,'H':15,'K':16,'R':17,'D':18,'E':19, '-':20} 
    seq_coded = np.zeros((MAX_LENGTH, NUMBER_OF_PARAMS + 1))
    for index, ch  in enumerate(seq): 
        ch_properties = get_amino_acid_properties_char(ch)
        #print(ch_properties)
        ch_blosum62 = get_blosum62_char(ch)
        
        ch_index = []
        if ch in AA: ch_index.append(AA[ch])
        else: ch_index.append(20)
        ch_index = np.array(ch_index)
        
        ch_pmbec = get_from_dict_char(ch, pmbec_dict)
        ch_strand_coil = get_from_dict_char(ch, strand_vs_coil_dict)
        ch_coil_strand = get_from_dict_char(ch, coil_vs_strand_dict)
        
        ch_helix_strand = get_from_dict_char(ch, helix_vs_strand_dict)
        ch_strand_helix = get_from_dict_char(ch, strand_vs_helix_dict)
        
        ch_helix_coil = get_from_dict_char(ch, helix_vs_coil_dict)
        ch_coil_helix = get_from_dict_char(ch, coil_vs_helix_dict)
        
        ch_coded = np.concatenate((ch_properties, ch_blosum62, ch_pmbec, ch_strand_coil, ch_coil_strand, ch_helix_strand, ch_strand_helix, ch_helix_coil, ch_coil_helix, ch_index))
                
        seq_coded[index] = ch_coded
    return seq_coded
    
# Numpy array: N x H x W
def array_seq_coded(arr_sequences):
    sequences_coded = np.zeros((len(arr_sequences), MAX_LENGTH, NUMBER_OF_PARAMS + 1))
    for index in range(len(arr_sequences)):
        seq_coded = get_seq_coded(arr_sequences[index])
        sequences_coded[index] = seq_coded        
    return sequences_coded

def param_norm(param_dict):
    TRAIN_SET_STATS = {'A': 5087, 'V': 4137, 'L': 6790, 'I': 4106, 'F': 2994, 'W': 939, 'M': 1144, 'P': 2987, 'G': 6127, 'S': 3637, 'T': 2579, 'C': 2525, 'Y': 1270, 'N': 2081, 'Q': 1494, 'H': 1272, 'K': 5610, 'R': 2850, 'D': 1982, 'E': 1971, '-': 0}
    
    values = []
    for key, value in param_dict.items():        
        values.append(value)        
    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values)    
    
    result = {}
    for key, value in param_dict.items():        
        new_value = (value - mean) / std
        result[key] = new_value        
    return result

def stat_chars(dataset):
    my_result = {}
    for ch in ALPHABET: 
        my_result[ch] = 0
    for seq in dataset:
        #print(seq)
        for ch in seq: 
            current_count = my_result[ch]
            my_result[ch] = current_count + 1
        #break
    print(my_result)

def chou_fasman_norm(attributes):
    result = {}
    for key, value in attributes.items():
        #print(key, value)
        letter = index_to_letter(key).letter
        #print(letter)
        result[letter] = value
    return result
    
def print_history(fold, history):
    print(history.history.keys())

    file_loss = TRAIN_DIR + "fold_" + str(fold) + "_loss.png"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    #plt.show()
    plt.savefig(file_loss, bbox_inches='tight')

def DeeAMP_model():
    seq_input = Input(shape=(MAX_LENGTH, NUMBER_OF_PARAMS + 1), name='seq_input')
    print("seq_input: ", seq_input.shape)
    input_1 = layers.Lambda(lambda x: x[:,:,:-1])(seq_input)
    print("input_1: ", input_1.shape)
    input_2 = layers.Lambda(lambda x: x[:,:, -1])(seq_input)
    print("input_2: ", input_2.shape)
    x = layers.BatchNormalization()(input_1)
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(MAX_LENGTH, NUMBER_OF_PARAMS))(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu')(x) 
    
    x = layers.Dropout(0.1)(x)
    
    x = layers.Conv1D(filters=128, kernel_size=3, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv1D(filters=128, kernel_size=3, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv1D(filters=128, kernel_size=3, activation='relu')(x)
    
    x = layers.Dropout(0.1)(x)    
    
    x = layers.Flatten()(x)
    
    embedded_seq = layers.Embedding(30, 10, input_length=30)(input_2)
    lstm1 = layers.LSTM(128)(embedded_seq) 
    
    x = layers.Concatenate()([lstm1, x])
    
    x = layers.Dense(128,activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128,activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    category_output = layers.Dense(1,activation='sigmoid')(x)
    model = Model(seq_input,category_output)
    
    METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
      tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
    ]

    model.compile(optimizer=Adam(lr=1e-4), 
        loss='binary_crossentropy',          
        metrics=['accuracy'],
        #metrics=METRICS
        )
    return model

def split_10_folds():
    #kf = KFold(n_splits=5, shuffle=True, random_state=MY_RANDOM_STATE)
    
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=MY_RANDOM_STATE)    
    
    fold = 1
    for train_index, val_index in kf.split(train_seqs, train_labels):
        #print("train_index: ", len(train_index))
        #print("val_index: ", len(val_index))
        
        data_train = [train_seqs[idx] for idx in train_index]        
        label_train = train_labels[train_index]
        
        data_val = [train_seqs[idx] for idx in val_index]        
        label_val = train_labels[val_index]
        
        train_one_fold(fold, data_train, label_train, data_val, label_val)
        
        fold = fold + 1
    
    print_history(fold, history)
    
def train_one_fold(fold, data_train, label_train, data_val, label_val):
    print("\n### TRAIN FOLD " + str(fold) + " ###")
    print("data_train: ", len(data_train))
    print("data_val: ", len(data_val))
    data_train_coded = array_seq_coded(data_train)
    data_val_coded = array_seq_coded(data_val)
    
    model = DeeAMP_model()
    
    print(model.summary())
    
    FILE_MODEL = TRAIN_DIR + "fold_" + str(fold) + "_model.hdf5"
    
    early_stopping = EarlyStopping(patience=20, verbose=1)
    model_checkpoint = ModelCheckpoint(FILE_MODEL, monitor='val_loss', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, 
        verbose=1, mode='auto', epsilon=0.0001, cooldown=1, min_lr=1e-8)
        
    history = model.fit(data_train_coded, label_train, 
                    batch_size = BATCH_SIZE,
                    epochs = EPOCHS, 
                    validation_data=(data_val_coded, label_val),
                    verbose = 1,
                    callbacks = [model_checkpoint, early_stopping, reduce_lr])
    
##################################
if __name__== "__main__":
    print("Short Antimicrobial Peptides Prediction")    
    
    # Amino acid properties
    hydropathy_norm = param_norm(hydropathy)
    volume_norm = param_norm(volume)
    polarity_norm = param_norm(polarity)
    pK_side_chain_norm = param_norm(pK_side_chain)
    prct_exposed_residues_norm = param_norm(prct_exposed_residues)
    hydrophilicity_norm = param_norm(hydrophilicity)
    accessible_surface_area_norm = param_norm(accessible_surface_area)
    local_flexibility_norm = param_norm(local_flexibility)
    accessible_surface_area_folded_norm = param_norm(accessible_surface_area_folded)
    refractivity_norm = param_norm(refractivity)
    mass_norm = param_norm(mass)
    solvent_exposed_area_norm = param_norm(solvent_exposed_area) 
    
    # Chou fasman table
    alpha_helix_score_norm = param_norm(chou_fasman_norm(alpha_helix_score))
    beta_sheet_score_norm = param_norm(chou_fasman_norm(beta_sheet_score))
    turn_score_norm = param_norm(chou_fasman_norm(turn_score))
    
    # Prepare data     
    train_seqs, train_labels, test_seqs, test_labels = prepare_data()
    
    stat_chars(train_seqs)
    
    data_test_coded = array_seq_coded(test_seqs)
    print("data_test_coded: ", data_test_coded.shape)
    #print(data_test_coded[:2])
    
    # Training 10 folds 
    split_10_folds()
    
    
