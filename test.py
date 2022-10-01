# Short Antimicrobial Peptides Prediction
# 
# @ PhD. Quang H. Nguyen
# School of Information and Communication Technology
# Hanoi University of Science and Technology
# Hanoi, September 09 2021

# Su dung ma tran BLOSUM62 (21 features) + Amino Acid Properties (12 features)
# + Chou Fasman (3 features)

import sys

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

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

# Thu vien params: tinh cac tham so cho proteins
from params.blosum import *
#print(blosum62_dict)
#print(blosum62_matrix)

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
#tf.random.set_seed(seed_value)

# Run on CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ALPHABET = "AVLIFWMPGSTCYNQHKRDE-"

BATCH_SIZE = 32
MAX_LENGTH = 30

NUMBER_OF_PARAMS = 183

TRAIN_DIR = "./models/"

# Format FASTA files:
# >unip30_cdh10_stdif_dpAmpepTr30_iamp2l_ampScan_ampepTr_cdh8_sample94_1
# NYIYSGHNYHQ
# >unip30_cdh10_stdif_dpAmpepTr30_iamp2l_ampScan_ampepTr_cdh8_sample94_2
# DPNATIIMLGTGTGIAPFR

def read_file(filename):
    text_file = open(filename)
    lines = text_file.readlines()
    text_file.close()
    #print("Number of lines: ", len(lines))
    #print(lines[:2])
    lst_seq = []
    for line in lines[1::2]:
        seq = line.strip()
        lst_seq.append(seq)
    
    #print(lst_seq[:2])    
    return lst_seq

# Bieu dien moi chuoi (33 ky tu) thanh mang bieu dien chi so cua tung ky tu trong chuoi 
# [18 19 12 16 10  9  2 17  7 19 10  1 19  0  2  3 11  0 17 19  5  2  2 19  9 13  0  9  9  9  0  0  0]
def coded(sequences):
    """char to int"""
    AA = {'A':0,'V':1,'L':2,'I':3,'F':4,'W':5,'M':6,'P':7,'G':8,'S':9,'T':10,'C':11,'Y':12,'N':13,'Q':14,'H':15,'K':16,'R':17,'D':18,'E':19, '-':20}  
    int_seq = []
    for index in range(len(sequences)):
        int_seq.append([])
        for i in range(len(sequences[index])):
            if sequences[index][i] in AA:
                int_seq[index].append(AA[sequences[index][i]])
            else:
                int_seq[index].append(20)          
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
    #print("train_labels: ", train_labels)
    
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

# Bieu dien moi chuoi (33 ky tu) thanh mang bieu dien chi so cua tung ky tu trong chuoi 
# + Input: sequence
# + Output: list index cua cac ky tu trong chuoi 
# [18 19 12 16 10  9  2 17  7 19 10  1 19  0  2  3 11  0 17 19  5  2  2 19  9 13  0  9  9  9  0  0  0]
def sequence_coded_v1(sequence):
    """char to int"""
    AA = {'A':20,'V':1,'L':2,'I':3,'F':4,'W':5,'M':6,'P':7,'G':8,'S':9,'T':10,'C':11,'Y':12,'N':13,'Q':14,'H':15,'K':16,'R':17,'D':18,'E':19, '-':0}  
    #int_seq = []
    int_seq = np.zeros(MAX_LENGTH)
    for i in range(len(sequence)):
        if sequence[i] in AA:
            #int_seq.append(AA[sequence[i]])
            int_seq[i] = AA[sequence[i]]
        else:
            #int_seq.append(20)
            int_seq[i] = 0
    return int_seq

def array_seq_coded_v1(arr_sequences):
    #sequences_coded = []
    sequences_coded = np.zeros((len(arr_sequences), MAX_LENGTH))
    for index in range(len(arr_sequences)):
        seq_coded = sequence_coded_v1(arr_sequences[index])
        #sequences_coded.append(seq_coded)
        sequences_coded[index] = seq_coded
    return sequences_coded

########## BLOSUM62 Parameters #################
# ALPHABET = ["AVLIFWMPGSTCYNQHKRDE-"]
# Ma hoa mot ky tu ve vec to chua cac tham so cua ma tran BLOSUM62 theo mang ALPHABET
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
    #print(ch_blosum62, ch_blosum62.shape)
    return ch_blosum62

########## BLOSUM50 Parameters #################
# ALPHABET = ["AVLIFWMPGSTCYNQHKRDE-"]
# Ma hoa mot ky tu ve vec to chua cac tham so cua ma tran BLOSUM62 theo mang ALPHABET
def get_blosum50_char(ch):
    ch = ch.upper()
    if ch in blosum50_dict.keys(): ch_blosum50_dict = blosum50_dict[ch]
    #else: ch_blosum50_dict = blosum50_dict["*"]
    #print(ch_blosum50_dict)
    ch_blosum50 = []
    for c in ALPHABET:
        if c in ch_blosum50_dict.keys(): ch_blosum50.append(ch_blosum50_dict[c] / 4.0)
        else: 
            #ch_blosum50.append(ch_blosum50_dict["*"] / 4.0)
            ch_blosum50.append(0.0)
        
    ch_blosum50 = np.array(ch_blosum50)
    #print(ch_blosum50, ch_blosum50.shape)
    return ch_blosum50

########## BLOSUM50 Parameters #################
# ALPHABET = ["AVLIFWMPGSTCYNQHKRDE-"]
# Ma hoa mot ky tu ve vec to chua cac tham so cua ma tran BLOSUM62 theo mang ALPHABET
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
            #ch_blosum30.append(0.0)
        
    ch_blosum30 = np.array(ch_blosum30)
    #print(ch_blosum30, ch_blosum30.shape)
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
    #print(ch_coded, ch_coded.shape)
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

# Moi chuoi se duoc bieu dien bang ma tran 35x21
# 35 amino acids, moi amino acid duoc bieu dien bang vec to BLOSUM 62 gom 21 gia tri     
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
        
        #ch_blosum50 = get_blosum50_char(ch)
        #ch_blosum30 = get_blosum30_char(ch)
        
        ch_pmbec = get_from_dict_char(ch, pmbec_dict)
        ch_strand_coil = get_from_dict_char(ch, strand_vs_coil_dict)
        ch_coil_strand = get_from_dict_char(ch, coil_vs_strand_dict)
        
        ch_helix_strand = get_from_dict_char(ch, helix_vs_strand_dict)
        ch_strand_helix = get_from_dict_char(ch, strand_vs_helix_dict)
        
        ch_helix_coil = get_from_dict_char(ch, helix_vs_coil_dict)
        ch_coil_helix = get_from_dict_char(ch, coil_vs_helix_dict)
        
        ch_coded = np.concatenate((ch_properties, ch_blosum62, ch_pmbec, ch_strand_coil, ch_coil_strand, ch_helix_strand, ch_strand_helix, ch_helix_coil, ch_coil_helix, ch_index))
        
        #ch_coded = np.concatenate((ch_properties, ch_blosum62, ch_blosum50, ch_pmbec, ch_strand_coil, ch_coil_strand, ch_helix_strand, ch_strand_helix, ch_helix_coil, ch_coil_helix))
        
        #ch_coded = np.concatenate((ch_properties, ch_blosum62, ch_blosum50, ch_blosum30, ch_pmbec, ch_strand_coil, ch_coil_strand, ch_helix_strand, ch_strand_helix, ch_helix_coil, ch_coil_helix))
                
        #ch_coded = ch_blosum62
        #ch_coded = ch_properties
        #ch_coded = np.concatenate((ch_properties, ch_blosum62))
        seq_coded[index] = ch_coded
    return seq_coded
    
# Numpy array: N x H x W
# N : so chuoi, H: chieu dai chuoi (35), W : so amino acids: 21 
def array_seq_coded(arr_sequences):
    sequences_coded = np.zeros((len(arr_sequences), MAX_LENGTH, NUMBER_OF_PARAMS + 1))
    for index in range(len(arr_sequences)):
        seq_coded = get_seq_coded(arr_sequences[index])
        sequences_coded[index] = seq_coded
        
    return sequences_coded

# Chuan hoa gia tri theo mean va variance
# hydropathy: {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}    
def param_norm(param_dict):
    TRAIN_SET_STATS = {'A': 5087, 'V': 4137, 'L': 6790, 'I': 4106, 'F': 2994, 'W': 939, 'M': 1144, 'P': 2987, 'G': 6127, 'S': 3637, 'T': 2579, 'C': 2525, 'Y': 1270, 'N': 2081, 'Q': 1494, 'H': 1272, 'K': 5610, 'R': 2850, 'D': 1982, 'E': 1971, '-': 0}

    #print(param_dict)
    
    values = []
    for key, value in param_dict.items():
        #print(key, value)
        #if key in TRAIN_SET_STATS.keys(): nb_count = TRAIN_SET_STATS[key]
        #else: nb_count = TRAIN_SET_STATS["-"]
        #for i in range(nb_count):
        #    values.append(value)
        values.append(value)    
    #print("values: ", len(values))
    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values)
    #print("mean : ", mean, " std: ", std)
    
    result = {}
    for key, value in param_dict.items():        
        new_value = (value - mean) / std
        result[key] = new_value
        #print(key, value, new_value)
    #print(result)
    return result

# Tinh phan bo cac ky tu trong tap Train
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
    

def quang_model():
    seq_input = Input(shape=(MAX_LENGTH, NUMBER_OF_PARAMS + 1), name='seq_input')
    print("seq_input: ", seq_input.shape)
    input_1 = layers.Lambda(lambda x: x[:,:,:-1])(seq_input)
    print("input_1: ", input_1.shape)
    input_2 = layers.Lambda(lambda x: x[:,:, -1])(seq_input)
    print("input_2: ", input_2.shape)
    x = layers.BatchNormalization()(input_1)
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(MAX_LENGTH, NUMBER_OF_PARAMS))(x)
    x = layers.Dropout(0.1)(x)
    #x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu')(x) 
    #x = layers.MaxPooling1D(pool_size=2, strides=1, padding='valid')(x)    
    x = layers.Dropout(0.1)(x)
    
    x = layers.Conv1D(filters=128, kernel_size=3, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv1D(filters=128, kernel_size=3, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv1D(filters=128, kernel_size=3, activation='relu')(x)
    #x = layers.MaxPooling1D(pool_size=2, strides=1, padding='valid')(x)
    x = layers.Dropout(0.1)(x)
    
    #x = layers.Conv1D(filters=64, kernel_size=3, activation='relu')(x)
    #x = layers.Conv1D(filters=64, kernel_size=3, activation='relu')(x)
    #x = layers.Conv1D(filters=64, kernel_size=3, activation='relu')(x)
    #x = layers.MaxPooling1D(pool_size=2, strides=1, padding='valid')(x)
    #x = layers.Dropout(0.1)(x)
    
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
    
    #METRICS = [
    #  tf.keras.metrics.TruePositives(name='tp'),
    #  tf.keras.metrics.FalsePositives(name='fp'),
    #  tf.keras.metrics.TrueNegatives(name='tn'),
    #  tf.keras.metrics.FalseNegatives(name='fn'), 
    #  tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    #  tf.keras.metrics.Precision(name='precision'),
    #  tf.keras.metrics.Recall(name='recall'),
    #  tf.keras.metrics.AUC(name='auc'),
    #  tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
    #]

    model.compile(optimizer=Adam(lr=1e-4), 
        loss='binary_crossentropy',          
        metrics=['accuracy'],
        #metrics=METRICS
        )
    return model

# Tinh accuracy, sensitivity, specificity
# Note that in binary classification, recall of the positive class is also known as "sensitivity"; recall of the negative class is "specificity".
def process_result(test_scores, test_labels, threshold = 0.5):
    #print("test_scores: ", test_scores)
    print("test_scores: ", test_scores.shape)
    #print("test_labels: ", test_labels)
    print("test_labels: ", test_labels.shape)
    
    test_hyp = []
    for score in test_scores:
        if score > threshold: test_hyp.append(1)
        else: test_hyp.append(0)
    
    conf_mat = confusion_matrix(test_labels, test_hyp)
    print("confusion_matrix: ", conf_mat.shape, "\n", conf_mat)
    
    acc = accuracy_score(test_labels, test_hyp)
    print("Accuracy: ", acc)
    
    target_names = ['class 0 : Negative', 'class 1: Positive']
    print(classification_report(test_labels, test_hyp, target_names=target_names))
    
    specificity = conf_mat[0][0] / (conf_mat[0][0] + conf_mat[0][1])
    sensitivity = conf_mat[1][1] / (conf_mat[1][0] + conf_mat[1][1])
    tp = conf_mat[1][1]
    tn = conf_mat[0][0]
    fp = conf_mat[1][0]
    fn = conf_mat[0][1]
    mcc = (tp * tn - fp * fn ) / math.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
    print("sensitivity: ", sensitivity)
    print("specificity: ", specificity)
    print("mcc: ", mcc)
    
    # Tinh Kappa
    #p0 = (tp + tn) / (tp + fn + tn + fp)
    #pe = (tp + fn) * (tp + fp) * (tn + fn) * (tn + fp) / (tp + fn + tn + fp) / (tp + fn + tn + fp)
    #kappa = (p0 - pe) / (1 - pe)
    #print("p0: ", p0, " pe: ", pe)
    #print("Kappa: ", kappa)
    
    kappa = 2 * (tp * tn - fn * fp) / ((tp + fp) * (fp + tn) + (tp + fn)*(fn + tn))
    print("Kappa: ", kappa)
    
    fpr, tpr, _ = roc_curve(test_labels, test_scores)
    auc_roc = auc(fpr,tpr)
    print("Test AUC: ", auc_roc)
    
    precision, recall, thresholds = precision_recall_curve(test_labels, test_scores)
    auc_precision_recall = auc(recall, precision)
    
    return acc,auc_roc,auc_precision_recall,kappa,sensitivity,specificity,mcc

def process_all_results():
    file_results_summary = open(TRAIN_DIR + "results_summary.csv", "w")
    file_results_summary.write("Fold,Accuracy,AUC-ROC,AUC-PR,Kappa,Sensitivity,Specificity,MCC\n")
    
    
    all_scores = np.load(TRAIN_DIR + "all_results.npy")
    print("all_scores: ", all_scores)
    
    nb_samples = all_scores.shape[1]
    print("nb_samples: ", nb_samples)
    
    for fold in range(10):
        acc,auc_roc,auc_pr,kappa,sensitivity,specificity,mcc = process_result(all_scores[fold], test_labels)
        results = str(fold) + "," + '{0:.4f}'.format(acc) + "," + '{0:.4f}'.format(auc_roc) + "," + '{0:.4f}'.format(auc_pr) + "," + '{0:.4f}'.format(kappa) + "," + '{0:.4f}'.format(sensitivity) + "," + '{0:.4f}'.format(specificity) + "," + '{0:.4f}'.format(mcc) + "\n"
        file_results_summary.write(results)
    
    nb_error_0 = 0
    nb_error_1 = 0
    
    nb_error_0_new = 0
    nb_error_1_new = 0
    
    lst_prob_mean = []
    lst_prob_median = []
    lst_prob_max = []
    for idx in range(nb_samples):
        probs = all_scores[:,idx]        
        
        lst_prob_mean.append(np.mean(probs))
        lst_prob_median.append(np.median(probs))
        lst_prob_max.append(np.max(probs))
        
        #if idx > 10: break
    #test_set["label"]    
    
    # Max ensemble 
    print("\n# Max ensemble ")
    acc,auc_roc,pr,kappa,sensitivity,specificity,mcc = process_result(np.array(lst_prob_max), test_labels)
    
    results = "Max," + '{0:.4f}'.format(acc) + "," + '{0:.4f}'.format(auc_roc) + "," + '{0:.4f}'.format(auc_pr) + "," + '{0:.4f}'.format(kappa) + "," + '{0:.4f}'.format(sensitivity) + "," + '{0:.4f}'.format(specificity) + "," + '{0:.4f}'.format(mcc) + "\n"
    
    file_results_summary.write(results)
        
    # Mean ensemble 
    print("\n# Mean ensemble")   
    
    acc,auc_roc,auc_pr,kappa,sensitivity,specificity,mcc = process_result(np.array(lst_prob_mean), test_labels)
    
    results = "Mean," + '{0:.4f}'.format(acc) + "," + '{0:.4f}'.format(auc_roc) + "," + '{0:.4f}'.format(auc_pr) + "," + '{0:.4f}'.format(kappa) + "," + '{0:.4f}'.format(sensitivity) + "," + '{0:.4f}'.format(specificity) + "," + '{0:.4f}'.format(mcc) + "\n"
    
    file_results_summary.write(results)
    
    # Median ensemble 
    print("\n# Median ensemble ")    
    acc,auc_roc,auc_precision_recall,kappa,sensitivity,specificity,mcc = process_result(np.array(lst_prob_median), test_labels)
    
    results = "Median," + '{0:.4f}'.format(acc) + "," + '{0:.4f}'.format(auc_roc) + "," + '{0:.4f}'.format(auc_pr) + "," + '{0:.4f}'.format(kappa) + "," + '{0:.4f}'.format(sensitivity) + "," + '{0:.4f}'.format(specificity) + "," + '{0:.4f}'.format(mcc) + "\n"
    
    file_results_summary.write(results)
    
    
    #file_results_summary.write(results)
    
    # For different Threshold 
    mcc_max = 0.0
    threshold_best = 0.0
    diff = 10.0
    threshold_best_diff = 0.0
    best_acc = 0 
    threshold_best_acc = 0 
    for i in range(100):
        threshold = 1.0 * i / 100
        acc,auc_roc,pr,kappa,sensitivity,specificity,mcc = process_result(np.array(lst_prob_median), test_labels, threshold)
        
        results = str(fold) + "," + '{0:.4f}'.format(acc) + "," + '{0:.4f}'.format(auc_roc) + "," + '{0:.4f}'.format(auc_pr) + "," + '{0:.4f}'.format(kappa) + "," + '{0:.4f}'.format(sensitivity) + "," + '{0:.4f}'.format(specificity) + "," + '{0:.4f}'.format(mcc) + "\n"
        
        print(results)
        if mcc_max < mcc: 
            mcc_max = mcc
            threshold_best = threshold
        if abs(sensitivity - specificity) < diff:
            diff = abs(sensitivity - specificity)
            threshold_best_diff = threshold
        if best_acc < acc:
            best_acc = acc
            threshold_best_acc = threshold
    
    print("\nBest MCC: ", mcc_max, " Threshold: ", threshold_best)
    acc,auc_roc,pr,kappa,sensitivity,specificity,mcc = process_result(np.array(lst_prob_median), test_labels, threshold_best)
    
    results = str(fold) + "," + '{0:.4f}'.format(acc) + "," + '{0:.4f}'.format(auc_roc) + "," + '{0:.4f}'.format(auc_pr) + "," + '{0:.4f}'.format(kappa) + "," + '{0:.4f}'.format(sensitivity) + "," + '{0:.4f}'.format(specificity) + "," + '{0:.4f}'.format(mcc) + "\n"
    
    print("\nBest MCC diff: ", diff, " Threshold: ", threshold_best_diff)
    acc,auc_roc,pr,kappa,sensitivity,specificity,mcc = process_result(np.array(lst_prob_median), test_labels, threshold_best_diff)
    
    results = str(fold) + "," + '{0:.4f}'.format(acc) + "," + '{0:.4f}'.format(auc_roc) + "," + '{0:.4f}'.format(auc_pr) + "," + '{0:.4f}'.format(kappa) + "," + '{0:.4f}'.format(sensitivity) + "," + '{0:.4f}'.format(specificity) + "," + '{0:.4f}'.format(mcc) + "\n"
    
    print("\nBest ACC: ", best_acc, " Threshold: ", threshold_best_acc)
    acc,auc_roc,pr,kappa,sensitivity,specificity,mcc = process_result(np.array(lst_prob_median), test_labels, threshold_best_acc)
    
    results = str(fold) + "," + '{0:.4f}'.format(acc) + "," + '{0:.4f}'.format(auc_roc) + "," + '{0:.4f}'.format(auc_pr) + "," + '{0:.4f}'.format(kappa) + "," + '{0:.4f}'.format(sensitivity) + "," + '{0:.4f}'.format(specificity) + "," + '{0:.4f}'.format(mcc) + "\n"
    
    file_results_summary.close()
    
def predict(file_request, file_result):
    MAX_NUMBER_OF_SEQUENCES = 10
    file_results_summary = open(file_result, "w")
    file_results_summary.write("Name,Sequence,Classification,Probability")
    
    text_file = open(file_request)
    lines = text_file.readlines()
    text_file.close()
    #print("Number of lines: ", len(lines))
    #print(lines[:2])
    lines_new = []
    for line in lines:
        line = line.strip()
        if len(line) > 0: lines_new.append(line)
        
    lst_seqs = []
    lst_names = []
    for line in lines_new[1::2]:
        seq = line.strip()
        lst_seqs.append(seq)
    for line in lines_new[::2]:
        seq_name = line.strip()
        lst_names.append(seq_name)    
        
    print("lst_seqs: ", lst_seqs)
    print("lst_names: ", lst_names)
    
    lst_seqs_ok = []
    lst_names_ok = []
    nb_seqs = 1
    for idx in range(len(lst_names)):
        seq_name = lst_names[idx].strip()
        seq = lst_seqs[idx].strip()
        print("seq_name: ", seq_name, " seq: ", seq)
        if seq_name == "": continue
        if seq_name[0] != ">": continue
        if len(seq) < 5: continue
        
        seq = seq.upper()
        
        if len(seq) > 30: seq = seq[:30]
        
        lst_names_ok.append(seq_name)
        lst_seqs_ok.append(seq)
        
        nb_seqs = nb_seqs + 1
        if nb_seqs == MAX_NUMBER_OF_SEQUENCES: break 
    
    print("lst_names_ok: ", lst_names_ok)
    print("lst_seqs_ok: ", lst_seqs_ok)
    
    if len(lst_seqs_ok) > 0: 
        data_test_coded = array_seq_coded(lst_seqs_ok)
        
        all_scores = []
        for fold in range(1, 11):
            print("\n Fold ", fold)
            FILE_MODEL = TRAIN_DIR + "fold_" + str(fold) + "_model.hdf5"
            
            model = quang_model()
            ind_scores = model.predict(data_test_coded)
            #print("ind_scores: ", ind_scores)
            
            model.load_weights(FILE_MODEL)
        
            ind_scores = model.predict(data_test_coded)
            #print("ind_scores: ", ind_scores)
            #print("ind_scores: ", ind_scores.shape)
            
            all_scores.append(ind_scores.reshape(-1).tolist())
            
            #if fold > 1: break
            #break
            
        all_scores = np.array(all_scores)
        
        nb_samples = all_scores.shape[1]
        print("nb_samples: ", nb_samples)
        
        lst_prob_median = []        
        for idx in range(nb_samples):
            probs = all_scores[:,idx]
            lst_prob_median.append(np.median(probs))
        
        #model = model_CNN()
        #FILE_MODEL = TRAIN_DIR + "fold_1_model.hdf5"
        #model.load_weights(FILE_MODEL)        
        
        #data_test_coded = array_seq_coded(test_seqs)
        #print("data_test_coded: ", data_test_coded.shape)
        #print(data_test_coded[:2])
        
        #scores = model.predict(data_test_coded)
        #print("scores: ", scores.shape, scores)
        
        for idx in range(len(lst_seqs_ok)):
            seq_name = lst_names_ok[idx]
            seq = lst_seqs_ok[idx]
            prob = lst_prob_median[idx]
            print("prob: ", prob)
            pred = "AMP"
            if prob < 0.5: pred = "Non AMP"
            result = seq_name + "," + seq + "," + pred + "," + '{0:.4f}'.format(prob)
            print(result)
            file_results_summary.write("\n" + result)
    
    file_results_summary.close()
    
##################################
if __name__== "__main__":
    print("Short Antimicrobial Peptides Prediction")    

    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv))
    
    if len(sys.argv) != 3:
        print("Error, please following Syntax: process_result_problem_amp.py file_request file_result")
        sys.exit(0)
        
    file_request = sys.argv[1]
    file_result = sys.argv[2]
    
    print("File request: ", file_request)
    print("File result: ", file_result)    
    
    # Amino acid properties
    #hydropathy_norm = param_norm(hydropathy)
    hydropathy_norm = hydropathy
    #volume_norm = param_norm(volume)
    volume_norm = volume
    #polarity_norm = param_norm(polarity)
    polarity_norm = polarity
    #pK_side_chain_norm = param_norm(pK_side_chain)
    pK_side_chain_norm = pK_side_chain
    #prct_exposed_residues_norm = param_norm(prct_exposed_residues)
    prct_exposed_residues_norm = prct_exposed_residues
    #hydrophilicity_norm = param_norm(hydrophilicity)
    hydrophilicity_norm = hydrophilicity
    #accessible_surface_area_norm = param_norm(accessible_surface_area)
    accessible_surface_area_norm = accessible_surface_area
    #local_flexibility_norm = param_norm(local_flexibility)
    local_flexibility_norm = local_flexibility
    #accessible_surface_area_folded_norm = param_norm(accessible_surface_area_folded)
    accessible_surface_area_folded_norm = accessible_surface_area_folded
    #refractivity_norm = param_norm(refractivity)
    refractivity_norm = refractivity
    #mass_norm = param_norm(mass)
    mass_norm = mass
    #solvent_exposed_area_norm = param_norm(solvent_exposed_area) 
    solvent_exposed_area_norm = solvent_exposed_area
    
    # Chou fasman table
    #alpha_helix_score_norm = param_norm(chou_fasman_norm(alpha_helix_score))
    alpha_helix_score_norm = chou_fasman_norm(alpha_helix_score)
    #beta_sheet_score_norm = param_norm(chou_fasman_norm(beta_sheet_score))
    beta_sheet_score_norm = chou_fasman_norm(beta_sheet_score)
    #turn_score_norm = param_norm(chou_fasman_norm(turn_score))
    turn_score_norm = chou_fasman_norm(turn_score)
    
    #stat_chars(train_seqs)
        
    predict(file_request, file_result)