# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 14:20:18 2023

@author: ayaha
"""

import pandas as pd

def data_preprocessing(source_data, target_data):
    target_data.rename(columns={"sent":"target_lang", "sentID.BTEC":"ID"}, inplace = True)
    source_data.rename(columns={"sent":"source_lang", "sentID.BTEC":"ID"}, inplace = True)
    target_data_train = target_data[target_data["split"] == "corpus-6-test-corpus-26-train"]
    target_data_test =target_data[target_data["split"] == "corpus-6-test-corpus-26-test"]
    target_data_dev = target_data[target_data["split"] == "corpus-6-test-corpus-26-dev"]

    source_data_train = source_data[source_data["split"] == "corpus-6-test-corpus-26-train"]
    source_data_test = source_data[source_data["split"] == "corpus-6-test-corpus-26-test"]
    source_data_dev = source_data[source_data["split"] == "corpus-6-test-corpus-26-dev"]
    
    data_train = pd.merge(source_data_train, target_data_train, on ="ID")
    data_test = pd.merge( source_data_test, target_data_test, on ="ID")
    data_dev = pd.merge( source_data_dev, target_data_dev, on ="ID")
    data_test = pd.concat([data_test, data_dev], ignore_index=True)
    data = pd.concat([data_train,data_test ], ignore_index=True)
    data.reset_index(inplace= True)
    # data_test.reset_index(inplace= True)
    return data

def data_preprocessing_corpus_6(source_data, target_data):
    target_data.rename(columns={"sent":"target_lang", "sentID.BTEC":"ID"}, inplace = True)
    source_data.rename(columns={"sent":"source_lang", "sentID.BTEC":"ID"}, inplace = True)
    target_data_train = target_data[target_data["split"] == "corpus-6-test-corpus-26-train"]
    target_data_test =target_data[target_data["split"] == "corpus-6-test-corpus-26-test"]
    target_data_dev = target_data[target_data["split"] == "corpus-6-test-corpus-26-dev"]
    target_data_train_corpus_6 = target_data[target_data["split"] == "corpus-6-train"]
    target_data_dev_corpus_6= target_data[target_data["split"] == "corpus-6-dev"]
    
    source_data_train = source_data[source_data["split"] == "corpus-6-test-corpus-26-train"]
    source_data_test = source_data[source_data["split"] == "corpus-6-test-corpus-26-test"]
    source_data_dev = source_data[source_data["split"] == "corpus-6-test-corpus-26-dev"]
    source_data_train_corpus_6 = source_data[source_data["split"] == "corpus-6-train"]
    source_data_dev_corpus_6= source_data[source_data["split"] == "corpus-6-dev"]
    
    
    data_train = pd.merge(source_data_train, target_data_train, on ="ID")
    data_test = pd.merge( source_data_test, target_data_test, on ="ID")
    data_dev = pd.merge( source_data_dev, target_data_dev, on ="ID")
    data_train_corpus_6 = pd.merge(source_data_train_corpus_6, target_data_train_corpus_6, on ="ID")
    data_dev_corpus_6 = pd.merge(source_data_dev_corpus_6, target_data_dev_corpus_6, on ="ID")
    
    data = pd.concat([data_train, data_dev], ignore_index=True)
    data = pd.concat([data,data_test ], ignore_index=True)
    data = pd.concat([data, data_train_corpus_6],  ignore_index=True )
    data = pd.concat([data, data_dev_corpus_6],  ignore_index=True )
    data.reset_index(inplace= True)
    data_test.reset_index(inplace= True)
    return data




