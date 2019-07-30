import pn
import pandas as pd
import numpy as np
import nltk
from collections import defaultdict
from nltk import bigrams, trigrams
import math
from pythainlp import word_tokenize
import time
import multiprocessing as mp
from pythainlp.corpus import tnc
from pythainlp.tag.named_entity import ThaiNameTagger
import json
from pythainlp.util import thai_digit_to_arabic_digit
import psutil
from utility import tokenize,delNonAlphabetAndEmpty,mergeOne,replaceEntities

ner = ThaiNameTagger()
def inputData(filename):
    '''
    Should implement for your dataset
    '''
    df = pd.read_csv(filename)
    df = df.rename(index=str, columns={"hyp": "input", "grammar": "output"})
    return preClean(df)

def sp(text):
    return text.split()

def preClean (df):
    '''
    Should implement for your dataset
    '''
    df = df[df.output != '']
    df = df[~df.input.str.contains('<->')]
    df = df.drop_duplicates(subset=(['input']),keep='first')
    df = df[df.output.str.contains('TrueCallSteering')]
    df = df.drop(columns=['output'])
    df = df.drop(df[df["input"].map(sp).map(len) <= 1].index)
    return df

def createDict(train_df):
    '''
    create dict word to index
    '''
    global max_sentence_len 
    global word2index 
    global index2word
    global vocab_size 
    global vocab_freqd
#     max_sentence_len = 0
#     word2index = {"keras_mask_zero" : 0}
#     index2word = {0 : "keras_mask_zero"}

    word2index["UNK"] = len(word2index)
    index2word[len(index2word)] = "UNK"
    for s in train_df.values:
        max_sentence_len = max(len(s),max_sentence_len)
        for w in s:
            vocab_freqd[w]+=1
            if w not in word2index:
                word2index[w] = len(word2index)
                index2word[len(index2word)] = w
    
    vocab_size = len(word2index)
    print("vocab size :",vocab_size)
    print("max sentence length :",max_sentence_len)
    
def rep(text):
    return text.replace(" ","")

def textPreProcess(df):
    '''
    pre-process text dataset
    '''
    af = df["input"]
    af = af.apply(rep)
    af = af.apply(tokenize)
    af = af.apply(delNonAlphabetAndEmpty)
    af = af.apply(replaceEntities)
    return af

def findSent(key):
    '''
    find sentence in dataset that contains key
    '''
    for sent in data:
        if key in sent:
            print(sent)

def getLnValue(x):
    if x >0.0:
        return math.log(x)
    else:
        return math.log(unk)
    
def calculate_sentence_ln_prob(sentence, model):
    ln_prob = 0
    sentence = ["<s>"] + sentence + ["</s>"]
    for w1, w2, w3 in trigrams(sentence):
        if w1 in model:
            if w2 in model[w1]:
                ln_prob += getLnValue(model[w1][w2][w3])
            else:
                ln_prob += getLnValue(0)
        else:
            ln_prob += getLnValue(0)
    return ln_prob

def perplexity(test,model):
    sm_invprob = 0
    cnt = 0
    for sentence in test:
        sm_invprob += calculate_sentence_ln_prob(sentence, model)
        cnt += len(sentence)
    return math.exp(-sm_invprob / cnt)

def getUnigramModel(data):
    model = defaultdict(lambda: 0)
    word_count =0
    for sentence in data:
        for w1 in sentence:
            model[w1] +=1.0
            word_count+=1
    for w1 in model:
        model[w1] = model[w1]/(word_count)
    return model

def getBigramModel(data):
    count = dict()
    model = dict()
    for sentence in data:
        for w1, w2 in bigrams(sentence):
            if w1 not in count:
                count[w1] = 0
                model[w1] = defaultdict(lambda: 0)
            
            if w2 not in model[w1]:
                model[w1][w2] = 0
                
            count[w1] += 1
            model[w1][w2] += 1
            
    for w1 in model:
        for w2 in model[w1]:
            model[w1][w2] /= count[w1]
    return model

def getTrigramModel(data):
    count = dict()
    model = dict()
    for sentence in data:
        for w1, w2, w3 in trigrams(sentence):
            if w1 not in count:
                count[w1] = defaultdict(lambda: 0)
                model[w1] = defaultdict(lambda: 0)
            
            if w2 not in model[w1]:
                count[w1][w2] = 0
                model[w1][w2] = defaultdict(lambda: 0)
            
            if w3 not in model[w1][w2]:
                model[w1][w2][w3] = 0
                
            count[w1][w2] += 1
            model[w1][w2][w3] += 1
            
    for w1 in model:
        for w2 in model[w1]:
            for w3 in model[w1][w2]:
                model[w1][w2][w3] /= count[w1][w2]
    return model

def getTrigramWithInterpolation(data):
    data = [["<s>"] + sen + ["</s>"] for sen in data]
    model_uni = getUnigramModel(data)
    model_bi = getBigramModel(data)
    model_tri = getTrigramModel(data)
    
            
    global unk
    unk = 1 / len(word2index)
    for w1 in model_tri:
        sm = 0
        for w2 in model_tri[w1]:
            for w3 in model_tri[w1][w2]:
                model_tri[w1][w2][w3] = 0.65*model_tri[w1][w2][w3] + 0.25*model_bi[w1][w2] + 0.07*model_uni[w2] + 0.03*unk
                sm += model_tri[w1][w2][w3]
    
    return model_tri

def lowestProb(data, model):
    minProb = float("inf")
    for sentence in data:
        tmp = calculate_sentence_ln_prob(sentence, model)
        minProb = min(tmp,minProb)
    return minProb

def save(): 
    s = dict()
    s["Model"] = model
    s["Vocab Frequency"] = vocab_freqd
    s["lowest prob"] = lprob
    json.dump(s, open('SpellCorrectionModel.json', 'w', encoding='utf8'), ensure_ascii=False)
  
df = inputData("mari-speech.csv")
df = textPreProcess(df)

max_sentence_len = 0
word2index = {"keras_mask_zero" : 0}
index2word = {0 : "keras_mask_zero"}
vocab_freqd = defaultdict(int)
vocab_size = 0
createDict(df)
# for k2, v2 in tnc.word_freqs():
#     vocab_freqd[k2]+=int(v2)
vocab_freq = [(k,v) for k, v in vocab_freqd.items()]
spellChecker = pn.NorvigSpellChecker(custom_dict=vocab_freq)

data = [token for token in df]

model = getTrigramWithInterpolation(data)
lprob = lowestProb(data, model)   

save()