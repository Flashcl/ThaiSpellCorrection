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
from utility import tokenize,delNonAlphabetAndEmpty,replaceEntities,mergeOne

def load():
    s = json.load(open('SpellCorrectionModel.json', 'r', encoding='utf8'))
    vocab_freqd =  s["Vocab Frequency"]
    vocab_freq = [(k,v) for k, v in vocab_freqd.items()]
    unk = 1 / len(vocab_freqd)
    model = defaultdict(lambda: 0) 
    for w1 in s["Model"]:
        if w1 not in model:
            model[w1] = defaultdict(lambda: 0) 
            
        for w2 in s["Model"][w1]:
            if w2 not in model[w1]:
                model[w1][w2] = defaultdict(lambda: 0) 
                
            for w3,val in s["Model"][w1][w2].items():
                if w3 not in model[w1][w2]:
                    model[w1][w2][w3] = val 
                
    lprob = s["lowest prob"]
    spellChecker = pn.NorvigSpellChecker(custom_dict=vocab_freq)
    return model, vocab_freq, unk, lprob, spellChecker

model, vocab_freq, unk, lprob, spellChecker = load()

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

def cartesianProduct(combi):
    cP = np.array(combi[0]).reshape((len(combi[0]),1))
    
    for i in range(len(combi)-1):
        a = np.tile(cP, (len(combi[i+1]), 1))
        b = np.repeat(np.array(combi[i+1]), len(cP)).reshape((len(cP)*len(combi[i+1]),1))
        cP = np.concatenate((a, b),axis=1)
        
    return cP

def takeFirst(elem):
    return elem[0]

def correct(sent,model=model,topK=5,threshold=lprob):
    # print(psutil.virtual_memory())

    sent = delNonAlphabetAndEmpty(tokenize(sent))
    sentences = mergeOne(sent)
    sentences = [replaceEntities(sent) for sent in sentences]
    combi = [list() for i in range(len(sentences))]
#     print(sentences,sent)

    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
            vocab = spellChecker.spell(sentences[i][j])
            if len(vocab) >= topK:
                combi[i].append(vocab[:topK])
            else:
                combi[i].append(vocab)

#     print(combi)
    cP = [cartesianProduct(c) for c in combi]
    del combi
    prob = list()
    
    for i in range(len(cP)):
        for j in range(len(cP[i])):
            prob.append((calculate_sentence_ln_prob(list(cP[i][j]),model),cP[i][j]))
    del cP

    prob = [p for p in prob if p[0] > threshold]
#     print(prob)
    if prob != []:
        prob.sort(key=takeFirst,reverse=True)
    #     maxp = prob[:topK]
        ans = prob[0][1]
    else:
        ans = sent
        prob.append((0,False))
        
    out =[]
    for i in range(len(ans)):
        if ans[i] in ['B-TIME', 'B-DATE', 'NUM',"PUNCT"]:
            out.append(sent[i])
        else:
            out.append(ans[i])
    return [(prob[0][0] or float("-inf"))," ".join(out)]

# x = input()
# start = time.time()
# print(correct(x))
# stop = time.time()
# print(stop-start)