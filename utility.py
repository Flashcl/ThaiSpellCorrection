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

ner = ThaiNameTagger()

def tokenize(text):
    return word_tokenize(text,engine='newmm')
#     return text.split()

def delNonAlphabetAndEmpty(s):
    out=[]
    th = [str(chr(a+ord('ก'))) for a in range(91)] 
    for w in s: 
        ss=''
        for i in range(len(w)):
            if w[i] in th or w[i].isalpha() or w[i].isnumeric() or w[i] in '-?.':
                ss+=w[i]
        if ss.strip() is not '' and (len(ss.strip())>=2 or ss in "ณa-"):
            out.append(ss.strip())
    return out

def replaceEntities(s):
    out=[]
    for w in s :
        if ner.get_ner(w)[0][1] in ['B-TIME', 'B-DATE', 'NUM',"PUNCT"]:
            out.append(ner.get_ner(w)[0][1])
        else:
            out.append(w)
    return out

def mergeOne(s):
    out = [s]
    cnt = 0
    for i in range(len(s)):
        if len(s[i])==1 and s[i] not in "-." and not s[i].isnumeric():
            for sen in out:
                tmp = []
                if i > 0:
                    tmp2 = sen.copy()
                    tmp2[i-1-cnt] = tmp2[i-1-cnt]+tmp2[i-cnt]
                    tmp2.pop(i-cnt)
                    tmp.append(tmp2)
                if i+1-cnt < len(sen):
                    tmp3 = sen.copy()
                    tmp3[i+1-cnt] = tmp3[i-cnt]+tmp3[i+1-cnt]
                    tmp3.pop(i-cnt)
                    tmp.append(tmp3)
            out = tmp
            cnt += 1 
    out = np.array(out)
    return out