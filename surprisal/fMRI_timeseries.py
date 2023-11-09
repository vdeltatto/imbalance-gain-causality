import numpy as np
import pandas as pd
from os import chdir, path
from scipy.stats import zscore
from tqdm import tqdm
import pickle
import re
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from scipy.special import softmax
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid.parasite_axes import SubplotHost

dirName = "/home/dev/Documents/PhD/MIT/project/fMRI"
chdir(dirName)

parentdir = path.dirname(dirName)
sys.path.insert(0, parentdir) 

SKIP_WORDS=20; END_WORDS=5176

words = np.load('fMRI/words_fmri.npy') #                               (5176,)
# s_wor = np.load('fMRI/stimuli_words.npy') #                          (5176,)
time = np.load('fMRI/time_fmri.npy') # [0,2,4,6,8,10 ... 2700]         (1351,)
runs = np.load('fMRI/runs_fmri.npy') # [1,1,1 ... 2,2,2 ... 4,4,4]     (1351,)
time_words = np.load('fMRI/time_words_fmri.npy') # [20,20.5,21 ..., 2692,2692.5,2693]  (5176,)
time_words = time_words[SKIP_WORDS:END_WORDS]    #                     (5156,)

part_f = np.load("fMRI/data_subject_F.npy") #                          (1211, 27905)
part_h = np.load("fMRI/data_subject_H.npy")
part_i = np.load("fMRI/data_subject_I.npy")
part_j = np.load("fMRI/data_subject_J.npy")
part_k = np.load("fMRI/data_subject_K.npy")
part_l = np.load("fMRI/data_subject_L.npy")
part_m = np.load("fMRI/data_subject_M.npy")
part_n = np.load("fMRI/data_subject_N.npy")

# The first 20TRs have been removed, as well as the last 15TR (tot = 35). 1211+(35*4)

words_id = np.zeros([len(time_words)]) # 5156
# w=find what TR each word belongs to; then I'll need to aggregate representations
for i in range(len(time_words)):
    words_id[i] = np.where(time_words[i]> time)[0][-1]
# I will need to aggregate SFL and EMB representations accordingly (see [1])

sentences = [[w for w in l.split(" _#_ ")+["+"] if w] for l in " _#_ ".join(list(words)).split("+")]
sentences = [" ".join(s) for s in sentences[:-1]]
sum([len(s.split()) for s in sentences]) # 5176, ok

#####################################################
# grouping sentences into passages (longer context) #
#####################################################

def divide_list(lst, sublist_length):
    sublists = [lst[i:i+sublist_length] for i in range(0, len(lst), sublist_length)]
    return [" ".join(lst) for lst in sublists]

def clean_string(string):
    # substituting out-of-vocabulary words
    str1 = re.sub("…", "...", string)
    str2 = re.sub("—", "-", str1)
    str3 = re.sub("‘", "'", str2)
    return str3

sentences = divide_list(sentences, 5) # context of 5 sentences!
sum([len(s.split()) for s in sentences]) # 5176, ok

with open("sentences.txt", 'w') as fp:
    for item in sentences:
        item = clean_string(item)
        fp.write("%s\n" % item)
    print('Done')

###############################################################################

def imputate_na(array):
    return np.where(np.isnan(array), ma.array(array, mask=np.isnan(array)).mean(axis=0), array)

def delay_one(mat, d):
        # delays a matrix by a delay d. Positive d ==> row t has row t-d
    new_mat = np.zeros_like(mat)
    if d>0:
        new_mat[d:] = mat[:-d]
    elif d<0:
        new_mat[:d] = mat[-d:]
    else:
        new_mat = mat
    return new_mat

def delay_mat(mat, delays):
        # delays a matrix by a set of delays d.
        # a row t in the returned matrix has the concatenated:
        # row(t-delays[0],t-delays[1]...t-delays[last] )
    new_mat = np.concatenate([delay_one(mat, d) for d in delays],axis = -1)
    return new_mat      # (1351,40)

def embed_words(embeddings):
    ids = words_id.astype(int)
    emb_words = []                         
    for i in range(time.shape[0]):
        emb = np.mean(embeddings[20:][ids==i], axis=0)
        emb_words.append(emb)
    emb_words = np.array(emb_words)
    tmp = delay_mat(emb_words, np.arange(1,5)) #  (1351, 3072) --> spillover
    # remove the edges of each run
    #tmp = np.vstack([zscore(tmp[runs==i][20:-15]) for i in range(1,5)]) # (1211, 40) ==> IMPORTANT how 1351 --> 1211
    tmp = np.vstack([tmp[runs==i][20:-15] for i in range(1,5)]) # (1211, 40) ==> IMPORTANT how 1351 --> 1211
    #tmp = np.nan_to_num(tmp) # changed this from Toneva - makes more sense to imputate
    tmp = imputate_na(tmp)
    return tmp

#######
# SFL #
#######

# frequency
freq = pd.read_excel("/home/dev/Documents/Datasets/subtlex.xlsx")
f = {row.Word : np.log(row.FREQcount) for index, row in freq.iterrows()}
minfreq = min(f.values())

def get_f(word):
    word = re.sub('[\.\,\:\-\?\!\)\(\"]', "", word)
    try:
        fr = f[word]
    except KeyError:
        fr = minfreq
    return fr

def get_surprisal(prompt):
    inputs = toker(prompt, return_tensors="pt")
    input_ids, output_ids = inputs["input_ids"], inputs["input_ids"][:, 1:]
    outputs = model(**inputs, labels=input_ids)
    logits = outputs.logits
    logprobs = torch.gather(F.log_softmax(logits, dim=2), 2, output_ids.unsqueeze(2))
    return [-item[0] for item in logprobs.tolist()[0]]

def tok_maker(a, sep, cased = False):
    # Credit to Ben S. https://stackoverflow.com/questions/74458282/match-strings-of-different-length-in-two-lists-of-different-length
    plainseq = " ".join(a)
    b = [re.sub(sep, "", item) for item in toker.tokenize(plainseq)]
    c = []
    if cased:
        for element in a:
            temp_list = []
            while "".join(temp_list) != element:
                temp_list.append(b.pop(0))
            c.append(temp_list)
    else:
        for element in a:
            temp_list = []
            while "".join(temp_list) != element.lower():
                temp_list.append(b.pop(0))
            c.append(temp_list)
    return c

def get_surprisal_tokens(tokens, sep, cased=False):
    s = get_surprisal(" ".join(tokens))
    toks = tok_maker(tokens, sep, cased)
    theindex = 0
    out = []
    for index, word in enumerate(toks[1:]):
        if len(word) == 1:
            surp = s[theindex]
            theindex += 1
            out.append(surp)
        else:
            surp = s[theindex:theindex+len(word)]
            theindex += len(word)
            out.append(sum(surp))
    return out

sent = [re.sub("\n", "", s).split() for s in open("sentences.txt").readlines()] 

freq = [get_f(word) for s in sent for word in s]
length = [len(word) for s in sent for word in s]

# @ VITTO non devi fare questa parte (ti mando la surprisal già calcolata così risparmi tempo)


# GPT
# model = AutoModelForCausalLM.from_pretrained("openai-gpt", return_dict_in_generate=True)
# toker = AutoTokenizer.from_pretrained("openai-gpt")
# s_gpt = [s for sent in tqdm(sent) for s in [np.nan]+get_surprisal_tokens(sent, sep = "</w>")] 
# # adding np.nan to keep track of beginning-of-sentence tokens

# # GPT2 SMALL (124M parameters)
# model = AutoModelForCausalLM.from_pretrained("gpt2", return_dict_in_generate=True)
# toker = AutoTokenizer.from_pretrained("gpt2")
# s_small = [s for sent in tqdm(sent) for s in [np.nan]+get_surprisal_tokens(sent, sep = "Ġ", cased=True)]

# # GPT2 medium
# model = AutoModelForCausalLM.from_pretrained("gpt2-medium", return_dict_in_generate=True)
# toker = AutoTokenizer.from_pretrained("gpt2-medium")
# s_med = [s for sent in tqdm(sent) for s in [np.nan]+get_surprisal_tokens(sent, sep = "Ġ", cased=True)]

# # GPT2 large
# model = AutoModelForCausalLM.from_pretrained("gpt2-large", return_dict_in_generate=True)
# toker = AutoTokenizer.from_pretrained("gpt2-large")
# s_large = [s for sent in tqdm(sent) for s in [np.nan]+get_surprisal_tokens(sent, sep = "Ġ", cased=True)]

# # GPT2 xl
# model = AutoModelForCausalLM.from_pretrained("gpt2-xl", return_dict_in_generate=True)
# toker = AutoTokenizer.from_pretrained("gpt2-xl")
# s_xl = [s for sent in tqdm(sent) for s in [np.nan]+get_surprisal_tokens(sent, sep = "Ġ", cased=True)]

# with open('SFL/gpt', 'wb') as handle:
#     pickle.dump(s_gpt, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('SFL/small', 'wb') as handle:
#     pickle.dump(s_small, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('SFL/med', 'wb') as handle:
#     pickle.dump(s_med, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('SFL/large', 'wb') as handle:
#     pickle.dump(s_large, handle, protocol=pickle.HIGHEST_PROTOCOL) 
# with open('SFL/xl', 'wb') as handle:
#     pickle.dump(s_xl, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('SFL/gpt', 'rb') as handle:
    s_gpt = pickle.load(handle)
with open('SFL/small', 'rb') as handle:
    s_small = pickle.load(handle)
with open('SFL/med', 'rb') as handle:
    s_med = pickle.load(handle)
with open('SFL/large', 'rb') as handle:
    s_large = pickle.load(handle)
with open('SFL/xl', 'rb') as handle:
    s_xl = pickle.load(handle)
    
#########
    
sfl_gpt = np.stack([np.array(s_gpt), np.array(freq), np.array(length)], axis=1)
sfl_gpt_X = embed_words(sfl_gpt)

sfl_small = np.stack([np.array(s_small), np.array(freq), np.array(length)], axis=1)
sfl_small_X = embed_words(sfl_small)

sfl_med = np.stack([np.array(s_med), np.array(freq), np.array(length)], axis=1)
sfl_med_X = embed_words(sfl_med)

sfl_large = np.stack([np.array(s_large), np.array(freq), np.array(length)], axis=1)
sfl_large_X = embed_words(sfl_large)

sfl_xl = np.stack([np.array(s_xl), np.array(freq), np.array(length)], axis=1)
sfl_xl_X = embed_words(sfl_xl)

#############################
# predicting BRAIN ACTIVITY #
#############################

# Important: differently from previous analyses, using Ridge regression (consistent with previous literature,
# also works better)
# IMPORTANT --> try with different penalty values

#from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV

def test_model_Ridge(X, y, n, saveto=None, save = True):
    kf = KFold(n_splits=n)
    out_reg = []
    out_coefs = []
    out_predictions = []
    for train_index, test_index in tqdm(kf.split(X), total=n):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        reg = RidgeCV(alphas=(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000), cv = 5)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        r, _ = pearsonr(y_test, y_pred)
        out_predictions.append([y_test, y_pred])
        coefs = reg.coef_#; print(coefs)
        out_coefs.append(coefs)
        out_reg.append(r)
    coefs_avg = np.mean(out_coefs, axis=0)
    coefs_std = np.std(out_coefs, axis=0)
    if save:
        with open("results/"+saveto+"_coefs_avg", 'wb') as handle:
            pickle.dump(coefs_avg, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open("results/"+saveto+"_coefs", 'wb') as handle:
            pickle.dump(out_coefs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open("results/"+saveto+"_coefs_std", 'wb') as handle:
            pickle.dump(coefs_std, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open("results/"+saveto+"_predictions", "wb") as handle:
            pickle.dump(out_reg, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return np.mean(out_reg), np.std(out_reg)

roi = np.load('fMRI/HP_subj_roi_inds.npy', allow_pickle=True)
roi = roi.tolist()

parts = [part_f, part_h, part_i, part_j, part_k, part_l, part_m, part_n]
partn = ["F", "H", "I", "J", "K", "L", "M", "N"]
roin =  ["all", "PostTemp", "AntTemp", "AngularG", "IFG", "MFG", "IFGorb", "pCingulate"]

######################################################
# averaging activity in each ROI across participants #
######################################################

brain_data_avg = {}
for the_roi in roin:
    out = []
    for part_act, part_name in zip(parts, partn):
        part = part_act[:,roi[part_name][the_roi]].mean(axis=1)
        out.append(part)
    out = np.array(out).mean(axis=0)
    brain_data_avg[the_roi] = out
    
results = {}
for roi in roin:
    print("\n\n\nProcessing", roi)
    base_gpt, base_gpt_sd = test_model_Ridge(sfl_gpt_X, brain_data_avg[roi], 10, "hp_"+roi+"_gpt_base")
    base_small, base_small_sd = test_model_Ridge(sfl_small_X, brain_data_avg[roi], 10, "hp_"+roi+"_small_base")
    base_med, base_med_sd = test_model_Ridge(sfl_med_X, brain_data_avg[roi], 10, "hp_"+roi+"_med_base")
    base_large, base_large_sd = test_model_Ridge(sfl_large_X, brain_data_avg[roi], 10, "hp_"+roi+"_large_base")
    base_xl, base_xl_sd = test_model_Ridge(sfl_xl_X, brain_data_avg[roi], 10, "hp_"+roi+"_xl_base")
    print(base_gpt, base_small, base_med, base_large, base_xl)