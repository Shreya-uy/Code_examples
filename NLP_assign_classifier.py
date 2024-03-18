#!/usr/bin/env python
# coding: utf-8

# # Generative model

# ## Data preprocessing

# In[3]:


import nltk
from sklearn.model_selection import train_test_split
import string, re, random
import sys
from nltk.lm.preprocessing import pad_both_ends, flatten,padded_everygram_pipeline
from nltk.util import bigrams,trigrams
from nltk.lm import MLE, Vocabulary
from nltk.lm.models import Laplace
from nltk.lm.models import KneserNeyInterpolated
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch
from transformers import RobertaTokenizer
import evaluate


# In[4]:

## Function to generate sample prompts from a given dataset

def generate_samples(seed_value, prompt_set):
    random.seed(seed_value)
    sample1 = prompt_set[random.randint(0,len(prompt_set))]
    sample2 = prompt_set[random.randint(0,len(prompt_set))]
    sample3 = prompt_set[random.randint(0,len(prompt_set))]
    sample4 = prompt_set[random.randint(0,len(prompt_set))]
    sample5 = prompt_set[random.randint(0,len(prompt_set))]
    return sample1,sample2,sample3,sample4,sample5


# In[5]:

## Function to train the model

def models_train(n,austen,dickens,tolstoy,wilde,vocab,model_type,discount = 1):
    if model_type == 'unigram' or 'bigram':
        model_aus,model_dickens,model_tol,model_wilde = MLE(n),MLE(n),MLE(n),MLE(n)
    if model_type == 'laplace':
        model_aus,model_dickens,model_tol,model_wilde = Laplace(n),Laplace(n),Laplace(n),Laplace(n)
    if model_type == 'kneser':
        model_aus,model_dickens,model_tol,model_wilde = KneserNeyInterpolated(n,discount),KneserNeyInterpolated(n,discount),KneserNeyInterpolated(n,discount),KneserNeyInterpolated(n,discount)
    model_aus.fit(austen,vocab)
    model_dickens.fit(dickens,vocab)
    model_tol.fit(tolstoy,vocab)
    model_wilde.fit(wilde,vocab)
    return model_aus,model_dickens,model_tol,model_wilde


# In[6]:


# Discriminative classifier

## Function to compute accuracy
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


# In[7]:


## Function to compute average perplexity of n samples
def average_perplexity(test_set,model,test_size):
    sum_p = 0
    for item in test_set:
        p = model.perplexity(item)
        sum_p += p
    avg_p = sum_p/test_size
    return avg_p


# In[8]:

## Function to generate average perplexity of the unigram model
def uni_avg_perp(test_set,model,test_size):
    count, sum_ug = 0,0
    seen = []
    for word in test_set:
        result = model.perplexity(word)
        if word not in seen:
            seen.append(word)
            if result < 1000000:
                sum_ug += result
    avg_perp = sum_ug / test_size
    return avg_perp


# In[97]:

## Classifier chosen and accuracy computed from 4 models based on the perplexity score
def classifier(test_set,model1,model2,model3,model4,t_a=1,t_d=1,t_t=1,t_w=1):
    Correct = {'Austen':0,'Dickens':0,'Tolstoy':0,'Wilde':0}
    sample_count = 0
    for [sentence,author] in test_set:
        min_perp = np.inf
        perp1 = model1.perplexity(sentence)
        if perp1 < min_perp:
            min_perp = perp1
            auth = 'Austen'
        perp2 = model2.perplexity(sentence)
        if perp2 < min_perp:
            min_perp = perp2
            auth = 'Dickens'
        perp3 = model3.perplexity(sentence)
        if perp3 < min_perp:
            min_perp = perp1
            auth = 'Tolstoy'
        perp4 = model4.perplexity(sentence)
        if perp4 < min_perp:
            min_perp = perp4
            auth = 'Wilde'
        if auth == author:
            Correct[auth] += 1
        sample_count += 1
    accuracy_aus = Correct['Austen']/t_a*100
    accuracy_dic = Correct['Dickens']/t_d*100
    accuracy_tol = Correct['Tolstoy']/t_t*100
    accuracy_wilde = Correct['Wilde']/t_w*100
    #accuracy_aus,accuracy_dic,accuracy_tol,accuracy_wilde
    return accuracy_aus,accuracy_dic,accuracy_tol,accuracy_wilde


# In[2]:

## Function to predict author based on perplexity
def predict(sentence,model1,model2,model3,model4):
    min_perp = np.inf
    perp1 = model1.perplexity(sentence)
    if perp1 < min_perp:
        min_perp = perp1
        auth = 'austen'
    perp2 = model2.perplexity(sentence)
    if perp2 < min_perp:
        min_perp = perp2
        auth = 'dickens'
    perp3 = model3.perplexity(sentence)
    if perp3 < min_perp:
        min_perp = perp1
        auth = 'tolstoy'
    perp4 = model4.perplexity(sentence)
    if perp4 < min_perp:
        min_perp = perp4
        auth = 'wilde'
    return auth


# In[98]:

## Preprocessing the datset for the discriminative model
def dataset_creation(sentlist1,author1,sentlist2,author2,sentlist3,author3,sentlist4,author4):
    token_sent1 = [' '.join(sentence) for sentence in sentlist1]
    token_sent2 = [' '.join(sentence) for sentence in sentlist2]
    token_sent3 = [' '.join(sentence) for sentence in sentlist3] 
    token_sent4 = [' '.join(sentence) for sentence in sentlist4]
    
    labels1 = len(sentlist1)*[author1]
    labels2 = len(sentlist2)*[author2]
    labels3 = len(sentlist3)*[author3]
    labels4 = len(sentlist4)*[author4]
    
    dataset = []
    first = list(zip(token_sent1,labels1))
    second = list(zip(token_sent2,labels2))
    third = list(zip(token_sent3,labels3))
    fourth = list(zip(token_sent4,labels4))
    
    dataset.extend(first)
    dataset.extend(second)
    dataset.extend(third)
    dataset.extend(fourth)
    
    train_data,test_data = train_test_split(dataset,test_size = 0.1,random_state=42)
    
    train_sent = list(map(lambda x:x[0],train_data))
    train_lab = list(map(lambda x:x[1],train_data))
    test_sent = list(map(lambda x:x[0],test_data))
    test_lab = list(map(lambda x:x[1],test_data))
    
    train_data_dict = {'text': train_sent,'label': train_lab}
    test_data_dict = {'text': test_sent,'label': test_lab}
    train_dataset = Dataset.from_dict(train_data_dict)
    test_dataset = Dataset.from_dict(test_data_dict)
    
    dataset_dict = DatasetDict({
     'train': train_dataset,
      'validation': test_dataset})
    return dataset_dict


# In[99]:

## Tokenize the data
def preprocess_function(inp_text):
    roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    return roberta_tokenizer(inp_text['text'], truncation=True)


# In[100]:


## Compute the accuracy for the discriminative classifier
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


# In[ ]:


## Function to remove special characters and convert text to lowercase
def clean(text):
    pattern = re.compile(r'[^a-zA-Z0-9\s]')
    remove_sc = pattern.sub(' ',text)
    allcase = remove_sc.lower()
    return allcase


# In[16]:

## Main function to run the 2 models
def main():    
    
    # Get the authorlist from the file
    filename = sys.argv[1]
    authorfile = open(filename,'r')
    authorlist_orig = authorfile.readlines()
    authorlist = [line.strip() for line in authorlist_orig]
    
    # Check whether we are using the given test set or generating the test set
    approach = sys.argv[3]
    flag = False
    
    if len(sys.argv)>3:
        flag = True
    if flag == True:
        test_name = sys.argv[5] 
        testfile = open(test_name,'r')
    
    # Data Preprocessing for both models
    ## Read the files
    aus = open(authorlist[0],'r')
    data_aus = aus.read()
    dickens = open(authorlist[1],'r')
    data_dickens = dickens.read()
    tol = open(authorlist[2],'r')
    data_tol = tol.read()
    wilde = open(authorlist[3],'r')
    data_wilde = wilde.read()

    ## Split text into sentences
    austen = data_aus.split('.')
    dickens = data_dickens.split('.')
    tolstoy = data_tol.split('.')
    wilde = data_wilde.split('.')

    # Remove special characters and lowercase
    data_austen,data_dickens,data_tolstoy,data_wilde = [],[],[],[]
    for sent in austen:
        sentence = clean(sent).split(' ')
        new_sentence = []
        for char in sentence:
                if char != '':
                    if '\n' in char:
                        newchar = char.replace('\n',' ')
                        new_sentence.append(newchar)
                    else:
                        new_sentence.append(char)
        data_austen.append(new_sentence)
    for sent in dickens:
        sentence = clean(sent).split(' ')
        new_sentence = []
        for char in sentence:
                if char != '':
                    if '\n' in char:
                        newchar = char.replace('\n',' ')
                        new_sentence.append(newchar)
                    else:
                        new_sentence.append(char)
        data_dickens.append(new_sentence)
    for sent in tolstoy:
        sentence = clean(sent).split(' ')
        new_sentence = []
        for char in sentence:
                if char != '':
                    if '\n' in char:
                        newchar = char.replace('\n',' ')
                        new_sentence.append(newchar)
                    else:
                        new_sentence.append(char)
        data_tolstoy.append(new_sentence)
    for sent in wilde:
        sentence = clean(sent).split(' ')
        new_sentence = []
        for char in sentence:
                if char != '':
                    if '\n' in char:
                        newchar = char.replace('\n',' ')
                        new_sentence.append(newchar)
                    else:
                        new_sentence.append(char)
        data_wilde.append(new_sentence)

    # Splitting into training and test data (if test set is not given)

    austen_train_, austen_test_ = train_test_split(data_austen,test_size=0.1, random_state=15)
    dickens_train_, dickens_test_ = train_test_split(data_dickens,test_size=0.1, random_state=20)
    tolstoy_train_, tolstoy_test_ = train_test_split(data_tolstoy,test_size=0.1, random_state=30)
    wilde_train_, wilde_test_ = train_test_split(data_wilde,test_size=0.1, random_state=42)
    
    if flag == True:
        austen_train_ = data_austen
        dickens_train_ = data_dickens
        tolstoy_train_ = data_tolstoy
        wilde_train_ = data_wilde
            
    if flag == True:
        ## Preprocess the test dataset

        ## Split text into sentences
        text = testfile.readlines()

        # Remove special characters and lowercase
        test_text = []
        for sent in text:
            sentence = clean(sent).split(' ')
            new_sentence = []
            for char in sentence:
                    if char != '':
                        if '\n' in char:
                            newchar = char.replace('\n',' ')
                            new_sentence.append(newchar)
                        else:
                            new_sentence.append(char)
            test_text.append(new_sentence)     

        prep_test_bg = list(flatten(pad_both_ends(sent, n=2) for sent in test_text))
        test_bg = list(bigrams(prep_test_bg))

        count = 0
        test_set_indices,bg_test = [],[]
        for word in prep_test_bg:
            if word == "<s>":
                s_ind = count
            count += 1
            if word == "</s>":
                e_ind = count
                test_set_indices.append([s_ind,e_ind])

        # Reconstruct the sentences from the tokenized dataset
        for ind in range(len(test_set_indices)):
            sentence = test_bg[test_set_indices[ind][0]:test_set_indices[ind][1]-1]
            bg_test.append(sentence)
    
    ## Remove empty sentences for training dataset
    austen_train,dickens_train,tolstoy_train,wilde_train = [],[],[],[]
    for sent in austen_train_:
        if sent != []:
            austen_train.append(sent)
    for sent in dickens_train_:
         if sent != []:
            dickens_train.append(sent)
    for sent in tolstoy_train_:
        if sent != []:
            tolstoy_train.append(sent)
    for sent in wilde_train_:
        if sent != []:
            wilde_train.append(sent)
    
    ## Remove empty sentences and collect sentence lengths for the test dataset
    austen_test,dickens_test,tolstoy_test,wilde_test = [],[],[],[]
    sentlen_austen,sentlen_dickens,sentlen_tolstoy,sentlen_wilde = [],[],[],[]
    for sent in austen_test_:
        if sent != []:
            austen_test.append(sent)
            sentlen_austen.append(len(sent))
    for sent in dickens_test_:
         if sent != []:
            dickens_test.append(sent)
            sentlen_dickens.append(len(sent))
    for sent in tolstoy_test_:
        if sent != []:
            tolstoy_test.append(sent)
            sentlen_tolstoy.append(len(sent))
    for sent in wilde_test_:
        if sent != []:
            wilde_test.append(sent)
            sentlen_wilde.append(len(sent))

    if approach == 'generative':
        ## Bigram Models

        ## Train data preprocessing (Bigram models)

        prep_austen_train_bg = list(flatten(pad_both_ends(sent, n=2) for sent in austen_train))
        prep_dickens_train_bg = list(flatten(pad_both_ends(sent, n=2) for sent in dickens_train))
        prep_tolstoy_train_bg = list(flatten(pad_both_ends(sent, n=2) for sent in tolstoy_train))
        prep_wilde_train_bg = list(flatten(pad_both_ends(sent, n=2) for sent in wilde_train))

        # Preprocessing pipeline of the train data for the model
        train_austen_bg,vocab_austen_bg = padded_everygram_pipeline(2, austen_train)
        train_dickens_bg,vocab_dickens_bg = padded_everygram_pipeline(2, dickens_train)
        train_tolstoy_bg,vocab_tolstoy_bg = padded_everygram_pipeline(2, tolstoy_train)
        train_wilde_bg,vocab_wilde_bg = padded_everygram_pipeline(2, wilde_train)

        # Train data preprocessed for perplexity computation
        train_austen_perp_bg = list(bigrams(prep_austen_train_bg))
        train_dickens_perp_bg = list(bigrams(prep_dickens_train_bg))
        train_tolstoy_perp_bg = list(bigrams(prep_tolstoy_train_bg))
        train_wilde_perp_bg = list(bigrams(prep_wilde_train_bg))
 
        ## Test data preprocessing

        prep_austen_test_bg = list(flatten(pad_both_ends(sent, n=2) for sent in austen_test))
        prep_dickens_test_bg = list(flatten(pad_both_ends(sent, n=2) for sent in dickens_test))
        prep_tolstoy_test_bg = list(flatten(pad_both_ends(sent, n=2) for sent in tolstoy_test))
        prep_wilde_test_bg = list(flatten(pad_both_ends(sent, n=2) for sent in wilde_test))

        test_austen_bg = list(bigrams(prep_austen_test_bg))
        test_dickens_bg = list(bigrams(prep_dickens_test_bg))
        test_tolstoy_bg = list(bigrams(prep_tolstoy_test_bg))
        test_wilde_bg = list(bigrams(prep_wilde_test_bg))

        ### Create the bigrams for sampling prompts 

        ### Train samples
        prompt_austen_bg, prompt_dickens_bg, prompt_tolstoy_bg, prompt_wilde_bg = [],[],[],[]
        for item in train_austen_perp_bg:
            if '<s>' in item or '</s>' in item:
                continue
            else:
                prompt_austen_bg.append(item)
        for item in train_dickens_perp_bg:
            if '<s>' in item or '</s>' in item:
                continue
            else:
                prompt_dickens_bg.append(item)
        for item in train_tolstoy_perp_bg:
            if '<s>' in item or '</s>' in item:
                continue
            else:
                prompt_tolstoy_bg.append(item)
        for item in train_wilde_perp_bg:
            if '<s>' in item or '</s>' in item:
                continue
            else:
                prompt_wilde_bg.append(item)

        ### Test samples

        for item in test_austen_bg:
            if '<s>' in item or '</s>' in item:
                continue
            else:
                prompt_austen_bg.append(item)
        for item in test_dickens_bg:
            if '<s>' in item or '</s>' in item:
                continue
            else:
                prompt_dickens_bg.append(item)
        for item in test_tolstoy_bg:
            if '<s>' in item or '</s>' in item:
                continue
            else:
                prompt_tolstoy_bg.append(item)
        for item in test_wilde_bg:
            if '<s>' in item or '</s>' in item:
                continue
            else:
                prompt_wilde_bg.append(item)

        ### Create indices for test set sentences

        count = 0
        test_set_indices_austen = []
        for word in prep_austen_test_bg:
            if word == "<s>":
                s_ind = count
            count += 1
            if word == "</s>":
                e_ind = count
                test_set_indices_austen.append([s_ind,e_ind])

        count = 0
        test_set_indices_dickens = []
        for word in prep_dickens_test_bg:
            if word == "<s>":
                s_ind = count
            count += 1
            if word == "</s>":
                e_ind = count
                test_set_indices_dickens.append([s_ind,e_ind])

        count = 0
        test_set_indices_tolstoy = []
        for word in prep_tolstoy_test_bg:
            if word == "<s>":
                s_ind = count
            count += 1
            if word == "</s>":
                e_ind = count
                test_set_indices_tolstoy.append([s_ind,e_ind])

        count = 0
        test_set_indices_wilde = []
        for word in prep_wilde_test_bg:
            if word == "<s>":
                s_ind = count
            count += 1
            if word == "</s>":
                e_ind = count
                test_set_indices_wilde.append([s_ind,e_ind])

        ### Test set of sentences (Bigram)

        bg_aus_test, bg_dic_test,bg_tol_test,bg_wilde_test = [],[],[],[]
        Test_set_global_bg = []
        for ind in range(len(test_set_indices_austen)):
            sentence = test_austen_bg[test_set_indices_austen[ind][0]:test_set_indices_austen[ind][1]-1]
            bg_aus_test.append(sentence)
            Test_set_global_bg.append([sentence,'Austen'])
        for ind in range(len(test_set_indices_dickens)):
            sentence = test_dickens_bg[test_set_indices_dickens[ind][0]:test_set_indices_dickens[ind][1]-1]
            bg_dic_test.append(sentence)
            Test_set_global_bg.append([sentence,'Dickens'])
        for ind in range(len(test_set_indices_tolstoy)):
            sentence = test_tolstoy_bg[test_set_indices_tolstoy[ind][0]:test_set_indices_tolstoy[ind][1]-1]
            bg_tol_test.append(sentence)
            Test_set_global_bg.append([sentence,'Tolstoy'])
        for ind in range(len(test_set_indices_wilde)):
            sentence = test_wilde_bg[test_set_indices_wilde[ind][0]:test_set_indices_wilde[ind][1]-1]
            bg_wilde_test.append(sentence)
            Test_set_global_bg.append([sentence,'Wilde'])

        ## Preprocessing test data with author labels

        test_size_aus = len(bg_aus_test)
        test_size_dic = len(bg_dic_test)
        test_size_tol = len(bg_tol_test)
        test_size_wilde = len(bg_wilde_test)

        # Creating a shared vocabulary

        train_austen_forvoc,vocab_austen_forvoc = padded_everygram_pipeline(2, austen_train)
        train_dickens_forvoc,vocab_dickens_forvoc = padded_everygram_pipeline(2, dickens_train)
        train_tolstoy_forvoc,vocab_tolstoy_forvoc = padded_everygram_pipeline(2, tolstoy_train)
        train_wilde_forvoc,vocab_wilde_forvoc = padded_everygram_pipeline(2, wilde_train)

        lm_aus_forvoc = MLE(2)
        lm_dickens_forvoc = MLE(2)
        lm_tol_forvoc = MLE(2)
        lm_wilde_forvoc = MLE(2)

        lm_aus_forvoc.fit(train_austen_forvoc, vocab_austen_forvoc)
        lm_dickens_forvoc.fit(train_dickens_forvoc, vocab_dickens_forvoc)
        lm_tol_forvoc.fit(train_tolstoy_forvoc, vocab_tolstoy_forvoc)
        lm_wilde_forvoc.fit(train_wilde_forvoc, vocab_wilde_forvoc)

        ausvoc = list(lm_aus_forvoc.vocab)
        dicvoc = list(lm_dickens_forvoc.vocab)
        tolvoc = list(lm_tol_forvoc.vocab)
        wilvoc = list(lm_wilde_forvoc.vocab)

        master_vocab = []
        master_vocab.extend(ausvoc)

        for word in dicvoc:
            if word not in master_vocab:
                master_vocab.append(word)
        for word in tolvoc:
            if word not in master_vocab:
                master_vocab.append(word)
        for word in wilvoc:
            if word not in master_vocab:
                master_vocab.append(word)

        Main_vocab = Vocabulary(master_vocab, unk_cutoff=1)

        # Randomly choosing bigram prompts from the corpus(train/test set)

        aus_prompt1_bg,aus_prompt2_bg,aus_prompt3_bg,aus_prompt4_bg,aus_prompt5_bg = generate_samples(41,prompt_austen_bg)
        dic_prompt1_bg,dic_prompt2_bg,dic_prompt3_bg,dic_prompt4_bg,dic_prompt5_bg = generate_samples(33,prompt_dickens_bg)
        tol_prompt1_bg,tol_prompt2_bg,tol_prompt3_bg,tol_prompt4_bg,tol_prompt5_bg = generate_samples(12,prompt_tolstoy_bg)
        wilde_prompt1_bg,wilde_prompt2_bg,wilde_prompt3_bg,wilde_prompt4_bg,wilde_prompt5_bg = generate_samples(20,prompt_wilde_bg)

        # Preprocessing the prompts

        prep_gen_aus_1 = list(bigrams(aus_prompt1_bg))
        prep_gen_aus_2 = list(bigrams(aus_prompt2_bg))
        prep_gen_aus_3 = list(bigrams(aus_prompt3_bg))
        prep_gen_aus_4 = list(bigrams(aus_prompt4_bg))
        prep_gen_aus_5 = list(bigrams(aus_prompt5_bg))

        prep_gen_dic_1 = list(bigrams(dic_prompt1_bg))
        prep_gen_dic_2 = list(bigrams(dic_prompt2_bg))
        prep_gen_dic_3 = list(bigrams(dic_prompt3_bg))
        prep_gen_dic_4 = list(bigrams(dic_prompt4_bg))
        prep_gen_dic_5 = list(bigrams(dic_prompt5_bg))

        prep_gen_tol_1 = list(bigrams(tol_prompt1_bg))
        prep_gen_tol_2 = list(bigrams(tol_prompt2_bg))
        prep_gen_tol_3 = list(bigrams(tol_prompt3_bg))
        prep_gen_tol_4 = list(bigrams(tol_prompt4_bg))
        prep_gen_tol_5 = list(bigrams(tol_prompt5_bg))

        prep_gen_wilde_1 = list(bigrams(wilde_prompt1_bg))
        prep_gen_wilde_2 = list(bigrams(wilde_prompt2_bg))
        prep_gen_wilde_3 = list(bigrams(wilde_prompt3_bg))
        prep_gen_wilde_4 = list(bigrams(wilde_prompt4_bg))
        prep_gen_wilde_5 = list(bigrams(wilde_prompt5_bg))

        ## Train an ordinary bigram MLE model
        lm_aus_bg, lm_dickens_bg, lm_tol_bg,lm_wilde_bg = models_train(2,train_austen_bg,train_dickens_bg,train_tolstoy_bg,train_wilde_bg,Main_vocab,'bigram')


        ## Compute average perplexity of the test set to compare models

        avg_bg_aus = average_perplexity(bg_aus_test,lm_aus_bg,test_size_aus)
        avg_bg_dic = average_perplexity(bg_dic_test,lm_dickens_bg,test_size_aus)
        avg_bg_tol = average_perplexity(bg_tol_test,lm_tol_bg,test_size_aus)
        avg_bg_wilde = average_perplexity(bg_wilde_test,lm_wilde_bg,test_size_aus)

        #print(f"The average perplexity of the bigram MLE for the austin test set is {avg_bg_aus}")

        ## Generate 5 sample sentences to compute perplexity

        sent_aus_bg_1 = lm_aus_bg.generate(1,aus_prompt1_bg,random_seed = 1)
        sent_aus_bg_2 = lm_aus_bg.generate(1,aus_prompt2_bg,random_seed = 5)
        sent_aus_bg_3 = lm_aus_bg.generate(1,aus_prompt3_bg,random_seed = 30)
        sent_aus_bg_4 = lm_aus_bg.generate(1,aus_prompt4_bg,random_seed = 45)
        sent_aus_bg_5 = lm_aus_bg.generate(1,aus_prompt5_bg,random_seed = 35)

        sent_dic_bg_1 = lm_dickens_bg.generate(1,dic_prompt1_bg, random_seed = 25)
        sent_dic_bg_2 = lm_dickens_bg.generate(1,dic_prompt2_bg,random_seed = 13)
        sent_dic_bg_3 = lm_dickens_bg.generate(1,dic_prompt3_bg,random_seed = 55)
        sent_dic_bg_4 = lm_dickens_bg.generate(1,dic_prompt4_bg,random_seed = 10)
        sent_dic_bg_5 = lm_dickens_bg.generate(1,dic_prompt5_bg,random_seed = 6)

        sent_tol_bg_1 = lm_tol_bg.generate(1,tol_prompt1_bg,random_seed = 20)
        sent_tol_bg_2 = lm_tol_bg.generate(1,tol_prompt2_bg,random_seed = 65)
        sent_tol_bg_3 = lm_tol_bg.generate(1,tol_prompt3_bg,random_seed = 41)
        sent_tol_bg_4 = lm_tol_bg.generate(1,tol_prompt4_bg,random_seed = 1)
        sent_tol_bg_5 = lm_tol_bg.generate(1,tol_prompt5_bg,random_seed = 8)

        sent_wilde_bg_1 = lm_wilde_bg.generate(1,wilde_prompt1_bg,random_seed = 33)
        sent_wilde_bg_2 = lm_wilde_bg.generate(1,wilde_prompt2_bg,random_seed = 18)
        sent_wilde_bg_3 = lm_wilde_bg.generate(1,wilde_prompt3_bg,random_seed = 53)
        sent_wilde_bg_4 = lm_wilde_bg.generate(1,wilde_prompt4_bg,random_seed = 8)
        sent_wilde_bg_5 = lm_wilde_bg.generate(1,wilde_prompt5_bg,random_seed = 29)

        ## Print perplexity of the generated sentences

        print("The Perplexities of the sentences generated using the un-smoothed bigram model for the Austen text are:")
        print(f"Generated instance 1: is {lm_aus_bg.perplexity(sent_aus_bg_1)}")
        print(f"Generated instance 2: is {lm_aus_bg.perplexity(sent_aus_bg_2)}")
        print(f"Generated instance 3: is {lm_aus_bg.perplexity(sent_aus_bg_3)}")
        print(f"Generated instance 4: is {lm_aus_bg.perplexity(sent_aus_bg_4)}")
        print(f"Generated instance 5: is {lm_aus_bg.perplexity(sent_aus_bg_5)}")

        print("The Perplexities of the sentences generated using the un-smoothed bigram model for the Dickens text are:")
        print(f"Generated instance 1: {lm_dickens_bg.perplexity(sent_dic_bg_1)}")
        print(f"Generated instance 2: {lm_dickens_bg.perplexity(sent_dic_bg_2)}")
        print(f"Generated instance 3: {lm_dickens_bg.perplexity(sent_dic_bg_3)}")
        print(f"Generated instance 4: {lm_dickens_bg.perplexity(sent_dic_bg_4)}")
        print(f"Generated instance 5: {lm_dickens_bg.perplexity(sent_dic_bg_5)}")

        print("The Perplexities of the sentences generated using the un-smoothed bigram model for the Tolstoy text are:")
        print(f"Generated instance 1: {lm_tol_bg.perplexity(sent_tol_bg_1)}")
        print(f"Generated instance 2: {lm_tol_bg.perplexity(sent_tol_bg_2)}")
        print(f"Generated instance 3: {lm_tol_bg.perplexity(sent_tol_bg_3)}")
        print(f"Generated instance 4: {lm_tol_bg.perplexity(sent_tol_bg_4)}")
        print(f"Generated instance 5: {lm_tol_bg.perplexity(sent_tol_bg_5)}")

        print("The Perplexities of the sentences generated using the un-smoothed bigram model for the Wilde text are:")
        print(f"Generated instance 1: {lm_wilde_bg.perplexity(sent_wilde_bg_1)}")
        print(f"Generated instance 2: {lm_wilde_bg.perplexity(sent_wilde_bg_2)}")
        print(f"Generated instance 3: {lm_wilde_bg.perplexity(sent_wilde_bg_3)}")
        print(f"Generated instance 4: {lm_wilde_bg.perplexity(sent_wilde_bg_4)}")
        print(f"Generated instance 5: {lm_wilde_bg.perplexity(sent_wilde_bg_5)}")


        ## Unigram model

        ### Data preprocessing

        prep_austen_train_ug = list(flatten(pad_both_ends(sent, n=1) for sent in austen_train))
        prep_dickens_train_ug = list(flatten(pad_both_ends(sent, n=1) for sent in dickens_train))
        prep_tolstoy_train_ug = list(flatten(pad_both_ends(sent, n=1) for sent in tolstoy_train))
        prep_wilde_train_ug = list(flatten(pad_both_ends(sent, n=1) for sent in wilde_train))

        prep_austen_test_ug = list(flatten(pad_both_ends(sent, n=1) for sent in austen_test))
        prep_dickens_test_ug = list(flatten(pad_both_ends(sent, n=1) for sent in dickens_test))
        prep_tolstoy_test_ug = list(flatten(pad_both_ends(sent, n=1) for sent in tolstoy_test))
        prep_wilde_test_ug = list(flatten(pad_both_ends(sent, n=1) for sent in wilde_test))

        train_austen_ug,vocab_austen_ug = padded_everygram_pipeline(1, austen_train)
        train_dickens_ug,vocab_dickens_ug = padded_everygram_pipeline(1, dickens_train)
        train_tolstoy_ug,vocab_tolstoy_ug = padded_everygram_pipeline(1, tolstoy_train)
        train_wilde_ug,vocab_wilde_ug = padded_everygram_pipeline(1, wilde_train)

        ### Generating 5 prompts

        aus_prompt1_ug,aus_prompt2_ug,aus_prompt3_ug,aus_prompt4_ug,aus_prompt5_ug = generate_samples(42,austen_test)
        dic_prompt1_ug,dic_prompt2_ug,dic_prompt3_ug,dic_prompt4_ug,dic_prompt5_ug = generate_samples(33,dickens_test)
        tol_prompt1_ug,tol_prompt2_ug,tol_prompt3_ug,tol_prompt4_ug,tol_prompt5_ug = generate_samples(12,tolstoy_test)
        wilde_prompt1_ug,wilde_prompt2_ug,wilde_prompt3_ug,wilde_prompt4_ug,wilde_prompt5_ug = generate_samples(20,wilde_test)

        ## Train the unigram models

        lm_aus_ug, lm_dickens_ug, lm_tol_ug,lm_wilde_ug = models_train(2,train_austen_ug,train_dickens_ug,train_tolstoy_ug,train_wilde_ug,Main_vocab,'unigram')

        # Check average perplexity for comparison
        avg_ug_aus = uni_avg_perp(prep_austen_test_ug,lm_aus_ug,test_size_aus)
        avg_ug_dic = uni_avg_perp(prep_dickens_test_ug,lm_dickens_ug,test_size_dic)
        avg_ug_tol = uni_avg_perp(prep_tolstoy_test_ug,lm_tol_ug,test_size_tol)
        avg_ug_wilde = uni_avg_perp(prep_wilde_test_ug,lm_wilde_ug,test_size_wilde)

        #print(f"The average perplexity of the unigram MLE for the Austen test set is {avg_ug_aus}")

        ### Preprocessing prompts for the unigram model

        sent_aus_ug_1 = lm_aus_ug.generate(1,aus_prompt1_ug,random_seed = 1)
        sent_aus_ug_2 = lm_aus_ug.generate(1,aus_prompt2_ug,random_seed = 5)
        sent_aus_ug_3 = lm_aus_ug.generate(1,aus_prompt3_ug,random_seed = 30)
        sent_aus_ug_4 = lm_aus_ug.generate(1,aus_prompt4_ug,random_seed = 45)
        sent_aus_ug_5 = lm_aus_ug.generate(1,aus_prompt5_ug,random_seed = 35)

        sent_dic_ug_1 = lm_dickens_ug.generate(1,dic_prompt1_ug, random_seed = 25)
        sent_dic_ug_2 = lm_dickens_ug.generate(1,dic_prompt2_ug,random_seed = 13)
        sent_dic_ug_3 = lm_dickens_ug.generate(1,dic_prompt3_ug,random_seed = 55)
        sent_dic_ug_4 = lm_dickens_ug.generate(1,dic_prompt4_ug,random_seed = 10)
        sent_dic_ug_5 = lm_dickens_ug.generate(1,dic_prompt5_ug,random_seed = 6)

        sent_tol_ug_1 = lm_tol_ug.generate(1,tol_prompt1_ug,random_seed = 20)
        sent_tol_ug_2 = lm_tol_ug.generate(1,tol_prompt2_ug,random_seed = 65)
        sent_tol_ug_3 = lm_tol_ug.generate(1,tol_prompt3_ug,random_seed = 41)
        sent_tol_ug_4 = lm_tol_ug.generate(1,tol_prompt4_ug,random_seed = 1)
        sent_tol_ug_5 = lm_tol_ug.generate(1,tol_prompt5_ug,random_seed = 8)

        sent_wilde_ug_1 = lm_wilde_ug.generate(1,wilde_prompt1_ug,random_seed = 33)
        sent_wilde_ug_2 = lm_wilde_ug.generate(1,wilde_prompt2_ug,random_seed = 18)
        sent_wilde_ug_3 = lm_wilde_ug.generate(1,wilde_prompt3_ug,random_seed = 53)
        sent_wilde_ug_4 = lm_wilde_ug.generate(1,wilde_prompt4_ug,random_seed = 8)
        sent_wilde_ug_5 = lm_wilde_ug.generate(1,wilde_prompt5_ug,random_seed = 29)

        ## Computing perplexity of the generated sentences

        print("The Perplexities of the sentences generated using the un-smoothed unigram model for the Austen text are")
        print(f"Generated instance 1: {lm_aus_ug.perplexity(prep_gen_aus_1)}")
        print(f"Generated instance 2: {lm_aus_ug.perplexity(prep_gen_aus_2)}")
        print(f"Generated instance 3: {lm_aus_ug.perplexity(prep_gen_aus_3)}")
        print(f"Generated instance 4: {lm_aus_ug.perplexity(prep_gen_aus_4)}")
        print(f"Generated instance 5: {lm_aus_ug.perplexity(prep_gen_aus_5)}")

        print("The Perplexities of the sentences generated using the un-smoothed unigram model for the Dickens text are")
        print(f"Generated instance 1: {lm_dickens_ug.perplexity(prep_gen_dic_1)}")
        print(f"Generated instance 2: {lm_dickens_ug.perplexity(prep_gen_dic_2)}")
        print(f"Generated instance 3: {lm_dickens_ug.perplexity(prep_gen_dic_3)}")
        print(f"Generated instance 4: {lm_dickens_ug.perplexity(prep_gen_dic_4)}")
        print(f"Generated instance 5: {lm_dickens_ug.perplexity(prep_gen_dic_5)}")

        print("The Perplexities of the sentences generated using the un-smoothed unigram model for the Tolstoy text are")
        print(f"Generated instance 1: {lm_tol_ug.perplexity(prep_gen_tol_1)}")
        print(f"Generated instance 2: {lm_tol_ug.perplexity(prep_gen_tol_2)}")
        print(f"Generated instance 3: {lm_tol_ug.perplexity(prep_gen_tol_3)}")
        print(f"Generated instance 4: {lm_tol_ug.perplexity(prep_gen_tol_4)}")
        print(f"Generated instance 5: {lm_tol_ug.perplexity(prep_gen_tol_5)}")

        print("The Perplexities of the sentences generated using the un-smoothed unigram model for the Wilde text are")
        print(f"Generated instance 1: {lm_wilde_ug.perplexity(prep_gen_wilde_1)}")
        print(f"Generated instance 2: {lm_wilde_ug.perplexity(prep_gen_wilde_2)}")
        print(f"Generated instance 3: {lm_wilde_ug.perplexity(prep_gen_wilde_3)}")
        print(f"Generated instance 4: {lm_wilde_ug.perplexity(prep_gen_wilde_4)}")
        print(f"Generated instance 5: {lm_wilde_ug.perplexity(prep_gen_wilde_5)}")

        ## Smoothing of the bigram model

        ## Preparing the data
        train_austen_bgs,vocab_austen_bgs = padded_everygram_pipeline(2, austen_train)
        train_dickens_bgs,vocab_dickens_bgs = padded_everygram_pipeline(2, dickens_train)
        train_tolstoy_bgs,vocab_tolstoy_bgs = padded_everygram_pipeline(2, tolstoy_train)
        train_wilde_bgs,vocab_wilde_bgs = padded_everygram_pipeline(2, wilde_train)

        ## Fitting the model
        lm_aus_bgs, lm_dickens_bgs, lm_tol_bgs,lm_wilde_bgs = models_train(2,train_austen_bgs,train_dickens_bgs,train_tolstoy_bgs,train_wilde_bgs,Main_vocab,'Laplace')

        ## Compute the average perplexity

        avg_bgs_aus = average_perplexity(bg_aus_test,lm_aus_bgs,test_size_aus)
        avg_bgs_dic = average_perplexity(bg_dic_test,lm_dickens_bgs,test_size_aus)
        avg_bgs_tol = average_perplexity(bg_tol_test,lm_tol_bgs,test_size_aus)
        avg_bgs_wilde = average_perplexity(bg_wilde_test,lm_wilde_bgs,test_size_aus)

        #print(f"The average perplexity of the laplace smoothed bigram MLE for the austin test set is {avg_bgs_aus}")

        ### Perplexity of the five generated sentences

        print("The Perplexities of the sentences generated using the smoothed bigram model for the Austen text are")
        print(f"Generated instance 1: {lm_aus_bgs.perplexity(prep_gen_aus_1)}")
        print(f"Generated instance 2: {lm_aus_bgs.perplexity(prep_gen_aus_2)}")
        print(f"Generated instance 3: {lm_aus_bgs.perplexity(prep_gen_aus_3)}")
        print(f"Generated instance 4: {lm_aus_bgs.perplexity(prep_gen_aus_4)}")
        print(f"Generated instance 5: {lm_aus_bgs.perplexity(prep_gen_aus_5)}")

        print("The Perplexities of the sentences generated using the smoothed bigram model for the Dickens text are")
        print(f"Generated instance 1: {lm_dickens_bgs.perplexity(prep_gen_dic_1)}")
        print(f"Generated instance 2: {lm_dickens_bgs.perplexity(prep_gen_dic_2)}")
        print(f"Generated instance 3: {lm_dickens_bgs.perplexity(prep_gen_dic_3)}")
        print(f"Generated instance 4: {lm_dickens_bgs.perplexity(prep_gen_dic_4)}")
        print(f"Generated instance 5: {lm_dickens_bgs.perplexity(prep_gen_dic_5)}")

        print("The Perplexities of the sentences generated using the smoothed bigram model for the Tolstoy text are")
        print(f"Generated instance 1: {lm_tol_bgs.perplexity(prep_gen_tol_1)}")
        print(f"Generated instance 2: {lm_tol_bgs.perplexity(prep_gen_tol_2)}")
        print(f"Generated instance 3: {lm_tol_bgs.perplexity(prep_gen_tol_3)}")
        print(f"Generated instance 4: {lm_tol_bgs.perplexity(prep_gen_tol_4)}")
        print(f"Generated instance 5: {lm_tol_bgs.perplexity(prep_gen_tol_5)}")

        print("The Perplexities of the sentences generated using the smoothed bigram model for the Wilde text are")
        print(f"Generated instance 1: {lm_wilde_bgs.perplexity(prep_gen_wilde_1)}")
        print(f"Generated instance 2: {lm_wilde_bgs.perplexity(prep_gen_wilde_2)}")
        print(f"Generated instance 3: {lm_wilde_bgs.perplexity(prep_gen_wilde_3)}")
        print(f"Generated instance 4: {lm_wilde_bgs.perplexity(prep_gen_wilde_4)}")
        print(f"Generated instance 5: {lm_wilde_bgs.perplexity(prep_gen_wilde_5)}")

        ## Interpolated model
        ## Preprocess the training data

        train_austen_bgi,vocab_austen_bgi = padded_everygram_pipeline(2, austen_train)
        train_dickens_bgi,vocab_dickens_bgi = padded_everygram_pipeline(2, dickens_train)
        train_tolstoy_bgi,vocab_tolstoy_bgi = padded_everygram_pipeline(2, tolstoy_train)
        train_wilde_bgi,vocab_wilde_bgi = padded_everygram_pipeline(2, wilde_train)

        ## Train an interpolated bigram model

        lm_aus_bgi, lm_dickens_bgi, lm_tol_bgi,lm_wilde_bgi = models_train(2,train_austen_bgi,train_dickens_bgi,train_tolstoy_bgi,train_wilde_bgi,Main_vocab,'kneser',0.5)

        ### Test set perplexity

        avg_bgi_aus = average_perplexity(bg_aus_test,lm_aus_bgi,test_size_aus)
        avg_bgi_dic = average_perplexity(bg_dic_test,lm_dickens_bgi,test_size_aus)
        avg_bgi_tol = average_perplexity(bg_tol_test,lm_tol_bgi,test_size_aus)
        avg_bgi_wilde = average_perplexity(bg_wilde_test,lm_wilde_bgi,test_size_aus)

        #print(f"The average perplexity of the interpolated bigram MLE for the Wilde test set is {avg_bgi_wilde}")

        ### Perplexity of the five generated sentences

        print("The Perplexities of the sentences generated using the interpolated bigram model for the Austen text are")
        print(f"Generated instance 1: {lm_aus_bgi.perplexity(prep_gen_aus_1)}")
        print(f"Generated instance 2: {lm_aus_bgi.perplexity(prep_gen_aus_2)}")
        print(f"Generated instance 3: {lm_aus_bgi.perplexity(prep_gen_aus_3)}")
        print(f"Generated instance 4: {lm_aus_bgi.perplexity(prep_gen_aus_4)}")
        print(f"Generated instance 5: {lm_aus_bgi.perplexity(prep_gen_aus_5)}")

        print("The Perplexities of the sentences generated using the interpolated bigram model for the Dickens text are")
        print(f"Generated instance 1: {lm_dickens_bgi.perplexity(prep_gen_dic_1)}")
        print(f"Generated instance 2: {lm_dickens_bgi.perplexity(prep_gen_dic_2)}")
        print(f"Generated instance 3: {lm_dickens_bgi.perplexity(prep_gen_dic_3)}")
        print(f"Generated instance 4: {lm_dickens_bgi.perplexity(prep_gen_dic_4)}")
        print(f"Generated instance 5: {lm_dickens_bgi.perplexity(prep_gen_dic_5)}")

        print("The Perplexities of the sentences generated using the interpolated bigram model for the Tolstoy text are")
        print(f"Generated instance 1: {lm_tol_bgi.perplexity(prep_gen_tol_1)}")
        print(f"Generated instance 2: {lm_tol_bgi.perplexity(prep_gen_tol_2)}")
        print(f"Generated instance 3: {lm_tol_bgi.perplexity(prep_gen_tol_3)}")
        print(f"Generated instance 4: {lm_tol_bgi.perplexity(prep_gen_tol_4)}")
        print(f"Generated instance 5: {lm_tol_bgi.perplexity(prep_gen_tol_5)}")

        print("The Perplexities of the sentences generated using the interpolated bigram model for the Wilde text are")
        print(f"Generated instance 1: {lm_wilde_bgi.perplexity(prep_gen_wilde_1)}")
        print(f"Generated instance 2: {lm_wilde_bgi.perplexity(prep_gen_wilde_2)}")
        print(f"Generated instance 3: {lm_wilde_bgi.perplexity(prep_gen_wilde_3)}")
        print(f"Generated instance 4: {lm_wilde_bgi.perplexity(prep_gen_wilde_4)}")
        print(f"Generated instance 5: {lm_wilde_bgi.perplexity(prep_gen_wilde_5)}")

        ## Interpolated trigram model

        ## Preprocessing trigram data

        prep_austen_train_tg = list(flatten(pad_both_ends(sent, n=3) for sent in austen_train))
        prep_dickens_train_tg = list(flatten(pad_both_ends(sent, n=3) for sent in dickens_train))
        prep_tolstoy_train_tg = list(flatten(pad_both_ends(sent, n=3) for sent in tolstoy_train))
        prep_wilde_train_tg = list(flatten(pad_both_ends(sent, n=3) for sent in wilde_train))

        # Train data preprocessed for perplexity computation
        train_austen_perp_tg = list(trigrams(prep_austen_train_tg))
        train_dickens_perp_tg = list(trigrams(prep_dickens_train_tg))
        train_tolstoy_perp_tg = list(trigrams(prep_tolstoy_train_tg))
        train_wilde_perp_tg = list(trigrams(prep_wilde_train_tg))

        ## Test data preprocessing for trigrams
        prep_austen_test_tg = list(flatten(pad_both_ends(sent, n=3) for sent in austen_test))
        prep_dickens_test_tg = list(flatten(pad_both_ends(sent, n=3) for sent in dickens_test))
        prep_tolstoy_test_tg = list(flatten(pad_both_ends(sent, n=3) for sent in tolstoy_test))
        prep_wilde_test_tg = list(flatten(pad_both_ends(sent, n=3) for sent in wilde_test))

        test_austen_tg = list(trigrams(prep_austen_test_tg))
        test_dickens_tg = list(trigrams(prep_dickens_test_tg))
        test_tolstoy_tg = list(trigrams(prep_tolstoy_test_tg))
        test_wilde_tg = list(trigrams(prep_wilde_test_tg))

        ### Test set of sentences (Trigram)

        tg_aus_test, tg_dic_test,tg_tol_test,tg_wilde_test = [],[],[],[]
        Test_set_global_tg = []
        for ind in range(len(test_set_indices_austen)):
            sentence = test_austen_tg[test_set_indices_austen[ind][0]:test_set_indices_austen[ind][1]-1]
            tg_aus_test.append(sentence)
            Test_set_global_tg.append([sentence,'Austen'])
        for ind in range(len(test_set_indices_dickens)):
            sentence = test_dickens_tg[test_set_indices_dickens[ind][0]:test_set_indices_dickens[ind][1]-1]
            tg_dic_test.append(sentence)
            Test_set_global_tg.append([sentence,'Dickens'])
        for ind in range(len(test_set_indices_tolstoy)):
            sentence = test_tolstoy_tg[test_set_indices_tolstoy[ind][0]:test_set_indices_tolstoy[ind][1]-1]
            tg_tol_test.append(sentence)
            Test_set_global_tg.append([sentence,'Tolstoy'])
        for ind in range(len(test_set_indices_wilde)):
            sentence = test_wilde_tg[test_set_indices_wilde[ind][0]:test_set_indices_wilde[ind][1]-1]
            tg_wilde_test.append(sentence)
            Test_set_global_tg.append([sentence,'Wilde'])

         ### Randomly choosing trigram prompts from the corpus(train/test set)

        random.seed(42)
        aus_prompt1_tg = test_austen_tg[random.randint(0,len(test_austen_tg))]
        aus_prompt2_tg = train_austen_perp_tg[random.randint(0,len(test_austen_tg))]
        aus_prompt3_tg = train_austen_perp_tg[random.randint(0,len(test_austen_tg))]
        aus_prompt4_tg = test_austen_tg[random.randint(0,len(test_austen_tg))]
        aus_prompt5_tg = train_austen_perp_tg[random.randint(0,len(test_austen_tg))]
        
        random.seed(33)
        dic_prompt1_tg = train_dickens_perp_tg[random.randint(0,len(test_dickens_tg))]
        dic_prompt2_tg = test_dickens_tg[random.randint(0,len(test_dickens_tg))]
        dic_prompt3_tg = test_dickens_tg[random.randint(0,len(test_dickens_tg))]
        dic_prompt4_tg = test_dickens_tg[random.randint(0,len(test_dickens_tg))]
        dic_prompt5_tg = train_dickens_perp_tg[random.randint(0,len(test_dickens_tg))]

        random.seed(12)
        tol_prompt1_tg = test_tolstoy_tg[random.randint(0,len(test_tolstoy_tg))]
        tol_prompt2_tg = test_tolstoy_tg[random.randint(0,len(test_tolstoy_tg))]
        tol_prompt3_tg = test_tolstoy_tg[random.randint(0,len(test_tolstoy_tg))]
        tol_prompt4_tg = train_tolstoy_perp_tg[random.randint(0,len(test_tolstoy_tg))]
        tol_prompt5_tg = test_tolstoy_tg[random.randint(0,len(test_tolstoy_tg))]

        random.seed(20)
        wilde_prompt1_tg = test_wilde_tg[random.randint(0,len(test_wilde_tg))]
        wilde_prompt2_tg = test_wilde_tg[random.randint(0,len(test_wilde_tg))]
        wilde_prompt3_tg = test_wilde_tg[random.randint(0,len(test_wilde_tg))]
        wilde_prompt4_tg = train_wilde_perp_tg[random.randint(0,len(test_wilde_tg))]
        wilde_prompt5_tg = train_wilde_perp_tg[random.randint(0,len(test_wilde_tg))]

        ## Training data for the model

        train_austen_tgi,vocab_austen_tgi = padded_everygram_pipeline(3, austen_train)
        train_dickens_tgi,vocab_dickens_tgi = padded_everygram_pipeline(3, dickens_train)
        train_tolstoy_tgi,vocab_tolstoy_tgi = padded_everygram_pipeline(3, tolstoy_train)
        train_wilde_tgi,vocab_wilde_tgi = padded_everygram_pipeline(3, wilde_train)

        ## Train the trigram model

        lm_aus_tgi, lm_dickens_tgi, lm_tol_tgi,lm_wilde_tgi = models_train(3,train_austen_tgi,train_dickens_tgi,train_tolstoy_tgi,train_wilde_tgi,Main_vocab,'kneser',0.8)

        ## Compute average test set perplexity

        avg_tgi_aus = average_perplexity(tg_aus_test,lm_aus_tgi,test_size_aus)
        avg_tgi_dic = average_perplexity(tg_dic_test,lm_dickens_tgi,test_size_aus)
        avg_tgi_tol = average_perplexity(tg_tol_test,lm_tol_tgi,test_size_aus)
        avg_tgi_wilde = average_perplexity(tg_wilde_test,lm_wilde_tgi,test_size_aus)

        print(f"The average perplexity of the interpolated trigram MLE for the austin test set is {avg_tgi_aus}")

        ### Generating sentences and computing perplexity

        sent_aus_tg_1 = lm_aus_tgi.generate(1,aus_prompt1_tg,random_seed = 1)
        sent_aus_tg_2 = lm_aus_tgi.generate(1,aus_prompt2_tg,random_seed = 5)
        sent_aus_tg_3 = lm_aus_tgi.generate(1,aus_prompt3_tg,random_seed = 30)
        sent_aus_tg_4 = lm_aus_tgi.generate(1,aus_prompt4_tg,random_seed = 45)
        sent_aus_tg_5 = lm_aus_tgi.generate(1,aus_prompt5_tg,random_seed = 35)

        sent_dic_tg_1 = lm_dickens_tgi.generate(1,dic_prompt1_tg, random_seed = 25)
        sent_dic_tg_2 = lm_dickens_tgi.generate(1,dic_prompt2_tg,random_seed = 13)
        sent_dic_tg_3 = lm_dickens_tgi.generate(1,dic_prompt3_tg,random_seed = 55)
        sent_dic_tg_4 = lm_dickens_tgi.generate(1,dic_prompt4_tg,random_seed = 10)
        sent_dic_tg_5 = lm_dickens_tgi.generate(1,dic_prompt5_tg,random_seed = 6)

        sent_tol_tg_1 = lm_tol_tgi.generate(1,tol_prompt1_tg,random_seed = 20)
        sent_tol_tg_2 = lm_tol_tgi.generate(1,tol_prompt2_tg,random_seed = 65)
        sent_tol_tg_3 = lm_tol_tgi.generate(1,tol_prompt3_tg,random_seed = 41)
        sent_tol_tg_4 = lm_tol_tgi.generate(1,tol_prompt4_tg,random_seed = 1)
        sent_tol_tg_5 = lm_tol_tgi.generate(1,tol_prompt5_tg,random_seed = 8)

        sent_wilde_tg_1 = lm_wilde_tgi.generate(1,wilde_prompt1_tg,random_seed = 33)
        sent_wilde_tg_2 = lm_wilde_tgi.generate(1,wilde_prompt2_tg,random_seed = 18)
        sent_wilde_tg_3 = lm_wilde_tgi.generate(1,wilde_prompt3_tg,random_seed = 53)
        sent_wilde_tg_4 = lm_wilde_tgi.generate(1,wilde_prompt4_tg,random_seed = 8)
        sent_wilde_tg_5 = lm_wilde_tgi.generate(1,wilde_prompt5_tg,random_seed = 29)

        ### Perplexity for the generated instances

        print("The Perplexities of the sentences generated using the interpolated trigram model for the Austen text are")
        print(f"Generated instance 1: {lm_aus_tgi.perplexity(prep_gen_aus_1)}")
        print(f"Generated instance 2: {lm_aus_tgi.perplexity(prep_gen_aus_2)}")
        print(f"Generated instance 3: {lm_aus_tgi.perplexity(prep_gen_aus_3)}")
        print(f"Generated instance 4: {lm_aus_tgi.perplexity(prep_gen_aus_4)}")
        print(f"Generated instance 5: {lm_aus_tgi.perplexity(prep_gen_aus_5)}")

        print("The Perplexities of the sentences generated using the interpolated trigram model for the Dickens text are")
        print(f"Generated instance 1: {lm_dickens_tgi.perplexity(prep_gen_dic_1)}")
        print(f"Generated instance 2: {lm_dickens_tgi.perplexity(prep_gen_dic_2)}")
        print(f"Generated instance 3: {lm_dickens_tgi.perplexity(prep_gen_dic_3)}")
        print(f"Generated instance 4: {lm_dickens_tgi.perplexity(prep_gen_dic_4)}")
        print(f"Generated instance 5: {lm_dickens_tgi.perplexity(prep_gen_dic_5)}")

        print("The Perplexities of the sentences generated using the interpolated trigram model for the Tolstoy text are")
        print(f"Generated instance 1: {lm_tol_tgi.perplexity(prep_gen_tol_1)}")
        print(f"Generated instance 2: {lm_tol_tgi.perplexity(prep_gen_tol_2)}")
        print(f"Generated instance 3: {lm_tol_tgi.perplexity(prep_gen_tol_3)}")
        print(f"Generated instance 4: {lm_tol_tgi.perplexity(prep_gen_tol_4)}")
        print(f"Generated instance 5: {lm_tol_tgi.perplexity(prep_gen_tol_5)}")

        print("The Perplexities of the sentences generated using the interpolated trigram model for the Wilde text are")
        print(f"Generated instance 1: {lm_wilde_tgi.perplexity(prep_gen_wilde_1)}")
        print(f"Generated instance 2: {lm_wilde_tgi.perplexity(prep_gen_wilde_2)}")
        print(f"Generated instance 3: {lm_wilde_tgi.perplexity(prep_gen_wilde_3)}")
        print(f"Generated instance 4: {lm_wilde_tgi.perplexity(prep_gen_wilde_4)}")
        print(f"Generated instance 5: {lm_wilde_tgi.perplexity(prep_gen_wilde_5)}")

        if flag == False:
            ## Classification

            acc_aus_bgi,acc_dic_bgi,acc_tol_bgi,acc_wil_bgi = classifier(Test_set_global_bg,lm_aus_bgi,lm_dickens_bgi,lm_tol_bgi,lm_wilde_bgi,test_size_aus,test_size_dic,test_size_tol,test_size_wilde)
            print(f"Accuracy of the interpolated bigram model on the Austen text is {acc_aus_bgi: .2f} %")
            print(f"Accuracy of the interpolated bigram model on the Dickens text is {acc_dic_bgi: .2f} %")
            print(f"Accuracy of the interpolated bigram model on the Tolstoy text is {acc_tol_bgi: .2f} %")
            print(f"Accuracy of the interpolated bigram model on the Wilde text is {acc_wil_bgi: .2f} %")

            acc_aus_tgi,acc_dic_tgi,acc_tol_tgi,acc_wil_tgi = classifier(Test_set_global_tg,lm_aus_tgi,lm_dickens_tgi,lm_tol_tgi,lm_wilde_tgi,test_size_aus,test_size_dic,test_size_tol,test_size_wilde)
            print(f"Accuracy of the interpolated trigram model on the Austen text is {acc_aus_tgi: .2f} %")
            print(f"Accuracy of the interpolated trigram model on the Dickens text is {acc_dic_tgi: .2f} %")
            print(f"Accuracy of the interpolated trigram model on the Tolstoy text is {acc_tol_tgi: .2f} %")
            print(f"Accuracy of the interpolated trigram model on the Wilde text is {acc_wil_tgi: .2f} %")
        if flag == True:
            for item in bg_test:
                print(predict(item,lm_aus_bgi,lm_dickens_bgi,lm_tol_bgi,lm_wilde_bgi))

    if approach == 'discriminative':
        # Discriminative classifier

        # Check if running on GPU
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f'device: {device}')

        #Create the dataset
        dataset = dataset_creation(data_austen,0,data_dickens,1,data_tolstoy,2,data_wilde,3)

        # Tokenise the data
        roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        data_collator = DataCollatorWithPadding(tokenizer=roberta_tokenizer)
        tokenized_austen = dataset.map(preprocess_function,batched=True)
        id2label = {0:"Austen",1:"Dickens",2:"Tolstoy",3:"Wilde"}
        label2id = {"Austen":0,"Dickens":1,"Tolstoy":2,"Wilde":3}
        model_name='roberta-base'
        num_labels=4

        # Create the model
        model = AutoModelForSequenceClassification.from_pretrained("roberta-base",num_labels=num_labels,id2label=id2label,label2id=label2id)
        # Evaluate accuracy
        accuracy = evaluate.load('accuracy')

        # Set training hyper-parameters
        training_args = TrainingArguments(
            output_dir='mymodel',
            learning_rate=5e-5,
            per_device_train_batch_size=128,
            per_device_eval_batch_size=128,
            num_train_epochs=5,
            #weight_decay=0.01,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            logging_steps=100,
        )

        # Creating the trainer object
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_austen['train'],
            eval_dataset=tokenized_austen['validation'],
            tokenizer=roberta_tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,)

        # Train model and print accuracy
        trainer.train()

if __name__ == "__main__":
    main()


# In[ ]:




