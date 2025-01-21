from transformers import BertTokenizer, BertModel
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from wikipedia.exceptions import DisambiguationError, PageError, HTTPTimeoutError
import h5py
import math
import os
import re
import statistics
import torch
import wikipedia
import numpy as np


class WikipediaText:
    def __init__(self, language:str):
        self.list_title = []
        wikipedia.set_lang(language)


    def random_text(self):
        random_title = wikipedia.random()
        page = wikipedia.page(random_title)
        text = page.content
        text = text.split('\n')  ## paragraph = line
        text = [x for x in text if x != '' and ' ']  ## remove blanks
        text = [x for x in text if '== ' not in x]  ## remove section titles
        self.list_title.append(page.title)
        return text



class Embedding:
    def __init__(self, model:str='bert-base-multilingual-cased', tokenizer:str='bert-base-multilingual-cased', gpu:bool=True):
        self.embeddings = {}
        self.dict_tsne = {}
        self.model = BertModel.from_pretrained(model)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        self.gpu = gpu

        # CPU -> GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        print(f"Using device: {self.device}")


    def embed(self, text:str):
        # embedding using GPU
        if self.gpu:
            for txt in text:  ## txt = paragraph
                # tokenize the text
                encoded = self.tokenizer(txt, return_tensors='pt', truncation=True, padding=True)  ## encoded = dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                subwords = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][0][1:-1])
                
                # get embeddings
                with torch.no_grad():
                    output = self.model(**encoded)  ## output = dict_keys(['last_hidden_state', 'pooler_output'])
                    embed = output.last_hidden_state.squeeze(0)
                    for sw, emb in zip(subwords, embed):
                        emb = emb.cpu().numpy()
                        if sw not in self.embeddings:
                            self.embeddings[sw] = [emb]
                        else:
                            self.embeddings[sw].append(emb)

        # embedding using CPU
        else:
            for txt in self.text:  ## txt = paragraph
                # get subword tokens
                encoded = self.tokenizer(txt, return_tensors='pt', truncation=True, padding=True)  ## encoded = dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
                subwords = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][0][1:-1])

                # get embeddings
                with torch.no_grad():
                    output = self.model(**encoded)  ## output = dict_keys(['last_hidden_state', 'pooler_output'])
                    embed = output.last_hidden_state.squeeze(0)
                    for sw, emb in zip(subwords, embed):
                        emb = emb.detach().numpy()
                        if sw not in self.embeddings:
                            self.embeddings[sw] = [emb]
                        else:
                            self.embeddings[sw].append(emb)

    # ~~~20250120 added comment-outs. functions below will be done the next day!~~~
    def tsne(self, min_emb:int, p_ratio:float, save_tsne:bool, path:str, language:str,n_components:int=2):
        # tsne with saving the result
        if save_tsne:
            if os.path.isfile(path):
                with h5py.File(path, 'r') as h:
                    cnt = len(h.keys())
                with h5py.File(path, 'a') as h:
                    g = h.create_group(name=f'{language}-{cnt+1}')
                    for sw in self.embeddings:
                        if len(self.embeddings[sw]) >= min_emb:
                            tsne = TSNE(n_components=n_components, perplexity=(len(self.embeddings[sw])*p_ratio))
                            self.dict_tsne[sw] = tsne.fit_transform(np.array(self.embeddings[sw]))
                            try:
                                if sw == '.':
                                    g.create_dataset(name='\u2024', data=self.dict_tsne[sw])
                                elif sw == '/':
                                    g.create_dataset(name='\u2044', data=self.dict_tsne[sw])
                                else:
                                    g.create_dataset(name=sw, data=self.dict_tsne[sw])
                            except:
                                print(f'SavingEmbeddingError: subword "{sw}". Skipping.')
                                print(self.dict_tsne[sw])
                                continue
            else:
                with h5py.File(path, 'w') as h:
                    g = h.create_group(name=f'{language}-1')
                    for sw in self.embeddings:
                        if len(self.embeddings[sw]) >= min_emb:
                            tsne = TSNE(n_components=n_components, perplexity=(len(self.embeddings[sw])*p_ratio))
                            self.dict_tsne[sw] = tsne.fit_transform(np.array(self.embeddings[sw]))
                            try:
                                if sw == '.':
                                    g.create_dataset(name='\u2024', data=self.dict_tsne[sw])
                                elif sw == '/':
                                    g.create_dataset(name='\u2044', data=self.dict_tsne[sw])
                                else:
                                    g.create_dataset(name=sw, data=self.dict_tsne[sw])
                            except:
                                print(f'SavingEmbeddingError: subword "{sw}". Skipping.')
                                continue
        
        # tsne without saving the result
        else:
            for sw in self.embeddings:
                if len(self.embeddings[sw]) >= min_emb:
                    tsne = TSNE(n_components=n_components, perplexity=(len(self.embeddings[sw])*p_ratio))
                    self.dict_tsne[sw] = tsne.fit_transform(np.array(self.embeddings[sw]))



class Cluster:
    def __init__(self, embeddings:numpy.ndarray, gpu:bool, min_emb:int, min_samples:int):
        self.dbscan = {}
        self.entropies = {}
        self.embeddings = embeddings
        self.gpu = gpu
        self.min_emb = min_emb
        self.min_samples = min_samples

    def cluster(self, eps:float, dif:float):
        if self.gpu:
            from cuml.cluster import DBSCAN as cuDBSCAN
        # emb corresponds to a set of embeddings of each subword
        for sw, emb in self.embeddings.items():
            e = eps
            # find the clusters the number of which is the greatest
            best_dbscan = numpy.full(len(emb), -1)
            if self.gpu:
                dbscan = cuDBSCAN(eps=e, min_samples=self.min_samples).fit_predict(emb)
            else:
                dbscan = DBSCAN(eps=e, min_samples=self.min_samples, metric='euclidean').fit_predict(emb)
            while max(dbscan) >= max(best_dbscan):
                best_dbscan = dbscan
                if len(best_dbscan)==numpy.sum(best_dbscan==0):
                    break
                e += dif
                if self.gpu:
                    dbscan = cuDBSCAN(eps=e, min_samples=self.min_samples).fit_predict(emb)
                else:
                    dbscan = DBSCAN(eps=e, min_samples=self.min_samples, metric='euclidean').fit_predict(emb)
            self.dbscan[sw] = best_dbscan
    
    def save_cluster(self, path:str, name:str):
        # identify the directory and the file
        if '/' not in path:
            hfile = path
            hdir = os.listdir(os.getcwd())
        else:
            match = re.search(r'(.+?\..+?/)(.+)', path[::-1])
            hfile = match.group(1)[::-1][:-1]
            hdir = match.group(2)[::-1]
        # save clusters
        if not os.path.exists(hdir):
            with h5py.File(path, 'w') as h:
                g = h.create_group(name=name)
                for sw in self.dbscan:
                    try:
                        if sw == '.':
                            g.create_dataset(name='\u2024', data=self.dbscan[sw])
                        elif sw == '/':
                            g.create_dataset(name='\u2044', data=self.dbscan[sw])
                        else:
                            g.create_dataset(name=sw, data=self.dbscan[sw])
                    except:
                        print(f'SavingClusterError: subword "{sw}". Skipping.')
                        continue
        else:
            with h5py.File(path, 'a') as h:
                g = h.create_group(name=name)
                for sw in self.dbscan:
                    try:
                        if sw == '.':
                            g.create_dataset(name='\u2024', data=self.dbscan[sw])
                        elif sw == '/':
                            g.create_dataset(name='\u2044', data=self.dbscan[sw])
                        else:
                            g.create_dataset(name=sw, data=self.dbscan[sw])
                    except:
                        print(f'SavingClusterError: subword "{sw}". Skipping.')
                        continue

    def entropy(self):
        self.entropy = {}
        # dbs corresponds to clusters for each subword
        for sw, dbs in self.dbscan.items():
            list_num = []
            num_minus = numpy.sum(dbs==-1)
            # list_num contains the number of how many are in each cluster
            for i in range(0, max(dbs)+1):
                list_num.append(numpy.sum(dbs==i))
            # list_entropy contains the entropy of each subword
            for i in list_num:
                self.entropy[sw] = -(i / (len(dbs)-num_minus)) * math.log(i / (len(dbs)-num_minus), 2)
        # the mean of the entropies is the average entropy of each subword in a language
        return statistics.mean(self.entropy.values())