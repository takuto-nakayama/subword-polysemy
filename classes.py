from transformers import BertTokenizer, BertModel
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from wikipedia.exceptions import DisambiguationError, PageError, HTTPTimeoutError
import os, h5py, re, numpy, torch, math, statistics, wikipedia, time, requests, numpy as np

class Dataset:
    def __init__(self, path:str):
        # path of a hdf5 file
        self.path = path
        # path is divided into file and directory name
        if '/' not in self.path:
            self.hfile = self.path
            self.hdir = os.listdir(os.getcwd())
        else:
            match = re.search(r'(.+?\..+?/)(.+)', self.path[::-1])
            self.hfile = match.group(1)[:-1][:-1]
            self.hdir = match.group(2)[::-1]
        # error messages and others
        if self.hfile not in os.listdir(self.hdir):
            print('Error: No such file in the directory')
        if self.hfile[self.hfile.index('.'):] != '.hdf5':
            print('Error: This class can cope with only ".hdf5"')

    def tree(self):
        with h5py.File(self.path, 'r') as h:
            print("HDF5 File Structure:")
            h.visititems(lambda name, obj: print(
                f"{'  ' * name.count('/')}[{'Group' if isinstance(obj, h5py.Group) else 'Dataset'}] {name}" +
                    (f", shape: {obj.shape}, dtype: {obj.dtype}" if isinstance(obj, h5py.Dataset) else "")
            ))

    def keys(self, key:str='/'):
        try:
            with h5py.File(self.path, 'r') as h:
                return list(h[key].keys())
        except KeyError:
            print(f'Key Error: The key "{key}" does not exist in the HDF5 file.')
        except Exception as e:
            print(f'Error: {e}')            
    
    def dataset(self, key:str='/'):
        try:
            with h5py.File(self.path, 'r') as h:
                return numpy.array([i.decode('utf-8') for i in h[key][:]])
        except KeyError:
            print(f'Key Error: The key "{key}" does not exist in the HDF5 file.')
        except Exception as e:
            print(f'Error: {e}')            

    def write(self, name:str=None, data:numpy.ndarray=None):
        if name and data:
            with h5py.File(self.path, 'a') as h:
                h.create_dataset(name=name, data=data)
        elif name:
            with h5py.File(self.path, 'a') as h:
                h.create_group(name=name)
        elif data:
            print('Error: Group name or dataset name is required')
        else:
            print('Error: Something is wrong in arguments')

class WikipediaText:
    def __init__(self, language:str):
        self.list_title = []
        wikipedia.set_lang(language)

    def random_text(self):
        random_title = wikipedia.random()
        page = wikipedia.page(random_title)
        text = page.content
        text = text.split('\n') # paragraph corresponds to a line
        text = [x for x in text if x != '' and ' '] # remove blanks
        text = [x for x in text if '== ' not in x] # remove section titles
        self.list_title.append(page.title)
        return text

class Embedding:
    def __init__(self, model:str='bert-base-multilingual-cased', tokenizer:str='bert-base-multilingual-cased', gpu:bool=True):
        self.embeddings = {}
        self.model = BertModel.from_pretrained(model)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        self.gpu = gpu
        if self.gpu:
            # CPU -> GPU
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            print(f"Using device: {self.device}")

    def embed(self, text:str):
        if self.gpu:
            # get subword tokens
            encoded = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            subwords = [self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][i]) for i in range(len(encoded['input_ids']))]
            # get embeddings
            with torch.no_grad():
                output = self.model(**encoded)
                embeddings = output.last_hidden_state
                for subword, embedding in zip(subwords, embeddings):
                    for sw, emb in zip(subword, embedding):
                        if sw not in ['[CLS]', '[SEP]', '[PAD]']:
                            emb = emb.cpu().numpy()
                            if sw not in self.embeddings:
                                self.embeddings[sw] = emb
                            else:
                                self.embeddings[sw] = np.vstack((self.embeddings[sw], emb))
        else:
            # get subword tokens
            encoded = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
            subwords = [self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][i]) for i in range(len(encoded['input_ids']))]
            # get embeddings
            with torch.no_grad():
                output = self.model(**encoded)
                embeddings = output.last_hidden_state
                for subword, embedding in zip(subwords, embeddings):
                    for sw, emb in zip(subword, embedding):
                        if sw not in ['[CLS]', '[SEP]', '[PAD]']:
                            emb = emb.detach().numpy()
                            if sw not in self.embeddings:
                                self.embeddings[sw] = emb
                            else:
                                self.embeddings[sw] = np.vstack((self.embeddings[sw], emb))

    def tsne(self, min_samples:int, n_components:int=2):
        if self.gpu:
            from cuml.manifold import cuTSNE
            for sw in self.embeddings:
                if len(self.embeddings[sw].shape) == 2 and self.embeddings[sw].shape[0] >= min_samples:
                    perplexity = len(self.embeddings[sw]) // 3 + 0.5
                    tsne = cuTSNE(n_components=n_components, random_state=42, perplexity=perplexity)
                    self.embeddings[sw] = tsne.fit_transform(self.embeddings[sw])
        else:
            for sw in self.embeddings:
                if len(self.embeddings[sw].shape) == 2 and self.embeddings[sw].shape[0] >= min_samples:
                    perplexity = len(self.embeddings[sw]) // 3 + 0.5
                    tsne = TSNE(n_components=n_components, random_state=42, perplexity=perplexity)
                    self.embeddings[sw] = tsne.fit_transform(self.embeddings[sw])
            
    def save_vector(self, path:str, name:str):
        # identify the directory and the file
        if '/' not in path:
            hfile = path
            hdir = os.listdir(os.getcwd())
        else:
            match = re.search(r'(.+?\..+?/)(.+)', path[::-1])
            hfile = match.group(1)[:-1][::-1]
            hdir = os.listdir(match.group(2)[::-1])
        # save vectors
        if hfile not in hdir:
            with h5py.File(path, 'w') as h:
                g = h.create_group(name=name)
                for sw in self.embeddings:
                    try:
                        if sw == '.':
                            g.create_dataset(name='\u2024', data=self.embeddings[sw])
                        elif sw == '/':
                            g.create_dataset(name='\u2044', data=self.embeddings[sw])
                        else:
                            g.create_dataset(name=sw, data=self.embeddings[sw])
                    except:
                        print(f'SavingEmbeddingError: subword "{sw}". Skipping.')
                        continue
        else:
            with h5py.File(path, 'a') as h:
                g = h.create_group(name=name)
                for sw in self.embeddings:
                    try:
                        if sw == '.':
                            g.create_dataset(name='\u2024', data=self.embeddings[sw])
                        elif sw == '/':
                            g.create_dataset(name='\u2044', data=self.embeddings[sw])
                        else:
                            g.create_dataset(name=sw, data=self.embeddings[sw])
                    except:
                        print(f'SavingEmbeddingError: subword "{sw}". Skipping.')
                        continue

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
            if len(emb.shape) == 2 and emb.shape[0] >= self.min_emb:
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
        if hfile not in os.listdir(hdir):
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