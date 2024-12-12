from transformers import BertTokenizer, BertModel
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from wikipedia.exceptions import DisambiguationError, PageError, HTTPTimeoutError
import os, h5py, re, numpy, torch, math, statistics, wikipedia, time

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
    def __init__(self):
        self.list_text = []
        self.list_title = []

    def load_text(self, language:str, num:int):
        wikipedia.set_lang(language)
        cnt = 1
        # roop for the input times
        while num >= cnt:
            try:
                random_title = wikipedia.random()
                page = wikipedia.page(random_title)
                text = page.content
                text = text.split('\n') # paragraph corresponds to a line
                text = [x for x in text if x != '' and ' '] # remove blanks
                text = [x for x in text if '== ' not in x] # remove section titles
                for t in text:
                    self.list_text.append(t)
                cnt += 1
                self.list_title.append(page.title)
                time.sleep(1)
            except (DisambiguationError, PageError, HTTPTimeoutError) as e:
                print(f'Error encountered: {e}. Skipping.')
                time.sleep(1)
                continue

class Embedding:
    def __init__(self, text=numpy.ndarray, model:str='bert-base-multilingual-cased', tokenizer:str='bert-base-multilingual-cased'):
        self.text = text
        self.embeddings = {}
        self.model = BertModel.from_pretrained(model)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)

    def embed(self, gpu:bool=True):
        if gpu:
            # CPU -> GPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(device)
            print(f"Using device: {device}")
            # txt corresponds to a sentence
            for txt in self.text:       
                # get subword tokens
                encoded = self.tokenizer(txt, return_tensors='pt', truncation=True, padding=True)
                encoded = {key: value.to(device) for key, value in encoded.items()}
                subwords = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][0][1:-1])
                # get embeddings
                with torch.no_grad():
                    output = self.model(**encoded)
                    embed = output.last_hidden_state.squeeze(0)
                    for sw, emb in zip(subwords, embed):
                        emb = emb.cpu().numpy()
                        if sw not in self.embeddings:
                            self.embeddings[sw] = [emb]
                        else:
                            self.embeddings[sw].append(emb)
        else:
            # txt corresponds to a sentence
            for txt in self.text:
                # get subword tokens
                encoded = self.tokenizer(txt, return_tensors='pt', truncation=True, padding=True)
                subwords = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][0][1:-1])
                # get embeddings
                with torch.no_grad():
                    output = self.model(**encoded)
                    embed = output.last_hidden_state.squeeze(0)
                    for sw, emb in zip(subwords, embed):
                        emb = emb.detach().numpy()
                        if sw not in self.embeddings:
                            self.embeddings[sw] = [emb]
                        else:
                            self.embeddings[sw].append(emb)
    
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
    def __init__(self, embeddings=numpy.ndarray):
        self.dbscan = {}
        self.entropies = {}
        self.embeddings = embeddings

    def cluster(self, min=2, pca=False, gpu=True, e=0.5, dif=0.5, brake=10):
        if gpu:
            from cuml.cluster import DBSCAN as cuDBSCAN
            import cuml
        # emb corresponds to a set of embeddings of each subword
        for sw, emb in self.embeddings.items():
            if len(emb) >= min:
                # pca version
                if pca:
                    pca = PCA()
                    emb = pca.fit_transform(emb)
                    index = numpy.where(numpy.cumsum(pca.explained_variance_ratio_) >= 0.9)[0][0] + 1
                    emb = emb[:index]
                # find the clusters the number of which is the greatest
                cnt = 0
                best_dbscan = numpy.full(len(emb), -1)
                dbscan = numpy.full(len(emb), -1)
                while max(dbscan) >= max(best_dbscan) and cnt < brake:
                    if max(dbscan) == max(best_dbscan):
                        cnt += 1
                    else:
                        cnt = 0
                    best_dbscan = dbscan
                    if gpu:
                        dbscan = cuDBSCAN(eps=e, min_samples=2).fit_predict(numpy.array(emb))
                    else:
                        dbscan = DBSCAN(eps=e, min_samples=2, metric='euclidean').fit_predict(emb)
                    e += dif
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