from transformers import BertTokenizer, BertModel
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import os, h5py, re, numpy, torch, math, statistics

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
            self.hfile = match.group(1)[:-1][::-1]
            self.hdir = match.group(2)[::-1]
        # error messages and others
        if self.hfile not in os.listdir(self.hdir):
            print('Error: No such file in the directory')
        if self.hfile[self.hfile.idnex('.'):] != '.hdf5':
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
    
    def savevec(self, path:str, name:str):
        # identify the directory and the file
        if '/' not in path:
            hfile = path
            hdir = os.listdir(os.getcwd())
        else:
            match = re.search(r'(.+?\..+?/)(.+)', path[::-1])
            hfile = match.group(1)[:-1][::-1]
            hdir = os.listdir(match.group(2))[::-1]
        # save vectors
        if hfile not in hdir:
            with h5py.File(path, 'w') as h:
                g = h.create_group(name=name)
                for sw in self.embeddings:
                    g.create_dataset(name=sw, data=self.embeddings[sw])
        else:
            with h5py.File(path, 'a') as h:
                g = h.create_group(name=name)
                for sw in self.embeddings:
                    g.create_dataset(name=sw, data=self.embeddings[sw])

class Cluster:
    def __init__(self, embedding=numpy.ndarray):
        self.dbscan = []
        self.embedding = embedding

    def cluster(self, min=2, pca=False, e=0.5, dif=0.5, git=0.1):
        # emb corresponds to a set of embeddings of each subword
        for emb in self.embeddings:
            if len(emb) >= min:
                # pca version
                if pca:
                    pca = PCA()
                    emb = pca.fit_transform(emb)
                    index = numpy.where(numpy.cumsum(pca.explained_variance_ratio_) >= 0.9)[0][0] + 1
                    emb = emb[:index]
                # find the clusters the number of which is the greatest
                best_dbscan = numpy.full(len(emb), -1)
                dbscan = DBSCAN(eps=e, min_samples=min, metric='euclidean').fit_predict(emb)
                while max(dbscan) >= max(best_dbscan):
                    best_dbscan = dbscan
                    e += dif
                    dbscan = DBSCAN(eps=e, min_samples=min, metric='euclidean').fit_predict(emb)
                self.dbscan.append(best_dbscan)

    def entropy(self):
        list_entropy = []
        # dbs corresponds to clusters for each subword
        for dbs in self.dbscan:
            list_num = []
            # list_num contains the number of how many are in each cluster
            for i in range(0, max(dbs)+1):
                list_num.append(numpy.sum(dbs==i))
            # list_entropy contains the entropy of each subword
            for i in list_num:
                list_entropy.append(-(i / len(dbs)) * math.log(i / len(dbs), 2))
        # the mean of the entropies is the average entropy of each subword
        return statistics.mean(list_entropy)