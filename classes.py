from transformers import BertTokenizer, BertModel
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from datetime import datetime
import os, h5py, re, numpy, torch, math, statistics

class Dataset:
    @classmethod
    def read_hdf5(cls, hpath=str, mode='tree', key='/'):
        if re.search(r'\..+', hpath).group()[1:] == 'hdf5':
            if mode == 'tree':
                with h5py.File(hpath, 'r') as h:
                    print("HDF5 File Structure:")
                    h.visititems(lambda name, obj: print(
                        f"{'  ' * name.count('/')}[{'Group' if isinstance(obj, h5py.Group) else 'Dataset'}] {name}" +
                            (f", shape: {obj.shape}, dtype: {obj.dtype}" if isinstance(obj, h5py.Dataset) else "")
                    ))
            elif mode == 'keys':
                with h5py.File(hpath, 'r') as h:
                    return list(h[key].keys())
            elif mode == 'dataset':
                with h5py.File(hpath, 'r') as h:
                    return numpy.array([i.decode('utf-8') for i in h[key][:]])
            else:
                print('Mode Error: "mode" argument should be "tree", "keys", or "dataset".')
        else:
            print('Extenshion Error: This method can handle only ".".hdf5".')

    @classmethod
    def to_hdf5(cls, hpath=str, name=str, data=None):
        if '/' not in hpath:
            hfile = hpath
            hdir = os.listdir(os.getcwd())
        else:
            match = re.search(r'(.+?\..+?/)(.+)', hpath[::-1])
            hfile = match.group(1)
            hdir = os.listdir(match.group(2))

        if hfile not in hdir:
            with h5py.File(hpath, 'w') as h:
                if data:
                    h.create_dataset(name=name, data=data)
                else:
                    h.create_group(name=name)
        else:
            with h5py.File(hpath, 'a') as h:
                if data:
                    h.create_dataset(name=name, data=data)
                else:
                    h.create_group(name=name)
 
class Embedding:
    def __init__(self, text=numpy.array, model='bert-base-multilingual-cased', tokenizer='bert-base-multilingual-cased'):
        self.text = text
        self.embeddings = {}
        self.model = BertModel.from_pretrained(model)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

    def embed(self):
        for txt in self.text:
            encoded = self.tokenizer(txt.decode('utf-8'), return_tensors='pt', truncation=True, padding=True)
            subwords = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][0][1:-1])

            with torch.no_grad():
                output = self.model(**encoded)
                embed = output.last_hidden_state.squeeze(0)
                for sw, emb in zip(subwords, embed):
                    emb = emb.detach().numpy()
                    if sw not in self.embeddings:
                        self.embeddings[sw] = [emb]
                    else:
                        self.embeddings[sw].append(emb)
    
    def add_embed(self, text=numpy.array):
        for txt in text:
            encoded = self.tokenizer(txt.decode('utf-8'), return_tensors='pt', truncation=True, padding=True)
            subwords = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][0][1:-1])

            with torch.no_grad():
                output = self.model(**encoded)
                embed = output.last_hidden_state.squeeze(0)
                for sw, emb in zip(subwords, embed):
                    emb = emb.detach().numpy()
                    if sw not in self.embeddings:
                        self.embeddings[sw] = [emb]
                    else:
                        self.embeddings[sw].append(emb)

    def savevec(self, hpath=str, name=str):
        if '/' not in hpath:
            hfile = hpath
            hdir = os.listdir(os.getcwd())
        else:
            match = re.search(r'(.+?\..+?/)(.+)', hpath[::-1])
            hfile = match.group(1)
            hdir = os.listdir(match.group(2))

        if hfile not in hdir:
            with h5py.File(hpath, 'w') as h:
                g = h.create_group(name=name)
                for sw in self.embeddings:
                    g.create_dataset(name=sw, data=self.embeddings[sw])
        else:
            with h5py.File(hpath, 'a') as h:
                g = h.create_group(name=name)
                for sw in self.embeddings:
                    g.create_dataset(name=sw, data=self.embeddings[sw])

class Cluster:
    def __init__(self, embpath=str):
        self.dbscan = []
        self.embpath = embpath
        with h5py.File(embpath, 'r') as h:
            self.langs = h.keys()

    def cluster(self, lang, min=2, pca=False, epsilon=0.5, range=0.5, ratio=0.1):
        with h5py.File(self.embpath, 'r') as h:
            for sw in h[lang].keys():
                emb = h[lang][sw][:]

                if len(emb) >= min:
                    if pca:
                        pca = PCA()
                        emb = pca.fit_transform(emb)
                        index = numpy.where(numpy.cumsum(pca.explained_variance_ratio_) >= 0.9)[0][0] + 1
                        emb = emb[:index]

                    now_clst = -1
                    pre_clst = -1
                    best_dbscan = numpy.full(len(emb), -1)
                    e = epsilon

                    while now_clst >= pre_clst:
                        dbscan = DBSCAN(eps=e, min_samples=min, metric='euclidean').fit_predict(emb)
                        now_clst = max(dbscan)
                        if now_clst >= pre_clst:
                            best_dbscan = dbscan
                            pre_clst = max(best_dbscan)
                        e += range
                    self.dbscan.append(best_dbscan)

    def entropy(self):
        list_entropy = []
        
        for dbs in self.dbscan:
            list_num = []
            for i in range(0, max(dbs)+1):
                list_num.append(numpy.sum(dbs==i))
            for i in list_num:
                list_entropy.append(-(i / len(dbs)) * math.log(i / len(dbs), 2))
        
        return statistics.mean(list_entropy)

