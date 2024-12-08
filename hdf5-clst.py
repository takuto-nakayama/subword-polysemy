from classes import Dataset, Embedding, Cluster
from datetime import datetime
from plotly import express as px
import argparse, h5py, os, pandas as pd

if __name__ == '__main__':
    #input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='the file path must be ".hdf5"')
    parser.add_argument('id', type=str)
    parser.add_argument('--gpu', default=True, type=bool)
    parser.add_argument('--save_vector', default=False, type=bool)
    parser.add_argument('--save_cluster', default=False, type=bool)
    parser.add_argument('--save_entropy', default=True, type=bool)
    args = parser.parse_args()

    #define variables
    path = args.path
    id = args.id
    # optional arguments
    gpu = args.gpu
    save_vector = args.save_vector
    save_cluster = args.save_cluster
    save_entropy = args.save_entropy

    # processing
    os.mkdir(f'result/{id}')
    data = Dataset(path)
    dict_entropy = {}
    
    for lang in data.keys():
        start = datetime.now() # starting time

        emb = Embedding(data.dataset(key=lang)) # embedding obj
        emb.embed(gpu=gpu) # embed the text
        if save_vector:
            emb.save_vector(path=f'result/embedding-{id}.hdf5', name=lang)
        time_emb = datetime.now() - start
        print(f'{lang}: Embedding is done. ({time_emb.seconds} seconds.)')

        start_clst = datetime.now()
        clst = Cluster(emb.embeddings) # cluster obj
        clst.cluster() # cluster the embeddings
        if save_cluster:
            clst.save_cluster(path=f'result/dbscan-{lang}.hdf5', name=lang)
        time_clst = datetime.now() - start_clst
        print(f'{lang}: Clustering is done. ({time_clst.seconds} seconds.)')

        start_ent = datetime.now()
        dict_entropy[lang] = clst.entropy() # calculate entropies
        end = datetime.now() # ending time
        time_ent = end - start_ent
        print(f'{lang}: Entropy is done. ({time_ent.seconds} seconds.)')

        # output amount of processing time        
        time = end - start
        print(f'{lang} is done. (Total: {time.seconds} seconds.)')

    # plot and save the result
    d = {'language': dict_entropy.keys(), 'entropy': dict_entropy.values()}
    df = pd.DataFrame(d)
    fig = px.scatter(df, x = 'language', y = 'entropy')
    if save_entropy:
        df.to_csv(f'result/{id}/entropy-{lang}.csv', index=False)
        fig.write_html(f'graph-{id}.html')
    fig.show()