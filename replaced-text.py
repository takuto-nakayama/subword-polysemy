from classes import WikipediaText, Embedding, Cluster
from datetime import datetime
from wikipedia.exceptions import DisambiguationError, PageError, HTTPTimeoutError
import argparse
import os
import csv
import requests
import time as time_module
import pandas as pd


# input the arguments
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('id', type=str)
    parser.add_argument('--gpu', action='store_false')
    parser.add_argument('--min_emb', default=100, type=int)
    parser.add_argument('--min_samples', default=5, type=int)
    parser.add_argument('--eps', default=0.5, type=float)
    parser.add_argument('--dif', default=0.5, type=float)
    parser.add_argument('--tsne', action='store_false')
    parser.add_argument('--p_ratio', default=0.3, type=float)
    parser.add_argument('--save_tsne', action='store_false')
    parser.add_argument('--save_cluster', action='store_true')
    args = parser.parse_args()


    # set the arguments
    path = args.path
    id = args.id
    gpu = args.gpu
    min_emb = args.min_emb
    min_samples = args.min_samples
    eps = args.eps
    dif = args.dif
    tsne = args.tsne
    p_ratio = args.p_ratio
    save_tsne = args.save_tsne
    save_cluster = args.save_cluster

    list_path = os.listdir(path)



    # make necessary directories
    if not os.path.exists('result'):
        os.mkdir('result')
    if not os.path.exists(f'result/{id}'):
        os.mkdir(f'result/{id}')
    if save_tsne and not os.path.exists(f'result/{id}/tsne-{id}'):
        os.mkdir(f'result/{id}/tsne-{id}')
    

    # process the data
    # create the instances
    for p in list_path:
        start = datetime.now()
        print(f'rocessing started at {start.time()}.')
        emb = Embedding(gpu=gpu)

        # embed the text
        with open(f'{path}/{p}', 'r') as f:
            text = f.readlines()
            emb.embed(text)

        # compress the embeddings with tSNE
        print(f'\ntSNE is processing...')
        emb.tsne(min_emb, p_ratio, save_tsne, f'result/{id}/tsne-{id}/{replace}.hdf5', replace)
        time_emb = datetime.now() - start
        print(f'Embedding is done ({len(emb.embeddings)} subwords). ({time_emb.seconds} seconds.)')

        # cluster the compressed embeddings
        start_clst =  datetime.now()
        clst = Cluster(emb.dict_tsne, gpu=gpu, min_emb=min_emb, min_samples=min_samples)
        clst.cluster(eps, dif)
        if save_cluster:
            clst.save_cluster(path=f'result/{id}/cluster-{id}.hdf5', name=replace)
        time_clst = datetime.now() - start_clst
        print(f'Clustering is done. ({time_clst.seconds} seconds.)')

        # calculate the entropy
        start_ent = datetime.now()
        ent = clst.entropy()
        end = datetime.now()
        time_ent = end - start_ent
        time = end - start
        print(f'Entropy is done (H={ent}). ({time_ent.seconds} seconds.)')
        list_result = [p, ent]
        
        # create the file
        if not os.path.exists(f'result/{id}/result-{id}.csv'):
            # save the results
            with open(f'result/{id}/result-{id}.csv', 'w', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['language', 'number of subwords', 'average paragraphs', 'entropy'])
                writer.writerow(list_result)
            # save the title
            with open(f'result/{id}/title-{id}.csv', 'w', encoding='utf-8') as f:
                writer = csv.writer(f)
        
        # open the existed file and write the results
        else:
            # save the results
            with open(f'result/{id}/result-{id}.csv', 'a', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(list_result)
            # save the title
            with open(f'result/{id}/title-{id}.csv', 'a', encoding='utf-8') as f:
                writer = csv.writer(f)

        print(f'All processing is done. ({time.seconds} seconds.)')



