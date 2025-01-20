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
    parser.add_argument('num', type=int)
    parser.add_argument('id', type=str)
    parser.add_argument('--gpu', action='store_false')
    parser.add_argument('--min_emb', default=10, type=int)
    parser.add_argument('--min_samples', default=2, type=int)
    parser.add_argument('--eps', default=0.5, type=float)
    parser.add_argument('--dif', default=0.5, type=float)
    parser.add_argument('--tsne', action='store_false')
    parser.add_argument('--p_ratio', default=0.3, type=float)
    parser.add_argument('--save_tsne', action='store_false')
    parser.add_argument('--save_cluster', action='store_true')
    args = parser.parse_args()


    # set the arguments
    path = args.path
    num = args.num
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
    codes = pd.read_csv(path)['ISO-code']


    # make necessary directories
    if not os.path.exists('result'):
        os.mkdir('result')
    if not os.path.exists(f'result/{id}'):
        os.mkdir(f'result/{id}')
    if save_tsne and not os.path.exists(f'result/{id}/tsne-{id}'):
        os.mkdir(f'result/{id}/tsne-{id}')
    

    # process the data
    for language in codes:
        # create the instances
        start = datetime.now()
        print(f'{language}: processing started at {start.time()}.')
        wiki = WikipediaText(language)
        paragraphs = 0
        cnt = 0
        emb = Embedding(gpu=gpu)

        # embed the text
        while cnt < num:
            try:
                text = wiki.random_text()  ## text:list()
                emb.embed(text)
                paragraphs += len(text)
                cnt += 1
                print(f'\rText & Embedding: {cnt}/{num} articles.', end='')
            except (DisambiguationError, PageError, HTTPTimeoutError) as e:
                time_module.sleep(1)
                continue
            except requests.exceptions.ConnectionError as e:
                time_module.sleep(3)
                continue
            except:
                time_module.sleep(1)
                continue
        list_title = wiki.list_title

        # compress the embeddings with tSNE
        print(f'\ntSNE is processing...')
        emb.tsne(min_emb, p_ratio, save_tsne, f'result/{id}/tsne-{id}/{language}.hdf5', language)
        time_emb = datetime.now() - start
        print(f'Text processnig is done ({paragraphs} ¶s, {len(list_title)} articles).')
        print(f'Embedding is done ({len(emb.embeddings)} subwords). ({time_emb.seconds} seconds.)')

        # cluster the compressed embeddings
        start_clst =  datetime.now()
        clst = Cluster(emb.dict_tsne, gpu=gpu, min_emb=min_emb, min_samples=min_samples)
        clst.cluster(eps, dif)
        if save_cluster:
            clst.save_cluster(path=f'result/{id}/cluster-{id}.hdf5', name=language)
        time_clst = datetime.now() - start_clst
        print(f'Clustering is done. ({time_clst.seconds} seconds.)')

        # calculate the entropy
        start_ent = datetime.now()
        ent = clst.entropy()
        end = datetime.now()
        time_ent = end - start_ent
        time = end - start
        print(f'Entropy is done (H={ent}). ({time_ent.seconds} seconds.)')
        list_result = [language,
                    len(emb.embeddings.keys()),
                        paragraphs / num,
                        ent]
        
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
                list_title.insert(0, language)
                writer.writerow(list_title)
        
        # open the existed file and write the results
        else:
            # save the results
            with open(f'result/{id}/result-{id}.csv', 'a', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(list_result)
            # save the title
            with open(f'result/{id}/title-{id}.csv', 'a', encoding='utf-8') as f:
                writer = csv.writer(f)
                list_title.insert(0, language)
                writer.writerow(list_title)

        print(f'All processing is done. ({time.seconds} seconds.)')



