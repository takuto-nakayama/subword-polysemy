from classes import WikipediaText, Embedding, Cluster
from datetime import datetime
from wikipedia.exceptions import DisambiguationError, PageError, HTTPTimeoutError

import argparse, os, csv, time, requests, pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('languages', type=str)
    parser.add_argument('num', type=int)
    parser.add_argument('id', type=str)
    parser.add_argument('--start_slice', type=int)
    parser.add_argument('--end_slice', type=int)
    parser.add_argument('--gpu', default=True, type=bool)
    parser.add_argument('--save_embedding', default=False, type=bool)
    parser.add_argument('--save_cluster', default=False, type=bool)
    args = parser.parse_args()

    languages = args.languages
    num = args.num
    id = args.id
    start_slice = args.start_slice
    end_slice = args.start_slice
    gpu = args.gpu
    save_embedding = args.save_embedding
    save_cluster = args.save_cluster

    df_languages = pd.read_csv(languages)
    codes = df_languages['ISO-code']

    if id not in os.listdir('result'):
        os.mkdir(f'result/{id}')
    for language in codes[start_slice:end_slice]:
        start = datetime.now()
        print(f'{language}: processing started at {start.time()}.')
        wiki = WikipediaText(language)
        paragraphs = 0
        cnt = 1
        emb = Embedding()
        while cnt <= num:
            if cnt % (num/10) == 0:
                process_time = datetime.now() - start
                print(f'Text & Embedding: {cnt % (num/10)}% is done ({process_time.seconds} seconds).')
            try:
                text = wiki.random_text()
                emb.embed(text)
                paragraphs += len(text)
                cnt += 1
            except (DisambiguationError, PageError, HTTPTimeoutError) as e:
                time.sleep(1)
                continue
            except requests.exceptions.ConnectionError as e:
                time.sleep(3)
                continue
        list_title = wiki.list_title
        if save_embedding:
            emb.save_vector(path=f'result/{id}/embedding-{id}.hdf5', name=f'{language}')
        time_emb = datetime.now() - start
        print(f'Text processnig is done ({paragraphs} ¶s, {len(list_title)} articles).')
        print(f'Embedding is done ({len(emb.embeddings)} subwords). ({time_emb.seconds} seconds.)')

        start_clst =  datetime.now()
        clst = Cluster(emb.embeddings)
        clst.cluster(gpu=gpu)
        if save_cluster:
            clst.save_cluster(path=f'result/{id}/cluster-{id}.hdf5', name=language)
        time_clst = datetime.now() - start_clst
        print(f'Clustering is done. ({time_clst.seconds} seconds.)')

        start_ent = datetime.now()
        ent = clst.entropy()
        end = datetime.now()
        time_ent = end - start_ent
        time = end - start
        print(f'Entropy is done (H={ent}). ({time_ent.seconds} seconds.)')

        list_result = [language,
                    len(emb.embeddings.keys()),
                        len(text) / num,
                        ent]
        
        if f'result-{id}.csv' not in os.listdir(f'result/{id}'):
            with open(f'result/{id}/result-{id}.csv', 'w', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['language', 'number of subwords', 'average paragraphs', 'entropy'])
                writer.writerow(list_result)
            with open(f'result/{id}/title-{id}.csv', 'w', encoding='utf-8') as f:
                writer = csv.writer(f)
                list_title.insert(0, language)
                writer.writerow(list_title)
        else:
            with open(f'result/{id}/result-{id}.csv', 'a', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(list_result)
            with open(f'result/{id}/title-{id}.csv', 'a', encoding='utf-8') as f:
                writer = csv.writer(f)
                list_title.insert(0, language)
                writer.writerow(list_title)

            print(f'All processing is done. ({time.seconds} seconds.)')
