from classes import WikipediaText, Embedding, Cluster
from datetime import datetime
from wikipedia.exceptions import DisambiguationError, PageError, HTTPTimeoutError

import argparse, os, csv, time, requests

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('language', type=str)
    parser.add_argument('num', type=int)
    parser.add_argument('id', type=str)
    parser.add_argument('--gpu', default=True, type=bool)
    parser.add_argument('--save_embedding', default=False, type=bool)
    parser.add_argument('--save_cluster', default=False, type=bool)
    args = parser.parse_args()

    language = args.language
    num = args.num
    id = args.id
    gpu = args.gpu
    save_embedding = args.save_embedding
    save_cluster = args.save_cluster

    if id not in os.listdir('result'):
        os.mkdir(f'result/{id}')

    start = datetime.now()
    print(f'{language}: processing started at {start.time()}.')
    wiki = WikipediaText(language)
    paragraphs = 0
    cnt = 1
    emb = Embedding(gpu=gpu)
    while cnt <= num:
        if cnt % (num/10) == 0:
            print(f'Text & Embedding: {cnt//(num/10)}/10 is done at {datetime.now().time()}.')
        try:
            text = wiki.random_text()
            emb.embed(text)
            paragraphs += len(text)
            cnt += 1
        except (DisambiguationError, PageError, HTTPTimeoutError) as e:
            time.sleep(1)
            continue
        except requests.exceptions.ConnectionError as e:
            print(f'RequestErrorEncountered: {e}.')
            time.sleep(3)
            continue
    list_title = wiki.list_title
    if save_embedding:
        emb.save_vector(path=f'result/{id}/embedding-{id}.hdf5', name=f'{language}')
    time_emb = datetime.now() - start
    print(f'Text processnig is done ({paragraphs} ¶s, {len(list_title)} articles).')
    print(f'Embedding is done ({len(emb.embeddings)} subwords). ({time_emb.seconds} seconds.)')

    start_clst =  datetime.now()
    clst = Cluster(emb.embeddings, gpu=gpu)
    clst.cluster()
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
                    paragraphs / num,
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



