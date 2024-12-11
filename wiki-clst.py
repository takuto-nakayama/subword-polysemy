from classes import WikipediaText, Embedding, Cluster
from datetime import datetime
from plotly import express as px
import argparse, os, pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('language', type=str)
    parser.add_argument('num', type=int)
    parser.add_argument('id', type=str)
    parser.add_argument('--gpu', default=True, type=bool)
    args = parser.parse_args()

    language = args.language
    num = args.num
    id = args.id
    gpu = args.gpu

    list_result = []
    list_title = []

    start = datetime.now()
    wiki = WikipediaText()
    wiki.load_text(language=language, num=num)
    text = wiki.list_text
    list_title.append(wiki.title)
    time_text = datetime.now() - start
    print(f'Text processnig is done. ({time_text.seconds} seconds.)')

    start_emb = datetime.now()
    emb = Embedding(text)
    emb.embed(gpu=gpu)
    time_emb = datetime.now() - start_emb
    print(f'Embedding is done. ({time_emb.seconds} seconds.)')

    start_clst =  datetime.now()
    clst = Cluster(emb.embeddings)
    clst.cluster(gpu=gpu)
    time_clst = datetime.now() - start_clst
    print(f'Clustering is done. ({time_clst.seconds} seconds.)')

    start_ent = datetime.now()
    ent = clst.entropy()
    end = datetime.now()
    time_ent = end - start_ent
    time = end - start
    print(f'Entropy is done ({ent}). ({time_ent.seconds} seconds.)')

    list_result.append([language,
                        len(emb.embeddings.keys()),
                        len(text) / num,
                        ent])
    
    if id not in os.listdir('result'):
        with open(f'result/{id}/result-{id}.csv', 'w') as f:
            f.write('language, number of subwords, average paragraphs, entropy\n')
            f.write(f'{list_result[0]},{list_result[1]}, {list_result[2]}, {list_result[3]}\n')
    else:
        with open(f'result/{id}/result-{id}.csv', 'a') as f:
            f.write(f'{list_result[0]},{list_result[1]}, {list_result[2]}, {list_result[3]}\n')

    print(f'All processing is done. ({time.seconds} seconds.)')



