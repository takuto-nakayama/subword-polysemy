from classes import Dataset, Embedding, Cluster
from datetime import datetime
from plotly import express as px
import argparse, pandas as pd

if __name__ == 'emb-clst':
    #input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='the file path must be ".hdf5"')
    parser.add_argument('save_to', type=str)
    args = parser.parse_args()

    #define variables
    path = args.path
    save_to = args.save_to

    clst = Cluster(embpath=path)
    dict_entropy = {}
    for lang in clst.langs:
        start = datetime.now()
        clst.cluster(lang)
        dict_entropy[lang] = clst.entropy()
        end = datetime.now()
        time = end - start
        print(f'{lang} is done. ({time.seconds} seconds.)')
    
    d = {'language': dict_entropy.keys(), 'entropy': dict_entropy.values()}
    df = pd.DataFrame(d)

    fig = px.scatter(
        df,
        x = 'language',
        y = 'entropy',
    )
    fig.write_html(f'{save_to}.html')
    fig.show()