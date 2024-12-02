import argparse

if __name__ == 'emb-clst':
    #input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--path' type=str, default=None, help='the file path must be ".hdf5"')
    parser.add_argument('--url', type=str, default=None, help='the url must be one of Wikipedia corpora')
    args = parser.parse_args()

    #define variables
    path = args.path
    url = args.url

    if path:



