import sys

from utils import data_handler

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Data loader',
                                     description='Use this tool to load your pdf data quickly into the weaviate ' +
                                                 'database for embedding and indexing.')
    parser.add_argument('-p', '--path', type=str, default='./data', help='Path to data directory')
    parser.add_argument('-m', '--model',
                        type=str, default='text-embedding-ada-002', help='Open AI embedding model')
    parser.add_argument('-n', '--name', type=str, default='ElteIk', help='Index name')
    parser.add_argument('-s', '--chunk_size',
                        type=int, default=1024, help='Size of one context node in tokens')
    parser.add_argument('-o', '--chunk_overlap',
                        type=int, default=200, help='Size of the overlap between context nodes in tokens')
    parser.add_argument('-e', '--empty', action='store_true', help='Use if db already has data')

    # TODO: 200 okenes átfedés, paragrafus és mondat határokon (SentenceSplitter()) 1000-2000 token között

    args = parser.parse_args(sys.argv[1:])

    if args.empty:
        data_handler.empty_db(args.name)

    data_handler.fill_db(args.path,
                         args.model,
                         args.name,
                         args.chunk_size,
                         args.chunk_overlap,
                         [data_handler.subject_filter])
