import sys

from utils import data_handler

import argparse


if __name__ == '__main__':
    """
    Main entry point for the data loader.
    This program will load the data from the specified path into the Weaviate database.
    """
    parser = argparse.ArgumentParser(prog='Data loader',
                                     description='Use this tool to load your pdf data quickly into the weaviate ' +
                                                 'database for embedding and indexing.')
    parser.add_argument('-p', '--path', type=str, default='./data/elte_ik', help='Path to data directory.')
    parser.add_argument('-m', '--model',
                        type=str, default='text-embedding-3-large', help='Open AI embedding model.')
    parser.add_argument('-n', '--name', type=str, default='ElteIk', help='Index name.')
    parser.add_argument('-s', '--chunk_size',
                        type=int, default=1024, help='Size of one context node in tokens.')
    parser.add_argument('-o', '--chunk_overlap',
                        type=int, default=200, help='Size of the overlap between context nodes in tokens.')
    parser.add_argument('-e', '--empty', action='store_true', help='Use if db already has data.')
    parser.add_argument('-l', '--local', action='store_true', help='Use for embedded Weaviate database.')

    args = parser.parse_args(sys.argv[1:])

    if args.empty:
        data_handler.empty_db(args.name, args.local)

    data_handler.fill_db(args.path,
                         args.model,
                         args.name,
                         args.chunk_size,
                         args.chunk_overlap,
                         args.local,
                         [data_handler.subject_filter])
