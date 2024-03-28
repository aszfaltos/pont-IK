from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.storage import StorageContext
from weaviate.embedded import EmbeddedOptions

import weaviate
from dotenv import load_dotenv

import argparse


def fill_db(doc_path: str, embedding_model: str, index_name: str, chunk_size: int, chunk_overlap: int, filters: list):
    load_dotenv()

    embed_model = OpenAIEmbedding(
        model=embedding_model,
    )

    client = weaviate.Client(embedded_options=EmbeddedOptions())

    documents = SimpleDirectoryReader(doc_path).load_data()

    for doc_filter in filters:
        documents = doc_filter(documents)

    parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = parser.get_nodes_from_documents(documents)

    vector_store = WeaviateVectorStore(weaviate_client=client, index_name=index_name, text_key='content')
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)
    index.storage_context.persist('embedded/')


def empty_db(index_name: str):
    load_dotenv()

    client = weaviate.Client(embedded_options=EmbeddedOptions())

    cursor = None
    while True:
        # Get the next batch of objects
        next_batch = get_batch_with_cursor(client, index_name, 100, cursor)

        # Break the loop if empty – we are done
        if len(next_batch) == 0:
            break

        cursor = next_batch[-1]["_additional"]["id"]

        for item in next_batch:
            client.data_object.delete(
                uuid=item['_additional']['id'],
            )


def get_batch_with_cursor(client, collection_name, batch_size, cursor=None):
    # First prepare the query to run through data
    query = (
        client.query.get(
            collection_name,         # update with your collection name
            [] # update with the required properties
        )
        .with_additional(["id"])
        .with_limit(batch_size)
    )

    # Fetch the next set of results
    if cursor is not None:
        result = query.with_after(cursor).do()
    # Fetch the first set of results
    else:
        result = query.do()

    return result["data"]["Get"][collection_name]


def subject_filter(documents: list) -> list:
    indexes = []
    for idx, doc in enumerate(documents):
        if not doc.metadata['file_name'].endswith('ELTE-intezmenyipont-unfiltered.pdf') or 'Informatika' in doc.text:
            indexes.append(idx)

    ret = []
    for idx in indexes:
        ret.append(documents[idx])

    return ret


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(prog='Data loader', description='Lorem ipsum', epilog='dolor sit amet')
    # parser.add_argument('command', choices=['load', 'empty'])
    # parser.add_argument('-p', '--path', type=str)
    # parser.add_argument('-m', '--model', type=str)
    # parser.add_argument('-n', '--name', type=str)
    # parser.add_argument('-s', '--chunk_size', type=int)
    # parser.add_argument('-o', '--chunk_overlap', type=int)

    fill_db("data/elte_ik/",
            'text-embedding-ada-002',
            'ElteIk',
            1024,
            20,
            [subject_filter])
