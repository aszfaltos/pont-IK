from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.storage import StorageContext
from llama_index.core.node_parser import SentenceSplitter
from weaviate.embedded import EmbeddedOptions

import weaviate
from dotenv import load_dotenv
import os


def fill_db(doc_path: str, embedding_model: str, index_name: str, chunk_size: int, chunk_overlap: int, local_db: bool,
            filters: list):
    """
    Fill the Weaviate database with the documents from the specified path.
    :param doc_path: The path to the documents.
    :param embedding_model: The OpenAI embedding model to use.
    :param index_name: The name of the database index.
    :param chunk_size: The size of one context node in tokens.
    :param chunk_overlap: The size of the overlap between context nodes in tokens.
    :param local_db: If True, use the embedded Weaviate database.
    :param filters: The list of filters to apply to the document nodes.
    """
    load_dotenv()

    embed_model = OpenAIEmbedding(
        model=embedding_model,
    )

    if local_db:
        client = weaviate.Client(embedded_options=EmbeddedOptions())
    else:
        client = weaviate.Client(url=os.environ['DB_URL'])

    documents = SimpleDirectoryReader(doc_path).load_data()

    for doc_filter in filters:
        documents = doc_filter(documents)

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(documents)

    vector_store = WeaviateVectorStore(weaviate_client=client, index_name=index_name, text_key='content')
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)
    index.storage_context.persist('embedded/')


def empty_db(index_name: str, local_db: bool):
    """
    Empty the Weaviate database.
    :param index_name: The name of the database index to empty.
    :param local_db: If True, use the embedded Weaviate database.
    """
    load_dotenv()

    if local_db:
        client = weaviate.Client(embedded_options=EmbeddedOptions())
    else:
        client = weaviate.Client(url=os.environ['DB_URL'])

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
    """
    Get a batch of objects from the Weaviate database.
    :param client: Weaviate client.
    :param collection_name: Name of the collection to get objects from.
    :param batch_size: The size of the batches to work with.
    :param cursor: The cursor to use for pagination.
    :return: The collected objects.
    """
    # First prepare the query to run through data
    query = (
        client.query.get(
            collection_name,         # update with your collection name
            []  # update with the required properties
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
    """
    Filter out unrelevant parts of ELTE-intezmenyipont-unfiltered.pdf.
    :param documents: The list of documents to filter.
    :return: The filtered list of documents.
    """
    indexes = []
    for idx, doc in enumerate(documents):
        if not doc.metadata['file_name'].endswith('ELTE-intezmenyipont-unfiltered.pdf') or 'Informatika' in doc.text:
            indexes.append(idx)

    ret = []
    for idx in indexes:
        ret.append(documents[idx])

    return ret
