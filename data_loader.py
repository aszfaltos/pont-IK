from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.service_context import ServiceContext
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.storage import StorageContext
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core import ChatPromptTemplate
from weaviate.embedded import EmbeddedOptions
from llama_index.readers.file import PDFReader

import weaviate
from dotenv import load_dotenv


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


def empty_db():
    pass


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
    fill_db("data/elte_ik/",
            'text-embedding-ada-002',
            'ElteIk',
            1024,
            20,
            [subject_filter])

