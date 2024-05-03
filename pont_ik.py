import argparse
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.service_context import ServiceContext
from llama_index.core.indices import VectorStoreIndex
from llama_index.vector_stores.weaviate import WeaviateVectorStore

import weaviate
from weaviate.embedded import EmbeddedOptions

from chat_engines import ControllerChatEngine
from tools import RerankQueryEngine, response_synthesizer, point_calc_regular, point_calc_double
from gui import GradioGUI

from configs import PontIkConfig

dir_ = Path(__file__).parent


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='ELTE IK assistant chatbot',
                                     description='This program will try to connect to a Weaviate database and run an ' +
                                                 'assistant chatbot with a Gradio frontend.')
    parser.add_argument('-l', '--local', action='store_true', help='Use this to use Weaviate embedded version')

    args = parser.parse_args(sys.argv[1:])

    load_dotenv()

    llm = LlamaOpenAI(model='gpt-4-turbo')
    embed_model = OpenAIEmbedding(model='text-embedding-3-large')

    if args.local:
        client = weaviate.Client(embedded_options=EmbeddedOptions())
    else:
        client = weaviate.Client(url=os.environ['DB_URL'])

    vector_store = WeaviateVectorStore(weaviate_client=client, index_name=PontIkConfig.db_index_name,
                                       text_key="content")
    service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)
    store_index = VectorStoreIndex.from_vector_store(vector_store, service_context=service_context)

    query_engine = RerankQueryEngine(store_index,
                                     PontIkConfig.retriever_top_k,
                                     PontIkConfig.reranker_top_n,
                                     PontIkConfig.do_rerank,
                                     PontIkConfig.prompt_path,
                                     PontIkConfig.question_forming_model)
    chat_engine = ControllerChatEngine(query_engine,
                                       response_synthesizer,
                                       PontIkConfig.prompt_path,
                                       PontIkConfig.chat_history_length,
                                       [point_calc_regular, point_calc_double])

    gui = GradioGUI(chat_engine)
    gui.launch()

