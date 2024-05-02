from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.service_context import ServiceContext
from llama_index.core.indices import VectorStoreIndex
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from weaviate.embedded import EmbeddedOptions

import weaviate
from dotenv import load_dotenv

from pathlib import Path

from chat_engines import ControllerChatEngine
from tools import RerankQueryEngine, response_synthesizer, point_calc_regular, point_calc_double
from gui import GradioGUI

dir_ = Path(__file__).parent


if __name__ == '__main__':
    load_dotenv()

    llm = LlamaOpenAI(model='gpt-4-turbo')
    embed_model = OpenAIEmbedding(model='text-embedding-3-large')

    client = weaviate.Client(embedded_options=EmbeddedOptions())
    vector_store = WeaviateVectorStore(weaviate_client=client, index_name="ElteIk", text_key="content")
    service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)
    store_index = VectorStoreIndex.from_vector_store(vector_store, service_context=service_context)

    query_engine = RerankQueryEngine(store_index,
                                     20,
                                     5,
                                     True,
                                     './prompts/preprocessor')
    chat_engine = ControllerChatEngine(query_engine,
                                       response_synthesizer,
                                       './prompts/controller_engine',
                                       15,
                                       [point_calc_regular, point_calc_double])

    gui = GradioGUI(chat_engine)
    gui.launch()

