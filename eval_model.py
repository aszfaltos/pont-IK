from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.service_context import ServiceContext
from llama_index.core.indices import VectorStoreIndex
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.evaluation import CorrectnessEvaluator
from llama_index.core import Response
from weaviate.embedded import EmbeddedOptions

import weaviate
from dotenv import load_dotenv

import json

from chat_engines.controller_chat_engine import ControllerChatEngine
from tools import RerankQueryEngine, response_synthesizer, point_calc_regular, point_calc_double

import unittest
import logging


class ChatEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super(ChatEngineTests, cls).setUpClass()

        load_dotenv()

        logging.basicConfig(level=logging.FATAL)

        llm = LlamaOpenAI(model='gpt-4-turbo')
        embed_model = OpenAIEmbedding(model='text-embedding-3-large')

        client = weaviate.Client(embedded_options=EmbeddedOptions())
        vector_store = WeaviateVectorStore(weaviate_client=client, index_name="ElteIk", text_key="content")
        service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)
        store_index = VectorStoreIndex.from_vector_store(vector_store, service_context=service_context)

        query_engine = RerankQueryEngine(store_index,
                                         20,
                                         5,
                                         False,
                                         .8,
                                         './prompts/preprocessor')
        cls.chat_engine = ControllerChatEngine(query_engine,
                                               response_synthesizer,
                                               './prompts/controller_engine',
                                               15,
                                               [point_calc_regular, point_calc_double])

        cls.evaluator = CorrectnessEvaluator(llm=llm)

    def test_correctness(self):
        with open('./eval/eval_qa.json', 'r') as f:
            qa_list = json.load(f)['qa_list']
            avg_score = 0
            for idx, qa in enumerate(qa_list):
                with self.subTest(idx=idx):
                    history = qa['history']
                    ground_truth = history[-1][1]
                    history[-1][1] = None

                    self.chat_engine.reload_history(history)
                    resp, ns, _ = self.chat_engine.chat()

                    eval_result = self.evaluator.evaluate(query=history[-1][0],
                                                          response=resp,
                                                          reference=ground_truth)
                    self.assertTrue(eval_result.passing, eval_result.feedback)

                    avg_score += eval_result.score

            avg_score /= len(qa_list)
            print('\nAverage eval score ' + str(avg_score))


if __name__ == '__main__':
    unittest.main()
