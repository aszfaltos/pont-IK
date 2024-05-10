from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.service_context import ServiceContext
from llama_index.core.indices import VectorStoreIndex
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.evaluation import CorrectnessEvaluator
from weaviate.embedded import EmbeddedOptions

import weaviate
from dotenv import load_dotenv

import json

from chat_engines.controller_chat_engine import ControllerChatEngine
from tools import RerankQueryEngine, response_synthesizer, point_calc_regular, point_calc_double

import logging
import time
import warnings
import os
import subprocess


# If you want to use the embedded database, set the USE_EMBEDDED_DATABASE to True
USE_EMBEDDED_DATABASE = True


class TestChatEngine:
    """
    A class to test the responses of the chat engine.
    """
    @classmethod
    def setup_class(cls):
        """
        Load the necessary tools and models for the chat engine.
        Create the chat engine, the query engine and the evaluator.
        """
        load_dotenv()

        logging.basicConfig(level=logging.FATAL)
        warnings.simplefilter('ignore')

        llm = LlamaOpenAI(model='gpt-4-turbo')
        embed_model = OpenAIEmbedding(model='text-embedding-3-large')

        if USE_EMBEDDED_DATABASE:
            client = weaviate.Client(embedded_options=EmbeddedOptions())
        else:
            client = weaviate.Client(url=os.environ['DB_URL'])
        vector_store = WeaviateVectorStore(weaviate_client=client, index_name="ElteIk", text_key="content")
        service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)
        store_index = VectorStoreIndex.from_vector_store(vector_store, service_context=service_context)

        query_engine = RerankQueryEngine(store_index,
                                         20,
                                         5,
                                         False,
                                         './prompts',
                                         'gpt-3.5-turbo-instruct')
        cls.chat_engine = ControllerChatEngine(query_engine,
                                               response_synthesizer,
                                               './prompts',
                                               15,
                                               [point_calc_regular, point_calc_double])

        cls.evaluator = CorrectnessEvaluator(llm=llm)

    def test_correctness(self, subtests):
        """
        Run the chat engine with the query-response pairs from the eval_qa.json file
        and check the correctness of the engine's responses against the example response from the file.
        """
        with open('./eval/eval_qa.json', 'r') as f:
            qa_list = json.load(f)['qa_list']
            avg_score = 0
            avg_runtime = 0
            for idx, qa in enumerate(qa_list):
                start = time.time()
                with subtests.test(idx=idx):
                    history = qa['history']
                    ground_truth = history[-1][1]
                    history[-1][1] = None

                    self.chat_engine.reload_history(history)
                    resp, ns, _ = self.chat_engine.chat()

                    eval_result = self.evaluator.evaluate(query=history[-1][0],
                                                          response=resp,
                                                          reference=ground_truth)

                    end = time.time()
                    avg_score += eval_result.score
                    avg_runtime += end - start
                    assert eval_result.passing, (f'\n\tQuery: {history[-1][0]}'
                                                 f'\n\tResponse: {resp}'
                                                 f'\n\tGround truth: {ground_truth}'
                                                 f'\n\tFeedback: {eval_result.feedback}'
                                                 f'\n\tTime: {end - start} seconds')
                    print(f'\t{idx}. - Time: {end - start} seconds - {"Passed." if eval_result.passing else "Failed."}')

            avg_score /= len(qa_list)
            avg_runtime /= len(qa_list)
            print('\nAverage eval score: ' + str(avg_score) + '\nAverage run time: ' + str(avg_runtime) + ' seconds')


if __name__ == '__main__':
    subprocess.call(['pytest', '-s', str(__file__)])
