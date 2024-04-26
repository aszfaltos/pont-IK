from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.service_context import ServiceContext
from llama_index.core.indices import VectorStoreIndex
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.core import Response
from weaviate.embedded import EmbeddedOptions

import weaviate
from dotenv import load_dotenv

import json

from chat_engines.controller_chat_engine import ControllerChatEngine
from tools import RerankQueryEngine, response_synthesizer, point_calc_regular, point_calc_double

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
                                     False,
                                     .8,
                                     './prompts/preprocessor')
    chat_engine = ControllerChatEngine(query_engine,
                                       response_synthesizer,
                                       './prompts/controller_engine',
                                       15,
                                       [point_calc_regular, point_calc_double])

    # define evaluator
    evaluator = FaithfulnessEvaluator(llm=llm)

    with open('./eval/eval_qa.json', 'r') as f:
        qa_list = json.load(f)['qa_list']
        avg_score = 0
        for qa in qa_list:
            history = qa['history']
            ground_truth = history[-1][1]
            history[-1][1] = None

            chat_engine.reload_history(history)
            resp, ns = chat_engine.chat()

            response = Response(response=resp, source_nodes=ns)

            eval_result = evaluator.evaluate_response(query=history[-1][0], response=response)

            eval_dict = {
                    'Query': [eval_result.query],
                    'Source': ['\n'.join(eval_result.contexts)[:1000] + '...'],
                    'Response': [eval_result.response],
                    'Ground Truth': [ground_truth],
                    'Evaluation Result': ['Pass' if eval_result.passing else 'Fail']
                }

            print(eval_dict)

            avg_score += eval_result.score

        avg_score /= len(qa_list)
        print('Average eval score ' + str(avg_score))

        # TODO: Use correctness evaluator
        # problem here is that its chatgpt it should be faithful but we need to check tool usage and stuff




