import os.path

from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.service_context import ServiceContext
from llama_index.core.indices import VectorStoreIndex
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.prompts import ChatMessage, MessageRole
from llama_index.core.schema import QueryBundle, NodeWithScore
from weaviate.embedded import EmbeddedOptions
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from InstructorEmbedding import INSTRUCTOR
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn

import gradio as gr

import weaviate
from dotenv import load_dotenv

from preprocessor import create_prompt, preprocess_question
import json
from os import path
from openai import OpenAI
from pathlib import Path
from gradio_pdf import PDF

dir_ = Path(__file__).parent


class ChatMemory:
    def __init__(self):
        self.memory: list[dict] = []

    def add_message(self, role: str, message: str):
        self.memory.append({'role': role, 'content': message})

    def get_last_n_message(self, n: int):
        n = max(len(self.memory), n)
        if n == 0:
            return []
        return self.memory[-n:]

    def clear(self):
        self.memory = []


class ChatEngine:
    def __init__(self,
                 index: VectorStoreIndex,
                 prompt_path: str,
                 history_len: int,
                 retriever_top_k: int,
                 rerank: bool,
                 hybrid_alpha: float):
        self.history_len = history_len
        self.index = index
        self.memory = ChatMemory()
        self.prompt_path = prompt_path
        with open(path.join(prompt_path, 'chat_engine', 'system_message.json'), 'r') as f:
            self.system_prompt = json.load(f)['message']
        self.chat_engine = OpenAI()
        self.retriever = VectorIndexRetriever(index,
                                              vector_store_query_mode=VectorStoreQueryMode.HYBRID,
                                              symilarity_top_k=retriever_top_k,
                                              alpha=hybrid_alpha,
                                              sparse_top_k=retriever_top_k)
        self.do_rerank = rerank
        if self.do_rerank:
            self.reranker = INSTRUCTOR('hkunlp/instructor-large')
        self.preprocessor_prompt = create_prompt(prompt_path)
        self.preprocessor_engine = OpenAI()

    def memory_from_gr_hist(self, gr_hist: list[list[str]]):
        look_back = max(self.history_len // 2, len(gr_hist))
        for [msg, rsp] in gr_hist[-look_back:]:
            if msg is not None:
                self.memory.add_message('user', msg)
            if rsp is not None:
                self.memory.add_message('assistant', rsp)

    def query(self):
        # create general query question
        history = self.memory.memory[:-1] + [{'role': 'user_last', 'content': self.memory.memory[-1]['content']}]
        preprocessed = preprocess_question(self.preprocessor_prompt, history, self.preprocessor_engine)

        # get context and metadata
        query_bundle = QueryBundle(preprocessed)
        nodes = self.retriever.retrieve(query_bundle)

        return nodes, preprocessed

    def rerank(self, query: str, nodes: list[NodeWithScore]):
        if not self.do_rerank:
            max_score_idx = np.argmax(list(map(lambda x: x.score, nodes)))
            return nodes[max_score_idx]

        corpus = list(map(lambda x: ['Represent the Hungarian document for retrieval: ', x.node.get_content()], nodes))
        query = ['Represent the Hungarian question for retrieving supporting documents: ', query]
        query_embed = self.reranker.encode(query)
        corpus_embed = self.reranker.encode(corpus)
        similarities = cosine_similarity(query_embed, corpus_embed)
        retrieved_doc_idx = np.argmax(similarities)

        return nodes[retrieved_doc_idx]

    def chat(self, context: NodeWithScore):
        # generate response
        system_prompt_with_context = self.system_prompt + context.node.get_content()
        history = [{'role': 'system', 'content': system_prompt_with_context}] + self.memory.memory
        resp = self.chat_engine.chat.completions.create(model='gpt-3.5-turbo', messages=history, stream=True)

        # return stream
        partial_message = ""
        for chunk in resp:
            if chunk.choices[0].delta.content is not None:
                partial_message = partial_message + chunk.choices[0].delta.content
                yield partial_message


def prepare_pdf_html_embed(src: str, page_num: int):
    return f'<embed src="/static/{os.path.basename(src)}#page={page_num}" width="700" height="900"></embed>'


if __name__ == '__main__':
    load_dotenv()

    llm = LlamaOpenAI(model='gpt-3.5-turbo')

    embed_model = OpenAIEmbedding(model='text-embedding-ada-002')

    client = weaviate.Client(embedded_options=EmbeddedOptions())

    vector_store = WeaviateVectorStore(weaviate_client=client, index_name="ElteIk", text_key="content")
    service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)

    store_index = VectorStoreIndex.from_vector_store(vector_store, service_context=service_context)

    chat_engine = ChatEngine(store_index,
                             './prompts', 6, 10, False, .8)

    app = FastAPI()
    static_dir = Path('./data/elte_ik')
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    with (gr.Blocks() as block):
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.ClearButton([msg, chatbot])

        document = gr.HTML(label='document')

        def user(user_message, history):
            return "", history + [[user_message, None]]

        def bot(history):
            if len(history) == 0:
                return
            chat_engine.memory_from_gr_hist(history)
            nodes, preprocessed = chat_engine.query()
            context = chat_engine.rerank(preprocessed, nodes)
            for partial_resp in chat_engine.chat(context):
                history[-1][1] = partial_resp
                yield (history,
                       prepare_pdf_html_embed(context.node.metadata['file_name'], context.node.metadata['page_label']))

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, [chatbot, document]
        )

    app = gr.mount_gradio_app(app, block, path="/")
    uvicorn.run(app, host="0.0.0.0", port=7860)
