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

import gradio as gr

import weaviate
from dotenv import load_dotenv

from preprocessor import create_prompt, preprocess_question
import json
from os import path
from openai import OpenAI


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
                 reranker_top_n: int,
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
        self.reranker = LLMRerank(choice_batch_size=5, top_n=reranker_top_n, service_context=index.service_context)
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
        print(len(nodes))
        try:
            r_nodes = self.reranker.postprocess_nodes(nodes, query_bundle)
            if len(r_nodes) > 0:
                nodes = r_nodes
        except Exception as _:
            pass
        print(len(nodes))
        context = nodes[0]
        file_name = context.node.metadata['file_name']
        page_number = context.node.metadata['page_label']

        return context, file_name, page_number

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


if __name__ == '__main__':
    load_dotenv()

    llm = LlamaOpenAI(model='gpt-3.5-turbo')

    embed_model = OpenAIEmbedding(model='text-embedding-ada-002')

    client = weaviate.Client(embedded_options=EmbeddedOptions())

    vector_store = WeaviateVectorStore(weaviate_client=client, index_name="ElteIk", text_key="content")
    service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)

    store_index = VectorStoreIndex.from_vector_store(vector_store, service_context=service_context)

    chat_engine = ChatEngine(store_index,
                             './prompts', 6, 10, 1, .8)

    with gr.Blocks() as interface:
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.ClearButton([msg, chatbot])

        file = gr.Textbox()
        page = gr.Number()

        # TODO: add clear and undo

        def user(user_message, history):
            return "", history + [[user_message, None]]

        def bot(history):
            if len(history) == 0:
                return
            chat_engine.memory_from_gr_hist(history)
            context, file_name, page_number = chat_engine.query()
            for partial_resp in chat_engine.chat(context):
                history[-1][1] = partial_resp
                yield history, file_name, page_number

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, [chatbot, file, page]
        )

    interface.launch()
