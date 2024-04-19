import os.path

from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.service_context import ServiceContext
from llama_index.core.indices import VectorStoreIndex
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from weaviate.embedded import EmbeddedOptions
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn

import gradio as gr

import weaviate
from dotenv import load_dotenv

import json
from pathlib import Path

from controller_chat_engine import ControllerChatEngine
from tools import RerankQueryEngine, response_synthesizer, point_calc_regular, point_calc_double

dir_ = Path(__file__).parent


def prepare_pdf_html_embed(src: str, page_num: int):
    return f'<embed id="doc" height="1000" width="700" src="/static/{os.path.basename(src)}#page={page_num}"></embed>'


def format_answer(resp: str, context):
    resp = json.loads(resp)["response"]
    ret = ""
    for piece in resp:
        if piece['source'] == "NONE":
            ret += piece['content'] + ' '
            continue
        ctx_idx = int(piece['source'].split('-')[1])
        src = context[ctx_idx].node.metadata['file_name']
        page_num = context[ctx_idx].node.metadata['page_label']
        ret += \
            f'<a class="chat-link" href="/static/{os.path.basename(src)}#page={page_num}">{piece["content"] + " "}</a>'

    return ret.strip()


def format_controller_answer(resp: dict):
    ret = ''
    for piece in resp['content']:
        if piece['file'] == 'None':
            ret += piece['text'] + ' '
            continue
        src = piece['file']
        page_num = piece['page']
        ret += \
            f'<a class="chat-link" href="/static/{os.path.basename(src)}#page={page_num}">{piece["text"] + " "}</a>'

    return ret.strip()


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

    app = FastAPI()
    static_dir = Path('./data/elte_ik')
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    with gr.Blocks() as block:
        chatbot = gr.Chatbot(elem_id='chatbot')
        msg = gr.Textbox()
        clear = gr.ClearButton([msg, chatbot])

        document = gr.HTML('<embed height="1000" width="700" id="doc"></embed>', label='document')

        def user(user_message, history):
            return "", history + [[user_message, None]]

        def bot(history):
            if len(history) == 0:
                return

            chat_engine.reload_history(history)
            resp, ns = chat_engine.chat()
            for n in ns:
                print(n.node.metadata['file_name'])
            history[-1][1] = resp
            print(history)

            return history


        INJECT_JS = """async () => {
                        setTimeout(() => {
                            var anchors = document.getElementsByClassName("chat-link");
                            let embed = document.getElementById("doc");
                            for (let a of anchors) {
                                a.addEventListener("click", (e) => {
                                    e.preventDefault();
                                    embed.src = e.target.href;
                                });
                            }
                        }, 10)
                    }"""

        msg.submit(user, [msg, chatbot], [msg, chatbot]).then(
            bot, chatbot, chatbot).then(
            None, None, None, js=INJECT_JS)

    app = gr.mount_gradio_app(app, block, path="/")
    uvicorn.run(app, host="0.0.0.0", port=7860)
