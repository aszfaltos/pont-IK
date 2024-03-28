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

dir_ = Path(__file__).parent

from basic_chat_engine import BasicChatEngine
from tools import RerankQueryEngine


def prepare_pdf_html_embed(src: str, page_num: int):
    return f'<embed id="doc" src="/static/{os.path.basename(src)}#page={page_num}" width="700" height="900"></embed>'


def format_answer(resp: str, context):
    resp = json.loads(resp)["response"]
    ret = ""
    for piece in resp:
        print(piece)
        if piece['source'] == "NONE":
            ret += piece['content'] + ' '
            continue
        ctx_idx = int(piece['source'].split('-')[1])
        src = context[ctx_idx].node.metadata['file_name']
        page_num = context[ctx_idx].node.metadata['page_label']
        ret += \
            f'<a class="chat-link" href="/static/{os.path.basename(src)}#page={page_num}">{piece["content"] + " "}</a>'

    return ret.strip()


if __name__ == '__main__':
    load_dotenv()

    llm = LlamaOpenAI(model='gpt-3.5-turbo')

    embed_model = OpenAIEmbedding(model='text-embedding-ada-002')

    client = weaviate.Client(embedded_options=EmbeddedOptions())

    vector_store = WeaviateVectorStore(weaviate_client=client, index_name="ElteIk", text_key="content")
    service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)

    store_index = VectorStoreIndex.from_vector_store(vector_store, service_context=service_context)

    query_engine = RerankQueryEngine(store_index, 10, 3, False, .8, './prompts/preprocessor')
    chat_engine = BasicChatEngine('./prompts/chat_engine', 15)

    app = FastAPI()
    static_dir = Path('./data/elte_ik')
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    with gr.Blocks() as block:
        chatbot = gr.Chatbot(elem_id='chatbot')
        msg = gr.Textbox()
        clear = gr.ClearButton([msg, chatbot])

        document = gr.HTML('<embed id="doc"></embed>', label='document')

        def user(user_message, history):
            return "", history + [[user_message, None]]

        def bot(history):
            if len(history) == 0:
                return
            chat_engine.reload_history(history[:-1])
            nodes, preprocessed = query_engine.query(history[-1], chat_engine.get_history())
            resp = chat_engine.chat(nodes)
            history[-1][1] = format_answer(resp, nodes)

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

    # action agentek
    # rag action vagy tool action
    # markdownban formázd be a gondolatmenetet a chatbe
