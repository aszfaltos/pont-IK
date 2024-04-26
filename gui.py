from chat_engines import ControllerChatEngine
import gradio as gr
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
from pathlib import Path


class GradioGUI:
    def __init__(self, chat_engine: ControllerChatEngine):
        self.chat_engine = chat_engine

    def bot_submit(self, history):
        if len(history) == 0:
            return
        print(history)
        self.chat_engine.reload_history(history)
        resp, _, thinking = self.chat_engine.chat()
        history[-1][1] = self.format_response(resp, thinking)

        return history

    @staticmethod
    def user_submit(user_message, history):
        return "", history + [[user_message, None]]

    @staticmethod
    def undo_click(history):
        return history[:-1]

    def retry_click(self, history):
        history[-1][1] = None
        history = self.bot_submit(history)
        return history

    @staticmethod
    def format_response(response: str, thinking: list[dict]):
        return ('<details>\n<summary>A gondolatmenetem</summary>\n<br>\n' +
                '<ul>' +
                '\n---\n'.join(['\n'.join([f'<li><strong>{key}: </strong>{value}</li>'
                                           for key, value in thought.items()]) for thought in thinking]) +
                '</ul></details>\n\n' + response)

    def launch(self):
        app = FastAPI()
        static_dir = Path('./data/elte_ik')
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

        inject_js = """async () => {
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

        css = """
            a {
                text-decoration: none !important;
                font-style: italic;
            }
            a:hover {
                border-bottom: 1px dotted white;
            }
        """

        with gr.Blocks(css=css) as block:
            with gr.Row() as row:
                with gr.Column() as col:
                    with gr.Row() as row2:
                        chatbot = gr.Chatbot(elem_id='chatbot', height=600)
                    with gr.Row() as row2:
                        msg = gr.Textbox()
                    with gr.Row() as row2:
                        undo = gr.Button(value='Undo')
                        retry = gr.Button(value='Retry')
                        clear = gr.ClearButton([msg, chatbot], variant='stop')
                with gr.Column() as col:
                    document = gr.HTML('<embed height="800" width="700" id="doc" src=""></embed>',
                                       label='document')

            undo.click(self.undo_click, [chatbot], [chatbot])
            retry.click(self.retry_click, [chatbot], [chatbot])
            msg.submit(self.user_submit, [msg, chatbot], [msg, chatbot]).then(
                       self.bot_submit, chatbot, chatbot).then(
                       None, None, None, js=inject_js)

        app = gr.mount_gradio_app(app, block, path="/")
        uvicorn.run(app, host="0.0.0.0", port=7860)