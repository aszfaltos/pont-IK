from chat_engines import ControllerChatEngine
import gradio as gr
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
from pathlib import Path


class GradioGUI:
    """
    A class to handle the gradio GUI.
    """
    def __init__(self, chat_engine: ControllerChatEngine):
        """
        :param chat_engine: The chat engine to use when messages are sent.
        """
        self.chat_engine = chat_engine

    def bot_submit(self, history):
        """
        Submits the user message to the chat engine and gets the response.
        :param history: The chat history in the format of gradio context.
        :return: The chat history updated with the response.
        """
        if len(history) == 0:
            return
        self.chat_engine.reload_history(history)
        resp, _, thinking = self.chat_engine.chat()
        history[-1][1] = self.format_response(resp, thinking)

        return history

    @staticmethod
    def user_submit(user_message, history):
        """
        Adds the user message to the chat history.
        :param user_message: The last message of the user from the input textbox.
        :param history: The previous chat history in the format of gradio context.
        :return: Empty string for the input textbox and the chat history updated with the user message.
        """
        return "", history + [[user_message, None]]

    @staticmethod
    def undo_click(history):
        """
        Removes the last message from the chat history.
        :param history: Chat history in the format of gradio context.
        :return: Updated chat history without the last user-assistant message pair.
        """
        if len(history) == 0:
            return []
        return history[:-1]

    def retry_click(self, history):
        """
        Removes the chat engine's last response and submits the chat history to the chat engine again.
        :param history: Chat history in the format of gradio context.
        :return: Updated chat history without the last assistant message and the new response from the chat engine.
        """
        if len(history) == 0:
            return []
        history[-1][1] = None
        history = self.bot_submit(history)
        return history

    @staticmethod
    def format_response(response: str, thinking: list[dict]):
        """
        Formats the response of the chat engine with the thinking process.
        :param response: The response of the chat engine.
        :param thinking: The thinking process of the chat engine.
        :return: The formatted markdown response.
        """
        return ('<details>\n<summary>A gondolatmenetem</summary>\n<br>\n' +
                '<ul>' +
                '\n---\n'.join(['\n'.join([f'<li><strong>{key}: </strong>{value}</li>'
                                           for key, value in thought.items()]) for thought in thinking]) +
                '</ul></details>\n\n' + response)

    def launch(self):
        """
        Defines the gradio GUI and launches the FastAPI server.
        """
        app = FastAPI()
        static_dir = Path('./data')
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
            with gr.Row() as _:
                with gr.Column() as _:
                    with gr.Row() as _:
                        chatbot = gr.Chatbot(label="Pont számító asszisztens", elem_id='chatbot', height=600)
                    with gr.Row() as _:
                        msg = gr.Textbox(label="Bemenet",
                                         placeholder="Kérdezz valamit az ELTE IK felvételi eljárással kapcsolatban!")
                    with gr.Row() as _:
                        undo = gr.Button(value='Vissza')
                        retry = gr.Button(value='Újra')
                        _ = gr.ClearButton([msg, chatbot], variant='stop', value='Törlés')
                with gr.Column() as _:
                    _ = gr.HTML('<embed height="800" width="700" id="doc" src="/static/placeholder.pdf"></embed>',
                                label='document')

            undo.click(self.undo_click, [chatbot], [chatbot])
            retry.click(self.retry_click, [chatbot], [chatbot])
            msg.submit(self.user_submit, [msg, chatbot], [msg, chatbot]).then(
                       self.bot_submit, chatbot, chatbot).then(
                       None, None, None, js=inject_js)

        app = gr.mount_gradio_app(app, block, path="/")
        uvicorn.run(app, host="0.0.0.0", port=7860)
