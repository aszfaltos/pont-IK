from llama_index.core.schema import NodeWithScore
from openai import OpenAI

import json
from os import path

from utils import ChatHistory


class BasicChatEngine:
    def __init__(self, prompt_path: str, max_history_length: int):
        self.max_history_length = max_history_length
        self._chat_engine = OpenAI()
        self._history = ChatHistory()
        with open(path.join(prompt_path, 'system_message.json'), 'r') as f:
            self.system_prompt = json.load(f)

    def reload_history(self, gradio_history: list[list[str]]):
        self._history = ChatHistory.from_gradio_context(gradio_history, self.max_history_length)

    def push_history(self, role: str, prompt: str):
        self._history.add_message(role, prompt)

    def pop_history(self):
        self._history.remove_message(-1)

    def get_history(self):
        return self._history

    def chat(self, context: list[NodeWithScore]):
        system_prompt_with_context = (self.system_prompt['message'] + self.system_prompt['source_instruction'] +
                                      'The sources you can use to answer questions:\n' +
                                      '\n'.join([f'SOURCE-{idx}:\n' + cont.node.get_content()
                                                 for idx, cont in enumerate(context)]))
        history = [{'role': 'system', 'content': system_prompt_with_context}] + self._history.get_all_messages()
        resp = (self._chat_engine.chat.completions
                .create(model='gpt-3.5-turbo', messages=history, stream=False, response_format={"type": "json_object"})
                .choices[0].message.content.strip())

        print(history)
        print(resp)

        return resp
