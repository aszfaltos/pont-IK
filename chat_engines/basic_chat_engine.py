from llama_index.core.schema import NodeWithScore
from openai import OpenAI

import json
from os import path

from utils import ChatHistory


class BasicChatEngine:
    """
    A class to handle the chat process of the chat engine.
    """
    def __init__(self, prompt_path: str, max_history_length: int):
        """
        :param prompt_path: The path to the directory containing the prompt files.
        :param max_history_length: The maximum number of messages to feed to the llm as chat history.
        """
        self.max_history_length = max_history_length
        self._chat_engine = OpenAI()
        self._history = ChatHistory()
        with open(path.join(prompt_path, 'system_message.json'), 'r') as f:
            self.system_prompt = json.load(f)

    def reload_history(self, gradio_history: list[list[str]]):
        """
        Reloads the chat history from a gradio context.
        :param gradio_history: A list of messages in the format of gradio context.
        """
        self._history = ChatHistory.from_gradio_context(gradio_history, self.max_history_length)

    def push_history(self, role: str, prompt: str):
        """
        Adds a message to the chat history.
        :param role: Role of the message.
        :param prompt: Content of the message.
        """
        self._history.add_message(role, prompt)

    def pop_history(self):
        """
        Removes the last message from the chat history.
        """
        self._history.remove_message(-1)

    def get_history(self):
        """
        Returns the chat history.
        :return: The chat history.
        """
        return self._history

    def chat(self, context: list[NodeWithScore]):
        """
        Chat with the llm.
        :param context: The context from the vector store.
        :return: The response from the llm.
        """
        system_prompt_with_context = (self.system_prompt['message'] + self.system_prompt['source_instruction'] +
                                      'The sources you can use to answer questions:\n' +
                                      '\n'.join([f'SOURCE-{idx}:\n' + cont.node.get_content()
                                                 for idx, cont in enumerate(context)]))
        history = [{'role': 'system', 'content': system_prompt_with_context}] + self._history.get_all_messages()
        resp = (self._chat_engine.chat.completions
                .create(model='gpt-3.5-turbo', messages=history, stream=False, response_format={"type": "json_object"})
                .choices[0].message.content.strip())

        return resp
