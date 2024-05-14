import os.path

from openai import OpenAI
import tiktoken
from dotenv import load_dotenv
import json
from typing import Any


class QuestionFormer:
    """
    This class is responsible for forming query questions from the chat history.
    """
    def __init__(self, prompt_path: str, model: str, tokenizer: str):
        """
        Initialize the QuestionFormer object.
        :param prompt_path: The path to the directory containing the prompt files.
        :param model: The OpenAI model to use.
        :param tokenizer: The OpenAI tokenizer that the model uses.
        """
        self.model = model
        self.sections: Any = None
        self.prompt_path = prompt_path
        self.prompt = self._create_prompt(prompt_path)
        self.client = OpenAI()
        self.encoding = tiktoken.get_encoding(tokenizer)

    def _create_prompt(self, prompt_path):
        """
        Create the prompt from the prompt files.
        :param prompt_path: The path to the directory containing the prompt files.
        :return: The prompt.
        """
        with open(os.path.join(prompt_path, 'examples.json'), 'r') as f:
            d = json.load(f)
            examples = [QuestionFormer._messages_to_string(example) for example in d['examples']]

        with open(os.path.join(prompt_path, 'system_message.json'), 'r') as f:
            d = json.load(f)
            system_message = d['role_description']
            self.sections = d['section_labels']
            for idx, task in enumerate(d['tasks']):
                system_message += f'{idx + 1}. {task}\n'

        return_char = "\n"

        return f'{self.sections["system"]}:\n{system_message}\n' + \
               f'{self.sections["examples"]}:\n{return_char.join(examples)}\n'

    @staticmethod
    def _message_to_string(message: dict):
        """
        Convert a message to a string.
        :param message: A message in the format of a dictionary.
        :return: The message as a string.
        """
        return f"{message['role'].upper()}: {message['content']}"

    @staticmethod
    def _messages_to_string(messages: list[dict]):
        """
        Convert a list of messages to a string.
        :param messages: The list of messages in the format of a dictionary.
        :return: The messages as a string.
        """
        return "\n".join([QuestionFormer._message_to_string(message) for message in messages])

    def add_history_to_prompt(self, history: list[dict]):
        """
        Add the chat history to the prompt.
        :param history: The chat history.
        :return: The prompt with the chat history.
        """
        return_char = "\n"
        history_str = return_char.join([QuestionFormer._message_to_string(message) for message in history])
        if len(self.encoding.encode(history_str)) > 2000:
            # 2000 characters won't be longer than 2000 tokens, but should be enough for generation.
            history_str = history_str[:2000]
        return (self.prompt +
                f'{self.sections["history"]}:\n' +
                f'{history_str}\n' +
                f'{self.sections["formed"]}:')

    def preprocess_question(self, history: list):
        """
        Preprocess the question.
        :param history: The chat history.
        :return: The generated query question.
        """
        load_dotenv()

        prompt = self.add_history_to_prompt(history)

        instruct_completion = self.client.completions.create(
            prompt=prompt,
            model=self.model,
            max_tokens=500,
            stop=None
        )

        return instruct_completion.choices[0].text


def test_question_former_token_limit():
    load_dotenv()

    qf = QuestionFormer('../prompts/question_former', 'gpt-3.5-turbo-instruct', 'cl100k_base')
    tokenizer = tiktoken.get_encoding('cl100k_base')
    history = [{'role': 'user_last', 'content': 'word ' * 8000}]  # 8000 tokens

    prompt = qf.add_history_to_prompt(history)
    tokenized = tokenizer.encode(prompt)
    assert len(tokenized) < 4096


