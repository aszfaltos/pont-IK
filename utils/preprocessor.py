import os.path

from openai import OpenAI
from dotenv import load_dotenv
import json
from os import path


class Preprocessor:
    def __init__(self, prompt_path: str):
        self.prompt_path = prompt_path
        self.prompt = Preprocessor._create_prompt(prompt_path)
        self.client = OpenAI()

    @staticmethod
    def _create_prompt(prompt_path):
        with open(os.path.join(prompt_path, 'examples.json'), 'r') as f:
            d = json.load(f)
            examples = [Preprocessor._messages_to_string(example) for example in d['examples']]

        with open(os.path.join(prompt_path, 'system_message.json'), 'r') as f:
            d = json.load(f)
            system_message = d['role_description'] + "Here are your tasks:\n"
            for idx, task in enumerate(d['tasks']):
                system_message += f'{idx + 1}. {task}\n'

        return_char = "\n"

        return f'SYSTEM:\n{system_message}\n' + \
            f'EXAMPLES:\n{return_char.join(examples)}\n'

    @staticmethod
    def _message_to_string(message: dict):
        return f"{message['role'].upper()}: {message['content']}"

    @staticmethod
    def _messages_to_string(messages: list[dict]):
        return "\n".join([Preprocessor._message_to_string(message) for message in messages])

    def add_history_to_prompt(self, history: list[dict]):
        return_char = "\n"
        return (self.prompt +
                f'HISTORY:\n{return_char.join([Preprocessor._message_to_string(message) for message in history])}' +
                f'\nFORMED:')

    def preprocess_question(self, history: list):
        load_dotenv()

        prompt = self.add_history_to_prompt(history)

        print(history)

        instruct_completion = self.client.completions.create(
            prompt=prompt,
            model='gpt-3.5-turbo-instruct',
            max_tokens=500,
            stop=None
        )

        return instruct_completion.choices[0].text
