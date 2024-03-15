import os.path

from openai import OpenAI
from dotenv import load_dotenv
import json
from os import path


def message_to_string(message: dict):
    return f"{message['role'].upper()}: {message['content']}"


def messages_to_string(messages: list[dict]):
    return "\n".join([message_to_string(message) for message in messages])


def create_prompt(prompt_path: str):
    examples = []
    with open(os.path.join(prompt_path, 'preprocessor', 'examples.json'), 'r') as f:
        d = json.load(f)
        examples = [messages_to_string(example) for example in d['examples']]

    system_message = ''
    with open(os.path.join(prompt_path, 'preprocessor', 'system_message.json'), 'r') as f:
        d = json.load(f)
        system_message = d['role_description'] + "Here are your tasks:\n"
        for idx, task in enumerate(d['tasks']):
            system_message += f'{idx+1}. {task}\n'

    return_char = "\n"

    return f'SYSTEM:\n{system_message}\n' + \
           f'EXAMPLES:\n{return_char.join(examples)}\n'


def add_history_to_prompt(prompt: str, history: list[dict]):
    return_char = "\n"
    return (prompt +
            f'HISTORY:\n{return_char.join([message_to_string(message) for message in history])}' +
            f'\nFORMED:')


def preprocess_question(prompt: str, history: list, openai_client: OpenAI):
    load_dotenv()

    prompt = add_history_to_prompt(prompt, history)

    instruct_completion = openai_client.completions.create(
        prompt=prompt,
        model='gpt-3.5-turbo-instruct',
        max_tokens=500,
        stop=None
    )

    return instruct_completion.choices[0].text


if __name__ == '__main__':
    load_dotenv()

    chat_history = [
        {
            'role': 'user',
            'content': 'Milyen pontszámra lenne szükségem, hogy bekerüljek az ELTE IK PTI Bsc-re?'
        },
        {
            'role': 'assistant',
            'content': 'Erre a szakra az előző évi felvételi ponthatár 430 pont volt, azonban a szabályok változása miatt ez valószínűleg változni fog az idei évben.'
        },
        {
            'role': 'user_last',
            'content': 'Milyen érettségi eredményekre lenne szükségem, hogy elérjem ezt a ponthatárt?'
        }
    ]

    completion = preprocess_question(chat_history, OpenAI(), './prompts/preprocessor')

    print(completion)


# instructhos a role description az úgy legyen hogy ez egy szöveg és csináld azt hogy és nem az hogy te ki vagy