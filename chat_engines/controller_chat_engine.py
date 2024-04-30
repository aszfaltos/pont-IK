from typing import Any, Callable, Dict, List

from llama_index.core.tools.function_tool import FunctionTool
from llama_index.core.schema import NodeWithScore
from openai import OpenAI

import json
from os import path

from utils import ChatHistory
from tools import RerankQueryEngine, rag_tool


class ControllerChatEngine:
    def __init__(self, query_engine: RerankQueryEngine, response_tool: Callable[[List[Dict]], str],
                 prompt_path: str, max_history_length: int, other_tools: List[Callable[..., Any]]):
        self._dummy_rag_tool = FunctionTool.from_defaults(rag_tool)
        self._rag_tool = query_engine
        self._response_tool = FunctionTool.from_defaults(response_tool)
        self._other_tools = [FunctionTool.from_defaults(tool) for tool in other_tools]
        self.max_history_length = max_history_length
        self._chat_engine = OpenAI()
        self._history = ChatHistory()
        with open(path.join(prompt_path, 'system_message.json'), 'r') as f:
            sys_prompt = json.load(f)
        with open(path.join(prompt_path, 'tool_examples', 'response_synthesizer.json'), 'r') as f:
            resp_tool_example = json.load(f)
        with open(path.join(prompt_path, 'tool_examples', 'rag_tool.json'), 'r') as f:
            rag_tool_example = json.load(f)
        with open(path.join(prompt_path, 'tool_examples', 'point_calc.json'), 'r') as f:
            point_calc_example = json.load(f)
        self.system_prompt = self._generate_system_prompt(sys_prompt,
                                                          resp_tool_example['examples'] +
                                                          rag_tool_example['examples'] +
                                                          point_calc_example['examples'])

    def reload_history(self, gradio_history: list[list[str]]):
        self._history = ChatHistory.from_gradio_context(gradio_history, self.max_history_length)

    def push_history(self, role: str, prompt: str):
        self._history.add_message(role, prompt)

    def pop_history(self):
        self._history.remove_message(-1)

    def get_history(self):
        return self._history

    def chat(self):
        context = {}

        history = [{'role': 'system', 'content': self.system_prompt}] + self._history.get_all_messages()

        instruction = ""
        resp_dict = []
        nodes = []
        for _ in range(5):
            history[0]['content'] = (self.system_prompt +
                                     '\n# Context got from search\n' + json.dumps(context) +
                                     ('\n# Instruction for next action\nIf you can read this you should always ' +
                                      'decide on your next move according to this instruction no matter what you ' +
                                      'think.\n' + instruction if instruction != '' else ''))

            try:
                history.append({
                    'role': 'assistant',
                    'content': str(resp_dict[-1])
                })
                history.append({
                    'role': 'user',
                    'content': str(resp_dict[-1]['observation'])
                })
            except IndexError:
                pass

            resp = (self._chat_engine.chat.completions
                    .create(model='gpt-4-turbo', messages=history, stream=False,
                            response_format={"type": "json_object"})
                    .choices[0].message.content.strip())
            resp_dict.append(json.loads(resp))
            if resp_dict[-1]['action'] == 'rag_tool':
                nodes, p_q = self._rag_tool.query(self._history)

                observation = {
                    'query': p_q,
                    'files': [{'file': node.node.metadata['file_name'], 'page': node.node.metadata['page_label']}
                              for node in nodes],
                }

                context = {'context': [
                    {'content': node.node.get_content(),
                     'file': node.node.metadata['file_name'], 'page': node.node.metadata['page_label']}
                    for idx, node in enumerate(nodes)]}

                instruction = ("You have context, answer the user question using this context, or try to answer " +
                               "it to the best of your abilities and tell the user that the context doesn't contain " +
                               "enough information to answer their question totally if that's the case.")
            elif resp_dict[-1]['action'] in [tool.metadata.name for tool in self._other_tools]:
                tool = list(filter(lambda tool: resp_dict[-1]['action'] == tool.metadata.name, self._other_tools))[0]
                observation = str(tool.call(**resp_dict[-1]['response']).content)
                if resp_dict[-1]['action'] == 'point_calc_regular' or resp_dict[-1]['action'] == 'point_calc_double':
                    instruction = ("You have calculated the user's acceptance points you should answer the user " +
                                   "by telling them that according to the information they have given they can " +
                                   "expect the points in your last observation.")
            elif resp_dict[-1]['action'] == 'response_synthesizer':
                observation = self._response_tool.call(**resp_dict[-1]['response']).content
                return observation, nodes, resp_dict
            else:
                break

            resp_dict[-1]['observation'] = observation

        print("Limit exceeded or invalid action.")

    @staticmethod
    def _prompt_from_func_tool(func_tool: FunctionTool) -> str:
        desc_split = func_tool.metadata.description.split('\n')
        func_desc = desc_split[0]
        desc = '\n'.join(desc_split[1:])
        return f'- name: {func_tool.metadata.name}\n- function_descriptor: {func_desc}\n- description: {desc}\n'

    def _generate_system_prompt(self, system_prompt: dict, examples: list[dict]) -> str:
        ret = f'# {system_prompt["task_description"]["title"]}\n' + system_prompt['task_description']['content'] + '\n'
        ret += (f'# {system_prompt["general_tool_description"]["title"]}\n' +
                system_prompt['general_tool_description']['content'] + '\n')
        ret += f'# {system_prompt["response_format"]["title"]}\n' + system_prompt['response_format']['content'] + '\n'
        ret += f'# {system_prompt["fix_tools"]["title"]}\n' + system_prompt['fix_tools']['content'] + '\n'
        ret += '# Examples on how to use the tools\n' + '\n\n'.join([json.dumps(example) for example in examples])
        ret += (f'# Tool list\n' +
                'Action should be their name and the response should be their parameters in a JSON format. ' +
                'Be careful to always use all of the parameters, ' +
                'and if your not sure about one of them you should ask the user for more information.'
                '\n'.join(['1.\n' + ControllerChatEngine._prompt_from_func_tool(self._dummy_rag_tool),
                           '2.\n' + ControllerChatEngine._prompt_from_func_tool(self._response_tool)] +
                          [f'{idx+3}\n' + ControllerChatEngine._prompt_from_func_tool(tool)
                           for idx, tool in enumerate(self._other_tools)]))

        return ret
