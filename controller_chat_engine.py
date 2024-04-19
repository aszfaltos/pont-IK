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
            self.system_prompt = self._generate_system_prompt(json.load(f))

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

        observation = {}
        resp_dict = []
        nodes = []
        for _ in range(5):
            history[0]['content'] = (self.system_prompt + '\n# Previous iterations\n' + json.dumps(resp_dict) +
                                     '\n# Context got from search:\n' + json.dumps(context))

            resp = (self._chat_engine.chat.completions
                    .create(model='gpt-4-turbo', messages=history, stream=False,
                            response_format={"type": "json_object"})
                    .choices[0].message.content.strip())

            resp_dict.append(json.loads(resp))
            print(resp_dict[-1])
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
            elif resp_dict[-1]['action'] in [tool.metadata.name for tool in self._other_tools]:
                tool = list(filter(lambda tool: resp_dict[-1]['action'] == tool.metadata.name, self._other_tools))[0]
                observation = str(tool.call(**resp_dict[-1]['response']).content)
            elif resp_dict[-1]['action'] == 'response_synthesizer':
                observation = self._response_tool.call(**resp_dict[-1]['response']).content
                return observation, nodes
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

    def _generate_system_prompt(self, system_prompt: dict) -> str:
        ret = f'# {system_prompt["task_description"]["title"]}\n' + system_prompt['task_description']['content'] + '\n'
        ret += (f'# {system_prompt["general_tool_description"]["title"]}\n' +
                system_prompt['general_tool_description']['content'] + '\n')
        ret += f'# {system_prompt["response_format"]["title"]}\n' + system_prompt['response_format']['content'] + '\n'
        ret += f'# {system_prompt["fix_tools"]["title"]}\n' + system_prompt['fix_tools']['content'] + '\n'
        ret += ('# Examples\n' +
                """
                1. The user has asked a question.
                    {
                        "thought": "Plusz információra van szükségem a kérdés megválaszolásához.",
                        "reason": "A felhasználó a felvételi eljárásról kérdezett, és a válasz információtartalmának pontossága
                        érdekében érdemes a dokumentumokban keresni. Ehhez nincs szükségem további információra.",
                        "action": "rag_tool",
                        "response": {}
                    }
                2. The user has asked a question but in a previous iteration you searched for additional information so now you should answer it.
                    {
                        "thought": "A kontextus alapján meg tudom válaszolni a kérdést.",
                        "reason": "Az előző megfigyelés által megadott kontextus elég információt tartalmaz a kérdés
                        megválaszolásához.",
                        "action": "response_synthesizer",
                        "response": { "content": [ 
                                        { "text": "Egy generált válasz rész a kontextus alapján.", "file": "file2.pdf", "page": "13"},
                                        { "text": "Egy generált válasz rész kontextus nélkül.", "file": "file1.pdf", "page": "5"} 
                                      ] 
                                    }
                    }
                3. The user have not asked a question, they only want to chat with you
                    {
                        "thought": "A felhasználó nem kérdezett még csak beszélgetést kezdeményez.",
                        "reason": "Nem szükséges kontextus a válasz generálásához, mivel még csak köszönt a felhasználó.",
                        "action": "response_synthesizer",
                        "response": { "content": [ 
                                        { "text": "Szia! Miben segíthetek?.", "file": "None", "page": "None"}
                                      ] 
                                    }
                    }
                4. The user has asked a question but in a previous iteration you searched for additional information so now you should answer it.
                    {
                        "thought": "Az előző ciklusban megkapott források alapján meg tudom válaszolni a kérdést.",
                        "reason": "Elegendő információ van az előző iterációban megszerzett dokumentumokban a kérdés
                        megválaszolásához.",
                        "action": "response_synthesizer",
                        "response": { "content": [ 
                                        { "text": "Válasz a kérdésre.", "file": "file1.pdf", "page": "2"},
                                        { "text": "Válasz folytatása.", "file": "file2.pdf", "page": "25"}
                                      ] 
                                    }
                    }
                5. The user has asked a question but in a previous iteration you searched for additional information so now you should answer it.
                    {
                        "thought": "Az előző ciklusban megkapott források alapján nem tudom megválaszolni a kérdést.",
                        "reason": "Nincs elegendő információ a forrásban a kérdés megválaszolásához, de már egyszer használtam a 
                        rag_tool-t, igy egy általános választ kell adnom a rendelkezésre álló információk alapján és ezt jelezni a 
                        kérdező felé.",
                        "action": "response_synthesizer",
                        "response": { "content": [ 
                                        { "text": "Általános válasz a rendelkezésre álló információk alapján.", 
                                          "file": "file3.pdf", "page": "12"}
                                        { "text": "A forrás alapján csak ennyi ifromációval tudok szolgálni, esetleg segíthetek még 
                                        valamiben?.", "file": "None", "page": "None"}
                                      ] 
                                    }
                    }
                """)
        ret += (f'# Tool list\n' +
                'Action should be their name and the response should be their parameters in a JSON format. ' +
                'Be careful to always use all of the parameters, ' +
                'and if your not sure about one of them you should ask the user for more information.'
                '\n'.join(['1.\n' + ControllerChatEngine._prompt_from_func_tool(self._dummy_rag_tool),
                           '2.\n' + ControllerChatEngine._prompt_from_func_tool(self._response_tool)] +
                          [f'{idx+3}\n' + ControllerChatEngine._prompt_from_func_tool(tool)
                           for idx, tool in enumerate(self._other_tools)]))

        return ret
