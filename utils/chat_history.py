import math

import pytest


class ChatHistory:
    def __init__(self):
        self._history: list[dict] = []

    @staticmethod
    def from_gradio_context(ctx: list[list[str]], max_history: int):
        ret = ChatHistory()

        look_back = min(math.ceil(max_history / 2), len(ctx))
        print(look_back)
        for [msg, rsp] in ctx[-look_back:]:
            if msg is not None:
                ret.add_message('user', msg)
            if rsp is not None:
                ret.add_message('assistant', rsp)

        return ret

    def add_message(self, role: str, message: str):
        self._history.append({'role': role, 'content': message})

    def remove_message(self, index: int):
        self._history.remove(self._history[index])

    def get_all_messages(self):
        return self._history

    def get_last_n_message(self, n: int):
        n = min(len(self._history), n)
        if n == 0:
            return []
        return self._history[-n:]

    def clear(self):
        self._history = []


@pytest.fixture
def history():
    yield ChatHistory()


def test_get_all_messages(history):
    assert history.get_all_messages() == []


def test_add_message(history):
    history.add_message('user', 'Hello')
    assert history.get_all_messages()[0]['role'] == 'user'
    assert history.get_all_messages()[0]['content'] == 'Hello'


def test_remove_message(history):
    history.add_message('user', '0')
    history.add_message('user', '1')
    history.add_message('user', '2')
    history.remove_message(0)

    assert len(history.get_all_messages()) == 2
    assert history.get_all_messages()[0]['content'] == '1'


def test_get_last_n_message(history):
    history.add_message('user', '0')
    history.add_message('user', '1')
    history.add_message('user', '2')
    assert history.get_last_n_message(2)[0]['content'] == '1'


def test_clear(history):
    history.add_message('user', '0')
    history.add_message('user', '1')
    history.add_message('user', '2')
    history.clear()
    assert history.get_all_messages() == []


def test_from_gradio_context_no_cutoff():
    history = ChatHistory.from_gradio_context([['0', '1'], ['2', '3']], 15)
    arr = history.get_all_messages()
    assert arr[0]['role'] == 'user'
    assert arr[0]['content'] == '0'
    assert arr[1]['role'] == 'assistant'
    assert arr[1]['content'] == '1'
    assert arr[2]['role'] == 'user'
    assert arr[2]['content'] == '2'
    assert arr[3]['role'] == 'assistant'
    assert arr[3]['content'] == '3'


def test_from_gradio_context_cutoff():
    history = ChatHistory.from_gradio_context([['0', '1'], ['2', '3']], 2)
    arr = history.get_all_messages()
    assert len(arr) == 2


def test_from_gradio_context_none():
    history = ChatHistory.from_gradio_context([['0', None]], 15)
    arr = history.get_all_messages()
    assert len(arr) == 1
