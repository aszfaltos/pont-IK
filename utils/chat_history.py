

class ChatHistory:
    def __init__(self):
        self._history: list[dict] = []

    @staticmethod
    def from_gradio_context(ctx: list[list[str]], max_history: int):
        ret = ChatHistory()

        look_back = max(max_history // 2, len(ctx))
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
