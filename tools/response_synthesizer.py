import os


def response_synthesizer(content: list[dict]) -> str:
    """
    Generates a response string from a list of dictionaries.

    :param content: List of dictionaries in a format:
        [
            { "text": "Answer section 1", "file": "context file path 1", "page": "page number 1"},
            { "text": "Answer section 2", "file": "None", "page": "None"}
        ]
        Here the "text" key represents the generated answer segment, and the "file" and "page" are the parameters of
        the context file which was used as source to generate that part of the answer, if no context file was used for a
        part "file" and"page" should be None.

    :returns: The concatenated answer where answer segments were converted into anchor tags that link to the proper
        source file if "file" and "page" are not None.
    """
    ret = ''
    for piece in content:
        try:  # If the file is None or there is no file just return the text.
            if piece['file'] is None or piece['file'] == 'None':
                ret += piece['text'] + ' '
                continue
        except KeyError:
            ret += piece['text'] + ' '
            continue
        src = piece['file']
        page_num = piece['page']
        ret += \
            (f'<a class="chat-link" href="file=data/elte_ik/{os.path.basename(src)}' +
             f'#page={page_num}">{piece["text"]} </a>')

    return ret.strip()


def test_rs_none_source():
    content = [{"text": "Segment 1", "file": None, "page": None}]
    resp = response_synthesizer(content)
    assert resp == 'Segment 1'


def test_rs_none_string_source():
    content = [{"text": "Segment 1", "file": "None", "page": "None"}]
    resp = response_synthesizer(content)
    assert resp == 'Segment 1'


def test_rs_no_source():
    content = [{"text": "Segment 1"}]
    resp = response_synthesizer(content)
    assert resp == 'Segment 1'


def test_rs_with_source():
    content = [{"text": "Segment 1", "file": "data/elte_ik/test.pdf", "page": 13}]
    resp = response_synthesizer(content)
    assert resp == '<a class="chat-link" href="/static/elte_ik/test.pdf#page=13">Segment 1 </a>'


def test_rs_with_source_string_page():
    content = [{"text": "Segment 1", "file": "data/elte_ik/test.pdf", "page": "13"}]
    resp = response_synthesizer(content)
    assert resp == '<a class="chat-link" href="/static/elte_ik/test.pdf#page=13">Segment 1 </a>'


def test_rs_multiple_segments():
    content = [
        {"text": "Segment 1", "file": "data/elte_ik/test.pdf", "page": "13"},
        {"text": "Segment 2", "file": "data/elte_ik/test.pdf", "page": "12"},
        {"text": "Segment 3", "file": None, "page": None},
        {"text": "Segment 4", "file": None, "page": None},
    ]
    resp = response_synthesizer(content)
    assert resp == ('<a class="chat-link" href="/static/elte_ik/test.pdf#page=13">Segment 1 </a>'
                    '<a class="chat-link" href="/static/elte_ik/test.pdf#page=12">Segment 2 </a>'
                    'Segment 3 Segment 4')
