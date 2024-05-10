import os
import json

from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.schema import NodeWithScore

from InstructorEmbedding import INSTRUCTOR
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from utils import ChatHistory, QuestionFormer


class RerankQueryEngine:
    def __init__(self,
                 index: VectorStoreIndex,
                 retriever_top_k: int,
                 reranker_top_n: int,
                 rerank: bool,
                 prompt_path: str,
                 question_forming_model: str):
        """
        :param index: VectorStoreIndex to retrieve data from.
        :param retriever_top_k: How many documents will be retrieved after a search.
        :param reranker_top_n: How many documents will remain after reranking.
        :param rerank: Whether to rerank documents.
        :param prompt_path: Path to the prompt's directory.
        :param question_forming_model: Open AI llm model used for question forming before search.
        """
        self._retriever = VectorIndexRetriever(index, similarity_top_k=retriever_top_k)
        self.reranker_top_n = reranker_top_n
        self.do_rerank = rerank
        if self.do_rerank:
            self._reranker = INSTRUCTOR('hkunlp/instructor-large')
        with open(os.path.join(prompt_path, 'reranker', 'instruct_prompts.json'), 'r') as f:
            self.reranker_prompts = json.load(f)
        self._preprocessor_engine = QuestionFormer(os.path.join(prompt_path, 'question_former'), question_forming_model)

    def _preprocess_query(self, history: ChatHistory) -> str:
        h = history.get_last_n_message(5)[:-1]  # limit history to fit in context length
        # Label the last message of the user according to few shot training. (See question_former prompts)
        h.append({'role': 'user_last', 'content': history.get_last_n_message(1)[0]['content']})
        return self._preprocessor_engine.preprocess_question(h)

    def _query_index(self, query: str) -> list[NodeWithScore]:
        return self._retriever.retrieve(query)

    def _rerank_nodes(self, query: str, nodes: list[NodeWithScore]) -> list[NodeWithScore]:
        if not self.do_rerank:  # Just return top_n when not using reranker.
            max_score_idx = np.argsort(list(map(lambda x: x.score, nodes)))[-self.reranker_top_n:].tolist()
            return [nodes[idx] for idx in max_score_idx]

        corpus = list(map(lambda x: [self.reranker_prompts['corpus'], x.node.get_content()], nodes))
        query = [self.reranker_prompts['query'], query]
        query_embed = self._reranker.encode(query)
        corpus_embed = self._reranker.encode(corpus)

        similarities = cosine_similarity(query_embed, corpus_embed)
        # The first element of similarities is the similarity between the query and the documents,
        # the second is the similarity between all the documents with each other.
        retrieved_doc_idx = np.argsort(similarities[0])[-self.reranker_top_n:].tolist()

        return [nodes[idx] for idx in retrieved_doc_idx]

    def query(self, history: ChatHistory) -> (list[NodeWithScore], str):
        """
        Searches for relevant context in the database
        :param history: The chat history of the user and the assistant
        :return: The retrieved nodes and the preprocessed user question according to witch the search was made
        """
        preprocessed_query = self._preprocess_query(history)
        nodes = self._query_index(preprocessed_query)
        nodes = self._rerank_nodes(preprocessed_query, nodes)

        return nodes, preprocessed_query


def rag_tool() -> (list[NodeWithScore]):  # Just an interface for the engine to interact with.
    """
    Searches for relevant context in the database
    :return: The retrieved context nodes and their source files and pages.
    """
    pass
