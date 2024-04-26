from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.schema import NodeWithScore

from llama_index.retrievers.bm25 import BM25Retriever

from InstructorEmbedding import INSTRUCTOR
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from utils import ChatHistory, Preprocessor

# spacy stemmer esetleg


class RerankQueryEngine:
    def __init__(self,
                 index: VectorStoreIndex,
                 retriever_top_k: int,
                 reranker_top_n: int,
                 rerank: bool,
                 hybrid_alpha: float,
                 preprocessor_prompt_path: str):
        self._retriever = VectorIndexRetriever(index, similarity_top_k=retriever_top_k)
        # self._bm25 = BM25Retriever.from_defaults(index=index, similarity_top_k=retriever_top_k)

        # githubon nézz rá hogy mi van
                          # VectorIndexRetriever(index,
                          #                      vector_store_query_mode=VectorStoreQueryMode.HYBRID,
                          #                      symilarity_top_k=retriever_top_k,
                          #                      alpha=hybrid_alpha,
                          #                      sparse_top_k=retriever_top_k)
        self.reranker_top_n = reranker_top_n
        self.do_rerank = rerank
        if self.do_rerank:
            self._reranker = INSTRUCTOR('hkunlp/instructor-large')
        self._preprocessor_engine = Preprocessor(preprocessor_prompt_path)

    def _preprocess_query(self, history: ChatHistory) -> str:
        h = history.get_all_messages()[:-1]
        h.append({'role': 'user_last', 'content': history.get_last_n_message(1)[0]['content']})
        return self._preprocessor_engine.preprocess_question(h)

    def _query_index(self, query: str) -> list[NodeWithScore]:
        return self._retriever.retrieve(query)

    def _rerank_nodes(self, query: str, nodes: list[NodeWithScore]) -> list[NodeWithScore]:
        print(len(nodes))
        if not self.do_rerank:
            max_score_idx = np.argsort(list(map(lambda x: x.score, nodes)))[-self.reranker_top_n:].tolist()
            return [nodes[idx] for idx in max_score_idx]

        corpus = list(map(lambda x: ['Represent the Hungarian document for retrieval: ', x.node.get_content()], nodes))
        query = ['Represent the Hungarian question for retrieving supporting documents: ', query]
        query_embed = self._reranker.encode(query)
        corpus_embed = self._reranker.encode(corpus)
        similarities = cosine_similarity(query_embed, corpus_embed)
        retrieved_doc_idx = np.argsort(similarities)[:self.reranker_top_n].tolist()

        return [nodes[idx] for idx in retrieved_doc_idx]

    def query(self, history: ChatHistory) -> (list[NodeWithScore], str):
        """Searches for relevant context in the database
        :param history: The chat history of the user and the assistant
        :return: The retrieved nodes and the preprocessed user question according to witch the search was made
        """
        preprocessed_query = self._preprocess_query(history)
        print(preprocessed_query)
        # nodes_test = self._bm25.retrieve(preprocessed_query)
        # print(len(nodes_test))
        # for n in nodes_test:
        #     print(n.node.metadata['file_name'])
        nodes = self._query_index(preprocessed_query)
        nodes = self._rerank_nodes(preprocessed_query, nodes)

        return nodes, preprocessed_query


def rag_tool() -> (list[NodeWithScore]):
    """
    Searches for relevant context in the database
    :return: The retrieved context nodes and their source files and pages.
    """
    pass
