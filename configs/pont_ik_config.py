from dataclasses import dataclass


@dataclass
class PontIkConfig:
    """Configuration for the ELTE IK assistant chatbot."""
    prompt_path: str = './prompts'
    db_index_name: str = 'ElteIk'
    embedding_model: str = 'text-embedding-3-large'
    llm_model: str = 'gpt-4o'
    question_forming_model = 'gpt-3.5-turbo-instruct'
    question_forming_tokenizer = 'cl100k_base'
    retriever_top_k: int = 20
    reranker_top_n: int = 5
    chat_history_length: int = 15
    do_rerank: bool = True
