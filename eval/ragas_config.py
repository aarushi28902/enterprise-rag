from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings


def get_ragas_llm():
    return LangchainLLMWrapper(Ollama(model="llama3.2"))


def get_ragas_embeddings():
    return LangchainEmbeddingsWrapper(OllamaEmbeddings(model="llama3.2"))
