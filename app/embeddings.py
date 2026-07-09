import pickle
import cohere as cohere_sdk
from pathlib import Path
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from typing import List, Any
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import COHERE_API_KEY, RUTA_FAISS, RUTA_DOCS


class VioletEmbeddings(Embeddings):
    """Wrapper directo sobre el cliente Cohere para embeddings."""

    def __init__(self, api_key: str, model: str = "embed-multilingual-v3.0"):
        self.client = cohere_sdk.Client(api_key)
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        texts = [t for t in texts if t.strip()]
        if not texts:
            return []
        resp = self.client.embed(
            texts=texts, model=self.model, input_type="search_document"
        )
        return resp.embeddings

    def embed_query(self, text: str) -> list[float]:
        resp = self.client.embed(
            texts=[text], model=self.model, input_type="search_query"
        )
        return resp.embeddings[0]


# Instancia global
embeddings = VioletEmbeddings(api_key=COHERE_API_KEY)


def cargar_vector_store():
    """Carga o reconstruye el índice híbrido FAISS + BM25."""
    ruta_fragmentos = Path(str(RUTA_FAISS)) / "fragmentos.pkl"

    if Path(RUTA_FAISS).exists() and ruta_fragmentos.exists():
        vs = FAISS.load_local(
            str(RUTA_FAISS), embeddings, allow_dangerous_deserialization=True
        )
        with open(ruta_fragmentos, "rb") as f:
            fragmentos = pickle.load(f)
    else:
        if not RUTA_DOCS.exists():
            raise FileNotFoundError(f"Carpeta no encontrada: {RUTA_DOCS}")

        loader = DirectoryLoader(
            str(RUTA_DOCS), glob="**/*.pdf", loader_cls=PyMuPDFLoader
        )
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
        fragmentos = splitter.split_documents(docs)
        fragmentos = [f for f in fragmentos if f.page_content.strip()]

        if not fragmentos:
            raise ValueError("Los PDFs no contienen texto extraíble.")

        vs = FAISS.from_documents(fragmentos, embeddings)
        vs.save_local(str(RUTA_FAISS))

        Path(RUTA_FAISS).mkdir(parents=True, exist_ok=True)
        with open(ruta_fragmentos, "wb") as f:
            pickle.dump(fragmentos, f)

    # Recuperadores
    faiss_retriever = vs.as_retriever(search_kwargs={"k": 3})
    bm25_retriever = BM25Retriever.from_documents(fragmentos)
    bm25_retriever.k = 3

    class VioletHybridRetriever(BaseRetriever):
        faiss_ret: BaseRetriever
        bm25_ret: BaseRetriever

        def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Any]:
            # Ejecuta ambas búsquedas en paralelo/secuencial
            docs_faiss = self.faiss_ret.invoke(query)
            docs_bm25 = self.bm25_ret.invoke(query)
            
            # Intercala y elimina duplicados por contenido para no saturar al agente
            vistos = set()
            docs_combinados = []
            for doc in (docs_faiss + docs_bm25):
                if doc.page_content not in vistos:
                    vistos.add(doc.page_content)
                    docs_combinados.append(doc)
            return docs_combinados

    # Retornamos tu nuevo recuperador híbrido inmune a errores de importación
    return VioletHybridRetriever(faiss_ret=faiss_retriever, bm25_ret=bm25_retriever)
