from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
from app.config import COHERE_API_KEY, RUTA_FAISS

# Inicializar embeddings globalmente
embeddings = CohereEmbeddings(
    model="embed-multilingual-v3.0", 
    cohere_api_key=COHERE_API_KEY
)

def get_vector_store():
    """
    Carga el índice FAISS. 
    Nota: La lógica de creación se moverá aquí en el siguiente paso.
    """
    if RUTA_FAISS.exists():
        return FAISS.load_local(
            str(RUTA_FAISS),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    return None