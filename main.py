import os
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from langchain_core.globals import set_debug

# Importaciones de tu lógica modular en la carpeta app/
from app.agente import procesar
from app.embeddings import cargar_vector_store
from app.config import cargar_dataframes
from app.router import clasificar

set_debug(os.getenv("VIOLET_DEBUG", "false").lower() == "true")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestión del ciclo de vida de la API.
    Carga los recursos analíticos pesados en la memoria del servidor
    ÚNICAMENTE al arrancar. Evita leer el disco en cada pregunta.
    """
    print("🚀 [Violet API] Inicializando recursos globales...")
    try:
        # Cargamos DataFrames (.csv) y base de datos vectorial en el estado global de la app
        app.state.dfs = cargar_dataframes()
        app.state.vs = cargar_vector_store()
        print(
            "✅ [Violet API] DataFrames y Vector Store cargados con éxito en memoria RAM."
        )
    except Exception as e:
        print(f"❌ [Violet API] Error crítico durante la inicialización: {str(e)}")
        # Fallback preventivo para evitar que el contenedor colapse por completo
        app.state.dfs = {}
        app.state.vs = None
    yield
    print("🛑 [Violet API] Liberando recursos y apagando servidor.")


# Instanciamos FastAPI con el manejador de ciclo de vida
app = FastAPI(
    title="ViolTech — Violet AI Agent Backend",
    description="API Stateless asíncrona para el procesamiento de IA, análisis de Churn y reportes corporativos.",
    version="2.0.0",
    lifespan=lifespan,
)

# Configuración de CORS: Vital para que el frontend en Hugging Face Spaces pueda comunicarse con Render
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # En producción se puede cambiar por la URL específica de HF Space
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- MODELOS DE VALIDACIÓN (PYDANTIC) ---
class ChatRequest(BaseModel):
    pregunta: str
    historial: List[Dict[str, Any]]


# --- ENDPOINTS ---


@app.get("/", status_code=status.HTTP_200_OK)
def health_check():
    """
    Endpoint de control de estado (Health Check).
    Indispensable para que el balanceador de carga de Render sepa que el servicio está operativo.
    """
    return {
        "status": "online",
        "agente": "Violet",
        "infraestructura": "FastAPI Stateless",
        "estado_recursos": {
            "dataframes_ready": bool(app.state.dfs),
            "vector_store_ready": bool(app.state.vs),
        },
    }


@app.post("/api/v1/chat", status_code=status.HTTP_200_OK)
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint principal de inferencia.
    Recibe la pregunta del usuario y el historial desde Streamlit, inyecta los recursos
    en memoria y retorna la respuesta procesada por el agente de LangChain.
    """
    # Verificación preventiva de infraestructura
    if not app.state.dfs or app.state.vs is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El servidor no ha terminado de inicializar sus bases de datos o modelos de embeddings.",
        )

    try:
        # Ejecutamos el router
        categoria = clasificar(request.pregunta)

        # Invocamos la función asíncrona pura de app/agente.py
        respuesta, _ = await procesar(
            pregunta=request.pregunta,
            dfs=app.state.dfs,
            vector_store=app.state.vs,
            categoria=categoria,
            historial=request.historial,
        )

        return {"respuesta": respuesta, "categoria": categoria}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno en el motor de ejecución del agente: {str(e)}",
        )
