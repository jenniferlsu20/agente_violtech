import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables e entorno
load_dotenv()

# Rutas base
BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_DATOS = BASE_DIR / "datos"
RUTA_DOCS = RUTA_DATOS / "docs"
RUTA_FAISS = RUTA_DATOS / "faiss_index"
RUTA_HISTORIAL = RUTA_DATOS / "historial_violet.json"

# Archivos de Datos
RUTA_CHURN = RUTA_DATOS / "clientes_scored.csv"
RUTA_STORE = RUTA_DATOS / "Superstore_cleaned.csv"

# Configuración del Sistema
VENTANA_MEMORIA = 5
MODELO_LLM = "command-r-plus-08-2024"

# Claves de API (Se leen desde .env)
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Credenciales de Servicio
SMTP_USER = os.getenv("SMTP_USER")
SMTP_APP_PASSWORD = os.getenv("SMTP_APP_PASSWORD")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Diccionarios de configuración UI/Lógica
BADGES = {
    "CHURN": ("📉", "Gestión de Retención"),
    "FINANZAS": ("💰", "Análisis Superstore"),
    "POLITICAS": ("🛡️", "Base de Conocimiento"),
}

# CSS global
CSS_VIOLTECH = """
<style>
    /* Estilo ejecutivo: sin degradados ni bordes redondeados excesivos */
    .metric-card {
        background-color: #f8f9fa;
        border-left: 4px solid #2c3e50;
        padding: 0.8rem;
        margin-bottom: 0.5rem;
        color: #34495e;
    }
    /* Estilo limpio para los headers de chat */
    .stChatFloatingInputContainer {
        border-radius: 0;
    }
</style>
"""
