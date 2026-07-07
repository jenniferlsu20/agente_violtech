import json
import pandas as pd
from pathlib import Path
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

# Importaciones locales (se adaptarán para recibir parámetros explícitos)
from envio import es_destino_seguro, manejar_datos_contacto
from agente import construir_agente, llm
from router import clasificar

# Rutas globales
RUTA_DATOS = Path("datos")
RUTA_HISTORIAL = RUTA_DATOS / "historial.json"
RUTA_CHURN = RUTA_DATOS / "churn.csv"
RUTA_STORE = RUTA_DATOS / "superstore.csv"


def procesar(pregunta: str, dfs: dict, vector_store, historial: list, contexto_sesion: dict) -> tuple:
    """
    Router central de Violet con gestión de estados asíncrona y stateless para FastAPI.
    
    Recibe 'contexto_sesion' (un diccionario con el estado enviado por el frontend)
    y devuelve una tupla: (respuesta_texto, categoria, contexto_sesion_actualizado)
    """
    p_lower = pregunta.lower().strip()
    nuevos_estados = contexto_sesion.copy() if contexto_sesion else {}

    # 1. INTERCEPTOR DE CONTACTO (Estado de envío activo)
    if nuevos_estados.get("proceso_envio", {}).get("activo"):
        # Se pasa nuevos_estados para que la función lo modifique de forma pura
        respuesta, categoria, nuevos_estados = es_destino_seguro(destino, canal)
        return respuesta, categoria, nuevos_estados

    # 2. DETECCIÓN DE INTENCIÓN DE ENVÍO
    if any(m in p_lower for m in ["gmail", "telegram"]):
        return _flujo_envio(p_lower, nuevos_estados)

    # 3. FILTRO DE SEGURIDAD (Scope)
    if not _es_relevante(p_lower):
        return (
            "Lo siento, mi especialidad es el análisis de datos de clientes y reportes de gestión "
            "para ViolTech. ¿Te gustaría analizar alguna métrica de retención o finanzas?",
            "FUERA_DE_CONTEXTO",
            nuevos_estados
        )

    # 4. CLASIFICACIÓN Y PROCESAMIENTO
    categoria = clasificar(pregunta)

    # Actualizar estado de reporte activo en el diccionario de retorno
    if categoria in ["CHURN", "FINANZAS"]:
        nuevos_estados["tipo_reporte_activo"] = (
            "financiero" if categoria == "FINANZAS" else "churn"
        )

    # Procesar según categoría
    if categoria == "POLITICAS" or (categoria == "FUERA_SCOPE" and "política" in p_lower):
        respuesta = _consultar_documentacion(pregunta, vector_store)
        return respuesta, categoria, nuevos_estados

    if categoria in ["CHURN", "FINANZAS"]:
        respuesta, cat = _procesar_analisis(pregunta, dfs, vector_store, categoria, historial)
        
        # Guardamos el último reporte generado en el estado para cuando el usuario pida enviarlo
        tipo = nuevos_estados.get("tipo_reporte_activo")
        if tipo:
            nuevos_estados[f"ultimo_reporte_{tipo}"] = respuesta
            
        return respuesta, cat, nuevos_estados

    return (
        "No pude clasificar tu solicitud, por favor intenta ser más específico.",
        categoria,
        nuevos_estados
    )


def _flujo_envio(p_lower: str, nuevos_estados: dict) -> tuple:
    """Orquestador de envíos puramente funcional."""
    tipo = nuevos_estados.get("tipo_reporte_activo")
    reporte = nuevos_estados.get(f"ultimo_reporte_{tipo}") if tipo else None

    if not reporte:
        return "⚠️ No hay un reporte activo. Genera uno primero.", "ACCION_ENVIO", nuevos_estados

    canal = "gmail" if "gmail" in p_lower else "telegram"

    if canal == "telegram":
        # Se asume que ahora devuelve la respuesta y la categoría de forma directa
        respuesta, cat = _manejar_confirmacion_envio(canal="telegram", reporte=reporte)
        return respuesta, cat, nuevos_estados

    # Activar estado de espera para Gmail en el diccionario que regresaremos a la API
    nuevos_estados["proceso_envio"] = {"activo": True, "canal": "gmail"}
    return (
        "Perfecto. Por favor, indícame tu correo de Gmail (ejemplo: nombre@gmail.com).",
        "ACCION_ENVIO",
        nuevos_estados,
    )


def _es_relevante(pregunta_lower: str) -> bool:
    temas = [
        "churn", "financiero", "datos", "cliente", "reporte", "grafico",
        "analisis", "ventas", "retencion", "política", "manual", "violet",
        "ingreso", "ganancia"
    ]
    return any(tema in pregunta_lower for tema in temas)


def _consultar_documentacion(pregunta, vector_store):
    try:
        docs = vector_store.invoke(pregunta)
        contexto = "\n\n".join(d.page_content for d in docs)
        plantilla = PromptTemplate.from_template(
            "Eres Violet, analista de ViolTech. Contexto:\n{contexto}\n\nPregunta: {pregunta}\n\nRespuesta:"
        )
        return (plantilla | llm | StrOutputParser()).invoke(
            {"contexto": contexto, "pregunta": pregunta}
        )
    except Exception as e:
        return f"Error al consultar políticas: {str(e)}"


def _procesar_analisis(pregunta, dfs, vector_store, categoria, historial):
    clave = "churn" if categoria == "CHURN" else "superstore"
    if clave not in dfs:
        return f"Dataset '{clave}' no disponible.", categoria

    try:
        agente = construir_agente(dfs[clave], vector_store, clave, historial)
        res = agente.invoke({"input": pregunta})
        return res.get("output", "No obtuve respuesta."), categoria
    except Exception as e:
        return f"Error técnico: {str(e)}", categoria


# --- FUNCIONES DE PERSISTENCIA Y CARGA ---
def cargar_historial():
    return (
        json.loads(RUTA_HISTORIAL.read_text(encoding="utf-8"))
        if RUTA_HISTORIAL.exists()
        else []
    )


def guardar_historial(mensajes):
    RUTA_DATOS.mkdir(exist_ok=True)
    RUTA_HISTORIAL.write_text(
        json.dumps(mensajes[-30:], ensure_ascii=False, indent=2), encoding="utf-8"
    )


def cargar_dataframes() -> dict:
    """
    Carga los dataframes en memoria. 
    En FastAPI, esto se ejecutará una sola vez en el evento 'startup' 
    para mantener el rendimiento óptimo de la API.
    """
    return {
        k: pd.read_csv(p)
        for k, p in [("churn", RUTA_CHURN), ("superstore", RUTA_STORE)]
        if p.exists()
    }
    
    