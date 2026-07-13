import json
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Importaciones locales (se adaptarán para recibir parámetros explícitos)
from app.config import (
    RUTA_CHURN,
    RUTA_STORE,
    RUTA_HISTORIAL,
    RUTA_DATOS,
    cargar_dataframes,
)
from app.envio import (
    es_destino_seguro,
    manejar_datos_contacto,
    procesar_confirmacion_envio,
)
from app.agente import construir_agente, truncar_historial, llm
from app.router import clasificar

def _es_error_limite_api(excepcion: Exception) -> bool:
    """
    Detecta si una excepcion corresponde a un límite de tasa/cuota de la API
    de Cohere (HTTP 429), para mostrar un mensaje claro al usuario en vez de
    un traceback técnico con headers HTTP.
    """
    texto = str(excepcion).lower()
    return "429" in texto or "trial key" in texto or "rate limit" in texto

MENSAJE_LIMITE_API = (
    "⏳ Violet está recibiendo demasiados solicitudes en este momento "
    "(se alcanzó el límite de la API). Por favor, intenta de nuevo en "
    "unos minutos."
)

async def procesar(
    pregunta: str, dfs: dict, vector_store, historial: list, contexto_sesion: dict
) -> tuple:
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
        respuesta, nuevos_estados = await manejar_datos_contacto(
            pregunta, nuevos_estados
        )
        return respuesta, "ACCION_ENVIO", nuevos_estados

    # 2. DETECCIÓN DE INTENCIÓN DE ENVÍO
    if any(m in p_lower for m in ["gmail", "telegram"]):
        return await _flujo_envio(p_lower, nuevos_estados)

    # 3. CLASIFICACIÓN Y PROCESAMIENTO
    # Le pasamos al router el tipo de reporte activo de la sesión, para que
    # respuestas ambiguas (ej. "sí", "genera el gráfico") no pierdan el
    # contexto de si veníamos de CHURN o FINANZAS.
    categoria_previa = nuevos_estados.get("tipo_reporte_activo")
    categoria_anterior = "CHURN" if categoria_previa == "churn" else "FINANZAS"
    categoria = clasificar(pregunta, categoria_anterior=categoria_anterior)

    # Actualizar estado de reporte activo en el diccionario de retorno
    if categoria in ["CHURN", "FINANZAS"]:
        nuevos_estados["tipo_reporte_activo"] = (
            "financiero" if categoria == "FINANZAS" else "churn"
        )

    # Procesar según categoría
    if categoria == "POLITICAS" or (
        categoria == "FUERA_SCOPE" and "política" in p_lower
    ):
        respuesta = _consultar_documentacion(pregunta, vector_store)
        return respuesta, categoria, nuevos_estados

    if categoria in ["CHURN", "FINANZAS"]:
        respuesta, cat = _procesar_analisis(
            pregunta, dfs, vector_store, categoria, historial
        )

        # Guardamos el último reporte generado en el estado para cuando el usuario pida enviarlo
        tipo = nuevos_estados.get("tipo_reporte_activo")

        if "[IMG_B64:" in respuesta:
            try:
                img_base64 = respuesta.split("[IMG_B64:")[1].rstrip("]").strip()
                if tipo:
                    nuevos_estados["grafico_pendiente_base64"] = img_base64
                    nuevos_estados["tipo_grafico_pendiente"] = tipo
            except IndexError:
                pass
        else:
            es_reporte_real = (
                "Reporte Financiero Ejecutivo" in respuesta
                or "Reporte de Clientes en Riesgo" in respuesta
                or "Clientes en Riesgo" in respuesta
            )

            if tipo and es_reporte_real:
                nuevos_estados[f"ultimo_reporte_{tipo}"] = respuesta

        return respuesta, cat, nuevos_estados

    return (
        "No pude clasificar tu solicitud, por favor intenta ser más específico.",
        categoria,
        nuevos_estados,
    )


async def _flujo_envio(p_lower: str, nuevos_estados: dict) -> tuple:
    """Orquestador de envíos puramente funcional."""
    tipo = nuevos_estados.get("tipo_reporte_activo")
    reporte = nuevos_estados.get(f"ultimo_reporte_{tipo}") if tipo else None

    if not reporte:
        return (
            "⚠️ No hay un reporte activo. Genera uno primero.",
            "ACCION_ENVIO",
            nuevos_estados,
        )

    canal = "gmail" if "gmail" in p_lower else "telegram"

    if canal == "telegram":
        # Se asume que ahora devuelve la respuesta y la categoría de forma directa
        respuesta, nuevos_estados = await procesar_confirmacion_envio(
            canal="telegram", contexto_sesion=nuevos_estados
        )
        return respuesta, "ACCION_ENVIO", nuevos_estados

    # Activar estado de espera para Gmail en el diccionario que regresaremos a la API
    nuevos_estados["proceso_envio"] = {"activo": True, "canal": "gmail"}
    return (
        "Perfecto. Por favor, indícame tu correo de Gmail (ejemplo: nombre@gmail.com).",
        "ACCION_ENVIO",
        nuevos_estados,
    )


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
        if _es_error_limite_api(e):
            return MENSAJE_LIMITE_API
        return f"Error al consultar políticas: {str(e)}"


def _procesar_analisis(pregunta, dfs, vector_store, categoria, historial):
    clave = "churn" if categoria == "CHURN" else "superstore"
    if clave not in dfs:
        return f"Dataset '{clave}' no disponible.", categoria

    try:
        agente = construir_agente(dfs[clave], vector_store, clave, historial)
        historial_formateado = truncar_historial(historial)
        res = agente.invoke({"input": pregunta, "chat_history": historial_formateado})
        return res.get("output", "No obtuve respuesta."), categoria
    except Exception as e:
        if _es_error_limite_api(e):
            return MENSAJE_LMITE_API, categoria        
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
