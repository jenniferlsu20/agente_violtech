from langchain_classic.agents.react.agent import create_react_agent
from langchain_classic.agents.agent import AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage
from langchain_cohere import ChatCohere
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.config import COHERE_API_KEY, MODELO_LLM, VENTANA_MEMORIA
from app.herramientas import crear_herramientas
from app.prompts import PROMPT_VIOLET

llm = ChatCohere(
    cohere_api_key=COHERE_API_KEY,
    model=MODELO_LLM,
    temperature=0.0,
    max_tokens=800,
)


def obtener_memoria(historial: list) -> list:
    mensajes_convertidos = []

    # Conservamos exactamente tu lógica de ventana protectora (k * 2 mensajes)
    # Ej: Si VENTANA_MEMORIA = 5, tomará los últimos 10 mensajes (5 preguntas y 5 respuestas)
    mensajes_filtrados = (
        historial[-(VENTANA_MEMORIA * 2) :] if VENTANA_MEMORIA > 0 else historial
    )

    for msg in mensajes_filtrados:
        # Soportamos tanto tus llaves nativas 'rol'/'contenido' como las de respaldo
        rol = msg.get("rol") or msg.get("role")
        contenido = msg.get("contenido") or msg.get("content")

        if rol == "user":
            mensajes_convertidos.append(HumanMessage(content=contenido))
        elif rol in ["assistant", "ai"]:
            mensajes_convertidos.append(AIMessage(content=contenido))

    return mensajes_convertidos


def construir_agente(df, vector_store, nombre_df: str, historial: list):
    """Construye el ejecutor del agente."""
    herramientas = crear_herramientas(df, vector_store, nombre_df, llm)

    # Construimos el agente nativo de LangChain
    agente = create_react_agent(llm, herramientas, PROMPT_VIOLET)

    # Retornamos el ejecutor real que procesará la consulta (Thought/Action/Observation)
    return AgentExecutor(
        agent=agente,
        tools=herramientas,
        verbose=True,
        handle_parsing_errors=True,  # Evita que la API caiga si la IA comete un error de formato
    )


async def procesar(
    pregunta: str, categoria: str, dfs: dict, vector_store, historial: list
):
    """Orquestador de lógica: Ejecuta el agente si la categoría es CHURN/FINANZAS/POLITICAS."""

    # ==========================================
    # 1. LÓGICA DE POLÍTICAS (RAG DIRECTO)
    # ==========================================
    if categoria == "POLITICAS":
        try:
            # Usamos el retriever para buscar en los documentos de ViolTech
            retriever = (
                vector_store.as_retriever()
                if hasattr(vector_store, "as_retriever")
                else vector_store
            )
            docs = retriever.invoke(pregunta)

            contexto = "\n\n".join(d.page_content for d in docs)
            plantilla = PromptTemplate.from_template(
                "Eres Violet, analista de ViolTech. Contexto:\n{contexto}\n\nPregunta: {pregunta}\nRespuesta:"
            )

            # Ejecución de la cadena LCEL para responder en base al documento
            respuesta = (plantilla | llm | StrOutputParser()).invoke(
                {"contexto": contexto, "pregunta": pregunta}
            )

            return respuesta, categoria

        except Exception as e:
            return f"Error técnico en el módulo de Políticas (RAG): {str(e)}", categoria

    # ==========================================
    # 2. VALIDACIÓN DE ALCANCE
    # ==========================================
    if categoria not in ["CHURN", "FINANZAS"]:
        return (
            "Lo siento, la consulta se encuentra fuera de mi alcance operativo actual.",
            "FUERA_SCOPE",
        )

    # Definimos qué dataset y prompt usar antes de entrar al try
    clave = "churn" if categoria == "CHURN" else "superstore"
    nombre = (
        "Churn — TelcoVenezuela"
        if categoria == "CHURN"
        else "SmartFinance — Superstore"
    )

    if clave not in dfs:
        return (
            f"Error: El dataset operativo '{clave}' no se encuentra cargado en el servidor.",
            categoria,
        )
        
    historial_filtrado = [
        msg for msg in historial 
        if "Reporte" not in msg.get("content", "")  # Excluimos reportes automáticos del historial de chat
        ]

    # ==========================================
    # 3. LÓGICA DE AGENTES (CHURN / FINANZAS)
    # ==========================================
    try:
        # Construimos el agente nativo de LangChain
        agente = construir_agente(dfs[clave], vector_store, nombre, historial_filtrado)

        # Mapeo del historial al estándar de mensajes
        mensajes_chat = []
        for msg in historial_filtrado:
            if msg.get("role") == "user":
                mensajes_chat.append(HumanMessage(content=msg["content"]))
            elif msg.get("role") == "assistant":
                mensajes_chat.append(AIMessage(content=msg["content"]))

        # Invocación nativa para LangChain AgentExecutor (Variables correctas)
        resultado = agente.invoke({
            "input": pregunta,
            "chat_history": mensajes_chat,
            "nombre_df": nombre
        })

        # Extraemos la salida final
        respuesta_final = resultado["output"]

        return respuesta_final, categoria

    except Exception as e:
        return f"Error técnico en el Agente: {str(e)}", categoria
