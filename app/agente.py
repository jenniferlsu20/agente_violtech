from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
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


def obtener_memoria(historial: list) -> ConversationBufferWindowMemory:
    memoria = ConversationBufferWindowMemory(
        k=VENTANA_MEMORIA,
        memory_key="chat_history",
        return_messages=True,
    )
    mensajes = historial[-(VENTANA_MEMORIA * 2) :]
    for msg in mensajes:
        if msg["rol"] == "user":
            memoria.chat_memory.add_user_message(msg["contenido"])
        elif msg["rol"] == "assistant":
            memoria.chat_memory.add_ai_message(msg["contenido"])
    return memoria


def construir_agente(df, vector_store, nombre_df: str, historial: list):
    """Construye el ejecutor del agente."""
    herramientas = crear_herramientas(df, vector_store, nombre_df, llm)
    memoria = obtener_memoria(historial)

    agente = create_react_agent(llm, herramientas, PROMPT_VIOLET)

    return AgentExecutor(
        agent=agente,
        tools=herramientas,
        memory=memoria,
        verbose=True,
        max_iterations=5,
        max_execution_time=60,
        handle_parsing_errors=True,
        early_stopping_method="force",
    )


def procesar(pregunta: str, categoria: str, dfs: dict, vector_store, historial: list):
    """Orquestador de lógica: Ejecuta el agente si la categoría es CHURN/FINANZAS/POLITICAS."""

    # --- Lógica de Políticas (RAG) ---
    if categoria == "POLITICAS":
        docs = vector_store.invoke(pregunta)
        contexto = "\n\n".join(d.page_content for d in docs)
        plantilla = PromptTemplate.from_template(
            "Eres Violet, analista de ViolTech. Contexto:\n{contexto}\n\nPregunta: {pregunta}\nRespuesta:"
        )
        respuesta = (plantilla | llm | StrOutputParser()).invoke(
            {"contexto": contexto, "pregunta": pregunta}
        )
        return respuesta, categoria

    # --- Lógica de Agentes (Churn / Finanzas) ---
    clave = "churn" if categoria == "CHURN" else "superstore"
    nombre = (
        "Churn — TelcoVenezuela"
        if categoria == "CHURN"
        else "SmartFinance — Superstore"
    )

    if clave not in dfs:
        return f"Error: Dataset {clave} no cargado.", categoria

    try:
        agente = construir_agente(dfs[clave], vector_store, nombre, historial)
        resultado = agente.invoke({"input": pregunta, "nombre_df": nombre})
        return resultado.get("output", "Sin respuesta."), categoria
    except Exception as e:
        return f"Error técnico: {str(e)}", categoria
