from langgraph.prebuilt.chat_agent_executor import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
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

    instrucciones_sistema = str(PROMPT_VIOLET)

    agente = create_react_agent(
        model=llm, tools=herramientas, state_modifier=instrucciones_sistema
    )

    return agente


async def procesar(
    pregunta: str, categoria: str, dfs: dict, vector_store, historial: list
):
    """Orquestador de lógica: Ejecuta el agente si la categoría es CHURN/FINANZAS/POLITICAS."""

    # --- Lógica de Políticas (RAG) ---
    if categoria == "POLITICAS":
        try:
            # Si vector_store es un objeto FAISS/Chroma nativo, usamos as_retriever() para invocarlo correctamente
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

            # Ejecución de la cadena LCEL (LangChain Expression Language)
            respuesta = (plantilla | llm | StrOutputParser()).invoke(
                {"contexto": contexto, "pregunta": pregunta}
            )

            # 🔥 Retornamos de inmediato la respuesta y la categoría.
            # Esto evita llamadas recursivas erróneas y detiene la ejecución aquí.
            return respuesta, categoria

        except Exception as e:
            return f"Error técnico en el módulo de Políticas (RAG): {str(e)}", categoria

    # --- Lógica de Agentes (Churn / Finanzas) ---
    if categoria not in ["CHURN", "FINANZAS"]:
        return (
            "Lo siento, la consulta se encuentra fuera de mi alcance operativo actual.",
            "FUERA_SCOPE",
        )

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

    try:
        agente = construir_agente(dfs[clave], vector_store, nombre, historial)

        # Mapeo del historial al estándar de mensajes LangGraph
        mensajes_input = []
        for msg in historial:
            if msg.get("role") == "user":
                mensajes_input.append(HumanMessage(content=msg["content"]))
            elif msg.get("role") == "assistant":
                mensajes_input.append(AIMessage(content=msg["content"]))

        # Agregamos la consulta actual al final de la lista
        mensajes_input.append(HumanMessage(content=pregunta))

        # Invocamos al grafo pasándole el estado inicial de mensajes.
        resultado = agente.invoke(
            {"messages": mensajes_input}, config={"recursion_limit": 15}
        )  # 'recursion_limit' actúa como salvaguarda frente a bucles infinitos (antiguo max_iterations)

        # LangGraph devuelve la lista completa de mensajes modificados por el flujo.
        respuesta_final = resultado["messages"][-1].content

        return respuesta_final, categoria

    except Exception as e:
        return f"Error técnico: {str(e)}", categoria
