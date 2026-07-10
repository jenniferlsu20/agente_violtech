from langchain_classic.agents.react.agent import create_react_agent
from langchain_classic.agents.agent import AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage
from langchain_cohere import ChatCohere
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.output_parsers import StrOutputParser
from app.config import COHERE_API_KEY, MODELO_LLM, VENTANA_MEMORIA
from app.herramientas import crear_herramientas
from app.prompts import PROMPT_VIOLET
from app.router import analizar_intencion_envio

llm = ChatCohere(
    cohere_api_key=COHERE_API_KEY,
    model=MODELO_LLM,
    temperature=0.0,
    max_tokens=800,
)


def obtener_memoria(historial: list) -> list:
    """Convierte el historial de chat en mensajes compatibles con LangChain.

    El historial se recorta usando la ventana de memoria configurada para
    conservar el mismo número de preguntas y respuestas.
    """
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
    tools = crear_herramientas(df, vector_store, nombre_df, llm)
    tool_names_str = ", ".join([tool.name for tool in tools])

    # 1. ChatPromptTemplate con la estructura EXACTA que exige un ReAct Agent
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PROMPT_VIOLET),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}\n\n{agent_scratchpad}"),
        ]
    )

    # 2. Rellenamos SOLO las variables que nosotros creamos
    prompt = prompt.partial(nombre_df=nombre_df, tool_names=tool_names_str)

    # 3. Creamos el agente
    agente = create_react_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agente,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )


async def procesar(
    pregunta: str, categoria: str, dfs: dict, vector_store, historial: list
):
    """Orquestador de lógica: Ejecuta el agente si la categoría es CHURN/FINANZAS/POLITICAS/ENVIO."""

    # 1. Validación de alcance
    if categoria not in ["CHURN", "FINANZAS", "POLITICAS", "ACCION_ENVIO"]:
        return (
            "Lo siento, la consulta se encuentra fuera de mi alcance operativo actual.",
            "FUERA_SCOPE",
        )

    # 2. LÓGICA DE POLÍTICAS (Retorno temprano)
    if categoria == "POLITICAS":
        try:
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
            respuesta = (plantilla | llm | StrOutputParser()).invoke(
                {"contexto": contexto, "pregunta": pregunta}
            )
            return respuesta, categoria
        except Exception as e:
            return f"Error técnico en el módulo de Políticas: {str(e)}", categoria

    # 3. PREPARACIÓN DE DATOS (Variables comunes)
    clave = "churn" if categoria in ["CHURN", "ACCION_ENVIO"] else "superstore"
    nombre = (
        "Agente de Envíos"
        if categoria == "ACCION_ENVIO"
        else ("Churn — TelcoVenezuela" if categoria == "CHURN" else "SmartFinance — Superstore")
    )

    if clave not in dfs:
        return f"Error: El dataset operativo '{clave}' no se encuentra cargado.", categoria

    # 4. PROCESAMIENTO DEL HISTORIAL
    historial_formateado = []
    for m in historial:
        # Filtro de seguridad: mantenemos el reporte si estamos en modo envío
        if "Reporte" in m.get("content", "") and categoria != "ACCION_ENVIO":
            continue
        
        if m["role"] == "user":
            historial_formateado.append(HumanMessage(content=m["content"]))
        else:
            historial_formateado.append(AIMessage(content=m["content"]))

    # 5. LÓGICA DE ACCIÓN DE ENVÍO
    pregunta_final = pregunta
    if categoria == "ACCION_ENVIO":
        intencion = analizar_intencion_envio(pregunta)
        if intencion["accion"] == "CANCELAR":
            return "Entendido, no realizaré el envío del reporte.", categoria
        if intencion["accion"] == "CONFIRMAR" and intencion["canal"]:
            pregunta_final = f"{pregunta} [INSTRUCCIÓN DEL SISTEMA: Usuario confirmó envío por {intencion['canal']}. Usa la herramienta 'enviar_reporte_directo'.]"

    # 6. INVOCACIÓN DEL AGENTE (CHURN / FINANZAS / ACCION_ENVIO)
    try:
        agente = construir_agente(dfs[clave], vector_store, nombre, historial)
        resultado = agente.invoke(
            {"input": pregunta_final, "chat_history": historial_formateado}
        )
        # Retornamos el resultado y la categoría consistentemente
        return resultado["output"], categoria

    except Exception as e:
        return f"Error técnico en el Agente: {str(e)}", categoria