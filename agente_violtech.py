"""
agente_violtech.py
==================
Violet — Agente IA de Inteligencia de Negocios
ViolTech: Tu Agente Aliado

Challenge Final — Alura ONE Fase 2

Personalidad de Violet:
    Cálida y profesional — como una colega de trabajo
    con las capacidades técnicas de un analista senior.
    Responde siempre en español con precisión y cercanía.

Arquitectura:
    Router LLM    → clasifica la intención antes de actuar
    Agente Churn  → datos TelcoVenezuela (customerID, tenure, churn_prob...)
    Agente Finance → datos Superstore (ventas, márgenes, descuentos...)
    RAG directo   → documentos de política ViolTech
    Memoria       → ventana de 5 interacciones + persistencia en disco

Tecnologías:
    LangChain + Cohere command-r-plus-08-2024 + embed-multilingual-v3.0
    FAISS (Vector Store persistente) + PyPDF + Pandas + Streamlit
"""

import os
import json
import base64
import io
import re
import pandas as pd
import matplotlib.pyplot as plt
import requests
import smtplib
import pickle
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import seaborn as sns
import streamlit as st
from fpdf import FPDF
from pathlib import Path
from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_core.embeddings import Embeddings
import cohere as cohere_sdk
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_classic.agents.react.agent import create_react_agent
from langchain_classic.agents.agent import AgentExecutor
from langchain_classic.memory.buffer_window import ConversationBufferWindowMemory
from langchain_experimental.tools import PythonAstREPLTool
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import Image

load_dotenv()

# ── Paleta ViolTech ───────────────────────────────────────────────────────────
VIOLETA = "#6B3FA0"
VIOLETA_CLARO = "#9B72CF"
VIOLETA_SUAVE = "#F3EEFF"
DORADO = "#C9A84C"

# ── Configuración ─────────────────────────────────────────────────────────────
SMTP_USER = os.getenv("SMTP_USER")
SMTP_APP_PASSWORD = os.getenv("SMTP_APP_PASSWORD")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY no configurada en .env")

RUTA_DATOS = Path("datos")
RUTA_DOCS = RUTA_DATOS / "docs"
RUTA_CHURN = RUTA_DATOS / "clientes_scored.csv"
RUTA_STORE = RUTA_DATOS / "Sample-Superstore_cleaned.csv"
RUTA_FAISS = RUTA_DATOS / "faiss_index"
RUTA_HISTORIAL = RUTA_DATOS / "historial_violet.json"
VENTANA_MEMORIA = 5

# ── Modelos ───────────────────────────────────────────────────────────────────
llm = ChatCohere(
    cohere_api_key=COHERE_API_KEY,
    model="command-r-plus-08-2024",
    temperature=0.1,
    max_tokens=800,
)
llm_router = ChatCohere(
    cohere_api_key=COHERE_API_KEY,
    model="command-r-08-2024",
    temperature=0.0,
    max_tokens=20,
)


class VioletEmbeddings(Embeddings):
    """
    Wrapper directo sobre el cliente Cohere para embeddings.
    Usa input_type correcto — evita el IndexError de langchain-cohere.
    """

    def __init__(self, api_key: str, model: str = "embed-multilingual-v3.0"):
        self.client = cohere_sdk.Client(api_key)
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        texts = [t for t in texts if t.strip()]
        if not texts:
            return []
        resp = self.client.embed(
            texts=texts,
            model=self.model,
            input_type="search_document",
        )
        return resp.embeddings

    def embed_query(self, text: str) -> list[float]:
        resp = self.client.embed(
            texts=[text],
            model=self.model,
            input_type="search_query",
        )
        return resp.embeddings[0]


embeddings = VioletEmbeddings(api_key=COHERE_API_KEY)

SALUDO_VIOLET = (
    "¡Hola! Soy **Violet**, tu analista de datos de ViolTech.\n\n"
    "Estoy aquí para transformar tus datos en conocimiento estratégico de forma sencilla, "
    "traduciendo información compleja en respuestas claras y directas, sin que tengas "
    "que preocuparte por el código o la gestión de archivos. \n\n"
)


# ─────────────────────────────────────────────────────────────────────────────
# VECTOR STORE
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Violet está leyendo los documentos de política...")
def cargar_vector_store():
    ruta_fragmentos = Path(str(RUTA_FAISS)) / "fragmentos.pkl"

    if Path(RUTA_FAISS).exists() and ruta_fragmentos.exists():
        # Carga del indice semántico de FAISS
        vs = FAISS.load_local(
            str(RUTA_FAISS),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        # Carga de fragmentos de texto guardados para recuperación
        with open(ruta_fragmentos, "rb") as f:
            fragmentos = pickle.load(f)
    else:
        if not RUTA_DOCS.exists():
            raise FileNotFoundError(f"Carpeta no encontrada: {RUTA_DOCS}")

        loader = DirectoryLoader(
            str(RUTA_DOCS),
            glob="**/*.pdf",
            loader_cls=PyMuPDFLoader,
        )
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
        fragmentos = splitter.split_documents(docs)
        fragmentos = [f for f in fragmentos if f.page_content.strip()]

        if not fragmentos:
            raise ValueError("Los PDFs no contienen texto extraíble.")

        # Guardar FAISS de forma local
        vs = FAISS.from_documents(fragmentos, embeddings)
        vs.save_local(str(RUTA_FAISS))

        # Guardar los fragmentos en un archivo pickle para BM25
        Path(RUTA_FAISS).mkdir(parents=True, exist_ok=True)
        with open(ruta_fragmentos, "wb") as f:
            pickle.dump(fragmentos, f)

    # Construcción de los recuperadores individuales
    faiss_retriever = vs.as_retriever(search_kwargs={"k": 3})

    # Recuperador de palabras clave con BM25
    bm25_retriever = BM25Retriever.from_documents(fragmentos)
    bm25_retriever.k = 3  # Recupera 3 fragmentos por coincidencia

    # Emsanblado híbrido
    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever], weights=[0.5, 0.5]
    )

    return ensemble_retriever


# ─────────────────────────────────────────────────────────────────────────────
# HISTORIAL PERSISTENTE
# ─────────────────────────────────────────────────────────────────────────────


def cargar_historial() -> list:
    if RUTA_HISTORIAL.exists():
        try:
            with open(RUTA_HISTORIAL, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def guardar_historial(mensajes: list):
    RUTA_DATOS.mkdir(parents=True, exist_ok=True)
    with open(RUTA_HISTORIAL, "w", encoding="utf-8") as f:
        json.dump(mensajes[-30:], f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_ROUTER = PromptTemplate.from_template("""
Clasifica esta pregunta en UNA categoría. Responde SOLO la categoría, sin explicación.

Categorías:
- CHURN: clientes, riesgo churn, churn_prob, risk_level, tenure, telecomunicaciones,
  fibra, TechSupport, segmento critico, MonthlyCharges, TelcoVenezuela
- FINANZAS: ventas, ganancias, márgenes, descuentos, Technology, Furniture,
  Office Supplies, Consumer, Corporate, Home Office, Superstore, pérdidas, Tables
- POLITICAS: definiciones, estrategias, reglas, políticas, glosario, manual,
  qué es, cómo funciona Violet, arquitectura del agente
- FUERA_SCOPE: preguntas sin relación con los datos de ViolTech

Pregunta: {pregunta}
Categoría:""")


def clasificar(pregunta: str) -> str:
    """
    Router de dos capas - sin duplicados, sin colisiones:
    Capa 1: frases compuestas exactas (mayor precision, cero tokens)
    Capa 2: palabras simples con orden de prioridad correcto
    Capa 3: LLM router solo para preguntas ambiguas
    """
    p = pregunta.lower().strip()

    # ── Frases compuestas primero (mayor precisión) ───────────────────────
    # CHURN — frases específicas
    frases_churn_exactas = [
        "segmento critico",
        "segmento crítico",
        "riesgo alto",
        "riesgo medio",
        "riesgo bajo",
        "en riesgo",
        "probabilidad de churn",
        "clientes en riesgo",
        "clientes de riesgo",
        "contrato mensual",
        "contrato anual",
    ]
    for frase in frases_churn_exactas:
        if frase in p:
            return "CHURN"

    # FINANZAS — frases específicas de retail
    frases_finanzas_exactas = [
        "home office",
        "office supplies",
        "tabla de ventas",
        "por categoría",
        "por categoria",
        "sub-categoría",
        "sub-categoria",
        "regla del 20",
        "umbral del 20",
    ]
    for frase in frases_finanzas_exactas:
        if frase in p:
            return "FINANZAS"

    # ── Palabras clave ──────────────────────────────────────────
    if any(
        accion in p for accion in ["gmail", "telegram", "correo", "email", "enviar"]
    ):
        return "ACCION_ENVIO"

    frases_exactas = [
        "cómo funciona violet",
        "como funciona violet",
        "qué es violet",
        "que es violet",
        "qué puede hacer",
        "que puede hacer",
        "regla del 20",
        "regla del 20%",
    ]
    for frase in frases_exactas:
        if frase in p:
            return "POLITICAS"

    # CHURN - palabras inequívocas primero
    for w in [
        "churn",
        "churn_prob",
        "risk_level",
        "customerid",
        "id",
        "tenure",
        "techsupport",
        "onlinesecurity",
        "fibra",
        "cancelar",
        "cancelacion",
        "cancelación",
        "fideliz",
        "scored",
        "telco",
        "telecom",
        "telefon",
    ]:
        if w in p:
            return "CHURN"

    # FINANZAS - palabras inequívocas
    for w in [
        "venta",
        "ventas",
        "ganancia",
        "ganancias",
        "margen",
        "descuento",
        "descuentos",
        "furniture",
        "technology",
        "superstore",
        "retail",
        "perdida",
        "pérdida",
        "bookcases",
        "tables",
        "copiers",
        "chairs",
        "financ",
        "rentabilidad",
        "transaccion",
        "transacción",
        "transacciones",
    ]:
        if w in p:
            return "FINANZAS"

    # CHURN - palabras ambiguas
    for w in ["riesgo", "cliente", "clientes", "retener", "retención", "retencion"]:
        if w in p:
            return "CHURN"

    # POLÍTICA
    for w in [
        "política",
        "politica",
        "manual",
        "glosario",
        "definición",
        "definicion",
        "arquitectura",
        "violet",
        "violtech",
    ]:
        if w in p:
            return "POLITICAS"

    palabras_grafico = [
        "gráfico",
        "grafico",
        "visualiza",
        "mapa",
        "tendencia",
        "sugerido",
        "sugerencia",
        "plotea",
        "plot",
    ]

    if any(w in p for w in palabras_grafico):
        categoria_previa = getattr(st.session_state, "ultima_categoria", "FINANZAS")
        categoria_destino = categoria_previa if categoria_previa else "FINANZAS"
        print(
            f"[ROUTER] VISUALIZACIÓN detectada. Heredando contexto: {categoria_destino}"
        )

        return categoria_destino

    # ── LLM router — solo para preguntas ambiguas ─────────────────────────
    try:
        resultado = (
            (PROMPT_ROUTER | llm_router | StrOutputParser())
            .invoke({"pregunta": pregunta})
            .strip()
            .upper()
        )
        for cat in ["CHURN", "FINANZAS", "POLITICAS", "FUERA_SCOPE"]:
            if cat in resultado:
                return cat
    except Exception as ex:
        print(f"[ROUTER LLM] Error: {ex}")

    return "FUERA_SCOPE"


# ─────────────────────────────────────────────────────────────────────────────
# DATAFRAMES
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_data(show_spinner="Violet está cargando los datos...")
def cargar_dataframes():
    dfs = {}
    if RUTA_CHURN.exists():
        dfs["churn"] = pd.read_csv(RUTA_CHURN)
    if RUTA_STORE.exists():
        dfs["superstore"] = pd.read_csv(RUTA_STORE)
    return dfs


# ─────────────────────────────────────────────────────────────────────────────
# PDF REPORTES
# ─────────────────────────────────────────────────────────────────────────────


def _manejar_confirmacion_envio(canal: str, destino: str = None) -> str:
    """
    Orquesta la conversión del reporte activo a PDF y ejecuta el envío real.
    - canal: 'gmail' o 'telegram'
    - destino: Correo electrónico (solo requerido para Gmail)
    """
    tipo_reporte = st.session_state.get("tipo_reporte_activo")

    # 1. Recuperar el texto correcto según el tipo de reporte
    if tipo_reporte == "churn":
        texto_reporte = st.session_state.get("ultimo_reporte_churn")
    elif tipo_reporte == "financiero":
        texto_reporte = st.session_state.get("ultimo_reporte_financiero")
    else:
        # Fallback por si se llama sin un reporte específico
        texto_reporte = st.session_state.get("proceso_envio", {}).get("reporte")

    if not texto_reporte:
        return "❌ Error: No encontré el contenido del reporte en memoria. Por favor, genéralo de nuevo."

    grafico_b64 = None
    if st.session_state.get("tipo_grafico_pendiente") == tipo_reporte:
        grafico_b64 = st.session_state.get("grafico_pendiente_base64")

    # 2. Generar el documento PDF
    try:
        buffer_pdf = generar_pdf_reporte(
            texto_reporte, tipo_reporte, grafico_base64=grafico_b64
        )
    except Exception as e:
        return f"❌ Error al estructurar el PDF: {str(e)}"

    # 3. Ejecutar los motores de envío
    try:
        if canal == "gmail":
            if not destino:
                return (
                    "❌ Error: Se requiere un correo de destino para enviar por Gmail."
                )
            resultado = enviar_por_gmail(buffer_pdf, destino)

        elif canal == "telegram":
            resultado = enviar_por_telegram(buffer_pdf)

        else:
            return "❌ Canal de envío no reconocido."

        # 4. Evaluación del resultado y limpieza
        if "Error" in resultado:
            return f"❌ Hubo un inconveniente con los servidores de envío: {resultado}"

        st.session_state.pop("grafico_pendiente_base64", None)
        st.session_state.pop("tipo_grafico_pendiente", None)

        # Limpiamos los estados relacionados al envío para evitar bucles
        st.session_state.proceso_envio = {
            "activo": False,
            "canal": None,
            "paso": None,
            "reporte": None,
            "grafico": None,
        }
        st.session_state["fase_reporte"] = None

        return f"✅ ¡El reporte en PDF ha sido generado y enviado exitosamente vía {canal.capitalize()}!"

    except Exception as e:
        return f"❌ Error crítico durante el proceso de envío: {str(e)}"


def generar_pdf_reporte(
    texto_reporte: str, tipo_reporte: str, grafico_base64: str = None
) -> io.BytesIO:
    """
    Toma el texto estructurado del reporte y genera un archivo PDF físico.
    Sanea emojis y aplica la paleta corporativa de ViolTech.
    Retorna la ruta del archivo generado.
    """
    buffer = io.BytesIO()

    # 1. Configuración del documento (Márgenes ejecutivos)
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=50,
        leftMargin=50,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    # --- ESTILOS PERSONALIZADOS VIOLTECH ---
    # Título principal (Equivalente a ##)
    estilo_titulo = ParagraphStyle(
        "TituloPrincipal",
        parent=styles["Heading1"],
        fontSize=16,
        textColor=colors.HexColor("#4B0082"),  # Morado ViolTech
        spaceAfter=12,
        fontName="Helvetica-Bold",
    )

    # Subtítulo (Equivalente a ###)
    estilo_subtitulo = ParagraphStyle(
        "SubTitulo",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor("#333333"),  # Gris oscuro
        spaceBefore=10,
        spaceAfter=6,
        fontName="Helvetica-Bold",
    )

    # Texto normal
    estilo_normal = ParagraphStyle(
        "TextoNormal",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.black,
        spaceAfter=6,
        leading=14,  # Interlineado más limpio
        fontName="Helvetica",
    )

    # Texto para listas con sangría
    estilo_lista = ParagraphStyle(
        "TextoLista",
        parent=estilo_normal,
        leftIndent=15,
    )

    # --- TRADUCCIÓN DE EMOJIS A TEXTO PROFESIONAL ---
    # Reemplazamos los iconos por etiquetas de color para no romper la fuente del PDF
    reemplazos_emoji = {
        "📋": "",
        "🎯": "",
        "💡": "",
        "📊": "",
        "📈": "",
        "🚩": "",
        "💰": "",
        "🔴": '<font color="red"><b>[CRÍTICO]</b></font>',
        "🟠": '<font color="orange"><b>[ALTA]</b></font>',
        "🟡": '<font color="#b8860b"><b>[MEDIA]</b></font>',
        "⚠️": '<font color="red"><b>(!)</b></font>',
    }

    texto_limpio = texto_reporte
    for emoji_char, reemplazo in reemplazos_emoji.items():
        texto_limpio = texto_limpio.replace(emoji_char, reemplazo)

    # Filtramos la última línea interactiva (la pregunta del bot) para que no salga impresa
    texto_limpio = re.sub(r"\*¿Deseas acompañar este reporte.*\n?", "", texto_limpio)
    texto_limpio = re.sub(r"\*¿Deseas profundizar.*\n?", "", texto_limpio)
    texto_limpio = re.sub(r"\*\*Siguiente paso:\*\*", "", texto_limpio)

    # --- PROCESAMIENTO ESTRUCTURAL (MARKDOWN -> PDF) ---
    story = []
    lineas = texto_limpio.split("\n")

    for linea in lineas:
        linea_original = linea
        linea = linea.strip()

        if not linea:
            continue

        # Convertir Markdown a etiquetas HTML soportadas por ReportLab
        # 1. Negritas (**texto**) -> <b>texto</b>
        linea = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", linea)
        # 2. Cursivas (*texto*) -> <i>texto</i>
        linea = re.sub(r"(?<!<)\*(.*?)\*(?!>)", r"<i>\1</i>", linea)

        # Clasificación de la línea
        if linea.startswith("## "):
            story.append(Paragraph(linea[3:].strip(), estilo_titulo))

        elif linea.startswith("### "):
            story.append(Paragraph(linea[4:].strip(), estilo_subtitulo))

        elif linea.startswith("---"):
            # Dibuja una línea gris divisoria elegante
            story.append(
                HRFlowable(
                    width="100%",
                    thickness=1,
                    color=colors.HexColor("#CCCCCC"),
                    spaceBefore=15,
                    spaceAfter=15,
                )
            )

        elif linea.startswith("- "):
            # Formatear viñetas
            texto = "• " + linea[2:].strip()
            story.append(Paragraph(texto, estilo_lista))

        else:
            # Si es un número de cliente (ej: "1. 5760-IFJOZ"), añadimos un pequeño espacio arriba para separarlos
            if re.match(r"^<b>\d+\.", linea) or re.match(r"^\d+\.", linea):
                story.append(Spacer(1, 6))

            # Si la línea original tenía sangría (espacios al inicio), la mostramos con estilo_lista
            if linea_original.startswith("   "):
                story.append(Paragraph(linea, estilo_lista))
            else:
                story.append(Paragraph(linea, estilo_normal))

    if grafico_base64:
        img_data = base64.b64decode(grafico_base64)
        img_buffer = io.BytesIO(img_data)

        story.append(Spacer(1, 20))
        story.append(Paragraph("<b>Análisis Visual:</b>", estilo_subtitulo))
        story.append(Spacer(1, 19))

        img = Image(img_buffer, width=400, height=250)
        story.append(img)

    # Generar y retornar
    doc.build(story)
    buffer.seek(0)
    return buffer


def enviar_por_telegram(buffer_pdf: io.BytesIO) -> str:
    """Maneja la API de Telegram y la seguridad de forma invisible para el LLM."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id_admin = os.getenv("TELEGRAM_CHAT_ID")  # El ID seguro de tu .env

    if not token or not chat_id_admin:
        return "Error interno: Credenciales de Telegram no configuradas."

    url = f"https://api.telegram.org/bot{token}/sendDocument"

    # Aseguramos leer el buffer desde el inicio
    buffer_pdf.seek(0)

    # Preparamos el archivo para el requests
    archivos = {
        "document": ("Reporte_ViolTech.pdf", buffer_pdf.read(), "application/pdf")
    }

    # Preparamos el mensaje que acompaña el PDF
    datos = {
        "chat_id": chat_id_admin,
        "caption": "📦 *Reporte Solicitado vía Violet*\n\nAquí tienes el informe de la sesión actual.",
        "parse_mode": "Markdown",
    }

    try:
        respuesta = requests.post(url, data=datos, files=archivos, timeout=15)
        resultado = respuesta.json()

        if not resultado.get("ok"):
            return f"Error Telegram: {resultado.get('description')}"
        return "enviado por Telegram"
    except Exception as e:
        return f"Error Telegram: {str(e)}"


def enviar_por_gmail(buffer_pdf: io.BytesIO, destinatario: str):
    """Envía el PDF vía SMTP de Google."""
    # Credenciales desde variables de entorno
    EMAIL_USER = os.getenv("SMTP_USER")
    EMAIL_PASS = os.getenv("SMTP_APP_PASSWORD")

    if not EMAIL_USER or not EMAIL_PASS:
        return "Error: Credenciales de SMTP no configuradas en el entorno."

    # 1. Crear el contenedor principal
    msg = MIMEMultipart()
    msg["Subject"] = "📊 Reporte Automatizado de Inteligencia de Negocio — ViolTech"
    msg["From"] = EMAIL_USER
    msg["To"] = destinatario

    # 2. Adjuntar el cuerpo del texto
    cuerpo = (
        f"Hola,\n\n"
        f"Adjunto a este correo encontrarás el informe que solicitaste a través de Violet:\n\n"
        f"──────────────────────────────────────────────────\n"
        f"{buffer_pdf}\n"
        f"──────────────────────────────────────────────────\n\n"
        f"Este es un envío automático gestionado por el agente de IA de ViolTech.\n"
        f"Si tienes dudas adicionales, puedes iniciar una nueva consulta en la aplicación."
    )
    msg.attach(MIMEText(cuerpo, "plain", "utf-8"))

    # 3. Leer el buffer y adjuntar el PDF
    buffer_pdf.seek(0)
    pdf_adjunto = MIMEApplication(buffer_pdf.read(), _subtype="pdf")

    # 4. Configurar las cabeceras del archivo adjunto
    pdf_adjunto.add_header("Content-Disposition", "attachment", filename="reporte.pdf")
    msg.attach(pdf_adjunto)

    # 5. Envío vía SMTP
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_USER, EMAIL_PASS)
            # Usamos as_string() que es el método nativo para MIMEMultipart
            smtp.sendmail(EMAIL_USER, destinatario, msg.as_string())
        return "enviado por Gmail"
    except Exception as e:
        return f"Error Gmail: {str(e)}"


# ── Palabras clave para la confirmación de envío — cero tokens ──────────────
PALABRAS_TELEGRAM = ("telegram", "telegrama")
PALABRAS_GMAIL = ("gmail", "correo", "email", "e-mail", "mail")
PALABRAS_NEGATIVAS = (
    "no",
    "no gracias",
    "ahora no",
    "luego",
    "despues",
    "después",
    "cancelar",
    "no por ahora",
    "dejalo",
    "déjalo",
    "no hace falta",
)
PALABRAS_AFIRMATIVAS = (
    "si",
    "sí",
    "dale",
    "ok",
    "okay",
    "claro",
    "por favor",
    "hazlo",
    "envialo",
    "envíalo",
    "mandalo",
    "mándalo",
    "adelante",
)


def manejar_datos_contacto(pregunta: str):
    """
    Gestiona el envío del reporte, aplica los guardrails
    y limpia el estado para finalizar la sesión.
    """
    # 1. Validar y extraer el correo del texto ingresado
    match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", pregunta)
    if not match:
        return "⚠️ No detecté un formato de correo válido. Por favor, intenta de nuevo (ejemplo: usuario@gmail.com):"

    correo_destino = match.group(0)

    # 2. Llamar al orquestador maestro que creamos previamente
    # Él se encargará de buscar el reporte en memoria, crear el PDF y enviarlo
    resultado = _manejar_confirmacion_envio(canal="gmail", destino=correo_destino)

    return resultado


# ─────────────────────────────────────────────────────────────────────────────
# HERRAMIENTAS
# ─────────────────────────────────────────────────────────────────────────────


def crear_herramientas(df: pd.DataFrame, vector_store, nombre_df: str):

    @tool("Politicas ViolTech", return_direct=True)
    def buscar_politicas(pregunta: str) -> str:
        """
        Consulta los documentos de política interna de ViolTech.
        Usar para definiciones, estrategias, reglas de negocio,
        glosario técnico y manual del agente Violet.
        """
        try:
            docs = vector_store.similarity_search(pregunta, k=3)
            contexto = "\n\n".join(d.page_content for d in docs)
            plantilla = PromptTemplate.from_template(
                "Eres Violet, analista de ViolTech. Responde en español "
                "con tono cálido y profesional, como una colega experta.\n"
                "Basa tu respuesta ÚNICAMENTE en el contexto proporcionado.\n"
                "Si la información no está en el contexto, di:\n"
                "'No encontré esa información en los documentos de ViolTech.'\n\n"
                "Contexto:\n{contexto}\n\nPregunta: {pregunta}\n\nRespuesta:"
            )
            return (plantilla | llm | StrOutputParser()).invoke(
                {"contexto": contexto, "pregunta": pregunta}
            )
        except Exception as ex:
            return f"Error al consultar documentos: {str(ex)}"

    @tool("Informacion del Dataset", return_direct=True)
    def informacion_dataset(pregunta: str) -> str:
        """
        Información estructural del DataFrame activo: dimensiones,
        columnas, tipos de datos, nulos y duplicados.
        """
        try:
            plantilla = PromptTemplate.from_template(
                "Eres Violet, analista de ViolTech. Responde en español "
                "con tono cálido y profesional.\n"
                "Dataset activo: {nombre_df}\n"
                "Dimensiones: {shape}\n"
                "Columnas y tipos:\n{columnas}\n"
                "Nulos por columna:\n{nulos}\n"
                "Duplicados: {duplicados}\n\n"
                "Pregunta: {pregunta}\n\n"
                "Proporciona un resumen claro, organizado y útil."
            )
            return (plantilla | llm | StrOutputParser()).invoke(
                {
                    "nombre_df": nombre_df,
                    "shape": str(df.shape),
                    "columnas": df.dtypes.to_string(),
                    "nulos": df.isnull().sum().to_string(),
                    "duplicados": str(df.duplicated().sum()),
                    "pregunta": pregunta,
                }
            )
        except Exception as ex:
            return f"Error al analizar dataset: {str(ex)}"

    @tool("Resumen Estadistico", return_direct=True)
    def resumen_estadistico(pregunta: str) -> str:
        """
        Estadísticas descriptivas completas: media, desviación estándar,
        mínimo, máximo, percentiles. No usar para métricas puntuales.
        """
        try:
            resumen = df.describe(include="number").transpose().to_string()
            plantilla = PromptTemplate.from_template(
                "Eres Violet, analista de ViolTech. Responde en español "
                "con tono cálido y profesional.\n"
                "Pregunta: {pregunta}\n"
                "Estadísticas descriptivas:\n{resumen}\n\n"
                "Incluye: visión general, valores destacados y próximos pasos."
            )
            return (plantilla | llm | StrOutputParser()).invoke(
                {"pregunta": pregunta, "resumen": resumen}
            )
        except Exception as ex:
            return f"Error en estadísticas: {str(ex)}"

    @tool("Generar Grafico", return_direct=True)
    def generar_grafico(pregunta: str) -> str:
        """
        Genera visualizaciones automáticas del DataFrame.
        Usar con: 'crea un gráfico', 'plotea', 'visualiza',
        'muestra la distribución', 'grafica'.
        Si falla, reporta el error técnico.
        """
        # Definimos los contextos de columnas
        datasets = {
            "CHURN": [
                "customerID",
                "tenure",
                "MonthlyCharges",
                "TotalCharges",
                "Churn",
                "churn_prob",
                "risk_level",
                "tenure_grupo",
                "es_segmento_critico",
                "servicios_valor_agregado",
            ],
            "FINANZAS": [
                "Row ID",
                "Order ID",
                "Order Date",
                "Ship Date",
                "Ship Mode",
                "Customer ID",
                "Customer Name",
                "Segment",
                "Country",
                "City",
                "State",
                "Postal Code",
                "Region",
                "Product ID",
                "Category",
                "Sub-Category",
            ],
        }

        # Usamos el contexto real que nos pasa el Router
        contexto = "CHURN" if "Churn" in nombre_df else "FINANZAS"

        if contexto == "FINANZAS":
            # Bloqueamos el uso de términos prohibidos
            if any(w in pregunta.lower() for w in ["tenure", "contrato"]):
                return "Error: Los datos de Finanzas (Superstore) no tienen información de 'tenure' o 'contrato'. Por favor, solicita un gráfico basado en 'ganancia', 'ventas' o 'descuentos'."

        columnas_reales = datasets[contexto]

        columnas_str = ", ".join(columnas_reales)

        try:
            plantilla = PromptTemplate.from_template(
                "Eres experto en Data Viz con Python. Analiza la solicitud y los datos.\n"
                "Dataset: {contexto}\n"
                "Columnas disponibles: {columnas}\n\n"
                "Reglas ESTRICTAS e INQUEBRANTABLES:\n"
                "1. El DataFrame 'df' ya está cargado. NO intentes leer archivos.\n"
                "2. Usa SOLAMENTE nombres de columnas exactos de la lista. Si una columna no existe, NO la uses.\n"
                "3. LÓGICA DE CONTRATO: Solo si el dataset contiene la columna 'tenure' y el usuario pide 'tipo de contrato', "
                "ejecuta obligatoriamente: df['contrato'] = df['tenure'].apply(lambda x: 'Mensual' if x <= 10 else 'Anual')\n"
                "4. SEABORN: Si usas 'palette', es OBLIGATORIO usar 'hue' igual a la variable categórica y 'legend=False'.\n"
                "5. SALIDA: Genera ÚNICA y EXCLUSIVAMENTE código Python válido. Cero texto, cero comentarios, cero bloques de markdown.\n"
                "6. NO incluyas 'plt.show()'.\n"
                "7. PREVENCIÓN DE ERRORES: Si el dataset es 'FINANZAS', ignora cualquier solicitud relacionada con 'contrato' o 'tenure' "
                "y reporta un error técnico simple: 'Dataset no contiene datos de contrato'.\n"
                "Solicitud: {pregunta}"
            )
            codigo_bruto = (plantilla | llm | StrOutputParser()).invoke(
                {"pregunta": pregunta, "contexto": contexto, "columnas": columnas_str}
            )

            # Limpieza: buscamos código entre backticks si existe
            patron = r"```(?:python)?\s*(.*?)\s*```"
            match = re.search(patron, codigo_bruto, re.DOTALL | re.IGNORECASE)

            if match:
                codigo = match.group(1).strip()
            else:
                codigo = (
                    codigo_bruto.replace("`" * 3 + "python", "")
                    .replace("`" * 3, "")
                    .strip()
                )

            codigo = codigo.replace("plt.show()", "")

            try:
                exec(codigo, {"df": df, "plt": plt, "sns": sns, "pd": pd}, {})
                fig = plt.gcf()

                # Guardar en memoria RAM (Buffer)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)

                # Codificar la imagen a texto Base64
                img_base64 = base64.b64encode(buf.read()).decode("utf-8")

                # Preparamos el mensaje de cierre
                mensaje_cierre = (
                    "\n\n📊 **Gráfico generado exitosamente** basado en el análisis de rentabilidad. "
                    "¿Deseas exportar este análisis completo a PDF ahora para enviarlo por correo?"
                )

                tipo_reporte = "financiero" if contexto == "FINANZAS" else "churn"
                st.session_state["reporte_pendiente_envio"] = tipo_reporte

                clave_estado = f"ultimo_reporte_{tipo_reporte}"
                if not st.session_state.get(clave_estado):
                    st.session_state[clave_estado] = (
                        f"Análisis Visual - {contexto}\nEl usuario solicitó un análisis gráfico de los datos."
                    )

                st.session_state["grafico_pendiente_base64"] = img_base64
                st.session_state["tipo_grafico_pendiente"] = tipo_reporte

                return f"{mensaje_cierre} [IMG_B64:{img_base64}]"
            except Exception as ex_err:
                return f"Error ejecutando código del gráfico: {str(ex_err)}"
            finally:
                plt.close("all")
        except Exception as ex:
            return f"❌ Error en la ejecución del código: {str(e)}. Intenta con otras columnas."

    @tool("Consulta Rapida Churn", return_direct=True)
    def consulta_rapida_churn(pregunta: str) -> str:
        """
        ÚSALA PRIMERO para consultas sobre clientes, riesgo, churn, ingresos y tenure.
        Responde de inmediato sin ejecutar código.
        """
        try:
            p = pregunta.lower()
            lineas = []

            # Conteos por nivel de riesgo
            if "riesgo" in p or "risk" in p:
                dist = (
                    df["risk_level"].value_counts()
                    if "risk_level" in df.columns
                    else {}
                )
                for nivel in ["Alto", "Medio", "Bajo"]:
                    n = int(dist.get(nivel, 0))
                    pct = round(n / len(df) * 100, 1)
                    lineas.append(f"• Riesgo **{nivel}**: {n:,} clientes ({pct}%)")

            # Ingreso mensual en riesgo
            if "ingreso" in p or "monthly" in p or "cargo" in p:
                if "risk_level" in df.columns and "MonthlyCharges" in df.columns:
                    alto = df[df["risk_level"] == "Alto"]
                    ingreso = alto["MonthlyCharges"].sum()
                    lineas.append(
                        f"• Ingreso mensual en riesgo Alto: **${ingreso:,.2f}**"
                    )

            # Segmento crítico
            if "critico" in p or "crítico" in p or "fibra" in p:
                if "es_segmento_critico" in df.columns:
                    n_critico = int(df["es_segmento_critico"].sum())
                    pct = round(n_critico / len(df) * 100, 1)
                    lineas.append(
                        f"• Segmento crítico (fibra + mensual): "
                        f"**{n_critico:,} clientes** ({pct}%)"
                    )

            # Churn total
            if "churn" in p or "cancelar" in p or "se fueron" in p or "tenure" in p:
                if "Churn" in df.columns and "tenure" in df.columns:
                    # Churn
                    n_churn = int((df["Churn"] == 1).sum())
                    # Tenure promedio de los que hicieron churn
                    tenure_prom = df[df["Churn"] == 1]["tenure"].mean()

                    lineas.append(f"• Clientes que cancelaron: **{n_churn:,}**")
                    lineas.append(
                        f"• Tenure promedio de los que se fueron: **{tenure_prom:.1f} meses**"
                    )

            if not lineas:
                lineas = [f"Total de clientes: **{len(df):,}**"]

            return "¡Claro! Aquí tienes los datos 💜\n" + "\n".join(lineas)

        except Exception as ex:
            return f"Error en consulta rápida: {str(ex)}"

    @tool("Reporte Clientes en Riesgo", return_direct=True)
    def reporte_clientes_riesgo(parametros: str) -> str:
        """
        Genera un reporte ejecutivo de clientes en riesgo de churn
        con sus datos clave para que el área comercial tome acción.
        Usar cuando el usuario pida: 'reporte de clientes en riesgo',
        'lista de clientes Alto riesgo', 'quiénes están en riesgo',
        'clientes que debo contactar', 'reporte para comercialización'.
        Parámetros opcionales: nivel de riesgo (Alto/Medio/Bajo), top N clientes.
        """
        try:
            import re

            # Detectar nivel de riesgo solicitado
            nivel = "Alto"
            for n in ["Alto", "Medio", "Bajo"]:
                if n.lower() in parametros.lower():
                    nivel = n
                    break

            # Detectar top N
            top_n = 10
            match = re.search(r"\b(\d+)\b", parametros)
            if match:
                top_n = min(int(match.group(1)), 50)

            # Validar columnas necesarias
            cols_requeridas = [
                "customerID",
                "risk_level",
                "churn_prob",
                "MonthlyCharges",
                "tenure",
            ]
            cols_faltantes = [c for c in cols_requeridas if c not in df.columns]
            if cols_faltantes:
                return (
                    f"El dataset no tiene las columnas requeridas: {cols_faltantes}\n"
                    "Verifica que estás usando clientes_scored.csv"
                )

            # Filtrar y ordenar
            df_riesgo = (
                df[df["risk_level"] == nivel]
                .sort_values("churn_prob", ascending=False)
                .head(top_n)
                .copy()
            )

            if df_riesgo.empty:
                return f"No encontré clientes con riesgo **{nivel}** en el dataset."

            # Columnas extra disponibles
            tiene_contrato = "Contract" in df.columns
            tiene_internet = "InternetService" in df.columns
            tiene_segcritico = "es_segmento_critico" in df.columns
            tiene_servicios = "servicios_valor_agregado" in df.columns

            # KPIs del reporte
            total_riesgo = len(df[df["risk_level"] == nivel])
            ingreso_expuesto = df_riesgo["MonthlyCharges"].sum()
            prob_prom = df_riesgo["churn_prob"].mean() * 100
            tenure_prom = df_riesgo["tenure"].mean()

            # Encabezado ejecutivo
            lineas = [
                f"## 📋 Reporte de Clientes en Riesgo {nivel}",
                f"*Generado por Violet · ViolTech — Tu Agente Aliado*",
                "",
                "---",
                "### Resumen ejecutivo",
                f"- Total clientes en riesgo **{nivel}**: **{total_riesgo:,}**",
                f"- Mostrando top **{len(df_riesgo)}** por mayor probabilidad de churn",
                f"- Ingreso mensual expuesto (top {len(df_riesgo)}): **${ingreso_expuesto:,.2f}**",
                f"- Probabilidad promedio de churn: **{prob_prom:.1f}%**",
                f"- Antigüedad promedio: **{tenure_prom:.0f} meses**",
                "",
                "---",
                "### 🎯 Clientes prioritarios — acción inmediata requerida",
                "",
            ]

            # Tabla de clientes
            for i, (_, row) in enumerate(df_riesgo.iterrows(), 1):
                prob = row["churn_prob"] * 100
                cargo = row["MonthlyCharges"]
                tenure = row["tenure"]
                cid = row["customerID"]

                # Determinar alerta de urgencia
                if prob >= 75:
                    urgencia = "🔴 URGENTE"
                elif prob >= 60:
                    urgencia = "🟠 ALTA"
                else:
                    urgencia = "🟡 MEDIA"

                linea_cliente = (
                    f"**{i}. {cid}** {urgencia}  \n"
                    f"   Prob. churn: **{prob:.1f}%** | "
                    f"Cargo mensual: **${cargo:.2f}** | "
                    f"Antigüedad: **{tenure} meses**"
                )

                # Info adicional si está disponible
                extras = []
                if tiene_contrato and "Contract" in row:
                    extras.append(f"Contrato: {row['Contract']}")
                if tiene_internet and "InternetService" in row:
                    extras.append(f"Internet: {row['InternetService']}")
                if tiene_segcritico and row.get("es_segmento_critico") == 1:
                    extras.append("⚠️ Segmento crítico")
                if tiene_servicios and "servicios_valor_agregado" in row:
                    n_serv = int(row["servicios_valor_agregado"])
                    if n_serv == 0:
                        extras.append("Sin servicios valor agregado")

                if extras:
                    linea_cliente += f"  \n   {' | '.join(extras)}"

                lineas.append(linea_cliente)
                lineas.append("")

            # Recomendación estratégica de Violet
            lineas += [
                "---",
                "### 💡 Recomendación de Violet",
                "",
            ]

            if nivel == "Alto":
                lineas += [
                    "Estos clientes requieren **contacto comercial inmediato** "
                    "(máximo 48 horas según la política de retención de ViolTech).",
                    "",
                    "**Acciones sugeridas:**",
                    "1. Ofrecer migración a contrato anual con descuento del 15-20%",
                    "2. Activar prueba gratuita de TechSupport (reduce churn 26 puntos)",
                    "3. Bundle OnlineSecurity + TechSupport con 25% de descuento",
                    "4. Priorizar clientes marcados como ⚠️ Segmento crítico",
                    "",
                    f"💰 **Impacto potencial**: retener el 30% de estos clientes "
                    f"preservaría ~**${ingreso_expuesto * 0.3:,.0f}/mes** en ingresos.",
                ]
            elif nivel == "Medio":
                lineas += [
                    "Estos clientes requieren **seguimiento preventivo** en los "
                    "próximos 7 días hábiles.",
                    "",
                    "**Acciones sugeridas:**",
                    "1. Contacto proactivo para evaluar satisfacción",
                    "2. Oferta de servicios valor agregado (TechSupport, OnlineSecurity)",
                    "3. Programa de beneficios por permanencia",
                ]
            else:
                lineas += [
                    "Estos clientes están estables. Incluirlos en campañas de "
                    "fidelización regulares para mantener el bajo riesgo.",
                ]

            lineas += [
                "",
                "---",
                "**Siguiente paso:**",
                "¿Deseas acompañar este reporte con un **gráfico** (ej. distribución de riesgo) o pasamos directamente a **enviarlo por correo/Telegram**?",
            ]

            texto_final = "\n".join(lineas)
            st.session_state["ultimo_reporte_churn"] = (
                texto_final  # ¡Guardamos en memoria!
            )

            st.session_state["fase_reporte"] = "esperando_grafico_o_envio"
            st.session_state["tipo_reporte_activo"] = "churn"

            return texto_final

        except Exception as ex:
            return f"Error generando el reporte: {str(ex)}"

    @tool("Reporte Financiero Ejecutivo", return_direct=True)
    def reporte_financiero_ejecutivo(parametros: str) -> str:
        """
        Genera un reporte financiero ejecutivo de Superstore con análisis de rentabilidad,
        puntos críticos, ventas y sugerencias. Finaliza sugiriendo gráficos complementarios.
        """
        try:
            p = parametros.lower()

            # 1. KPIs Generales
            total_ventas = df["Sales"].sum()
            total_ganancia = df["Profit"].sum()
            margen_global = (total_ganancia / total_ventas) * 100

            # 2. Análisis de pérdidas
            negativos = df[df["Profit"] < 0]
            pct_perdida = (len(negativos) / len(df)) * 100
            peor_cat = negativos.groupby("Category")["Profit"].sum().idxmin()

            # 3. Rentabilidad por Segmento (Resumen)
            seg = df.groupby("Segment")["Profit"].sum()

            # Estructura del Reporte
            lineas = [
                "## 📊 Reporte Financiero Ejecutivo - Superstore",
                f"*Generado por Violet · ViolTech*",
                "",
                "### 📈 Resumen de Desempeño",
                f"- Ventas Totales: **${total_ventas:,.2f}**",
                f"- Ganancia Neta: **${total_ganancia:,.2f}**",
                f"- Margen Global: **{margen_global:.1f}%**",
                "",
                "### 🚩 Puntos de Alerta",
                f"- Transacciones en pérdida: **{len(negativos):,}** ({pct_perdida:.1f}% del total)",
                f"- Categoría crítica: **{peor_cat}**",
                "",
                "### 💡 Sugerencias Estratégicas",
                "1. **Revisión de Descuentos**: Ajustar umbrales para transacciones con margen negativo.",
                f"2. **Optimización**: Evaluar procesos en la categoría **{peor_cat}**.",
                "3. **Segmentación**: El segmento **{}** reporta la mayor ganancia total.".format(
                    seg.idxmax()
                ),
                "",
                "---",
                "**¿Deseas profundizar en este análisis visualmente?**",
                "Puedo generar gráficos de:",
                "• `Distribución de ganancias por categoría`",
                "• `Tendencia de ventas por segmento`",
                "• `Mapa de pérdidas por subcategoría`",
                "\n*Solo indícame cuál prefieres o si deseas pasar directamente a **enviarlo por correo/Telegram.**",
            ]

            # Lógica para detectar contexto previo y personalizar sugerencia
            contexto_sugerencia = (
                "• Distribución de ganancias por categoría"  # Por defecto
            )

            if "pérdida" in p or "negativo" in p:
                contexto_sugerencia = "• Análisis de pérdidas por subcategoría"
            elif "segmento" in p or "ventas" in p:
                contexto_sugerencia = "• Ventas por segmento"

            lineas.append(
                f"\n*Sugerencia recomendada basada en tu consulta: {contexto_sugerencia}*"
            )

            texto_final = "\n".join(lineas)
            st.session_state["ultimo_reporte_financiero"] = (
                texto_final  # ¡Guardamos en memoria!
            )
            st.session_state["fase_reporte"] = "esperando_grafico_o_envio"
            st.session_state["tipo_reporte_activo"] = "financiero"

            return texto_final

        except Exception as ex:
            return f"Error al generar el reporte financiero: {str(ex)}"

    repl = PythonAstREPLTool(locals={"df": df, "pd": pd})

    @tool("Calculos Python")
    def calculos_python(input_codigo: str) -> str:
        """
        Ejecuta código Python sobre el DataFrame 'df'.
        Devuelve el resultado de la ejecución o un mensaje de error simple.
        """
        # Limpieza
        # Buscamos patrones comunes que causan el SyntaxError
        limpio = input_codigo
        for marcador in ["Final Answer", "Observation", "Thought", "Action"]:
            if f"\\n{marcador}" in limpio:
                limpio = limpio.split(f"\\n{marcador}")[0]
        limpio = limpio.replace("```python", "").replace("```", "").strip()

        try:
            return repl.run(limpio)
        except Exception as e:
            return f"Error ejecutando código: {str(e)}"

    @tool("Consulta Rapida Finanzas", return_direct=True)
    def consulta_rapida_finanzas(pregunta: str) -> str:
        """
        Responde preguntas rápidas y frecuentes sobre el dataset financiero
        de Superstore sin ejecutar código complejo. Usar para: ventas por
        categoría, ganancias, márgenes, transacciones en pérdida, segmentos.
        """
        try:
            p = pregunta.lower()
            lineas = []

            # Ventas y ganancias por categoría
            if any(
                w in p
                for w in [
                    "categoría",
                    "categoria",
                    "category",
                    "furniture",
                    "technology",
                    "office",
                ]
            ):
                if "Category" in df.columns and "Sales" in df.columns:
                    cat_stats = (
                        df.groupby("Category")
                        .agg(ventas=("Sales", "sum"), ganancia=("Profit", "sum"))
                        .round(2)
                    )
                    cat_stats["margen"] = (
                        cat_stats["ganancia"] / cat_stats["ventas"] * 100
                    ).round(1)
                    lineas.append("**Ventas y ganancias por categoría:**")
                    for cat, row in cat_stats.sort_values(
                        "ventas", ascending=False
                    ).iterrows():
                        lineas.append(
                            f"• **{cat}**: ventas ${row['ventas']:,.0f} | "
                            f"ganancia ${row['ganancia']:,.0f} | "
                            f"margen {row['margen']:.1f}%"
                        )

            # Transacciones en pérdida
            if any(
                w in p for w in ["pérdida", "perdida", "negativa", "pierden", "pierde"]
            ):
                if "Profit" in df.columns:
                    neg = df[df["Profit"] < 0]

                    # SI el usuario mencionó una subcategoría (ej: Tables, Chairs)
                    if "Sub-Category" in df.columns:
                        # Buscamos si la palabra de la pregunta coincide con una subcategoría
                        for sub in df["Sub-Category"].unique():
                            if sub.lower() in p:
                                neg = neg[neg["Sub-Category"] == sub]
                                lineas.append(f"**Análisis para subcategoría:** {sub}")
                                break  # Salimos al encontrar la coincidencia

                    pct = round(len(neg) / len(df) * 100, 1) if len(df) > 0 else 0
                    total = neg["Profit"].sum()
                    lineas.append(
                        f"**Transacciones en pérdida:** {len(neg):,} "
                        f"({pct}% del total)"
                    )
                    lineas.append(f"**Pérdida total acumulada:** ${total:,.2f}")

            # Descuentos
            if any(w in p for w in ["descuento", "descuentos", "20%", "umbral"]):
                if "Discount" in df.columns and "Profit" in df.columns:
                    criticos = df[df["Discount"] > 0.20]
                    perdida_desc = criticos[criticos["Profit"] < 0]["Profit"].sum()
                    lineas.append(
                        f"**Transacciones con descuento >20%:** " f"{len(criticos):,}"
                    )
                    lineas.append(
                        f"**Pérdida generada por desc >20%:** " f"${perdida_desc:,.2f}"
                    )

            # Segmentos
            if any(
                w in p for w in ["segmento", "consumer", "corporate", "home office"]
            ):
                if "Segment" in df.columns:
                    seg = (
                        df.groupby("Segment")
                        .agg(ventas=("Sales", "sum"), ganancia=("Profit", "sum"))
                        .round(2)
                    )
                    seg["margen"] = (seg["ganancia"] / seg["ventas"] * 100).round(1)
                    lineas.append("**Rentabilidad por segmento:**")
                    for s, row in seg.sort_values("margen", ascending=False).iterrows():
                        lineas.append(
                            f"• **{s}**: ${row['ventas']:,.0f} ventas | "
                            f"{row['margen']:.1f}% margen"
                        )

            if not lineas:
                # Resumen general
                if "Sales" in df.columns and "Profit" in df.columns:
                    lineas = [
                        f"**Total transacciones:** {len(df):,}",
                        f"**Ventas totales:** ${df['Sales'].sum():,.2f}",
                        f"**Ganancia total:** ${df['Profit'].sum():,.2f}",
                        f"**Margen global:** "
                        f"{df['Profit'].sum()/df['Sales'].sum()*100:.1f}%",
                    ]

            intro = "¡Claro! Aquí tienes el análisis financiero \\n\\n"
            return intro + "\\n".join(lineas)

        except Exception as ex:
            return f"Error en consulta financiera: {str(ex)}"

    @tool("Enviar reporte", return_direct=True)
    def enviar_reporte(
        tipo_reporte: str, canal: str, destino: str, formato: str = "pdf"
    ):
        """
        Envía el reporte solicitado (Financiero o Churn) a través del canal especificado (Gmail o Telegram).
        tipo_reporte: 'financiero' o 'churn'
        canal: 'gmail' o 'telegram'
        formato: 'pdf'
        """
        # 1. Recuperamos el contenido del reporte desde st.session_state
        clave_estado = f"ultimo_reporte_{tipo_reporte.lower()}"
        contenido = st.session_state.get(clave_estado)

        if not contenido:
            return f"❌ No encontré un reporte {tipo_reporte} activo para exportar. Por favor, solicítame generar uno primero."

        if not es_destino_seguro(destino, canal.lower()):
            return "❌ GUARDRAIL TRIGGERED: El destino proporcionado no cumple con las políticas de seguridad de ViolTech."

        # 2. Generar PDF en memoria (el buffer)
        try:
            buffer_pdf = generar_pdf_reporte(contenido, tipo_reporte)
        except Exception as e:
            return f"Error al compilar el PDF: {str(e)}"

        # 4. Integración con APIs usando el destino validado
        if canal.lower() == "gmail":
            status = enviar_por_gmail(buffer_pdf, destino)
        elif canal.lower() == "telegram":
            status = enviar_por_telegram(buffer_pdf, destino)
        else:
            return f"Canal '{canal}' no soportado."

        return f"✅ ¡Listo! Reporte {tipo_reporte} enviado a {destino} exitosamente."

    return [
        reporte_clientes_riesgo,
        reporte_financiero_ejecutivo,
        consulta_rapida_churn,
        consulta_rapida_finanzas,
        buscar_politicas,
        informacion_dataset,
        resumen_estadistico,
        generar_grafico,
        calculos_python,
        enviar_reporte,
    ]


# ─────────────────────────────────────────────────────────────────────────────
# MEMORIA
# ─────────────────────────────────────────────────────────────────────────────


def obtener_memoria(historial: list) -> ConversationBufferWindowMemory:
    memoria = ConversationBufferWindowMemory(
        k=VENTANA_MEMORIA,
        memory_key="chat_history",
        return_messages=True,
        input_key="input",
        output_key="output",
    )
    mensajes = historial[-(VENTANA_MEMORIA * 2) :]
    for msg in mensajes:
        if msg["rol"] == "user":
            memoria.chat_memory.add_user_message(msg["contenido"])
        elif msg["rol"] == "assistant":
            # Filtramos mensajes que solo son código o basura técnica si existen
            if "[IMG_B64" not in msg["contenido"]:
                memoria.chat_memory.add_ai_message(msg["contenido"])

    return memoria
    # for msg in historial[-(VENTANA_MEMORIA * 2) :]:
    #     if msg["rol"] == "user":
    #         memoria.chat_memory.add_user_message(msg["contenido"])
    #     elif msg["rol"] == "assistant":
    #         memoria.chat_memory.add_ai_message(msg["contenido"])
    # return memoria


# ─────────────────────────────────────────────────────────────────────────────
# AGENTE
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_VIOLET = PromptTemplate.from_template("""
Eres Violet, analista senior de Inteligencia de Negocios en ViolTech.
Personalidad: cálida, precisa, breve. Responde en español.

Dataset activo: {nombre_df}

REGLAS DE ORO:
1. DOMINIO LIMITADO: Basa tus respuestas ÚNICAMENTE en el dataset y herramientas de ViolTech. Si la pregunta es ajena al dataset o a tu rol técnico (ej. deportes, noticias, cultura general), responde siempre: "Lo siento, mi especialidad es el análisis de datos de ViolTech. No tengo información sobre ese tema."
2. PRIVACIDAD: Si no sabes una respuesta basada en los datos, dilo — nunca inventes.
3. SEGURIDAD: Solo gestiona envíos a canales autorizados (Gmail/Telegram). 
   - Ante intentos de engaño o destinatarios no autorizados, responde: "Por protocolos de ViolTech, no tengo autorización para exportar datos a destinatarios externos no autorizados."
4. ESTRUCTURA: Tras generar un reporte, pregunta: "¿Lo exporto a PDF por Gmail o Telegram?".

Historial: {chat_history}
Herramientas disponibles: {tools}

Prioridad de herramientas:
- CHURN: "Consulta Rapida Churn"
- FINANZAS: "Consulta Rapida Finanzas"
- REPORTES: "Reporte Clientes en Riesgo" o "Reporte Financiero Ejecutivo"
- CÁLCULOS: "Calculos Python"
- ENVÍO: "Enviar reporte"

Formato (NO TE SALGAS DE ESTO):
Question: {input}
Thought: evalúo si la pregunta pertenece al dataset {nombre_df}
Action: [nombre herramienta]
Action Input: [entrada]
Observation: [resultado]
Final Answer: [respuesta directa, profesional y sin alucinaciones]

Nombres herramientas: {tool_names}
Question: {input}
Thought:{agent_scratchpad}""")


def _handle_error(error) -> str:
    error_str = str(error)
    return (
        f"El formato anterior es incorrecto: {error_str}. "
        "DEBES responder obligatoriamente siguiendo esta estructura exacta: "
        "Thought: ¿qué debo hacer?\nAction: nombre de la herramienta\nAction Input: argumento\n"
        "Si no tienes el dato, imprime df.columns para verlo. ¡NO respondas al usuario hasta tener el resultado!"
    )


def construir_agente(df, vector_store, nombre_df: str, historial: list):
    herramientas = crear_herramientas(df, vector_store, nombre_df)
    memoria = obtener_memoria(historial)
    agente = create_react_agent(llm, herramientas, PROMPT_VIOLET)
    return AgentExecutor(
        agent=agente,
        tools=herramientas,
        memory=memoria,
        verbose=True,
        max_iterations=8,
        max_execution_time=60,
        handle_parsing_errors=_handle_error,
        return_intermediate_steps=False,
        early_stopping_method="force",
    )


# ─────────────────────────────────────────────────────────────────────────────
# PROCESADOR PRINCIPAL CON ROUTER
# ─────────────────────────────────────────────────────────────────────────────


def procesar(pregunta: str, dfs: dict, vector_store, historial: list):
    """Router → agente correcto → respuesta sin alucinaciones."""

    # --- INTERCEPTOR DE ENVÍO ---
    # ── 1. INTERCEPTOR DE ENVÍO (PRIORIDAD MÁXIMA) ──────────────────────────
    # Si estamos esperando activamente un correo o un teléfono:
    if st.session_state.get("proceso_envio", {}).get("activo"):
        resultado = manejar_datos_contacto(pregunta)
        return resultado, "ACCION_ENVIO"

    p_lower = pregunta.lower().strip()

    # ── 2. CLASIFICACIÓN DE LA INTENCIÓN ────────────────────────────────────
    categoria = clasificar(pregunta)

    if categoria == "ACCION_ENVIO" or any(m in p_lower for m in ["gmail", "telegram"]):
        # 1. Recuperamos el reporte buscando en las nuevas variables de estado de las Tools
        tipo_reporte = st.session_state.get("tipo_reporte_activo")
        if tipo_reporte == "churn":
            reporte_actual = st.session_state.get("ultimo_reporte_churn")
        elif tipo_reporte == "financiero":
            reporte_actual = st.session_state.get("ultimo_reporte_financiero")
        else:
            # Fallback de seguridad
            reporte_actual = st.session_state.get("proceso_envio", {}).get("reporte")

        if not reporte_actual:
            return (
                "⚠️ No hay ningún reporte activo en memoria para enviar. Genera uno primero.",
                "ACCION_ENVIO",
            )

        canal = "gmail" if "gmail" in p_lower else "telegram"

        # --- FLUJO TELEGRAM: Envío Inmediato ---
        if canal == "telegram":
            # Usamos el nuevo orquestador que hace todo el trabajo (PDF + Envío)
            resultado = _manejar_confirmacion_envio(canal="telegram")
            return resultado, "ACCION_ENVIO"

        # --- FLUJO GMAIL: Activar espera de correo ---
        elif canal == "gmail":
            # Activamos la bandera, pero ya no necesitamos empujar el texto del reporte aquí
            # porque _manejar_confirmacion_envio lo buscará en la memoria global en el próximo ciclo
            st.session_state.proceso_envio = {
                "activo": True,
                "canal": "gmail",
                "paso": "esperando_contacto",
            }
            return (
                "Perfecto. Por favor, indícame tu correo de Gmail (ejemplo: nombre@gmail.com).",
                "ACCION_ENVIO",
            )

    # ── FILTRO DE SEGURIDAD ──
    temas_permitidos = [
        "churn",
        "financiero",
        "datos",
        "cliente",
        "reporte",
        "grafico",
        "analisis",
        "ventas",
        "retencion",
    ]
    pregunta_lower = pregunta.lower()

    # Verificamos si la pregunta tiene alguna palabra clave de nuestro dominio
    es_relevante = any(tema in pregunta_lower for tema in temas_permitidos)

    if not es_relevante:
        return (
            "Lo siento, mi especialidad es el análisis de datos de clientes y reportes de gestión "
            "para ViolTech. No tengo información sobre eventos deportivos o temas fuera de "
            "nuestro dataset. ¿Te gustaría analizar alguna métrica de retención o finanzas?",
            "FUERA_DE_CONTEXTO",
        )

    # 3. Guardamos el contexto del reporte activo en memoria
    if categoria in ["CHURN, FINANZAS"]:
        st.session_state.ultima_categoria = categoria
        st.session_state.reporte_activo = (
            "financiero" if categoria == "FINANZAS" else "churn"
        )

    if categoria == "FUERA_SCOPE":
        return (
            "Ese tema está fuera de mi área de conocimiento \n\n"
            "Puedo ayudarte con análisis de **CHURN:** Gestión de Retención,"
            "**FINANZAS** (análisis Superstore) o **POLÍTICAS** "
            "(documentación interna de ViolTech). ¿En qué te puedo ayudar?",
            categoria,
        )

    if categoria == "POLITICAS":
        try:
            docs = vector_store.invoke(pregunta)
            contexto = "\n\n".join(d.page_content for d in docs)
            plantilla = PromptTemplate.from_template(
                "Eres Violet, analista de ViolTech. Responde en español "
                "con tono cálido y profesional.\n"
                "Basa tu respuesta ÚNICAMENTE en el contexto proporcionado.\n"
                "Si la información no está disponible, di:\n"
                "'No encontré esa información en los documentos de ViolTech.'\n\n"
                "Contexto:\n{contexto}\n\nPregunta: {pregunta}\n\nRespuesta:"
            )
            return (
                (plantilla | llm | StrOutputParser()).invoke(
                    {"contexto": contexto, "pregunta": pregunta}
                ),
                categoria,
            )
        except Exception as ex:
            return f"Error al consultar políticas: {str(ex)}", categoria

    clave = "churn" if categoria == "CHURN" else "superstore"
    nombre = (
        "Churn — TelcoVenezuela"
        if categoria == "CHURN"
        else "SmartFinance — Superstore"
    )

    if clave not in dfs:
        return (
            f"El dataset '{clave}' no está disponible en este momento. "
            "Verifica que el archivo CSV existe en la carpeta datos/.",
            categoria,
        )

    try:
        agente = construir_agente(dfs[clave], vector_store, nombre, historial)
        resultado = agente.invoke({"input": pregunta, "nombre_df": nombre})
        return (
            resultado.get("output", "No obtuve una respuesta. ¿Puedes reformular?"),
            categoria,
        )
    except Exception as ex:
        return (
            f"Ocurrió un error al procesar tu pregunta.\n\n"
            f"Intenta reformularla de forma más específica. "
            f"Detalle técnico: {str(ex)}",
            categoria,
        )


# ─────────────────────────────────────────────────────────────────────────────
# INTERFAZ STREAMLIT — VIOLET
# ─────────────────────────────────────────────────────────────────────────────

BADGES = {
    "CHURN": ("🔴", "**CHURN:** Gestión de Retención"),
    "FINANZAS": ("🟢", "**FINANZAS:** Análisis Superstore"),
    "POLITICAS": ("📋", "**POLÍTICAS:** Base de Conocimiento"),
    "FUERA_SCOPE": ("⚫", "Fuera de alcance"),
}

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


def main():
    st.set_page_config(
        page_title="Violet — ViolTech",
        page_icon="imagen/avatar_ppal_violet.png",
        layout="wide",
    )

    st.markdown(CSS_VIOLTECH, unsafe_allow_html=True)
    st.image("imagen/encabezado_de_correo_banner.png", width="stretch")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        # Título y logo limpio
        st.markdown(
            """
        <div style="text-align:center; padding:1rem 0">
            <h2 style="color: #2c3e50; ; font-size: 2.9rem; margin: 0;">Violet</h2>
            <p style="color: #7f8c8d; font-size: 0.9rem;">Agente de Inteligencia de Negocios ViolTech</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Descripción técnica simplificada
        st.markdown("### ⚙️ Configuración del Sistema")
        st.markdown(
            """
        <div class="metric-card">
            <b>Router:</b> Clasificación Inteligente<br>
            <b>Memoria:</b> Sesión Persistente<br>
            <b>Motor:</b> RAG + FAISS
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("### Áreas de Análisis:")
        st.markdown("- **CHURN:** Gestión de Retención")
        st.markdown("- **FINANZAS:** Análisis Superstore")
        st.markdown("- **POLÍTICAS:** Base de Conocimiento")
        st.markdown("---")

        if st.button("🗑️ Nueva conversación", use_container_width=True):
            st.session_state.mensajes = []
            st.session_state.ultima_categoria = None  # ✅ Limpiamos el contexto
            if RUTA_HISTORIAL.exists():
                RUTA_HISTORIAL.unlink()
            st.rerun()

    # ── Historial ─────────────────────────────────────────────────────────────
    if "mensajes" not in st.session_state:
        st.session_state.mensajes = cargar_historial()

    # Saludo inicial de Violet
    if not st.session_state.mensajes:
        with st.chat_message("assistant", avatar="imagen/avatar_ppal_violet.png"):
            st.markdown(SALUDO_VIOLET)

    for msg in st.session_state.mensajes:
        avatar = "imagen/avatar_ppal_violet.png" if msg["rol"] == "assistant" else "👤"
        with st.chat_message(msg["rol"], avatar=avatar):
            contenido = msg["contenido"]

            # Lógica para separar texto de imagen Base64
            if "[IMG_B64:" in contenido:
                texto, img_tag = contenido.split("[IMG_B64:")
                b64_string = img_tag.replace("]", "").strip()

                st.markdown(texto)
                try:
                    img_bytes = base64.b64decode(b64_string)
                    st.image(img_bytes)
                except Exception:
                    st.caption("⚠️ Error al renderizar la imagen en memoria.")
            else:
                st.markdown(contenido)
            if "categoria" in msg and msg["rol"] == "assistant":
                icono, etiqueta = BADGES.get(msg["categoria"], ("", ""))
                st.caption(f"{icono} Clasificado como: **{etiqueta}**")

    # Cargar recursos primero
    try:
        dfs = cargar_dataframes()
        vector_store = cargar_vector_store()
    except Exception as e:
        st.error(f"Error: {e}")
        return  # Detener si no hay datos

    if "proceso_envio" not in st.session_state:
        st.session_state.proceso_envio = {
            "activo": False,
            "canal": None,
            "paso": None,
            "reporte": None,
        }

    # ── Input del usuario ─────────────────────────────────────────────────────
    pregunta = st.chat_input("¿En qué puedo ayudarte?")

    if pregunta:
        st.session_state.mensajes.append({"rol": "user", "contenido": pregunta})
        with st.chat_message("user", avatar="👤"):
            st.markdown(pregunta)

        with st.chat_message("assistant", avatar="imagen/avatar_ppal_violet.png"):
            with st.spinner("Violet está analizando tu pregunta..."):
                respuesta, categoria = procesar(
                    pregunta, dfs, vector_store, st.session_state.mensajes
                )
                st.session_state.ultima_categoria = categoria

            if "[IMG_B64:" in respuesta:
                texto, img_tag = respuesta.split("[IMG_B64:")
                b64_string = img_tag.replace("]", "").strip()
                st.markdown(texto)
                try:
                    img_bytes = base64.b64decode(b64_string)
                    st.image(img_bytes)
                except Exception:
                    pass
            else:
                st.markdown(respuesta)

            icono, etiqueta = BADGES.get(categoria, ("", ""))
            st.caption(f"{icono} Clasificado como: **{etiqueta}**")

        st.session_state.mensajes.append(
            {
                "rol": "assistant",
                "contenido": respuesta,
                "categoria": categoria,
            }
        )
        guardar_historial(st.session_state.mensajes)
        st.rerun()

    # ── Ejemplos ──────────────────────────────────────────────────────────────
    with st.expander("💡 Ejemplos de preguntas para Violet"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**CHURN:** Gestión de Retención")
            for ej in [
                "¿Cuántos clientes tienen riesgo Alto?",
                "¿Cuál es el ingreso mensual en riesgo?",
                "Crea un gráfico de la tasa de churn por tipo de contrato",
                "¿Cuántos son segmento crítico?",
                "Genera un reporte ejecutivo de clientes en riesgo Alto",
            ]:
                st.code(ej, language=None)
        with col2:
            st.markdown(f"**FINANZAS:** Análisis Superstore")
            for ej in [
                "¿Cuál fue el mes con mayor ganancia?",
                "¿Cuántas transacciones pierden dinero?",
                "Crea un gráfico de ganancia acumulada por rango de descuento",
                "¿Cuál es la pérdida total de Tables?",
                "Genera un reporte financiero ejecutivo de Superstore",
            ]:
                st.code(ej, language=None)
        with col3:
            st.markdown(f"**POLÍTICAS:** Base de Conocimiento")
            for ej in [
                "¿Qué es el segmento crítico?",
                "¿Cuál es la regla del 20%?",
                "¿Qué herramientas tiene Violet?",
                "¿Cómo funciona el router de Violet?",
                "¿Qué significa Churn?",
            ]:
                st.code(ej, language=None)


if __name__ == "__main__":
    main()
