import io
import re
import smtplib
import asyncio
import base64
import pandas as pd
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import httpx

# Importaciones de ReportLab
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, HRFlowable

# Importamos las credenciales desde tu módulo centralizado de configuración
from app.config import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    SMTP_SERVER,
    SMTP_PORT,
    SMTP_USER,
    SMTP_APP_PASSWORD,
)

# --- 1. LIMPIEZA DE EMOJIS Y PARSEO DE MARKDOWN ---


def _limpiar_emojis(texto: str) -> str:
    """
    Elimina emojis y caracteres Unicode especiales que ReportLab no puede
    renderizar con fuentes estándar (Helvetica/Times), evitando excepciones fatales.
    """
    if not texto:
        return ""
    # Mantiene caracteres alfanuméricos, puntuación estándar y codificación Latin-1 (acentos, ñ, Ñ)
    return re.sub(r"[^\x00-\x7F\x80-\xFF]", "", texto)


def _convertir_markdown_a_tags_reportlab(texto: str) -> str:
    """
    Convierte sintaxis básica de Markdown (**bold**, *italic*) a las etiquetas
    estilo HTML soportadas nativamente por el objeto Paragraph de ReportLab.
    """
    # **negrita** -> <b>negrita</b>
    texto = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", texto)
    # *itálica* -> <i>itálica</i>
    texto = re.sub(r"\*(.*?)\*", r"<i>\1</i>", texto)
    return texto


def es_destino_seguro(destino: str, canal: str) -> bool:
    """Validador de seguridad para evitar fugas de información."""
    canal = canal.lower()

    if canal == "gmail":
        # Se requiere que el usuario proporcione un correo válido
        return bool(destino and destino.endswith("@gmail.com"))

    elif canal == "telegram":
        # Para Telegram, el destino es irrelevante porque siempre enviamos al grupo autorizado
        return True

    return False


def generar_pdf_reporte(
    texto_reporte: str, tipo_reporte: str, grafico_base64: str = None
) -> bytes:
    """Genera el reporte PDF con el branding de ViolTech."""
    buffer = io.BytesIO()

    # Configuración de página y márgenes confortables
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45,
    )

    styles = getSampleStyleSheet()
    story = []

    # Definición de Paleta Corporativa Muted para ViolTech
    COLOR_PRIMARIO = colors.HexColor("#1A365D")  # Azul Oscuro Ejecutivo
    COLOR_SECUNDARIO = colors.HexColor("#2B6CB0")  # Azul Inteligencia
    COLOR_TEXTO = colors.HexColor("#2D3748")  # Gris Pizarra Oscuro

    # Estilos Personalizados Únicos
    estilo_titulo = ParagraphStyle(
        "RepTitle",
        parent=styles["Heading1"],
        fontSize=22,
        leading=26,
        textColor=COLOR_PRIMARIO,
        spaceAfter=15,
        keepWithNext=True,
    )
    estilo_cuerpo = ParagraphStyle(
        "RepBody",
        parent=styles["BodyText"],
        fontSize=11,
        leading=16,
        textColor=COLOR_TEXTO,
        spaceAfter=10,
    )

    # Encabezado Principal del Reporte
    nombre_reporte = (
        "ANÁLISIS DE RETENCIÓN (CHURN)"
        if tipo_reporte == "churn"
        else "RENDIMIENTO FINANCIERO"
    )
    story.append(Paragraph(f"ViolTech - {nombre_reporte}", estilo_titulo))
    story.append(Spacer(1, 10))

    # Procesar línea por línea el Markdown del Agente
    lineas = texto_reporte.split("\n")
    for linea in lineas:
        linea_limpia = _limpiar_emojis(linea).strip()
        if not linea_limpia:
            story.append(Spacer(1, 6))
            continue

        # Convertir títulos Markdown (#, ##, ###) a estilos ReportLab
        if linea_limpia.startswith("###"):
            estilo_h3 = ParagraphStyle(
                "H3",
                parent=styles["Heading3"],
                fontSize=13,
                leading=16,
                textColor=COLOR_SECUNDARIO,
                spaceBefore=8,
                spaceAfter=4,
                keepWithNext=True,
            )
            texto_fmt = _convertir_markdown_a_tags_reportlab(
                linea_limpia.replace("###", "").strip()
            )
            story.append(Paragraph(texto_fmt, estilo_h3))
        elif linea_limpia.startswith("##"):
            estilo_h2 = ParagraphStyle(
                "H2",
                parent=styles["Heading2"],
                fontSize=15,
                leading=18,
                textColor=COLOR_PRIMARIO,
                spaceBefore=12,
                spaceAfter=6,
                keepWithNext=True,
            )
            texto_fmt = _convertir_markdown_a_tags_reportlab(
                linea_limpia.replace("##", "").strip()
            )
            story.append(Paragraph(texto_fmt, estilo_h2))
        elif linea_limpia.startswith("#"):
            texto_fmt = _convertir_markdown_a_tags_reportlab(
                linea_limpia.replace("#", "").strip()
            )
            story.append(Paragraph(texto_fmt, estilo_titulo))
        else:
            # Líneas regulares o viñetas
            texto_fmt = _convertir_markdown_a_tags_reportlab(linea_limpia)
            story.append(Paragraph(texto_fmt, estilo_cuerpo))

    # Inyección del Gráfico si está presente en formato Base64
    if grafico_base64:
        try:
            img_data = base64.b64decode(grafico_base64)
            img_buffer = io.BytesIO(img_data)
            # Dimensiones escaladas de forma segura para la página letter
            img = Image(img_buffer, width=420, height=240)
            story.append(Spacer(1, 15))
            story.append(img)
        except Exception as e:
            story.append(Spacer(1, 10))
            story.append(
                Paragraph(
                    f"<i>(Nota: El gráfico adjunto no pudo ser procesado en el PDF: {str(e)})</i>",
                    estilo_cuerpo,
                )
            )

    # Construcción física del documento
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


async def enviar_por_telegram(
    buffer_pdf: io.BytesIO, filename: str = "reporte_violtech.pdf"
) -> str:
    """Envía el PDF de forma asíncrona mediante la API de Bots de Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return "Error: Credenciales de Telegram no configuradas en el servidor."

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument"
    buffer_pdf.seek(0)
    contenido_pdf = buffer_pdf.read() if hasattr(buffer_pdf, "read") else buffer_pdf
    files = {"document": (filename, contenido_pdf, "application/pdf")}
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "caption": (
            "📊 *¡Hola!* éste es el reporte interactivo en PDF que me solicitaste,"
            "lo puedes visualizar ingresando al grupo Violet_Reportes."
        ),
        "parse_mode": "Markdown",
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, data=data, files=files, timeout=160)
            resultado = response.json()

            if not resultado.get("ok"):
                return f"Error Telegram: {resultado.get('description')}"
        return "Exitoso"
    except Exception as e:
        return f"Error de conexión con Telegram: {str(e)}"


def _enviar_smtp_sincrono(buffer_pdf, destino: str, filename: str) -> str:
    """Lógica bloqueante de SMTP empaquetada para ser ejecutada en un hilo seguro."""
    if not SMTP_USER or not SMTP_APP_PASSWORD:
        return "Error: Credenciales SMTP no configuradas en el servidor."

    msg = MIMEMultipart()
    msg["From"] = SMTP_USER
    msg["To"] = destino
    msg["Subject"] = "📊 Reporte de Inteligencia de Negocios - ViolTech"

    cuerpo = (
        "Estimado usuario,\n\n"
        "Adjunto encontrarás el reporte detallado en formato PDF que ha sido solicitado "
        "a Violet (Agente IA de Inteligencia de Negocios de ViolTech).\n\n"
        "Este correo electrónico fue generado automáticamente de manera segura.\n\n"
        "Atentamente,\nSistemas Inteligentes - ViolTech"
    )
    msg.attach(MIMEText(cuerpo, "plain", "utf-8"))
    adjunto = MIMEApplication(buffer_pdf, Name=filename)
    adjunto["Content-Disposition"] = f'attachment; filename="{filename}"'
    msg.attach(adjunto)

    try:
        # Uso estricto de SSL/TLS para seguridad corporativa
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, timeout=160) as server:
            server.login(SMTP_USER, SMTP_APP_PASSWORD)
            server.sendmail(SMTP_USER, destino, msg.as_string())
        return "Exitoso"
    except Exception as e:
        return f"Excepción SMTP: {str(e)}"


async def enviar_por_gmail(
    buffer_pdf: bytes, destino: str, filename: str = "reporte_violtech.pdf"
) -> str:
    """Envía el PDF por correo delegando la tarea síncrona de SMTP a un hilo secundario (non-blocking)."""
    return await asyncio.to_thread(_enviar_smtp_sincrono, buffer_pdf, destino, filename)


# async def enviar_reporte(tipo_reporte: str, canal: str, destino: str, contexto_sesion: dict) -> str:
#     """Punto de entrada principal para el agente (Stateless para FastAPI)."""
#     canal = canal.lower()

#     # 1. Ajuste de destino para Telegram
#     if canal == "telegram":
#         destino = TELEGRAM_CHAT_ID

#     # 2. Validación de seguridad
#     if not es_destino_seguro(destino, canal):
#         if canal == "gmail":
#             return "⚠️ Para enviarte el reporte por Gmail, por favor facilítame tu dirección de correo electrónico (ejemplo: usuario@gmail.com)."
#         return (
#             "❌ Error: Canal de envío no soportado o configuración de grupo inválida."
#         )

# # 3. Lógica de generación y envío
# contenido = contexto_sesion.get(f"ultimo_reporte_{tipo_reporte.lower()}")
# if not contenido:
#     return f"❌ No encontré un reporte {tipo_reporte} activo."

# buffer = generar_pdf_reporte(contenido, tipo_reporte)

# if canal == "gmail":
#     status = await enviar_por_gmail(buffer, destino)
# else:
#     status = await enviar_por_telegram(buffer)  # Telegram ya tiene su ID configurado internamente

# return f"✅ {status}"

# --- 4. ORQUESTADOR CENTRAL DE ENVÍO (STATELESS) ---


async def procesar_confirmacion_envio(
    canal: str, destino: str = None, contexto_sesion: dict = None
) -> tuple:
    """
    Orquesta de manera pura y asíncrona la conversión y transmisión del reporte activo.

    Retorna: (mensaje_resultado, contexto_sesion_actualizado)
    """
    nuevos_estados = contexto_sesion.copy() if contexto_sesion else {}
    tipo_reporte = nuevos_estados.get("tipo_reporte_activo")

    # 1. Recuperar el texto correcto según el tipo de reporte en el payload de la sesión
    if tipo_reporte == "churn":
        texto_reporte = nuevos_estados.get("ultimo_reporte_churn")
    elif tipo_reporte == "financiero":
        texto_reporte = nuevos_estados.get("ultimo_reporte_financiero")
    else:
        texto_reporte = nuevos_estados.get("proceso_envio", {}).get("reporte")

    if not texto_reporte:
        return (
            "❌ Error: No encontré el contenido del reporte en el contexto de sesión. Por favor, genéralo de nuevo.",
            nuevos_estados,
        )

    canal_norm = canal.lower()
    destino_final = TELEGRAM_CHAT_ID if canal_norm == "telegram" else destino

    if not es_destino_seguro(destino_final, canal_norm):
        return (
            "⚠️ Para enviarte el reporte por Gmail necesito una dirección "
            "@gmail.com válida (ejemplo: usuario@gmail.com)."
        ), nuevos_estados

    grafico_b64 = None
    if nuevos_estados.get("tipo_grafico_pendiente") == tipo_reporte:
        grafico_b64 = nuevos_estados.get("grafico_pendiente_base64")

    # 2. Generar el documento PDF mediante ReportLab
    try:
        buffer_pdf = generar_pdf_reporte(
            texto_reporte, tipo_reporte, grafico_base64=grafico_b64
        )
    except Exception as e:
        return f"❌ Error al estructurar el PDF: {str(e)}", nuevos_estados

    # 3. Ejecutar los motores asíncronos de envío según el canal elegido
    try:
        if canal == "gmail":
            if not destino:
                return (
                    "❌ Error: Se requiere un correo de destino válido para realizar el envío por Gmail.",
                    nuevos_estados,
                )
            resultado = await enviar_por_gmail(buffer_pdf, destino)
        elif canal == "telegram":
            resultado = await enviar_por_telegram(buffer_pdf)
        else:
            return "❌ Canal de envío no reconocido.", nuevos_estados

        # 4. Evaluación del resultado final y reinicio/limpieza de banderas de control
        if resultado != "Exitoso":
            return (
                f"❌ Hubo un inconveniente con los servidores de envío: {resultado}",
                nuevos_estados,
            )

        # Limpieza pura del diccionario de sesión
        nuevos_estados.pop("grafico_pendiente_base64", None)
        nuevos_estados.pop("tipo_grafico_pendiente", None)
        nuevos_estados["fase_reporte"] = None
        nuevos_estados["proceso_envio"] = {
            "activo": False,
            "canal": None,
            "paso": None,
            "reporte": None,
            "grafico": None,
        }

        return (
            f"✅ ¡El reporte en PDF ha sido generado y enviado exitosamente vía {canal.capitalize()}!",
            nuevos_estados,
        )

    except Exception as e:
        return (
            f"❌ Error crítico durante la ejecución del envío: {str(e)}",
            nuevos_estados,
        )


async def manejar_datos_contacto(pregunta: str, contexto_session: dict) -> tuple:
    """
    Gestiona el flujo completo cuando el usuario proporciona sus datos de contacto.
    """
    # 1. Validar y extraer el correo del texto ingresado
    match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", pregunta)
    if not match:
        return "⚠️ No detecté un formato de correo válido. Por favor, intenta de nuevo (ejemplo: usuario@gmail.com):"

    correo_destino = match.group(0)

    if not es_destino_seguro(correo_destino, "gmail"):
        return (
            "❌ Error: La dirección de correo no cumple con las políticas de seguridad.",
            contexto_session,
        )

    resultado, nuevo_contexto = await procesar_confirmacion_envio(
        canal="gmail", destino=correo_destino, contexto_sesion=contexto_session
    )

    return resultado, nuevo_contexto
