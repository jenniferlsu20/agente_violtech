import io
import os
import re
import smtplib
import requests
import base64
import streamlit as st
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

# Importaciones de ReportLab
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, HRFlowable


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
) -> io.BytesIO:
    """Genera el reporte PDF con el branding de ViolTech."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=50,
        leftMargin=50,
        topMargin=40,
        bottomMargin=40,
    )
    styles = getSampleStyleSheet()

    # Estilos corporativos
    estilo_titulo = ParagraphStyle(
        "Titulo",
        parent=styles["Heading1"],
        fontSize=16,
        textColor=colors.HexColor("#4B0082"),
        fontName="Helvetica-Bold",
    )
    estilo_subtitulo = ParagraphStyle(
        "SubTitulo",
        parent=styles["Heading2"],
        fontSize=13,
        spaceBefore=10,
        fontName="Helvetica-Bold",
    )
    estilo_normal = ParagraphStyle(
        "Normal", parent=styles["Normal"], fontSize=10, leading=14
    )
    estilo_lista = ParagraphStyle("Lista", parent=estilo_normal, leftIndent=15)

    # Limpieza de Emojis y elementos interactivos del LLM
    reemplazos = {"🔴": "[CRÍTICO]", "🟠": "[ALTA]", "🟡": "[MEDIA]", "⚠️": "(!)"}
    texto_limpio = texto_reporte
    for char, rep in reemplazos.items():
        texto_limpio = texto_limpio.replace(char, rep)

    # Limpieza de prompts residuales
    texto_limpio = re.sub(r"\*¿Deseas.*\n?", "", texto_limpio)
    texto_limpio = re.sub(r"\*\*Siguiente paso:\*\*", "", texto_limpio)

    story = []
    for linea in texto_limpio.split("\n"):
        linea = linea.strip()
        if not linea:
            continue

        # Limpieza simple de markdown para ReportLab
        linea_procesada = re.sub(r"\*\*(.*?)\*\*", r"\1", linea)

        if linea.startswith("## "):
            story.append(Paragraph(linea_procesada[3:], estilo_titulo))
        elif linea.startswith("### "):
            story.append(Paragraph(linea_procesada[4:], estilo_subtitulo))
        elif linea.startswith("- "):
            story.append(Paragraph("• " + linea_procesada[2:], estilo_lista))
        else:
            story.append(Paragraph(linea_procesada, estilo_normal))

    if grafico_base64:
        img_data = base64.b64decode(grafico_base64)
        story.append(Image(io.BytesIO(img_data), width=400, height=250))

    doc.build(story)
    buffer.seek(0)
    return buffer


def enviar_por_gmail(buffer_pdf: io.BytesIO, destinatario: str) -> str:
    """Envío SMTP profesional."""
    try:
        msg = MIMEMultipart()
        msg["Subject"] = "📊 Reporte Inteligencia de Negocio — ViolTech"
        msg["From"] = os.getenv("SMTP_USER")
        msg["To"] = destinatario

        msg.attach(
            MIMEText("Adjunto encontrarás el informe solicitado.", "plain", "utf-8")
        )

        pdf_adjunto = MIMEApplication(buffer_pdf.read(), _subtype="pdf")
        pdf_adjunto.add_header(
            "Content-Disposition", "attachment", filename="reporte.pdf"
        )
        msg.attach(pdf_adjunto)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(os.getenv("SMTP_USER"), os.getenv("SMTP_APP_PASSWORD"))
            smtp.sendmail(msg["From"], destinatario, msg.as_string())
        return "¡Reporte enviado exitosamente por Gmail!"
    except Exception as e:
        return f"Error Gmail: {str(e)}"


def enviar_por_telegram(buffer_pdf: io.BytesIO) -> str:
    """Envío vía Bot de Telegram."""
    try:
        url = f"https://api.telegram.org/bot{os.getenv('TELEGRAM_BOT_TOKEN')}/sendDocument"
        files = {"document": ("Reporte.pdf", buffer_pdf.read(), "application/pdf")}
        data = {
            "chat_id": os.getenv("TELEGRAM_CHAT_ID"),
            "caption": "📦 Reporte generado por Violet",
        }

        res = requests.post(url, data=data, files=files, timeout=15)
        return (
            "¡Reporte enviado exitosamente por Telegram al grupo Violet_Reportes!"
            if res.json().get("ok")
            else f"Error Telegram: {res.json().get('description')}"
        )
    except Exception as e:
        return f"Error Telegram: {str(e)}"


def enviar_reporte(tipo_reporte: str, canal: str, destino: str) -> str:
    """Punto de entrada principal para el agente."""
    canal = canal.lower()

    # 1. Ajuste de destino para Telegram
    if canal == "telegram":
        destino = os.getenv("TELEGRAM_CHAT_ID")

    # 2. Validación de seguridad
    if not es_destino_seguro(destino, canal):
        if canal == "gmail":
            return "⚠️ Para enviarte el reporte por Gmail, por favor facilítame tu dirección de correo electrónico (ejemplo: usuario@gmail.com)."
        return (
            "❌ Error: Canal de envío no soportado o configuración de grupo inválida."
        )

    # 3. Lógica de generación y envío (continúa igual...)
    contenido = st.session_state.get(f"ultimo_reporte_{tipo_reporte.lower()}")
    if not contenido:
        return f"❌ No encontré un reporte {tipo_reporte} activo."

    buffer = generar_pdf_reporte(contenido, tipo_reporte)

    if canal == "gmail":
        status = enviar_por_gmail(buffer, destino)
    else:
        status = enviar_por_telegram(
            buffer
        )  # Telegram ya tiene su ID configurado internamente

    return f"✅ {status}"
