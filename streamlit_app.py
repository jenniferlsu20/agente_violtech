import streamlit as st
import time
import requests
import json
import base64
import pathlib

# URL de tu API de FastAPI.
URL_API_VIOLET = st.sidebar.text_input(
    "URL del Backend (Render)", value="https://violtech-violet-backend.onrender.com"
)

# Constantes de configuración locales de la UI
RUTA_HISTORIAL = pathlib.Path("historial.json")
SALUDO_VIOLET = "Hola, soy Violet, tu analista de Inteligencia de Negocios de ViolTech. ¿En qué puedo ayudarte hoy?"

BADGES = {
    "CHURN": ("🔴", "**CHURN:** Gestión de Retención"),
    "FINANZAS": ("🟢", "**FINANZAS:** Análisis Superstore"),
    "POLITICAS": ("📋", "**POLÍTICAS:** Base de Conocimiento"),
    "FUERA_SCOPE": ("⚫", "Fuera de alcance"),
}

CSS_VIOLTECH = """
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-left: 4px solid #2c3e50;
        padding: 0.8rem;
        margin-bottom: 0.5rem;
        color: #34495e;
    }
    .footer-container {
        font-size: 12px;
        color: gray;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #ddd;
    }
</style>
"""


# Funciones ligeras de historial mantenidas en el frontend
def cargar_historial():
    if RUTA_HISTORIAL.exists():
        try:
            with open(RUTA_HISTORIAL, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _sanear_para_persistencia(mensajes: list) -> list:
    """
    Devuelve una copia de los mensajes sin el Base64 de los gráficos.
    Se usa SOLO para lo que se escribe a disco o se reenvia al backend
    la sesión en memoria (st.session_state.mensajes) conserva el tag
    completo para que Streamlit pueda seguir renderizando la imagen mientras
    dure la sesión activa.
    """
    saneados = []
    for m in mensajes:
        contenido = m.get("contenido", m.get("content", ""))
        if "[IMG_B64:" in contenido:
            texto_visible = contenido.split("[IMG_B64:")[0].strip()
            contenido = f"{texto_visible}\n[Gráfico generado en un turno anterior - no se conserva en el historial]"
        nuevo_msg = dict(m)
        if "contenido" in nuevo_msg:
            nuevo_msg["contenido"] = contenido
        elif "content" in nuevo_msg:
            nuevo_msg["content"] = contenido
        saneados.append(nuevo_msg)
    return saneados


def guardar_historial(mensajes):
    mensajes_saneados = _sanear_para_persistencia(mensajes)
    with open(RUTA_HISTORIAL, "w", encoding="utf-8") as f:
        json.dump(mensajes_saneados, f, ensure_ascii=False, indent=4)


def render_footer():
    st.markdown(
        '<div class="footer-container">© 2026 ViolTech | Violet es una IA y puede cometer errores. En periodo de prueba</div>',
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="Violet | ViolTech",
        page_icon="imagen/avatar_ppal_violet.png",
        layout="wide",
    )

    st.markdown(CSS_VIOLTECH, unsafe_allow_html=True)

    # 1. ENCABEZADO
    (
        col1,
        col2,
    ) = st.columns([0.65, 0.15])
    with col1:
        st.image("imagen/encabezado_de_correo_banner.png", width="stretch")
    with col2:
        if st.button("ℹ️ Acerca de Violet", use_container_width=True):
            st.session_state.show_about = True

        if st.session_state.get("show_about", False):
            st.info("""
                ### Violet v1.0.0\n
                **Analista de Inteligencia de Negocios**\n\n
            
                Especialista en:\n
                - 📊 Churn Predictivo\n
                - 📈 Análisis Financiero\n
                - 📑 Políticas e Información Violtech\n
                - 🛡️ Gestión de Datos Segura\n\n
            
                Desarrollado por: Jennifer | 
                Alura ONE Challenge
                """)
            time.sleep(5)
            st.session_state.show_about = False
            st.rerun()

        if st.button("❓ Pregúntale a Violet", use_container_width=True):
            st.session_state.show_preguntas = True
        st.session_state.show_about = False

        if st.session_state.get("show_preguntas", False):
            st.info("""
                ### Ejemplos de consulta:
                - ¿Cuántos clientes tienen riesgo Alto? (Churn)
                - Crea un gráfico de la tasa de churn por tipo de contrato (Churn)
                - Genera un reporte ejecutivo de clientes en riesgo Alto (Churn)
                - ¿Cuántas transacciones pierden dinero? (Finanzas)
                - Genera un reporte financiero ejecutivo de Superstore (Finanzas)
                - ¿Cuál es la regla del 20%? (Políticas)
                - ¿Cómo funciona el router de Violet? (Política)   
            """)
            time.sleep(9)
            st.session_state.show_preguntas = False
            st.rerun()

    # 2. SIDEBAR
    with st.sidebar:
        st.markdown(
            """
        <div style="text-align:center; padding:1rem 0">
            <h2 style="color: #2c3e50; font-size: 2.9rem; margin: 0;">Violet</h2>
            <p style="color: #7f8c8d; font-size: 0.9rem;">Agente de Inteligencia de Negocios ViolTech</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
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
        st.markdown(
            "- **CHURN:** Gestión de Retención\n- **FINANZAS:** Análisis Superstore\n- **POLÍTICAS:** Base de Conocimiento"
        )
        st.markdown("---")

        if st.button("🗑️ Nueva conversación", use_container_width=True):
            st.session_state.mensajes = [
                {
                    "rol": "assistant",
                    "contenido": SALUDO_VIOLET,
                    "categoria": "POLITICAS",
                }
            ]
            st.session_state.ultima_categoria = None
            st.session_state.contexto_sesion = {}
            if RUTA_HISTORIAL.exists():
                RUTA_HISTORIAL.unlink()
            st.rerun()

    # 3. LÓGICA DE HISTORIAL Y SALUDO INICIAL
    if "mensajes" not in st.session_state:
        historial_guardado = cargar_historial()
        if not historial_guardado:
            historial_guardado = [
                {
                    "rol": "assistant",
                    "contenido": SALUDO_VIOLET,
                    "categoria": "POLITICAS",
                }
            ]
        st.session_state.mensajes = historial_guardado

    if "contexto_sesion" not in st.session_state:
        st.session_state.contexto_sesion = {}

    # Renderizar conversación
    for msg in st.session_state.mensajes:
        # 1. Definimos el avatar según el rol sin dibujar nada todavía
        avatar = (
            "imagen/avatar_ppal_violet.png"
            if msg["rol"] == "assistant"
            else "imagen/usuario.gif"
        )

        # 2. Dibujamos un solo contenedor con el rol y avatar correctos
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

    pregunta = st.chat_input("¿En qué puedo ayudarte?")

    if pregunta:
        # Mantén las llaves en español para que tu UI no se rompa al renderizar
        st.session_state.mensajes.append({"rol": "user", "contenido": pregunta})
        with st.chat_message("user", avatar="imagen/usuario.gif"):
            st.markdown(pregunta)

        with st.chat_message("assistant", avatar="imagen/avatar_ppal_violet.png"):
            with st.spinner("Violet está analizando..."):
                try:
                    # Traducimos el historial al formato que exige el backend
                    historial_saneado = _sanear_para_persistencia(
                        st.session_state.mensajes[:-1]
                    )
                    historial_api = []
                    for m in historial_saneado:
                        # Usamos .get() por si quedó algún diccionario viejo en memoria
                        rol_api = m.get("rol", m.get("role"))
                        contenido_api = m.get("contenido", m.get("content"))
                        historial_api.append(
                            {"role": rol_api, "content": contenido_api}
                        )

                    # Construimos el payload estructurado para FastAPI
                    payload = {
                        "pregunta": pregunta,
                        "historial": historial_api,
                        "contexto_sesion": st.session_state.contexto_sesion,
                    }

                    # Consumo asíncrono simulado mediante un request HTTP POST con timeout
                    endpoint = f"{URL_API_VIOLET.rstrip('/')}/api/v1/chat"
                    response = requests.post(endpoint, json=payload, timeout=180)

                    if response.status_code == 200:
                        data = response.json()
                        respuesta = data["respuesta"]
                        categoria = data["categoria"]
                        st.session_state.contexto_sesion = data.get(
                            "contexto_sesion", st.session_state.contexto_sesion
                        )
                    else:
                        data = {
                            "error": f"HTTP {response.status_code}",
                            "detalle": response.text,
                        }
                        respuesta = f"❌ Error en el servidor backend (Código {response.status_code}): {response.text}"
                        categoria = "FUERA_SCOPE"

                except requests.exceptions.RequestException as e:
                    data = {"error": "RequestException", "detalle": str(e)}
                    respuesta = f"❌ No se pudo conectar con el servidor de Violet en Render. Verifica la URL. Detalles: {str(e)}"
                    categoria = "FUERA_SCOPE"

                # Desplegar resultados de la API
                if "[IMG_B64:" in respuesta:
                    texto_resp, img_tag_resp = respuesta.split("[IMG_B64:")
                    b64_resp = img_tag_resp.replace("]", "").strip()
                    st.markdown(texto_resp)
                    try:
                        img_bytes_resp = base64.b64decode(b64_resp)
                        st.image(img_bytes_resp)
                    except Exception:
                        st.caption("⚠️ Error al renderizar la imagen en memoria.")
                else:
                    st.markdown(respuesta)

                with st.expander("🛠️ Ver JSON de depuración"):
                    st.write("Payload enviado a FastAPI:")
                    st.json(payload)  # Muestra lo que Streamlit mandó
                    st.write("Respuesta cruda del servidor:")
                    st.json(data)  # Muestra el JSON exacto que devolvió FastAPI

                if categoria in BADGES:
                    emoji, texto = BADGES[categoria]
                    st.caption(f"{emoji} {texto}")

        # Guardar estado con las llaves consistentes en español
        st.session_state.mensajes.append(
            {"rol": "assistant", "contenido": respuesta, "categoria": categoria}
        )
        guardar_historial(st.session_state.mensajes)
        st.rerun()

    render_footer()


if __name__ == "__main__":
    main()
