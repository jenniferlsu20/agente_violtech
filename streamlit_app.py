import streamlit as st
import time
import requests
import json
import pathlib

# URL de tu API de FastAPI.
# En local usas "http://127.0.0.1:8000", cuando despliegues en Render la cambias por tu URL de producción.
URL_API_VIOLET = st.sidebar.text_input(
    "URL del Backend (Render)", value="http://127.0.0.1:8000"
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


def guardar_historial(historial):
    with open(RUTA_HISTORIAL, "w", encoding="utf-8") as f:
        json.dump(historial, f, ensure_ascii=False, indent=4)


def render_footer():
    st.markdown(
        '<div class="footer-container">© 2026 ViolTech | Violet es una IA y puede cometer errores. En periodo de prueba</div>',
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="Violet — ViolTech",
        page_icon="imagen/avatar_ppal_violet.png",
        layout="wide",
    )

    st.markdown(CSS_VIOLTECH, unsafe_allow_html=True)

    # 1. ENCABEZADO
    col1, col2 = st.columns([0.85, 0.15])
    with col1:
        st.image("imagen/encabezado_de_correo_banner.png", width=None)
    with col2:
        if st.button("ℹ️ Acerca de Violet", use_container_width=True):
            st.session_state.show_about = True

        if st.session_state.get("show_about", False):
            st.info(
                """### Violet v1.0.0\n**Analista de Inteligencia de Negocios**\n\nDesarrollado por: Jennifer | Alura ONE Challenge"""
            )
            time.sleep(3)
            st.session_state.show_about = False
            st.rerun()

    # 2. SIDEBAR
    with st.sidebar:
        st.markdown(
            '<h2 style="text-align:center;">Violet</h2>', unsafe_allow_html=True
        )
        if st.button("🗑️ Nueva conversación", use_container_width=True):
            st.session_state.mensajes = []
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

    # Renderizar conversación
    for msg in st.session_state.mensajes:
        with st.chat_message(msg["rol"]):
            st.markdown(msg["contenido"])
            if msg.get("categoria") in BADGES and msg["rol"] == "assistant":
                emoji, texto = BADGES[msg["categoria"]]
                st.caption(f"{emoji} {texto}")

    # 4. PROCESAMIENTO VÍA API (RENDERING STATLESS)
    pregunta = st.chat_input("¿En qué puedo ayudarte?")
    if pregunta:
        st.session_state.mensajes.append({"rol": "user", "contenido": pregunta})
        with st.chat_message("user"):
            st.markdown(pregunta)

        with st.chat_message("assistant"):
            with st.spinner("Violet está analizando..."):
                try:
                    # Construimos el payload estructurado para FastAPI (main.py)
                    payload = {
                        "pregunta": pregunta,
                        "historial": st.session_state.mensajes[
                            :-1
                        ],  # Enviamos el historial previo
                    }

                    # Consumo asíncrono simulado mediante un request HTTP POST con timeout
                    endpoint = f"{URL_API_VIOLET.rstrip('/')}/api/v1/chat"
                    response = requests.post(endpoint, json=payload, timeout=60)

                    if response.status_code == 200:
                        data = response.json()
                        respuesta = data["respuesta"]
                        categoria = data["categoria"]
                    else:
                        respuesta = f"❌ Error en el servidor backend (Código {response.status_code}): {response.text}"
                        categoria = "FUERA_SCOPE"

                except requests.exceptions.RequestException as e:
                    respuesta = f"❌ No se pudo conectar con el servidor de Violet en Render. Verifica la URL. Detalles: {str(e)}"
                    categoria = "FUERA_SCOPE"

                # Desplegar resultados de la API
                st.markdown(respuesta)
                if categoria in BADGES:
                    emoji, texto = BADGES[categoria]
                    st.caption(f"{emoji} {texto}")

        # Guardar estado y persistir de forma ligera
        st.session_state.mensajes.append(
            {"rol": "assistant", "contenido": respuesta, "categoria": categoria}
        )
        guardar_historial(st.session_state.mensajes)
        st.rerun()

    render_footer()


if __name__ == "__main__":
    main()
