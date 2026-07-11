import io
import re
import base64
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import asyncio
import json
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.tools import PythonAstREPLTool
from app.envio import procesar_confirmacion_envio
from app.config import TELEGRAM_CHAT_ID


def crear_herramientas(df: pd.DataFrame, vector_store, nombre_df: str, llm):
    """
    Fábrica de herramientas para el agente.
    Se inyecta el DataFrame, el Vector Store y el LLM para mantener el módulo puro.
    """

    @tool("Politicas ViolTech", return_direct=True)
    def buscar_politicas(pregunta: str) -> str:
        """
        Consulta los documentos de política interna de ViolTech.
        """
        try:
            docs = vector_store.invoke(pregunta)
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
        """Información estructural del DataFrame activo."""
        try:
            plantilla = PromptTemplate.from_template(
                "Eres Violet, analista de ViolTech. Responde en español con tono cálido y profesional.\n"
                "Dataset activo: {nombre_df}\nDimensiones: {shape}\n"
                "Columnas y tipos:\n{columnas}\nNulos por columna:\n{nulos}\n"
                "Duplicados: {duplicados}\n\nPregunta: {pregunta}\n\n"
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
        """Estadísticas descriptivas completas."""
        try:
            resumen = df.describe(include="number").transpose().to_string()
            plantilla = PromptTemplate.from_template(
                "Eres Violet. Pregunta: {pregunta}\nEstadísticas descriptivas:\n{resumen}\n\n"
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
        Genera visualizaciones automáticas del DataFrame y devuelve Base64.
        """
        # Usamos el contexto real que nos pasa el Router
        contexto = "CHURN" if "Churn" in nombre_df else "FINANZAS"

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

        if contexto == "FINANZAS" and any(
            x in pregunta.lower() for x in ["tenure", "contrato"]
        ):
            return "Error: Los datos de Finanzas (Superstore) no tienen información de 'tenure' o 'contrato'. Por favor, solicita un gráfico basado en 'ganancia', 'ventas' o 'descuentos'."

        columnas_reales = datasets[contexto]

        columnas_str = ", ".join(columnas_reales)

        try:
            plantilla = PromptTemplate.from_template(
                "Eres experto en Data Viz con Python. Analiza la solicitud y los datos.\n"
                "Genera código Python válido para graficar. df ya está cargado.\n"
                "Dataset: {contexto}\nColumnas: {columnas}\nSolicitud: {pregunta}\n"
                "Solo código, sin plt.show()."
            )
            codigo_bruto = (plantilla | llm | StrOutputParser()).invoke(
                {"pregunta": pregunta, "contexto": contexto, "columnas": columnas_str}
            )

            # Limpieza: buscamos código entre backticks si existe
            patron = r"```(?:python)?\s*(.*?)\s*```"
            match = re.search(patron, codigo_bruto, re.DOTALL | re.IGNORECASE)
            codigo = match.group(1).strip() if match else codigo_bruto.strip()
            codigo = codigo.replace("plt.show()", "")

            exec(codigo, {"df": df, "plt": plt, "sns": sns, "pd": pd}, {})
            fig = plt.gcf()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode("utf-8")
            plt.close("all")

            return f"📊 **Gráfico generado exitosamente**\n[IMG_B64:{img_base64}]"
        except Exception as e:
            return f"❌ Error generando gráfico: {str(e)}"

    @tool("Consulta Churn", return_direct=True)
    def consulta_rapida_churn(pregunta: str) -> str:
        """
        Útil para responder consultas sobre abandono de clientes (Churn) y niveles de riesgo.

        IMPORTANTE: El dataset tiene estas columnas:
        customerID, tenure, MonthlyCharges, TotalCharges, Churn, churn_prob,
        risk_level, tenure_grupo, es_segmento_critico, servicios_valor_agregado.

        Reglas de oro:
        1. Para medir el abandono, usa la columna 'Churn' (Yes/No).
        2. Para el nivel de riesgo, usa la columna 'risk_level' (Alto, Medio, Bajo).
        3. Si preguntan por antigüedad, usa 'tenure' o 'tenure_grupo'.
        """
        try:
            p = pregunta.lower()

            # 1. ANÁLISIS DE RIESGO (Enriquecido)
            if any(k in p for k in ["riesgo", "risk", "nivel"]):
                dist = df.get("risk_level", pd.Series()).value_counts()
                lineas = []
                total = len(df)
                for nivel in ["Alto", "Medio", "Bajo"]:
                    n = int(dist.get(nivel, 0))
                    pct = round(n / total * 100, 1) if total > 0 else 0
                    lineas.append(f"• Riesgo **{nivel}**: {n:,} clientes ({pct}%)")
                return "### Distribución de Riesgo:\n" + "\n".join(lineas)

            # 2. ANÁLISIS DE CHURN (Abandono)
            if any(
                k in p
                for k in ["churn", "abandono", "se van", "retención", "retencion"]
            ):
                # Convertimos a string, minúsculas y eliminamos espacios laterales
                df_temp = df["Churn"].astype(str).str.lower().str.strip()

                # Ahora contamos cuántos coinciden con "yes" y "no" (o "1" y "0")
                # Adaptado para aceptar ambas posibilidades
                si = len(df_temp[df_temp.isin(["yes", "1", "true"])])
                no = len(df_temp[df_temp.isin(["no", "0", "false"])])

                total = len(df)
                tasa = round((si / total) * 100, 1) if total > 0 else 0

                return (
                    f"### Resumen de Abandono (Churn):\n"
                    f"- Clientes que abandonaron: {si:,}\n"
                    f"- Clientes activos: {no:,}\n"
                    f"- Tasa de abandono global: **{tasa}%**"
                )

            # 3. FALLBACK: RESUMEN GENERAL
            return f"Total de clientes analizados en Churn: **{len(df):,}**."

        except Exception as ex:
            return f"Error técnico al consultar el dataset de Churn: {str(ex)}"

    @tool("Reporte Clientes en Riesgo", return_direct=False)
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
                "*Generado por Violet · ViolTech — Tu Agente Aliado*",
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
                    urgencia = "🔴 ALTA"
                elif prob >= 60:
                    urgencia = "🟡 MEDIA"
                else:
                    urgencia = "🟢 BAJA"

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

            return texto_final

        except Exception as ex:
            return f"Error generando reporte: {str(ex)}"

    @tool("Reporte Financiero Ejecutivo", return_direct=False)
    def reporte_financiero_ejecutivo(parametros: str) -> str:
        """
        Genera un reporte financiero detallado del dataset de Superstore.
        Argumento 'parametros': Palabras clave de la consulta del usuario.
        """
        global ultimo_reporte_generado

        try:
            # Limpiamos espacios en blanco de los nombres de columnas
            df.columns = df.columns.str.strip()

            # Validación crítica antes de operar
            if "Sales" not in df.columns or "Profit" not in df.columns:
                columnas_detectadas = df.columns.tolist()
                return f"Error: Dataset incorrecto. Columnas encontradas: {columnas_detectadas}"
            # ----------------------------

            p = parametros.lower() if parametros else ""

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
            mejor_segmento = seg.idxmax()

            # Estructura del Reporte
            lineas = [
                "## 📊 Reporte Financiero Ejecutivo - Superstore",
                "*Generado por Violet · ViolTech*",
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
                f"3. **Segmentación**: El segmento **{mejor_segmento}** reporta la mayor ganancia total.",
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

            ultimo_reporte_generado = texto_final

            return texto_final

        except Exception as ex:
            return f"Error al generar el reporte financiero: {str(ex)}"

    class EstadoBot:
        ultimo_reporte_financiero = ""
        ultimo_reporte_churn = ""

    GRUPO_VIOLET = TELEGRAM_CHAT_ID

    @tool("Enviar Reporte a Canal", return_direct=False)
    def tool_enviar_reporte(parametros_json: str) -> str:
        """
        Envía el último reporte generado por el sistema.

        ¡IMPORTANTE! El Action Input DEBE ser estrictamente un string JSON válido.
        Canales aceptados: 'telegram' o 'gmail'.

        Ejemplos de Action Input:
        - Para Telegram: {"canal": "telegram", "destino": "TELEGRAM_CHAT_ID"}
        - Para Gmail: {"canal": "gmail", "destino": "correo@ejemplo.com"}

        NO incluyas el texto del reporte en el JSON. El sistema ya lo tiene en memoria.
        """
        # 1. Parsear el JSON — limpiar markdown que el LLM pueda añadir
        try:
            texto_limpio = parametros_json.strip("`").replace("json\n", "").strip()
            # Extraer SOLO el objeto JSON válido: desde el primer '{' hasta
            # su '}' de cierre correspondiente — ignora cualquier carácter
            inicio = texto_limpio.find("{")
            fin = texto_limpio.rfind("}")
            if inicio == -1 or fin == -1 or fin < inicio:
                raise json.JSONDecodeError("No se encontró un objeto JSON", texto_limpio, 0)

            texto_json = texto_limpio[inicio:fin + 1]
            argumentos = json.loads(texto_json)
        except json.JSONDecodeError:
            return (
                'Error de formato en el JSON recibido. Vuelve a intentar con '
                'EXACTAMENTE este formato, sin texto adicional antes ni después: '
                '{"canal": "gmail", "destino": "usuario@gmail.com"}'
            )
            
        canal = argumentos.get("canal")
        destino = argumentos.get("destino")

        if not canal or not destino:
            return "Error: Faltan las claves 'canal' o 'destino' en el JSON."

        canal = canal.lower().strip()
        if canal not in ("telegram", "gmail"):
            return f"Error: Canal '{canal}' no soportado. Usa 'telegram' o 'gmail'."

        if canal == "telegram" and destino == "TELEGRAM_CHAT_ID":
            destino = GRUPO_VIOLET

        # 2. Fuente única de verdad para el reporte activo — EstadoBot
        if EstadoBot.ultimo_reporte_churn:
            tipo = "churn"
        elif EstadoBot.ultimo_reporte_financiero:
            tipo = "financiero"
        else:
            return "❌ No hay ningún reporte generado previamente para enviar."

        contexto_sesion = {
            "tipo_reporte_activo": tipo,
            f"ultimo_reporte_{tipo}": (
                EstadoBot.ultimo_reporte_churn
                if tipo == "churn"
                else EstadoBot.ultimo_reporte_financiero
            ),
        }

        # 3. procesar_confirmacion_envio ya genera el PDF
        mensaje, _ = asyncio.run(
            procesar_confirmacion_envio(
                canal=canal, destino=destino, contexto_sesion=contexto_sesion
            )
        )

        # 4. Limpiar el reporte enviado para no reenviarlo por error
        if tipo == "churn":
            EstadoBot.ultimo_reporte_churn = ""
        else:
            EstadoBot.ultimo_reporte_financiero = ""

        return mensaje

    @tool("Consulta Finanzas", return_direct=True)
    def consulta_rapida_finanzas(pregunta: str) -> str:
        """
        Útil para responder preguntas sobre métricas financieras del Superstore.
        IMPORTANTE: El dataset tiene EXACTAMENTE estas columnas: Row ID, Order ID, Order Date,
        Ship Date, Ship Mode, Customer ID, Customer Name, Segment, Country, City, State,
        Postal Code, Region, Product ID, Category, Sub-Category, Product Name, Sales,
        Quantity, Discount, Profit, margen_pct, es_perdida, rango_descuento, dias_envio,
        anio_orden, mes_orden, periodo.

        Reglas de oro:
        1. Si el usuario busca productos específicos (ej. 'Tablets', 'Phones'), filtra por 'Sub-Category'.
        2. 'Profit' negativo indica pérdida; positivo indica ganancia.
        3. 'Sales' indica los ingresos totales.
        """

        # Convertimos a minúsculas para hacer un cruce (matching) seguro
        pregunta_lower = pregunta.lower()

        try:
            # 1. BÚSQUEDA DINÁMICA POR SUB-CATEGORÍA (Ej: "tablets", "phones", "art")
            # Obtenemos los valores únicos de la columna y verificamos si están en la pregunta
            subcategorias_unicas = [
                str(sub).lower() for sub in df["Sub-Category"].dropna().unique()
            ]
            sub_encontradas = [
                sub for sub in subcategorias_unicas if sub in pregunta_lower
            ]

            if sub_encontradas:
                resultados = []
                for sub in sub_encontradas:
                    # Filtramos el DataFrame ignorando el case sensitive
                    filtro = df[df["Sub-Category"].str.lower() == sub]
                    ventas = filtro["Sales"].sum()
                    ganancia = filtro["Profit"].sum()

                    estado_financiero = "Pérdida" if ganancia < 0 else "Ganancia"

                    resultados.append(
                        f"- {sub.capitalize()}: Ventas ${ventas:,.2f} | {estado_financiero} ${ganancia:,.2f}"
                    )
                return "**Análisis por Sub-Categoría:**\n" + "\n".join(resultados)

            # 2. BÚSQUEDA A NIVEL CATEGORÍA PRINCIPAL
            if any(
                k in pregunta_lower
                for k in ["categoría", "category", "categorías", "categories"]
            ):
                stats = df.groupby("Category")[["Sales", "Profit"]].sum()

                # Formateo elegante de la tabla
                stats["Sales"] = stats["Sales"].apply(lambda x: f"${x:,.2f}")
                stats["Profit"] = stats["Profit"].apply(lambda x: f"${x:,.2f}")

                return "**Resumen por Categoría:**\n" + stats.to_string()

            # 3. FALLBACK: RESUMEN GENERAL (Si no se especifica producto/categoría)
            ventas_totales = df["Sales"].sum()
            ganancia_total = df["Profit"].sum()
            return (
                f"**Resumen General del Superstore:**\n"
                f"Ventas Totales: ${ventas_totales:,.2f} | Ganancia Total: ${ganancia_total:,.2f}"
            )

        except Exception as ex:
            return f"Error técnico al consultar el dataset financiero: {str(ex)}"

    repl = PythonAstREPLTool(locals={"df": df, "pd": pd})

    @tool("Calculos Python")
    def calculos_python(input_codigo: str) -> str:
        """Ejecuta codigo Python arbitrario."""
        limpio = re.sub(r"```python|```", "", input_codigo).strip()
        try:
            return repl.run(limpio)
        except Exception as e:
            return f"Error de ejecución: {str(e)}"

    return [
        buscar_politicas,
        resumen_estadistico,
        informacion_dataset,
        generar_grafico,
        consulta_rapida_churn,
        reporte_clientes_riesgo,
        reporte_financiero_ejecutivo,
        tool_enviar_reporte,
        consulta_rapida_finanzas,
        calculos_python,
    ]
