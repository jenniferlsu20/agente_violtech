import io
import re
import base64
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.tools import PythonAstREPLTool


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

    @tool("Reporte Clientes en Riesgo", return_direct=True)
    def reporte_clientes_riesgo(parametros: str) -> str:
        """Reporte ejecutivo de clientes en riesgo."""
        try:
            nivel = next(
                (
                    n
                    for n in ["Alto", "Medio", "Bajo"]
                    if n.lower() in parametros.lower()
                ),
                "Alto",
            )
            df_riesgo = (
                df[df["risk_level"] == nivel]
                .sort_values("churn_prob", ascending=False)
                .head(10)
            )

            lineas = [f"## 📋 Reporte Clientes en Riesgo {nivel}", "---"]
            for i, row in enumerate(df_riesgo.itertuples(), 1):
                lineas.append(
                    f"**{i}. {row.customerID}** | Prob: **{row.churn_prob*100:.1f}%**"
                )

            lineas.append("\n¿Quieres un gráfico o enviarlo por PDF?")
            return "\n".join(lineas)
        except Exception as ex:
            return f"Error generando reporte: {str(ex)}"

    @tool("Reporte Financiero Ejecutivo", return_direct=True)
    def reporte_financiero_ejecutivo(parametros: str) -> str:
        """Reporte financiero de Superstore."""
        try:
            total_ventas = df["Sales"].sum()
            total_ganancia = df["Profit"].sum()
            lineas = [
                "## 📊 Reporte Financiero Ejecutivo",
                f"Ventas Totales: **${total_ventas:,.2f}**",
                f"Ganancia Neta: **${total_ganancia:,.2f}**",
                "\n¿Gráfico o PDF?",
            ]
            return "\n".join(lineas)
        except Exception as ex:
            return f"Error: {str(ex)}"

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
        informacion_dataset,
        generar_grafico,
        consulta_rapida_churn,
        reporte_clientes_riesgo,
        reporte_financiero_ejecutivo,
        consulta_rapida_finanzas,
        calculos_python,
    ]
