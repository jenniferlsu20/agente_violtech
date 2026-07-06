from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_cohere import ChatCohere
from app.config import COHERE_API_KEY

llm_router = ChatCohere(
    cohere_api_key=COHERE_API_KEY,
    model="command-r-08-2024",
    temperature=0.0,
    max_tokens=20,
)

PROMPT_ROUTER = PromptTemplate.from_template("""
Clasifica esta pregunta en UNA categoría. Responde SOLO la categoría, sin explicación.

Categorías:
- POLITICAS: Úsala para cualquier pregunta que involucre CONCEPTOS, TEORÍAS, DEFINICIONES,
SIGNIFICADOS o REGLAS (por ejemplo, "regla del 20%"). También aplica para estrategias de la 
empresa, manuales, glosarios y preguntas sobre la arquitectura o funcionamiento de Violet.
- CHURN: Úsala SOLO para consultar DATOS NUMÉRICOS, MÉTRICAS o ESTADÍSTICAS sobre retención 
de clientes, riesgo de abandono (churn, churn_prob, risk_level), telecomunicaciones, fibra, 
soporte técnico, o variables de usuarios (tenure, MonthlyCharges, TelcoVenezuela).
- FINANZAS: Úsala SOLO para consultar DATOS NUMÉRICOS, MÉTRICAS o REPORTES sobre reportes de 
ventas, ganancias, márgenes, o categorías de productos, sugcategorias de productos, descuentos, 
Technology, Furniture, Office Supplies, Consumer, Corporate, Home Office, Superstore, pérdidas, 
Tables.
- FUERA_SCOPE: Úsala SOLO Y ESTRICTAMENTE para temas que no tengan absolutamente nada que 
ver con análisis de datos, ViolTech, finanzas, telecomunicaciones o políticas de la empresa 
(ej. deportes, clima, recetas de cocina).

Pregunta: {pregunta}
Categoría:""")


def clasificar(pregunta: str, categoria_anterior: str = "FINANZAS") -> str:
    """
    Router jerárquico.
    categoria_anterior: Se usa para el contexto de visualizaciones cuando no se especifica.
    """
    p = pregunta.lower().strip()

    if any(
        accion in p for accion in ["gmail", "telegram", "correo", "email", "enviar"]
    ):
        return "ACCION_ENVIO"

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
        "probabilidad de abandono",
        "clientes en riesgo",
        "clientes de riesgo",
        "contrato mensual",
        "contrato anual",
        "tasa de churn",
        "tasa de abandono",
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
        "ventas totales",
        "ganancia total",
        "ganancias totales",
    ]
    for frase in frases_finanzas_exactas:
        if frase in p:
            return "FINANZAS"

    frases_exactas = [
        "qué significa",
        "que significa",
        "qué es ",
        "que es ",
        "qué son",
        "que son",
        "qué hace",
        "que hace",
        "cómo funciona routerdatos",
        "como funciona router",
        "cómo funciona violet",
        "como funciona violet",
        "cómo se define",
        "como se define",
        "definición de",
        "definicion de",
        "explica qué",
        "explica que",
        "qué herramientas",
        "que herramientas",
        "qué puede",
        "que puede",
        "para qué sirve",
        "para que sirve",
        "cuál es la diferencia",
        "cual es la diferencia",
        "qué diferencia",
        "que diferencia",
        "qué es violet",
        "que es violet",
        "cómo funciona violet",
        "como funciona violet",
        "qué es violet",
        "que es violet",
        "manual de violet",
        "manual del agente",
        "política de retención",
        "politica de retencion",
        "política de descuentos",
        "politica de descuentos",
        "regla del 20",
        "regla del 20%",
        "umbral del 20",
    ]
    for frase in frases_exactas:
        if frase in p:
            return "POLITICAS"

    # ── palabras clave ──────────────────────────────────────────
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
        return categoria_anterior

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
