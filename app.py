
import os
import random
import re
import secrets

import sympy as sp
from flask import Flask, jsonify, request, session
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

app = Flask(__name__, static_folder="static", static_url_path="")
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)

X = sp.Symbol("x")
TRANSFORMACIONES = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)
MAX_LARGO = 100


class EntradaInvalida(Exception):
    """Error que se le puede mostrar al usuario."""


def normalizar(texto):
    """Limpia el texto y verifica que solo tenga caracteres seguros."""
    t = texto.strip().lower()
    t = t.replace("²", "^2").replace("³", "^3").replace("**", "^")
    t = t.replace("×", "*").replace("÷", "/").replace("−", "-").replace("–", "-")
    t = t.replace("[", "(").replace("]", ")").replace(",", ".")

    if not t:
        raise EntradaInvalida("Escribe una ecuación, por ejemplo: x^2 - 3x + 2 = 0")
    if len(t) > MAX_LARGO:
        raise EntradaInvalida("La ecuación es demasiado larga.")
    if not re.fullmatch(r"[0-9x+\-*/^().=\s√π]+", t):
        raise EntradaInvalida(
            "Solo se permite la variable x, números, + - * / ^ ( ) √ π y el signo =."
        )
    if t.count("=") > 1:
        raise EntradaInvalida("La ecuación solo puede tener un signo =.")

    if len(re.findall(r"\^", t)) != len(re.findall(r"\^\s*\d{1,2}(?![\d.])", t)):
        raise EntradaInvalida("Los exponentes deben ser números enteros, por ejemplo x^2.")
    if re.search(r"\^\s*\d+\s*\^", t):
        raise EntradaInvalida("No se permiten potencias encadenadas.")

    t = re.sub(r"√\s*(\d+(?:\.\d+)?|x|π)", r" sqrt(\1)", t)
    t = t.replace("√", " sqrt").replace("π", "(pi)")
    return t


def leer_lado(texto):
    try:
        return parse_expr(texto, local_dict={"x": X, "pi": sp.pi, "sqrt": sp.sqrt}, transformations=TRANSFORMACIONES)
    except Exception:
        raise EntradaInvalida("No pude interpretar la ecuación. Revisa los paréntesis y los signos.")


def obtener_polinomio(texto):
    """Normaliza una ecuación o función y devuelve el polinomio resultante."""
    t = normalizar(texto)
    izq, _, der = t.partition("=")
    expr = leer_lado(izq) - (leer_lado(der) if der.strip() else 0)

    try:
        expr = sp.expand(expr.xreplace({f: sp.Rational(str(f)) for f in expr.atoms(sp.Float)}))
        poli = sp.Poly(expr, X)
    except Exception:
        raise EntradaInvalida("La expresión debe ser un polinomio en x (sin x en denominadores ni dentro de raíces).")

    grado = poli.degree() if not poli.is_zero else 0
    return expr, poli, grado


def clasificar_funcion(texto):
    """Determina si una función es lineal, cuadrática o de otro tipo."""
    _, _, grado = obtener_polinomio(texto)

    if grado == 1:
        return {
            "tipo": "lineal",
            "grado": 1,
            "mensaje": "La función es lineal.",
        }
    if grado == 2:
        return {
            "tipo": "cuadratica",
            "grado": 2,
            "mensaje": "La función es cuadrática.",
        }
    if grado == 0:
        return {
            "tipo": "constante",
            "grado": 0,
            "mensaje": "La función es constante.",
        }
    return {
        "tipo": "no_aplica",
        "grado": grado,
        "mensaje": f"La función no es lineal ni cuadrática: es de grado {grado}.",
    }


def extraer_coeficientes(texto):
    """Pasa todo a un lado, expande y devuelve (a, b, c) exactos."""
    _, poli, grado = obtener_polinomio(texto)

    if grado > 2:
        raise EntradaInvalida(f"Esa ecuación es de grado {grado}, no cuadrática.")
    if grado < 2:
        raise EntradaInvalida("Esa ecuación no es cuadrática: después de simplificar, x² desaparece.")

    a, b, c = (poli.coeff_monomial(X**k) for k in (2, 1, 0))
    return a, b, c


def texto(valor):
    """Convierte una expresión de SymPy en texto legible."""
    s = re.sub(r"\bpi\b", "π", str(valor)).replace("**", "^")
    s = re.sub(r"sqrt\(", "√(", s)
    s = s.replace("*I", "i").replace("I", "i").replace("*", "·")
    return s


def forma_estandar(a, b, c):
    partes = []
    for coef, var in ((a, "x²"), (b, "x"), (c, "")):
        if coef == 0:
            continue
        signo = "-" if coef < 0 else "+"
        mag = abs(coef)
        if mag == 1 and var:
            num = ""
        elif var and not mag.is_integer:
            num = f"({texto(mag)})"
        else:
            num = texto(mag)
        cuerpo = num + var
        partes.append((signo, cuerpo))
    primero = partes[0]
    s = ("-" if primero[0] == "-" else "") + primero[1]
    for signo, cuerpo in partes[1:]:
        s += f" {signo} {cuerpo}"
    return s + " = 0"


def forma_lineal(a, b):
    partes = []
    for coef, variable in ((a, "x"), (b, "")):
        if coef == 0:
            continue
        signo = "-" if coef < 0 else "+"
        magnitud = abs(coef)
        numero = "" if variable and magnitud == 1 else texto(magnitud)
        partes.append((signo, numero + variable))

    primero = partes[0]
    resultado = ("-" if primero[0] == "-" else "") + primero[1]
    for signo, cuerpo in partes[1:]:
        resultado += f" {signo} {cuerpo}"
    return resultado


def resolver(a, b, c):
    d = sp.simplify(b**2 - 4 * a * c)
    signo = 0 if d == 0 else (1 if sp.N(d, 30) > 0 else -1)
    vertice = (-b / (2 * a), c - b**2 / (4 * a))
    raiz_d = sp.sqrt(d)
    exactas = [sp.simplify((-b + raiz_d) / (2 * a)), sp.simplify((-b - raiz_d) / (2 * a))]

    if signo > 0:
        tipo, exactas = "reales_distintas", exactas
    elif signo == 0:
        tipo, exactas = "real_doble", exactas[:1]
    else:
        tipo = "complejas"

    raices = []
    for r in exactas:
        num = complex(sp.N(r, 15))
        raices.append({
            "exacta": texto(r),
            "re": num.real,
            "im": num.imag,
        })

    pasos = [
        f"Forma estándar: {forma_estandar(a, b, c)}",
        f"Coeficientes: a = {texto(a)}, b = {texto(b)}, c = {texto(c)}",
        f"Discriminante: Δ = b² − 4ac = ({texto(b)})² − 4·({texto(a)})·({texto(c)}) = {texto(d)}",
        "Fórmula general: x = (−b ± √Δ) / (2a)",
        f"Vértice: ({texto(vertice[0])}, {texto(vertice[1])})",
    ]

    tipo_funcion = clasificar_funcion(f"{forma_estandar(a, b, c)}")
    return {
        "estandar": forma_estandar(a, b, c),
        "a": texto(a), "b": texto(b), "c": texto(c),
        "coef": {"a": float(a), "b": float(b), "c": float(c)},
        "discriminante": texto(d),
        "tipo": tipo,
        "tipo_funcion": tipo_funcion["tipo"],
        "tipo_funcion_label": tipo_funcion["mensaje"],
        "raices": raices,
        "vertice": {"x": float(vertice[0]), "y": float(vertice[1])},
        "pasos": pasos,
    }


@app.post("/api/definir_tipo")
def api_definir_tipo():
    datos = request.get_json(silent=True) or {}
    ecuacion = datos.get("ecuacion")
    if not isinstance(ecuacion, str):
        return jsonify(error="Envía la función como texto."), 400

    try:
        resultado = clasificar_funcion(ecuacion)
        return jsonify(resultado)
    except EntradaInvalida as error:
        return jsonify(error=str(error)), 400


@app.post("/api/juego/nueva")
def api_juego_nueva():
    a = random.choice([n for n in range(-8, 9) if n != 0])
    b = random.randint(-10, 10)
    session["juego_lineal"] = {"a": a, "b": b}
    valores = [{"x": x, "y": a * x + b} for x in range(-3, 4)]
    return jsonify(valores=valores)


@app.post("/api/juego/comprobar")
def api_juego_comprobar():
    objetivo = session.get("juego_lineal")
    if not objetivo:
        return jsonify(error="Inicia una nueva partida antes de comprobar."), 400

    datos = request.get_json(silent=True) or {}
    if not isinstance(datos, dict):
        return jsonify(error="Envía la respuesta como texto."), 400
    respuesta = datos.get("respuesta")
    if not isinstance(respuesta, str):
        return jsonify(error="Escribe una función lineal para comprobar."), 400

    try:
        _, poli, grado = obtener_polinomio(respuesta)
    except EntradaInvalida as error:
        return jsonify(error=str(error)), 400

    correcta = (
        grado == 1
        and poli.coeff_monomial(X) == objetivo["a"]
        and poli.coeff_monomial(1) == objetivo["b"]
    )
    return jsonify(correcta=correcta)


@app.post("/api/juego/finalizar")
def api_juego_finalizar():
    objetivo = session.pop("juego_lineal", None)
    if not objetivo:
        return jsonify(error="No hay una partida activa para finalizar."), 400

    funcion = f"f(x) = {forma_lineal(objetivo['a'], objetivo['b'])}"
    return jsonify(funcion=funcion)


@app.get("/")
def inicio():
    return app.send_static_file("index.html")


@app.post("/api/resolver")
def api_resolver():
    datos = request.get_json(silent=True) or {}
    ecuacion = datos.get("ecuacion")
    if not isinstance(ecuacion, str):
        return jsonify(error="Envía la ecuación como texto."), 400

    try:
        a, b, c = extraer_coeficientes(ecuacion)
        return jsonify(resolver(a, b, c))
    except EntradaInvalida as error:
        return jsonify(error=str(error)), 400


if __name__ == "__main__":
    app.run(debug=True)