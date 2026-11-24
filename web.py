import re
import numpy as np
import sympy as sp
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------- Métodos numéricos ----------------
def punto_medio(f, a, b, N):
    h = (b - a) / N
    suma = 0
    for i in range(N):
        suma += f(a + (i + 0.5) * h)
    return h * suma


def trapecio(f, a, b, N):
    h = (b - a) / N
    suma = (f(a) + f(b)) / 2
    for i in range(1, N):
        suma += f(a + i * h)
    return h * suma


def simpson_13(f, a, b, N):
    if N % 2 != 0:
        raise ValueError("N debe ser par para Simpson 1/3")
    h = (b - a) / N
    suma = f(a) + f(b)
    for i in range(1, N):
        x = a + i * h
        suma += 2 * f(x) if i % 2 == 0 else 4 * f(x)
    return (h / 3) * suma


def simpson_38(f, a, b, N):
    if N % 3 != 0:
        raise ValueError("N debe ser múltiplo de 3 para Simpson 3/8")
    h = (b - a) / N
    suma = f(a) + f(b)
    for i in range(1, N):
        x = a + i * h
        suma += 2 * f(x) if i % 3 == 0 else 3 * f(x)
    return (3 * h / 8) * suma


METODOS = {
    "Punto medio": punto_medio,
    "Trapecio": trapecio,
    "Simpson 1/3": simpson_13,
    "Simpson 3/8": simpson_38,
}

# ---------------- Parseo seguro de la función ----------------
x = sp.Symbol("x")
FUNCIONES = {"sin": sp.sin, "cos": sp.cos, "exp": sp.exp, "log": sp.log}
NOMBRES_PERMITIDOS = {"x", "sin", "cos", "exp", "log"}
CARACTERES_PERMITIDOS = re.compile(r"^[0-9a-z+\-*/().,^\s]+$")


def parsear_funcion(texto):
    texto = texto.strip().replace("^", "**")
    if not texto or len(texto) > 200:
        raise ValueError("Escribe una función válida (máx. 200 caracteres).")

    # Validar ANTES de sympify (que internamente usa eval)
    if not CARACTERES_PERMITIDOS.match(texto):
        raise ValueError("La expresión contiene caracteres no permitidos.")
    for nombre in re.findall(r"[a-z]+", texto):
        if nombre not in NOMBRES_PERMITIDOS:
            raise ValueError(f"Nombre no permitido: '{nombre}'.")

    expr = sp.sympify(texto, locals={"x": x, **FUNCIONES})

    if not expr.free_symbols.issubset({x}):
        raise ValueError("Solo puedes usar la variable x.")
    return expr


# ---------------- Gráfica ----------------
def graficar(f_vec, a, b, N, metodo, resultado):
    fig, ax = plt.subplots(figsize=(9, 5))

    xs = np.linspace(a, b, 600)
    ys = f_vec(xs)
    ax.plot(xs, ys, color="black", lw=2, label="f(x)")

    h = (b - a) / N
    nodos = a + h * np.arange(N + 1)
    color = "tab:blue"

    if N <= 200:
        if metodo == "Punto medio":
            medios = nodos[:-1] + h / 2
            ax.bar(medios, f_vec(medios), width=h, alpha=0.35,
                   color=color, edgecolor="navy")
        elif metodo == "Trapecio":
            fn = f_vec(nodos)
            for i in range(N):
                ax.fill([nodos[i], nodos[i], nodos[i + 1], nodos[i + 1]],
                        [0, fn[i], fn[i + 1], 0],
                        alpha=0.35, color=color, edgecolor="navy")
        else:
            paso = 2 if metodo == "Simpson 1/3" else 3
            for i in range(0, N, paso):
                xi = nodos[i:i + paso + 1]
                coef = np.polyfit(xi, f_vec(xi), paso)
                xx = np.linspace(xi[0], xi[-1], 60)
                yy = np.polyval(coef, xx)
                ax.fill_between(xx, 0, yy, alpha=0.35, color=color)
                ax.plot(xx, yy, color="navy", lw=1)
        ax.plot(nodos, f_vec(nodos), "o", color="red", ms=3, label="nodos")
    else:
        ax.fill_between(xs, 0, ys, alpha=0.25, color=color)

    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title(f"{metodo} con N = {N}  →  integral ≈ {resultado:.8f}")
    ax.grid(alpha=0.3)
    ax.legend()
    return fig


# ---------------- Interfaz ----------------
st.set_page_config(page_title="Integración numérica", page_icon="∫")
st.title("∫ Integración numérica")
st.caption("Funciones permitidas: polinomios, sin(x), cos(x), exp(x), log(x). "
           "Ejemplos: x**2, 3*x + 1, sin(x), exp(x), log(x)")

with st.sidebar:
    st.header("Parámetros")
    texto = st.text_input("f(x)", value="x**2")
    a = st.number_input("Límite izquierdo a", value=0.0, format="%.6f")
    b = st.number_input("Límite derecho b", value=1.0, format="%.6f")
    N = st.number_input("Número de subintervalos N", min_value=1,
                        max_value=100000, value=6, step=1)
    nombre = st.selectbox("Método", list(METODOS.keys()))
    calcular = st.button("Calcular", type="primary")

if calcular:
    try:
        expr = parsear_funcion(texto)

        if a >= b:
            raise ValueError("El límite inferior debe ser menor que el superior.")

        N = int(N)
        f = sp.lambdify(x, expr, modules=["numpy"])

        def f_vec(arr):
            arr = np.asarray(arr, dtype=float)
            return np.asarray(f(arr), dtype=float) * np.ones_like(arr)

        with np.errstate(all="ignore"):
            if not np.all(np.isfinite(f_vec(np.linspace(a, b, 400)))):
                raise ValueError("La función no está definida (o es infinita) "
                                 "en parte del intervalo. Revisa log(x) o "
                                 "divisiones entre cero.")

        with np.errstate(all="ignore"):
            resultado = float(METODOS[nombre](f, a, b, N))

        st.subheader("Resultado")
        col1, col2 = st.columns(2)
        col1.metric("Integral aproximada", f"{resultado:.10f}")

        try:
            exacto = float(sp.integrate(expr, (x, a, b)))
            col2.metric("Error absoluto", f"{abs(exacto - resultado):.3e}")
            st.write(f"Valor exacto (sympy): **{exacto:.10f}**")
        except Exception:
            st.info("No se pudo calcular el valor exacto para comparar.")

        st.latex(r"\int_{%s}^{%s} %s \, dx" % (a, b, sp.latex(expr)))
        st.write(f"**Método:** {nombre}  |  **N:** {N}  |  "
                 f"**Intervalo:** [{a}, {b}]")

        fig = graficar(f_vec, a, b, N, nombre, resultado)
        st.pyplot(fig)
        plt.close(fig)
        if N > 200:
            st.caption("Con N > 200 se omite el dibujo detallado de cada "
                       "subintervalo para que la gráfica sea legible.")

    except Exception as error:
        st.error(f"No se pudo calcular: {error}")
else:
    st.info("Escribe f(x), elige los límites, N y el método en la barra "
            "lateral, y pulsa **Calcular**.")