from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
CSV_ORIGINAL = ROOT / "data" / "grupo_12.csv"
CSV_LIMPIO = ROOT / "data" / "grupo_12_limpio.csv"



print("Primeras 5 filas:")
print(original.head())
print("\nForma original:", original.shape)
print("\nTipos originales:")
print(original.dtypes)
print(
    "\nValores de humedad que contienen %:",
    original["humedad_relativa_pct"].astype("string").str.contains("%", na=False).sum(),
)
print(
    "\nClases antes de limpiar:\n",
    original["riesgo_incendio"].value_counts().sort_index(),
)
print("\nNulos originales:")
print(original.isna().sum())
print("\nFilas duplicadas exactas:", original.duplicated().sum())


media_temperatura_original = pd.to_numeric(
    original["temperatura_c"], errors="coerce"
).mean()
print(
    "\nMedia original de temperatura:",
    round(media_temperatura_original, 3),
)


df = original.drop_duplicates().copy()

# 3. Limpia humedad: quita '%' y convierte la columna a número
df["humedad_relativa_pct"] = (
    df["humedad_relativa_pct"]
    .astype("string")
    .str.replace("%", "", regex=False)
    .str.strip()
)
df["humedad_relativa_pct"] = pd.to_numeric(
    df["humedad_relativa_pct"], errors="coerce"
)


temperaturas_validas = df["temperatura_c"].between(10, 45)
mediana_zaragoza = df.loc[
    (df["vereda"] == "Zaragoza") & temperaturas_validas,
    "temperatura_c",
].median()

df.loc[df["registro_id"] == "I070", "temperatura_c"] = mediana_zaragoza

media_temperatura_corregida = df["temperatura_c"].mean()
print("Temperatura sospechosa I070 corregida a:", mediana_zaragoza)
print(
    "Media de temperatura después de corregir I070:",
    round(media_temperatura_corregida, 3),
)


mediana_humedad = df["humedad_relativa_pct"].median()
df["humedad_relativa_pct"] = df["humedad_relativa_pct"].fillna(
    mediana_humedad
)


df.to_csv(CSV_LIMPIO, index=False)

print("\nForma después de limpiar:", df.shape)
print("Nulos finales:")
print(df.isna().sum())
print("Registros por clase:")
print(df["riesgo_incendio"].value_counts().sort_index())


sns.scatterplot(
    data=df,
    x="temperatura_c",
    y="humedad_relativa_pct",
    hue="riesgo_incendio",
)
plt.title("Temperatura y humedad según riesgo de incendio")
plt.xlabel("Temperatura (°C)")
plt.ylabel("Humedad relativa (%)")
plt.tight_layout()
plt.savefig(ROOT / "data" / "grafico_dispersion.png", dpi=150)
plt.close()


sns.boxplot(data=df, x="riesgo_incendio", y="temperatura_c")
plt.title("Temperatura por clase de riesgo")
plt.xlabel("Riesgo de incendio (0 = bajo, 1 = alto)")
plt.ylabel("Temperatura (°C)")
plt.tight_layout()
plt.savefig(ROOT / "data" / "grafico_boxplot.png", dpi=150)
plt.close()


columnas_x = ["temperatura_c", "humedad_relativa_pct"]
X = df[columnas_x].to_numpy(dtype=float)
y = df["riesgo_incendio"].to_numpy(dtype=float)

media_X = X.mean(axis=0)
desviacion_X = X.std(axis=0)
desviacion_X[desviacion_X == 0] = 1
X_std = (X - media_X) / desviacion_X


def sigmoide(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def predecir_probabilidad(X_entrada, w, b):
    return sigmoide(X_entrada @ w + b)


def perdida_logaritmica(y_real, probabilidades):
    epsilon = 1e-12
    probabilidades = np.clip(probabilidades, epsilon, 1 - epsilon)
    return -np.mean(
        y_real * np.log(probabilidades)
        + (1 - y_real) * np.log(1 - probabilidades)
    )


def exactitud(y_real, probabilidades):
    predicciones = (probabilidades >= 0.5).astype(int)
    return np.mean(predicciones == y_real)



np.random.seed(42)
w = np.zeros(X_std.shape[1])
b = 0.0

w_inicial = w.copy()
b_inicial = b
probabilidades_iniciales = predecir_probabilidad(X_std, w, b)
perdida_inicial = perdida_logaritmica(y, probabilidades_iniciales)

epocas = 1000
tasa_aprendizaje = 0.1
historial_perdida = []
historial_exactitud = []

for epoca in range(epocas):
    probabilidades = predecir_probabilidad(X_std, w, b)
    error = probabilidades - y

    dw = (X_std.T @ error) / len(y)
    db = np.mean(error)

    w -= tasa_aprendizaje * dw
    b -= tasa_aprendizaje * db

    probabilidades = predecir_probabilidad(X_std, w, b)
    historial_perdida.append(perdida_logaritmica(y, probabilidades))
    historial_exactitud.append(exactitud(y, probabilidades))

probabilidades_finales = predecir_probabilidad(X_std, w, b)
perdida_final = perdida_logaritmica(y, probabilidades_finales)
exactitud_final = exactitud(y, probabilidades_finales)
predicciones = (probabilidades_finales >= 0.5).astype(int)
errores = int(np.sum(predicciones != y))

print("\nPesos iniciales:", w_inicial, "sesgo:", b_inicial)
print("Pesos finales:", w, "sesgo:", b)
print("Pérdida inicial:", round(perdida_inicial, 4))
print("Pérdida final:", round(perdida_final, 4))
print("Exactitud final:", round(exactitud_final, 4))
print("Registros del lado equivocado:", errores, "de", len(y))
print("Media usada para estandarizar:", media_X)
print("Desviación usada para estandarizar:", desviacion_X)


plt.plot(historial_perdida)
plt.title("Curva de aprendizaje")
plt.xlabel("Época")
plt.ylabel("Pérdida logarítmica")
plt.tight_layout()
plt.savefig(ROOT / "data" / "curva_aprendizaje.png", dpi=150)
plt.close()


plt.figure()
sns.scatterplot(
    data=df,
    x="temperatura_c",
    y="humedad_relativa_pct",
    hue="riesgo_incendio",
)

if abs(w[1]) > 1e-10:
    temperaturas = np.linspace(
        df["temperatura_c"].min(),
        df["temperatura_c"].max(),
        200,
    )
    temp_std = (temperaturas - media_X[0]) / desviacion_X[0]
    humedad_std = -(w[0] * temp_std + b) / w[1]
    humedad = humedad_std * desviacion_X[1] + media_X[1]

    plt.plot(
        temperaturas,
        humedad,
        color="black",
        label="Frontera de decisión",
    )

plt.title("Frontera de decisión del perceptrón")
plt.xlabel("Temperatura (°C)")
plt.ylabel("Humedad relativa (%)")
plt.legend()
plt.tight_layout()
plt.savefig(ROOT / "data" / "frontera_decision.png", dpi=150)
plt.close()


nuevos = np.array([
    [36.0, 25.0],  # Se espera riesgo alto
    [21.0, 80.0],  # Se espera riesgo bajo
])
nuevos_std = (nuevos - media_X) / desviacion_X
probabilidades_nuevos = predecir_probabilidad(nuevos_std, w, b)

print("\nEjemplos nuevos [temperatura, humedad] y probabilidad de riesgo=1:")
for registro, probabilidad in zip(nuevos, probabilidades_nuevos):
    print(registro, "->", round(float(probabilidad), 4))
