# fase3_entrenamiento_svc.py

import logging
import numpy as np
from pathlib import Path
import joblib # Para guardar el modelo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuración ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_PROJECT_DIR = Path.cwd()
# Directorio donde se guardó el dataset de la fase anterior
DATASET_DIR = BASE_PROJECT_DIR / "dataset_final_vlm"
# Directorio donde se guardará el modelo final entrenado
MODEL_OUTPUT_DIR = BASE_PROJECT_DIR / "modelo_final"

# --- PASO 1: Cargar el Dataset ---
logger.info("===== INICIO FASE 3: ENTRENAMIENTO DEL CLASIFICADOR SVC =====")
logger.info("Cargando dataset desde los ficheros .npy...")

try:
    X = np.load(DATASET_DIR / "dataset_X.npy")
    y = np.load(DATASET_DIR / "dataset_y.npy")
    logger.info(f"Dataset cargado correctamente. Forma de X: {X.shape}, Forma de y: {y.shape}")
except FileNotFoundError:
    logger.error(f"Error: No se encontraron los ficheros 'dataset_X.npy' o 'dataset_y.npy' en la carpeta '{DATASET_DIR}'.")
    logger.error("Asegúrate de haber ejecutado el script 'generar_vectores.py' primero.")
    exit()

# --- PASO 2: División de Datos (Entrenamiento y Prueba) ---
# Dividimos el dataset para poder evaluar el modelo de forma objetiva.
# 80% para entrenar, 20% para probar.
# stratify=y es MUY importante para que la proporción de clases (FDT, MTF, TOR) sea la misma en ambos conjuntos.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
logger.info(f"Datos divididos. Muestras de entrenamiento: {len(X_train)}, Muestras de prueba: {len(X_test)}")

# --- PASO 3: Escalar los Datos ---
# Los modelos SVC son sensibles a la escala de los datos.
# Creamos el escalador y lo 'ajustamos' SÓLO con los datos de entrenamiento.
logger.info("Escalando los datos...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Aplicamos la misma transformación a los datos de prueba.
X_test_scaled = scaler.transform(X_test)

# --- PASO 4: Entrenar el Modelo SVC ---
# 'kernel='linear'' es un excelente punto de partida para datos de alta dimensión como los embeddings.
# 'C=1' es un buen valor por defecto para el parámetro de regularización.
# 'probability=True' nos permite obtener la confianza de cada predicción más adelante.
logger.info("Entrenando el modelo SVC...")
model = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
model.fit(X_train_scaled, y_train)
logger.info("¡Modelo entrenado con éxito!")

# --- PASO 5: Evaluar el Rendimiento del Modelo ---
logger.info("Evaluando el rendimiento del modelo con el conjunto de prueba...")
y_pred = model.predict(X_test_scaled)

# Precisión global
accuracy = accuracy_score(y_test, y_pred)
logger.info(f"\n--- Precisión Global del Modelo: {accuracy:.2%} ---")

# Reporte de clasificación detallado (precisión, recall, f1-score por clase)
logger.info("\n--- Reporte de Clasificación ---")
print(classification_report(y_test, y_pred))

# Matriz de Confusión: una forma visual de ver los aciertos y errores
logger.info("\n--- Matriz de Confusión ---")
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Matriz de Confusión')
plt.ylabel('Clase Real')
plt.xlabel('Clase Predicha')
plt.savefig(MODEL_OUTPUT_DIR / "matriz_de_confusion.png")

# --- PASO 6: Guardar la Herramienta Final (Modelo y Escalador) ---
logger.info("Guardando el modelo y el escalador para uso futuro...")
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Es CRUCIAL guardar también el escalador, ya que cualquier dato nuevo
# deberá ser escalado de la misma manera antes de hacer una predicción.
joblib.dump(model, MODEL_OUTPUT_DIR / "clasificador_svc.joblib")
joblib.dump(scaler, MODEL_OUTPUT_DIR / "escalador.joblib")

logger.info(f"Modelo guardado en: {MODEL_OUTPUT_DIR / 'clasificador_svc.joblib'}")
logger.info(f"Escalador guardado en: {MODEL_OUTPUT_DIR / 'escalador.joblib'}")
logger.info("===== FIN FASE 3: ENTRENAMIENTO COMPLETADO =====")