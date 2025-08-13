# fase4_inferencia.py

import logging
from pathlib import Path
import joblib
import pydicom
import numpy as np
from PIL import Image as PilImage

# --- Importaciones para el modelo ---
import torch
import torchvision.models as models
import torchvision.transforms as transforms

# --- Configuración ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_PROJECT_DIR = Path.cwd()
# Directorio donde se encuentra el modelo guardado
MODEL_DIR = BASE_PROJECT_DIR / "modelo_final"
# --- ¡IMPORTANTE! Aquí pones la ruta a la nueva imagen que quieres clasificar ---
RUTA_IMAGEN_NUEVA = BASE_PROJECT_DIR / "imagen_para_probar.dcm"

# --- PASO 1: REUTILIZAR LA LÓGICA DE EXTRACCIÓN DE VECTORES ---
# Es buena práctica tener esta lógica en un solo lugar. Aquí la replicamos para
# tener un script autónomo.

logger.info("Cargando modelo ResNet50 para extracción de características...")
try:
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    feature_extractor = torch.nn.Sequential(*list(resnet50.children())[:-1])
    feature_extractor.eval()
    logger.info("Modelo ResNet50 cargado.")
except Exception as e:
    logger.error(f"No se pudo cargar el modelo ResNet50. Error: {e}")
    exit()

# Definimos la transformación necesaria para ResNet50
preprocess_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extraer_vector_de_un_dicom(filepath: Path, crop_pixels: int = 20) -> np.ndarray:
    """
    Toma la ruta de un fichero DICOM, lo procesa y devuelve su vector de características.
    """
    logger.info(f"Procesando imagen: {filepath.name}")
    ds = pydicom.dcmread(str(filepath), force=True)
    pixel_array = ds.pixel_array

    min_val, max_val = np.min(pixel_array), np.max(pixel_array)
    img_array_8bit = np.interp(pixel_array, (min_val, max_val), (0, 255)).astype(np.uint8) if max_val > min_val else np.zeros_like(pixel_array, dtype=np.uint8)
    
    pil_img = PilImage.fromarray(img_array_8bit)
    
    bbox = pil_img.getbbox()
    if bbox: pil_img = pil_img.crop(bbox)
    
    w, h = pil_img.size
    if w > crop_pixels * 2 and h > crop_pixels * 2:
        pil_img = pil_img.crop((crop_pixels, crop_pixels, w - crop_pixels, h - crop_pixels))
    
    if pil_img.size[0] == 0 or pil_img.size[1] == 0:
        raise ValueError("La imagen no tiene contenido para procesar después del recorte.")

    # Aplicar transformaciones y extraer vector
    img_tensor = preprocess_transform(pil_img)
    batch_t = torch.unsqueeze(img_tensor, 0)
    with torch.no_grad():
        features = feature_extractor(batch_t)
    
    return features.squeeze().numpy()


# --- PASO 2: EJECUTAR LA INFERENCIA ---
def ejecutar_inferencia():
    """
    Función principal que carga el modelo y clasifica una nueva imagen.
    """
    logger.info("===== INICIO FASE 4: INFERENCIA SOBRE UNA NUEVA IMAGEN =====")

    # Comprobar si existe la imagen a clasificar
    if not RUTA_IMAGEN_NUEVA.exists():
        logger.error(f"La imagen a clasificar no se encuentra en: {RUTA_IMAGEN_NUEVA}")
        logger.error("Por favor, crea una imagen de prueba o ajusta la variable 'RUTA_IMAGEN_NUEVA'.")
        # Creamos una imagen dummy para que el ejemplo pueda correr
        logger.info("Creando una imagen DICOM de prueba vacía ('sin objeto')...")
        pydicom.dcmwrite(RUTA_IMAGEN_NUEVA, pydicom.dataset.Dataset())


    # Cargar el clasificador y el escalador guardados
    try:
        logger.info(f"Cargando modelo desde {MODEL_DIR}...")
        model = joblib.load(MODEL_DIR / "clasificador_svc.joblib")
        scaler = joblib.load(MODEL_DIR / "escalador.joblib")
        logger.info("Modelo y escalador cargados correctamente.")
    except FileNotFoundError:
        logger.error(f"Error: No se encontró el modelo o el escalador en la carpeta '{MODEL_DIR}'.")
        logger.error("Asegúrate de haber ejecutado el script 'fase3_entrenamiento_svc.py' primero.")
        return

    try:
        # 1. Extraer el vector de la nueva imagen
        vector_nuevo = extraer_vector_de_un_dicom(RUTA_IMAGEN_NUEVA)
        logger.info(f"Vector extraído. Dimensiones: {vector_nuevo.shape}")

        # 2. Escalar el vector
        # El escalador y el modelo esperan un array 2D, por eso usamos reshape(1, -1)
        vector_nuevo_2d = vector_nuevo.reshape(1, -1)
        vector_escalado = scaler.transform(vector_nuevo_2d)
        logger.info("Vector escalado correctamente.")

        # 3. Realizar la predicción
        prediccion = model.predict(vector_escalado)
        probabilidades = model.predict_proba(vector_escalado)
        
        clases_modelo = model.classes_
        
        logger.info("\n--- PREDICCIÓN FINAL ---")
        print(f"\n✅ La imagen ha sido clasificada como: '{prediccion[0]}'")
        
        print("\n📊 Confianza por clase:")
        for i, clase in enumerate(clases_modelo):
            print(f"  - {clase}: {probabilidades[0][i]:.2%}")

    except Exception as e:
        logger.error(f"Ha ocurrido un error durante la inferencia: {e}", exc_info=True)


if __name__ == "__main__":
    ejecutar_inferencia()