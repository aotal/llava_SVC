# fase2_generate_vectors_vlm.py (COMPLETE VERSION WITH PROMPT-BASED DIFFERENTIATED AUGMENTATION)

import asyncio
import logging
from pathlib import Path
import shutil
import pydicom
from typing import List, Tuple
import numpy as np
from PIL import Image as PilImage
import base64
import io
import random

# --- NEW! Import for the Ollama client ---
import ollama

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODELO_VLM = 'llava-phi3'
BASE_PROJECT_DIR = Path.cwd()
INPUT_DIR = BASE_PROJECT_DIR / "datos_etiquetados" # Labeled data in subfolders
OUTPUT_PARENT_DIR = BASE_PROJECT_DIR / "output_processed_vlm"
DATASET_FINAL_DIR = BASE_PROJECT_DIR / "dataset_final_vlm"
ERROR_DIR = OUTPUT_PARENT_DIR / "CONVERSION_ERRORS"

# --- Configuration for prompt-based augmentation ---
DESCRIPTIONS_PER_IMAGE = 5
VLM_TEMPERATURE = 0.7

# --- NEW! Differentiated prompt lists ---
OBJECT_PROMPTS = [
    "Describe the geometric shape of the main object in this radiographic image.",
    "In which part of the image is the object located? Be brief.",
    "Describe the main object's orientation. Is it vertical, horizontal, or tilted?",
    "Focus on the object's contour and describe it in a few words.",
    "What is the most prominent visual feature of the object in the image?",
    "Describe the silhouette of the object appearing in this grayscale image.",
]

BACKGROUND_PROMPTS = [
    "Describe the general appearance of this image. Does it contain any clear objects?",
    "Analyze the texture and background pattern in this image. Is it uniform?",
    "Describe this image as if there were no object of interest in it.",
    "Describe the general distribution of light and dark areas in the image.",
    "If the image were empty, how would you describe it?",
    "Look for subtle patterns or artifacts in the background and describe them briefly.",
]

# --- Functions to interact with the VLM ---
def pil_to_base64(pil_img: PilImage) -> str:
    """Converts a Pillow library image to a base64 string."""
    buffered = io.BytesIO()
    img_to_save = pil_img.convert("L")
    img_to_save.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

async def extract_vector_with_vlm(pil_img: PilImage, prompt: str, temperature: float) -> np.ndarray | None:
    """Takes a PIL image, generates a description using the VLM, and returns its embedding."""
    try:
        encoded_image = pil_to_base64(pil_img)
        
        chat_response = await ollama.AsyncClient().chat(
            model=MODELO_VLM,
            messages=[{'role': 'user', 'content': prompt, 'images': [encoded_image]}],
            options={'temperature': temperature}
        )
        description = chat_response['message']['content']
        
        embedding_response = await ollama.AsyncClient().embeddings(
            model=MODELO_VLM, prompt=description
        )
        return np.array(embedding_response['embedding'])
    except Exception as e:
        logger.error(f"Error during interaction with Ollama: {e}")
        return None

async def process_file(filepath: Path, class_name: str, crop_pixels: int = 20) -> List[Tuple[np.ndarray, str]]:
    """Processes a DICOM file, generates 50 varied descriptions, and extracts their vectors."""
    original_stem = filepath.stem
    logger.info(f"Processing '{filepath.name}' to generate {DESCRIPTIONS_PER_IMAGE} descriptions (Class: {class_name})...")
    
    file_results = []
    try:
        ds = pydicom.dcmread(str(filepath), force=True)
        pixel_array = ds.pixel_array

        # 1. Initial image cleanup
        min_val, max_val = np.min(pixel_array), np.max(pixel_array)
        img_array_8bit = np.interp(pixel_array, (min_val, max_val), (0, 255)).astype(np.uint8) if max_val > min_val else np.zeros_like(pixel_array, dtype=np.uint8)
        pil_img_base = PilImage.fromarray(img_array_8bit)
        
        bbox = pil_img_base.getbbox()
        if bbox: pil_img_base = pil_img_base.crop(bbox)
        
        w, h = pil_img_base.size
        if w > crop_pixels * 2 and h > crop_pixels * 2:
            pil_img_base = pil_img_base.crop((crop_pixels, crop_pixels, w - crop_pixels, h - crop_pixels))
        
        if pil_img_base.size[0] == 0 or pil_img_base.size[1] == 0:
            raise ValueError("Image has no content after cropping.")

        # 2. Select the correct prompt list
        if class_name.upper() in ['MTF', 'TOR']:
            prompt_list = OBJECT_PROMPTS
        else:
            prompt_list = BACKGROUND_PROMPTS

        # 3. Loop to generate 50 descriptions and vectors
        for i in range(DESCRIPTIONS_PER_IMAGE):
            random_prompt = random.choice(prompt_list)
            vector = await extract_vector_with_vlm(pil_img_base, random_prompt, VLM_TEMPERATURE)
            if vector is not None:
                file_results.append((vector, class_name))
        
        logger.info(f"SUCCESS: Generated {len(file_results)} descriptions/vectors for '{original_stem}'.")

    except Exception as e:
        logger.exception(f"Critical exception while processing '{filepath.name}': {e}. Moving to errors.")
        ERROR_DIR.mkdir(parents=True, exist_ok=True)
        shutil.move(str(filepath), str(ERROR_DIR / filepath.name))
        return []

    return file_results

async def worker(semaphore, filepath, class_name):
    """
    Una función 'wrapper' que adquiere el semáforo antes de procesar un fichero
    y lo libera cuando termina.
    """
    async with semaphore:
        # Llama a tu función de procesamiento original
        return await process_file(filepath, class_name)


async def run_vlm_extraction():
    """
    Main function that orchestrates the extraction and consolidates the dataset.
    NOW WITH A CONCURRENCY LIMIT TO PROTECT MEMORY.
    """
    logger.info(f"===== START PHASE 2: EXTRACTION WITH VLM AND PROMPTS =====")

    if not INPUT_DIR.exists():
        logger.error(f"Input directory '{INPUT_DIR}' does not exist.")
        return
        
    if OUTPUT_PARENT_DIR.exists(): shutil.rmtree(OUTPUT_PARENT_DIR)
    if DATASET_FINAL_DIR.exists(): shutil.rmtree(DATASET_FINAL_DIR)
    OUTPUT_PARENT_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_FINAL_DIR.mkdir(parents=True, exist_ok=True)

    class_folders = [d for d in INPUT_DIR.iterdir() if d.is_dir()]
    if not class_folders:
        logger.warning(f"No class folders found inside '{INPUT_DIR}'.")
        return

    logger.info(f"Found {len(class_folders)} classes: {[d.name for d in class_folders]}")
    logger.info(f"Will generate {DESCRIPTIONS_PER_IMAGE} vectors for each original image.")

    # --- Lógica de concurrencia con semáforo ---
    CONCURRENCY_LIMIT = 5
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    logger.info(f"Concurrency limit set to {CONCURRENCY_LIMIT} simultaneous tasks.")
    
    tasks = []
    for class_dir in class_folders:
        files_in_class = [f for f in class_dir.iterdir() if f.is_file() and f.name.lower().endswith('.dcm')]
        logger.info(f"-> Found {len(files_in_class)} files for class '{class_dir.name}'.")
        for filepath in files_in_class:
            tasks.append(worker(semaphore, filepath, class_dir.name))

    list_of_results = []
    if tasks:
        list_of_results = await asyncio.gather(*tasks)

    # --- ¡CÓDIGO RESTAURADO! Consolidación y guardado del dataset ---
    logger.info("===== CONSOLIDATING FINAL DATASET FOR SVC =====")
    X_final = []
    y_final = []

    for file_result in list_of_results:
        for vector, label in file_result:
            X_final.append(vector)
            y_final.append(label)

    if not X_final:
        logger.error("No vectors were generated. The final dataset is empty.")
    else:
        # Convertir listas a arrays de NumPy para un guardado eficiente
        X_np = np.array(X_final)
        y_np = np.array(y_final)

        # Definir las rutas de salida para los ficheros del dataset
        x_path = DATASET_FINAL_DIR / "dataset_X.npy"
        y_path = DATASET_FINAL_DIR / "dataset_y.npy"
        
        # Guardar los arrays en disco
        np.save(x_path, X_np)
        np.save(y_path, y_np)

        logger.info("--- Dataset ready for Phase 3! ---")
        logger.info(f"Data dataset (X) saved to: {x_path}")
        logger.info(f" -> Shape of X array: {X_np.shape}")
        logger.info(f"Label dataset (y) saved to: {y_path}")
        logger.info(f" -> Shape of y array: {y_np.shape}")

    logger.info("===== END OF EXECUTION =====")
if __name__ == "__main__":
    # Before running, make sure the Ollama service is running
    # and you have downloaded the model: ollama run llava-phi3
    asyncio.run(run_vlm_extraction())