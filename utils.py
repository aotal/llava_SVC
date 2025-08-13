# utils.py (VERSIÓN SIMPLIFICADA)

import os
import logging
import re
from typing import Any, Optional

# Obtener un logger específico para este módulo de utilidades.
logger = logging.getLogger(__name__)


def configurar_logging_aplicacion(log_file_path: Optional[str] = None,
                                   level: int = logging.INFO,
                                   log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
    """
    Configura el logging para la aplicación.
    Escribe en un archivo (opcional) y siempre en la consola.
    """
    handlers = [logging.StreamHandler()]
    if log_file_path:
        try:
            log_dir = os.path.dirname(log_file_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            handlers.append(logging.FileHandler(log_file_path, encoding='utf-8'))
            msg_log_file = log_file_path
        except OSError as e:
            print(f"ADVERTENCIA DE LOGGING: No se pudo crear el directorio para el log {log_file_path}: {e}. Logueando solo a consola.")
            msg_log_file = "No configurado (error al crear directorio)"
    else:
        msg_log_file = "No configurado"

    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers,
        force=True # Para asegurar que se reconfigure si se llama varias veces.
    )
    logging.getLogger().info(f"Logging configurado. Nivel: {logging.getLevelName(level)}. "
                             f"Archivo de log: {msg_log_file}")


def clean_filename_part(part_value: Any, allowed_chars: str = "._-") -> str:
    """
    Limpia una cadena para ser usada como parte de un nombre de fichero,
    permitiendo solo caracteres alfanuméricos y los especificados en allowed_chars.
    Reemplaza los no permitidos por un guion bajo.
    """
    if part_value is None:
        return "Desconegut"

    s_part_value = str(part_value)
    # Escapa los caracteres permitidos para usarlos en la expresión regular
    escaped_allowed_chars = re.escape(allowed_chars)
    # Crea un patrón que encuentra cualquier caracter que NO sea alfanumérico o de los permitidos
    pattern = r'[^a-zA-Z0-9' + escaped_allowed_chars + r']'

    cleaned_value = re.sub(pattern, '_', s_part_value)
    cleaned_value = re.sub(r'_+', '_', cleaned_value) # Reemplaza múltiples guiones bajos por uno solo
    cleaned_value = cleaned_value.strip('_') # Elimina guiones bajos al principio o al final

    return cleaned_value if cleaned_value else "valor_net"