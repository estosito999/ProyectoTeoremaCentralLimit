# tcl_config.py

# Subconjunto del dataset SST-2 (Glue) para generar errores de la IA
# Puedes aumentar la cantidad si tu PC aguanta más.
SUBSET_TRAIN = "train[:20000]"   # 2000 frases

# Tamaño de ventana (n) para el Teorema Central del Límite
TAM_VENTANA = 100

# Archivo donde guardaremos los micro-errores de la IA
ERRORS_FILE = "errores_decision.npy"

# Cuántos "pasos" quieres ver en la animación (frames lógicos)
N_STEPS = 20

# Número de bins del histograma de medias
N_BINS_MEDIAS = 30
