# generate_errors.py

import numpy as np
from transformers import pipeline
from datasets import load_dataset
from tcl_config import SUBSET_TRAIN, ERRORS_FILE

def compute_ia_errors():
    print("Cargando modelo DistilBERT SST-2 desde HuggingFace...")
    clf = pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )

    print(f"Cargando dataset GLUE SST-2 ({SUBSET_TRAIN})...")
    dataset = load_dataset("glue", "sst2", split=SUBSET_TRAIN)

    texts = dataset["sentence"]
    labels = np.array(dataset["label"], dtype=int)

    n_total = len(texts)
    print(f"Total de textos a evaluar: {n_total}")

    errores = []
    batch_size = 32

    print("Calculando micro-errores (0 = acierto, 1 = error)...")
    for i in range(0, n_total, batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]

        outputs = clf(batch_texts)

        for out, etiqueta_real in zip(outputs, batch_labels):
            pred_label_str = out["label"]   # 'POSITIVE' o 'NEGATIVE'
            etiqueta_pred = 1 if pred_label_str == "POSITIVE" else 0
            error = 1.0 if etiqueta_pred != etiqueta_real else 0.0
            errores.append(error)

        print(f"Procesados: {min(i + batch_size, n_total)}/{n_total}", end="\r")

    errores = np.array(errores, dtype=float)
    print("\nListo. Micro-errores calculados.")

    tasa_global = errores.mean()
    print(f"Tasa de error global de la IA: {tasa_global:.4f}")

    np.save(ERRORS_FILE, errores)
    print(f"Guardado en '{ERRORS_FILE}'.")

if __name__ == "__main__":
    compute_ia_errors()
