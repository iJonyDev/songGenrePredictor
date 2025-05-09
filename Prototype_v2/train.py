import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


# Importar funciones de los nuevos módulos
from model import GenreClassifier
from data_utils import prepare_data_with_bert, create_stratified_sample
from training_utils import train_val_loop, predict_on_test_data, save_model, load_model
from evaluation_utils import evaluate_model # Opcional, si se quiere evaluar después

# Configurar el dispositivo (GPU, MPS o CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Usando el dispositivo: {device}")

# --- Funciones eliminadas (movidas a otros módulos) ---
# train_model, save_model, evaluate_model (versión antigua), prepare_data_with_bert

def main():
    # --- Configuración ---
    TRAIN_CSV = 'train.csv'
    TEST_CSV = 'test_data.csv'
    SAMPLED_TRAIN_CSV = 'train_sampled.csv'
    MODEL_SAVE_PATH = "trained_model.pth"
    USE_SAMPLED_DATA = True # Cambiar a False para usar el dataset completo
    CREATE_SAMPLE = False # Cambiar a True si no existe train_sampled.csv
    EPOCHS = 4 # Número de épocas para el entrenamiento
    # --- Hiperparámetros ---
    BATCH_SIZE = 16 # Podemos probar a aumentar un poco el batch con bert-base
    LEARNING_RATE = 2e-5 # Podemos probar a aumentar un poco lr con bert-base
    TEST_BATCH_SIZE = 16 # Ajustar también para predicción
    EARLY_STOPPING_PATIENCE = 2 # Número de épocas a esperar si no hay mejora
    WARMUP_PROPORTION = 0.1 # Proporción de pasos de entrenamiento para el calentamiento

    # --- 1. Creación de Muestra (Opcional) ---
    train_file_to_use = TRAIN_CSV
    if USE_SAMPLED_DATA:
        if CREATE_SAMPLE:
            print("Creando muestra estratificada...")
            train_file_to_use = create_stratified_sample(TRAIN_CSV, SAMPLED_TRAIN_CSV)
        else:
            train_file_to_use = SAMPLED_TRAIN_CSV
            print(f"Usando muestra existente: {train_file_to_use}")
    else:
        print(f"Usando dataset completo: {train_file_to_use}")

    # --- 2. Preparación de Datos ---
    print("\n--- Preparando Datos de Entrenamiento y Validación ---")
    # Preparar datos de entrenamiento (ajusta LabelEncoder)
    train_val_inputs, train_val_labels, label_encoder = prepare_data_with_bert(
        file_path=train_file_to_use,
        fit_label_encoder=True, 
        is_test_data=False
    )
    num_genres = len(label_encoder.classes_)

    # Dividir datos preparados en entrenamiento y validación
    print("Dividiendo datos en entrenamiento y validación...")
    train_indices, val_indices = train_test_split(
        range(len(train_val_labels)),
        test_size=0.2, 
        random_state=42,
        stratify=train_val_labels # Estratificar la división también
    )

    train_inputs = {key: val[train_indices] for key, val in train_val_inputs.items()}
    val_inputs = {key: val[val_indices] for key, val in train_val_inputs.items()}
    train_labels = train_val_labels[train_indices]
    val_labels = train_val_labels[val_indices]

    # --- Calcular Pesos de Clase ---
    print("Calculando pesos de clase dinámicamente...")
    # Asegúrate de que train_labels esté en la CPU y sea un array de numpy para compute_class_weight
    train_labels_numpy = train_labels.cpu().numpy() if isinstance(train_labels, torch.Tensor) else train_labels
    unique_classes = np.unique(train_labels_numpy)
    
    class_weights_computed = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=train_labels_numpy
    )
    # Convertir a tensor y mover al dispositivo
    class_weights_tensor = torch.tensor(class_weights_computed, dtype=torch.float).to(device)
    print(f"Pesos de clase calculados: {class_weights_tensor}")
    # -----------------------------

    # Crear TensorDatasets
    train_dataset = TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], train_labels)
    val_dataset = TensorDataset(val_inputs['input_ids'], val_inputs['attention_mask'], val_labels)
    print(f"Tamaño Dataset Entrenamiento: {len(train_dataset)}")
    print(f"Tamaño Dataset Validación: {len(val_dataset)}")

    print("\n--- Preparando Datos de Prueba ---")
    test_inputs, _, _ = prepare_data_with_bert(
        file_path=TEST_CSV, 
        label_encoder=label_encoder, # Usar el mismo encoder
        is_test_data=True
    )

    # --- 3. Definición y Entrenamiento del Modelo ---
    print("\n--- Definiendo el Modelo ---")
    model = GenreClassifier(num_genres=num_genres).to(device)

    print("\n--- Iniciando Entrenamiento --- ")
    # Usar train_val_loop del módulo training_utils
    model = train_val_loop(
        model=model, 
        train_data=train_dataset, 
        val_data=val_dataset, 
        device=device,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        early_stopping_patience=EARLY_STOPPING_PATIENCE, # Pasar el parámetro
        class_weights=class_weights_tensor, # Pasar los pesos calculados
        warmup_proportion=WARMUP_PROPORTION # Pasar el nuevo parámetro
    )

    # --- 4. Guardar Modelo Entrenado ---
    save_model(model, path=MODEL_SAVE_PATH)

    # --- 5. Evaluación Opcional en Conjunto de Validación ---
    # Si se quiere una evaluación más detallada post-entrenamiento
    # print("\n--- Evaluación Final en Conjunto de Validación ---")
    # val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    # evaluate_model(model, val_dataloader, device, label_encoder)

    # # --- 6. Predicción en Datos de Prueba ---
    # # Usar predict_on_test_data del módulo training_utils
    # predicted_label_indices = predict_on_test_data(
    #     model=model, 
    #     test_inputs=test_inputs, 
    #     device=device, 
    #     batch_size=TEST_BATCH_SIZE
    # )

    # # Decodificar las etiquetas predichas
    # predicted_genres = label_encoder.inverse_transform(predicted_label_indices)
    
    # print("\n--- Predicciones para los datos de prueba ---")
    # # Opcional: Guardar predicciones en un archivo
    # predictions_df = pd.DataFrame({
    #     'lyrics': pd.read_csv(TEST_CSV)['lyrics'], # Asumiendo que quieres las letras originales
    #     'predicted_genre': predicted_genres
    # })
    # predictions_df.to_csv('test_predictions.csv', index=False)
    # print("Predicciones guardadas en test_predictions.csv")
    # print("Primeras 50 predicciones:")
    # print(predictions_df.head(50))

if __name__ == "__main__":
    main()
