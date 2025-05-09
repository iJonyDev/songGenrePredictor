import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch.nn as nn
from torch import tensor # Mantener si se usa directamente
import copy # Necesario para deepcopy
# --- Añadir import para el scheduler ---
from torch.optim.lr_scheduler import LambdaLR
# -------------------------------------

# Añadir warmup_proportion al signature, default None
def train_val_loop(model, train_data, val_data, device, epochs=3, batch_size=16, learning_rate=2e-5, early_stopping_patience=None, class_weights=None, warmup_proportion=None):
    """
    Handles the training and validation loops with optional early stopping, class weighting, and LR scheduler.

    Args:
        model (nn.Module): The model to train.
        train_data (TensorDataset): Training dataset.
        val_data (TensorDataset): Validation dataset.
        device (torch.device): Device to run training on.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training and validation.
        learning_rate (float): Learning rate for the optimizer.
        early_stopping_patience (int, optional): Number of epochs to wait for improvement 
                                                 before stopping. Defaults to None (disabled).
        class_weights (torch.Tensor, optional): Tensor containing class weights for the loss function.
                                                Defaults to None.
        warmup_proportion (float, optional): Proportion of total training steps for linear warmup.
                                             Defaults to None (no warmup).

    Returns:
        nn.Module: The best model found during training (if early stopping enabled), 
                   or the model from the last epoch.
    """
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=1e-7, weight_decay=0.01) # Añadir weight_decay
    # --- Usar class_weights en CrossEntropyLoss ---
    loss_fn = nn.CrossEntropyLoss(weight=class_weights) # Pasar los pesos aquí
    # --------------------------------------------

    # --- Configuración del LR Scheduler con Warmup Lineal y Decaimiento Lineal ---
    scheduler = None
    if warmup_proportion is not None and warmup_proportion > 0:
        total_training_steps = len(train_loader) * epochs
        warmup_steps = int(total_training_steps * warmup_proportion)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps)) # Aumento lineal
            # Decaimiento lineal desde el LR máximo hasta 0 después del warmup
            return max(0.0, float(total_training_steps - current_step) / float(max(1, total_training_steps - warmup_steps)))

        scheduler = LambdaLR(optimizer, lr_lambda)
        print(f"  Planificador de LR con calentamiento lineal por {warmup_steps} pasos y luego decaimiento lineal habilitado.")
    # ----------------------------------------------------------------------

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    print(f"Iniciando entrenamiento por hasta {epochs} épocas...")
    if class_weights is not None:
        # Mover a CPU para imprimir si está en MPS/CUDA
        weights_to_print = class_weights.cpu().numpy() if isinstance(class_weights, torch.Tensor) else class_weights
        print(f"  Usando pesos de clase: {weights_to_print}") # Mostrar pesos usados
    if early_stopping_patience is not None:
        print(f"  Early stopping habilitado con paciencia={early_stopping_patience}")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        # ... (bucle de entrenamiento existente sin cambios internos) ...
        for batch_idx, batch in enumerate(train_loader):
            # Mover batch al dispositivo
            input_ids, attention_mask, labels = [t.to(device) for t in batch]
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            
            # Verificar si la pérdida es NaN antes de backward()
            if torch.isnan(loss):
                print("¡Pérdida NaN detectada! Deteniendo el entrenamiento.")
                # Devolver el mejor modelo encontrado hasta ahora si existe
                if best_model_state:
                    model.load_state_dict(best_model_state)
                    print("Cargando el estado del mejor modelo encontrado antes del NaN.")
                return model 

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            optimizer.step()
            # --- Actualizar LR si el scheduler está activo ---
            if scheduler is not None:
                scheduler.step() # Actualizar LR en cada paso (batch)
            # --------------------------------------------
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation loop
        model.eval()
        total_val_loss = 0
        val_preds, val_true = [], []
        with torch.no_grad():
            # ... (bucle de validación existente sin cambios internos) ...
            for batch in val_loader:
                input_ids, attention_mask, labels = [t.to(device) for t in batch]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs, labels)
                total_val_loss += loss.item()
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().tolist())
                val_true.extend(labels.cpu().tolist())

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"  Loss Entrenamiento: {avg_train_loss:.4f}")
        print(f"  Loss Validación: {avg_val_loss:.4f}")
        print("  Informe de Clasificación (Validación):")
        print(classification_report(val_true, val_preds, zero_division=0))

        # --- Lógica de Early Stopping ---
        if early_stopping_patience is not None:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                # Guardar el estado del mejor modelo
                best_model_state = copy.deepcopy(model.state_dict())
                print(f"  Mejora en validación encontrada. Guardando modelo.")
            else:
                epochs_no_improve += 1
                print(f"  No hubo mejora en validación por {epochs_no_improve} épocas.")
                if epochs_no_improve >= early_stopping_patience:
                    print(f"\n¡Early stopping activado después de {epoch + 1} épocas!")
                    # Cargar el mejor estado antes de salir
                    if best_model_state:
                        model.load_state_dict(best_model_state)
                        print("Cargando el mejor modelo encontrado.")
                    else:
                        print("Advertencia: Early stopping activado pero no se guardó ningún estado (¿problema en la primera época?).")
                    break # Salir del bucle de épocas
        # --------------------------------

    print("Entrenamiento completado.")
    return model

def predict_on_test_data(model, test_inputs, device, batch_size=16):
    """
    Performs prediction on test data in batches.

    Args:
        model (nn.Module): The trained model.
        test_inputs (dict): Dictionary containing 'input_ids' and 'attention_mask' tensors for test data.
        device (torch.device): Device to run prediction on.
        batch_size (int): Batch size for prediction.

    Returns:
        list: A list of predicted label indices.
    """
    model.eval()
    all_predictions = []
    
    # Crear DataLoader para los datos de prueba
    test_dataset = TensorDataset(test_inputs['input_ids'], test_inputs['attention_mask'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    print(f"\nIniciando predicción en lotes para {len(test_dataset)} muestras de prueba...")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # Mover el lote al dispositivo
            input_ids, attention_mask = [t.to(device) for t in batch]
            
            # Realizar predicción para el lote
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_predictions = torch.argmax(outputs, dim=1).cpu().tolist()
            all_predictions.extend(batch_predictions)
            if (i + 1) % 100 == 0: # Imprimir progreso cada 100 lotes
                 print(f"  Procesados { (i + 1) * batch_size } / {len(test_dataset)} muestras")

    print("Predicción completada.")
    return all_predictions

def save_model(model, path="trained_model.pth"):
    """
    Saves the model state dictionary.
    """
    torch.save(model.state_dict(), path)
    print(f"Modelo guardado en {path}")

def load_model(model_class, path, num_genres, device):
    """
    Loads a model state dictionary.
    """
    model = model_class(num_genres=num_genres)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval() # Poner en modo evaluación por defecto
    print(f"Modelo cargado desde {path}")
    return model
