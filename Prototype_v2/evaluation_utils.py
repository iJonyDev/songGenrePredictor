import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

def calculate_metrics(y_true, y_pred, label_encoder):
    """
    Calculates and prints various classification metrics.

    Args:
        y_true (list or np.array): True labels (numeric).
        y_pred (list or np.array): Predicted labels (numeric).
        label_encoder (LabelEncoder): Fitted LabelEncoder to decode labels for reports.

    Returns:
        dict: A dictionary containing calculated metrics.
    """
    print("\n--- Informe de Evaluación ---")
    # Usar zero_division=0 para evitar warnings
    report = classification_report(y_true, y_pred, target_names=label_encoder.classes_, zero_division=0)
    print("Informe de Clasificación:")
    print(report)

    conf_matrix = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    # Usar average='weighted' para métricas multiclase generales
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # ROC AUC requiere probabilidades para multiclase, o usar One-vs-Rest
    # Calcular probabilidades si es posible, o usar OvR sobre las etiquetas predichas
    # Nota: roc_auc_score con etiquetas predichas directamente puede ser menos informativo
    # que con probabilidades. Aquí usamos OvR sobre etiquetas.
    try:
        # Asegurarse de que y_true y y_pred sean adecuados para roc_auc_score multiclase
        # Puede requerir binarizar las etiquetas si no se proporcionan puntuaciones
        # Para simplificar, calcularemos OvR sobre las etiquetas predichas
        # Esto puede no ser estándar, pero da una idea general.
        # Una mejor aproximación requeriría las salidas de probabilidad del modelo.
        roc_auc = roc_auc_score(y_true, torch.nn.functional.one_hot(torch.tensor(y_pred), num_classes=len(label_encoder.classes_)).numpy(), multi_class='ovr')
        print(f"ROC AUC (OvR): {roc_auc:.4f}")
    except Exception as e:
        print(f"No se pudo calcular ROC AUC: {e}")
        roc_auc = None # O manejar de otra forma

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Weighted): {precision:.4f}")
    print(f"Recall (Weighted): {recall:.4f}")

    metrics = {
        'accuracy': accuracy,
        'precision_weighted': precision,
        'recall_weighted': recall,
        'roc_auc_ovr': roc_auc,
        'confusion_matrix': conf_matrix,
        'classification_report': report
    }
    return metrics

def plot_confusion_matrix(conf_matrix, class_names):
    """
    Plots the confusion matrix.
    """
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

# Nota: La curva ROC multiclase es más compleja. 
# Se puede hacer una por clase (OvR) o micro/macro-average.
# La implementación actual en train.py para ROC era incorrecta para multiclase.
# Se omite aquí por simplicidad, pero se podría agregar una función 
# plot_multiclass_roc_curve si se necesitan las probabilidades del modelo.

def evaluate_model(model, dataloader, device, label_encoder):
    """
    Evaluates the model on a given dataloader and calculates metrics.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader containing the evaluation data.
        device (torch.device): Device to run evaluation on.
        label_encoder (LabelEncoder): Fitted LabelEncoder.

    Returns:
        dict: A dictionary containing calculated metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []

    print(f"\nIniciando evaluación en {len(dataloader.dataset)} muestras...")
    with torch.no_grad():
        for batch in dataloader:
            # Asumiendo que el dataloader devuelve (input_ids, attention_mask, labels)
            input_ids, attention_mask, labels = [t.to(device) for t in batch]
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_preds = torch.argmax(outputs, dim=1).cpu().tolist()
            
            all_preds.extend(batch_preds)
            all_labels.extend(labels.cpu().tolist())
    print("Evaluación completada.")

    # Calcular y mostrar métricas
    metrics = calculate_metrics(all_labels, all_preds, label_encoder)
    
    # Graficar matriz de confusión
    plot_confusion_matrix(metrics['confusion_matrix'], label_encoder.classes_)

    return metrics
