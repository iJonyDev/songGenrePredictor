import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
import torch

def create_stratified_sample(input_csv_path, output_csv_path, sample_fraction=0.2, stratify_column='genre', random_state=42):
    """
    Creates a stratified sample from a CSV file.

    Args:
        input_csv_path (str): Path to the input CSV file.
        output_csv_path (str): Path to save the sampled CSV file.
        sample_fraction (float): Fraction of the dataset to sample (e.g., 0.1 for 10%).
        stratify_column (str): The column to use for stratification.
        random_state (int): Random state for reproducibility.
    """
    # Cargar el dataset original
    df = pd.read_csv(input_csv_path)

    # Verificar si la columna de estratificación existe
    if stratify_column not in df.columns:
        raise ValueError(f"La columna '{stratify_column}' no se encuentra en el archivo {input_csv_path}.")

    # Obtener las etiquetas para la estratificación
    y = df[stratify_column]

    # Crear la muestra estratificada
    _, df_sample = train_test_split(
        df, 
        test_size=sample_fraction, 
        stratify=y, 
        random_state=random_state
    )

    # Guardar la muestra en un nuevo archivo CSV
    df_sample.to_csv(output_csv_path, index=False)
    print(f"Muestra estratificada guardada en {output_csv_path}")
    print(f"Tamaño de la muestra: {len(df_sample)} filas")
    print("Distribución de géneros en la muestra:")
    print(df_sample[stratify_column].value_counts(normalize=True))
    return output_csv_path # Devolver la ruta del archivo creado


def prepare_data_with_bert(file_path, label_encoder=None, fit_label_encoder=False, is_test_data=False, lyrics_col='lyrics', genre_col='genre'):
    """
    Prepares data using BERT tokenizer and encodes labels if available.

    Args:
        file_path (str): Path to the CSV file.
        label_encoder (LabelEncoder): Optional, pre-fitted LabelEncoder.
        fit_label_encoder (bool): Whether to fit the LabelEncoder on the genres.
        is_test_data (bool): Whether the data is test data (lacks labels).
        lyrics_col (str): Name of the column containing lyrics.
        genre_col (str): Name of the column containing genres (ignored if is_test_data=True).

    Returns:
        inputs (dict): Tokenized inputs for BERT ('input_ids', 'attention_mask').
        labels (torch.Tensor or None): Encoded labels as tensors, or None if is_test_data is True.
        label_encoder (LabelEncoder): Fitted or passed LabelEncoder.
    """
    # Cargar los datos
    data = pd.read_csv(file_path)

    # Verificar si la columna de letras existe
    if lyrics_col not in data.columns:
        raise KeyError(f"La columna '{lyrics_col}' no se encuentra en el archivo {file_path}.")

    lyrics = data[lyrics_col].tolist()

    # Tokenizar las letras de canciones con BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print(f"Tokenizando {len(lyrics)} textos de {file_path}...")
    inputs = tokenizer(lyrics, padding=True, truncation=True, return_tensors="pt", max_length=512)
    print("Tokenización completada.")

    labels = None
    # Si los datos no son de prueba, procesar las etiquetas
    if not is_test_data:
        if genre_col not in data.columns:
            raise KeyError(f"La columna '{genre_col}' no se encuentra en el archivo {file_path} (se esperaba para datos de entrenamiento/validación).")
        
        genres = data[genre_col].tolist()
        print(f"Procesando {len(genres)} etiquetas de {file_path}...")

        # Codificar las etiquetas
        if fit_label_encoder:
            print("Ajustando LabelEncoder...")
            label_encoder = LabelEncoder()
            encoded_genres = label_encoder.fit_transform(genres)
            print("Clases encontradas:", label_encoder.classes_)
        elif label_encoder is None:
             raise ValueError("Se requiere un label_encoder preajustado si fit_label_encoder es False.")
        else:
            print("Transformando etiquetas con LabelEncoder existente...")
            # Manejar etiquetas desconocidas en datos de validación/prueba si es necesario
            known_labels_mask = pd.Series(genres).isin(label_encoder.classes_)
            if not known_labels_mask.all():
                 unknown_labels = pd.Series(genres)[~known_labels_mask].unique()
                 print(f"Advertencia: Se encontraron etiquetas desconocidas: {unknown_labels}. Se asignará un valor predeterminado o se omitirán.")
            encoded_genres = label_encoder.transform(genres)


        labels = torch.tensor(encoded_genres)
        print("Etiquetas procesadas.")

    return inputs, labels, label_encoder
