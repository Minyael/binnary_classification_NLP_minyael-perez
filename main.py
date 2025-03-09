# Importar la función de entrenamiento y evaluación desde el código fuente
from src.binary_classification_NLP import train_imdb_model

# Verifica si el script se ejecuta directamente
if __name__ == "__main__":
   train_imdb_model()  # Llama a la función para entrenar y evaluar la red neuronal