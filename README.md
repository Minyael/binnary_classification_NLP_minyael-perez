# README - Clasificación binaria

---

## Descripción General

Este proyecto implementa una red neuronal utilizando TensorFlow y Keras para la clasificación binaria en el dataset IMDB, enfocándose en el análisis de sentimientos de reseñas de películas. El objetivo es entrenar un modelo que pueda distinguir reseñas positivas de negativas. Este proyecto fue realizado con Python 3.12.

---

## Requisitos Previos

Antes de ejecutar el código, asegúrate de tener instaladas las siguientes bibliotecas:

```bash
pip install numpy keras tensorflow matplotlib
```

Opcionalmente, puedes usar un entorno virtual para aislar dependencias:

```bash
python -m venv env
source env/bin/activate  # En Linux/macOS
env\Scripts\activate  # En Windows
pip install -r requirements.txt
```

---

## Estructura del Proyecto

```
📂 binary_classification_NLP/
├── 📂 src/                      
│   ├── binary_classification_NLP.py   # Código fuente
├── main.py                            # Script principal para ejecutar el modelo
├── requirements.txt                   # Dependencias del proyecto
├── README.md                          # Documentación del proyecto
```

---

## Implementación del Modelo

### Carga de Datos

Se utiliza el dataset IMDB, limitando el vocabulario a las 10,000 palabras más frecuentes para procesar las reseñas.

```python
from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
```

### Preprocesamiento y Visualización

Se decodifica una reseña de ejemplo para visualizar su contenido y se realiza el one-hot encoding de las secuencias de reseñas.

```python
word_index = imdb.get_word_index()                   # Diccionario palabra-índice
reverse_word_index = {value: key for key, value in word_index.items()}
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
print(\"Primera reseña decodificada:\")
print(decoded_review)
print(\"Etiqueta (1=Positiva, 0=Negativa):\", train_labels[0])
```

Para vectorizar las reseñas:

```python
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))  # Matriz de ceros
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # Marca la presencia de palabras
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
```

Convertir las etiquetas a tipo float32:

```python
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
```

### Separación de Datos de Validación

Se separa un subconjunto de validación (los primeros 10,000 ejemplos) del conjunto de entrenamiento.

```python
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]
```

### Definición del Modelo

El modelo consiste en:

- Una capa oculta con 64 neuronas y activación ReLU.
- Una segunda capa oculta con 16 neuronas y activación ReLU.
- Una capa de salida con 1 neurona y activación Sigmoid para la clasificación binaria.

```python
from keras import models, layers

model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10000,)),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
```

### Compilación y Entrenamiento del Modelo

Se compila el modelo con el optimizador RMSprop y la función de pérdida binary_crossentropy. Luego se entrena durante 20 épocas con un tamaño de lote de 512, utilizando el conjunto de validación para monitorear el desempeño.

```python
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    partial_x_train, partial_y_train,
    epochs=20, batch_size=512,
    validation_data=(x_val, y_val)
)
```

### Evaluación y Visualización de Resultados

Se evalúa el modelo en el conjunto de prueba:

```python
results = model.evaluate(x_test, y_test)
print("\nResultados del modelo en el conjunto de prueba (Loss, Accuracy):", results)
```

Además, se grafican la pérdida y la precisión tanto en entrenamiento como en validación:

```python
# Graficar pérdida
history_dict = history.history
plt.plot(range(1, len(history_dict['loss']) + 1), history_dict['loss'], 'bo', label='Training loss')
plt.plot(range(1, len(history_dict['val_loss']) + 1), history_dict['val_loss'], 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Graficar precisión
plt.clf()
plt.plot(range(1, len(history_dict['accuracy']) + 1), history_dict['accuracy'], 'bo', label='Training Accuracy')
plt.plot(range(1, len(history_dict['val_accuracy']) + 1), history_dict['val_accuracy'], 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

---

## Ejecución

Para entrenar y evaluar el modelo, ejecuta en la terminal:

```bash
python main.py
```

---

## Conclusión

Este proyecto demuestra cómo utilizar Keras para entrenar un modelo de clasificación binaria en el dataset IMDB, permitiendo analizar el sentimiento en reseñas de películas. Los gráficos generados facilitan la evaluación de la pérdida y la precisión durante el entrenamiento y la validación.
