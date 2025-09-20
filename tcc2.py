import matplotlib.pyplot as plt
import numpy as np 
import cv2
import os
import PIL
import tensorflow as tf
import zipfile
import pathlib
import seaborn as sns

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import  classification_report

#lendo e extraindo o arquivo archive.zip onde o dataset está localizando
#reading and extracting the file archive.zip where the dataset is located 
# data_dir = tf.keras.utils.get_file("combined_images", 'file:///mnt/d/OneDrive - Universidade do Estado do Rio de Janeiro/tcc/archive.zip', cache_dir='.', extract=True)

#definindo caminhos
#defining paths
source_zip_path = pathlib.Path('/mnt/d/OneDrive - Universidade do Estado do Rio de Janeiro/tcc/archive.zip')
destination_dir = pathlib.Path('./combined_images') 

#criando o diretorio de destino se ele não existir
#create the destination directory if it doesn't exist
destination_dir.mkdir(parents=True, exist_ok=True)

#verificando se o caminho existe e se está vazio, se exitir e estiver vazio, abre o arquivo zip e extrai ele
#cverifying if the path exist and it's empty, if so, then open the zip file and extract it
if destination_dir.exists() and any(destination_dir.iterdir()):
    print(f'Path {destination_dir} is ready')
else:
    print(f"Extracting {source_zip_path} to {destination_dir}...")
    with zipfile.ZipFile(source_zip_path, 'r') as zip_ref:
        zip_ref.extractall(destination_dir)
    print("Extraction complete.")
    

#printando para ver se o local onde o dataset foi extraido está corretp
#printing to see if the path where the dataset was extracted is the right path
print("testeeee",destination_dir)

#adicionando "combined_images" no caminho porque ao criar o cache, o caminho foi setado como ./datasets/combined_images, mas o certo é ./datasets/combined_images/combined_images
#adding "combined_images" to the path because when the cache dir was criated, the path was set to ./datasets/combined_images, but the right path is ./datasets/combined_images/combined_images
data_dir= pathlib.Path(destination_dir) / "combined_images"
data_test_dir= pathlib.Path(destination_dir) / "Test"

MildDemented = list(data_dir.glob('MildDemented/*'))

# img = Image.open(str(MildDemented[0]))
# img.show()

# Criando datasets direto do diretório
batch_size = 16
img_height = 180
img_width = 180


# class_names = full_ds.class_names
# print("Classes:", class_names)
        


#carregando os conjuntos de treino, validacao e teste
#loading the train, validation and test sets
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset="training",
    label_mode='int',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset="validation",
    label_mode='int',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    data_test_dir,
    labels='inferred',
    label_mode='int',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False 
)


# selected_samples_per_class = 10000 # Desired number of samples per class
# filtered_dataset_elements = []

print(train_ds.class_names)

# Normalização dados
#Normalizing data
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))


early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

#definindo o modelo
#defining the model
def get_model(layer1, layer2):
    model = Sequential([
        layers.Conv2D(64, 3, padding='same', input_shape=(180, 180, 3), activation='relu'), #16 filtros, kernel 3x3, padding same (same = com padding / valid = sem padding), funcao de ativacao relu
        layers.MaxPooling2D(), #pooling padrao de 2x2 | usando para selecionar o maior valor dentro da regiao 2x2 em um mapa criado pela camada conv2d
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(layer1, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(layer2, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4 , activation='softmax'),
        
    ])


    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy,
                metrics=['accuracy']
    )
    
    return model

#selecionano a GPU para o treimento
#selecting the GPU to train
with tf.device('/GPU:0'):
    #instanciando o modelo
    #instantiating the model
    gpu_model = get_model(layer1=256, layer2=256)
    # model.fit(X_train_fold, y_train_fold, epochs=50, batch_size=10,
    #                     validation_data=(X_val_fold, y_val_fold),
    #                     class_weight=class_weights_dict, callbacks=[early_stopping], verbose=0)
    #treinando o modelo
    #training the model
    gpu_model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=[early_stopping])
    #avaliando o modelo
    #evaluating the model
    y_pred = gpu_model.predict(test_ds)

#extraido os labels verdadeiros e fazendo a matriz de confusão
#extracting the true labels and making the confusion matrix
y_true = np.concatenate([y for x, y in test_ds], axis=0)
cm = tf.math.confusion_matrix(labels=y_true, predictions=y_pred.argmax(axis=1))

# for i , j in cm:

print(cm)

classes = ['MildDemented', 'ModerateDemented',  'NonDemented', 'VeryMildDemented']
#plotando a matriz de confusão
#plotting the confusion matrix 
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)  
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

print(classification_report(y_true, y_pred.argmax(axis=1), target_names=classes))




































































































































































