import matplotlib.pyplot as plt
import numpy as np 
import cv2
import os
import PIL
import tensorflow as tf
import zipfile
import pathlib

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

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

MildDemented = list(data_dir.glob('MildDemented/*'))

img = Image.open(str(MildDemented[0]))
img.show()

Alzheimer_stage_imagens_dic = {
    'NonDemented': list(data_dir.glob('NonDemented/*')),
    'MildDemented': list(data_dir.glob('MildDemented/*')),
    'ModerateDemented': list(data_dir.glob('ModerateDemented/*')),
    'VeryMildDemented': list(data_dir.glob('VeryMildDemented/*')),
}

Alzheimer_stage_labels_dic = {
    'NonDemented': 0,
    'MildDemented': 1,
    'ModerateDemented': 2,
    'VeryMildDemented': 3,
}

img2 = cv2.imread(str(Alzheimer_stage_imagens_dic['NonDemented'][0]))
print(img2.shape)

#inicializado variaveis de features e targes
#initializing features and targets variable
X, y = [], []


#iterando entre as chaves do dicionario
for alzheimer_stage, images in Alzheimer_stage_imagens_dic.items():
    #fazendo o downsampling para balancear as classes em 10k 
    downsampling = images[:10000]
    print(f"Processando {len(downsampling)} imagens para o estágio: {alzheimer_stage}")
    #iterando pela lista de caminho das imagens dentro do dicionario
    for image in tqdm(downsampling, desc=f"Processando {alzheimer_stage}"):
        #abrindo imagem
        img2 = cv2.imread(str(image))
        #verificando a imagem foi selecionada
        if img2 is not None:
            #redimensionando a imagem para o padrao 
            resized_img = cv2.resize(img2, (180, 180))
            X.append(resized_img)
            y.append(Alzheimer_stage_labels_dic[alzheimer_stage])
            
#convertendo para arrays numpy
X = np.array(X)
y = np.array(y)

#criando sets de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9,  random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,  random_state=0)

#normalizando as features de treino e teste
X_train_scaled = X_train.astype('float32') / 255
X_test_scaled = X_test.astype('float32') / 255

#definindo o modelo
def get_model(layer1, layer2):
    model = Sequential([
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(layer1, activation='relu'),
        layers.Dense(layer2, activation='relu'),
        layers.Dense(4),
        
    ])


    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
    )
    
    return model

#selecionano a GPU para o treimento
with tf.device('/GPU:0'):
    gpu_model = get_model(layer1=128, layer2=64)
    gpu_model.fit(X_train_scaled, y_train, epochs=10)
    print(gpu_model.evaluate(X_test_scaled, y_test))
    








































































































































































