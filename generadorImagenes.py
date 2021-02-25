import os
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator, load_img, image, img_to_array


folderDataAugmentation = 'DataAug_San'
numeroImagenesDA = 6

try:
    os.mkdir(folderDataAugmentation)
except:
    print('Error')

train_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False)

data_path= 'C:/Users/eduar/Escritorio/San'    
data_dir_list = os.listdir(data_path)

ancho_shape, alto_shape = 224,224

contador=0
numero_de_imagenes=0

for image_file in data_dir_list:
    image_list = os.listdir(data_path)

    img_path = data_path + '/' + image_file

    imagen=load_img(img_path)
    imagen=cv2.resize(image.img_to_array(imagen), (ancho_shape, alto_shape), interpolation= cv2.INTER_AREA)

    x = imagen/255
    x = np.expand_dims(x,axis=0)
    t=1
    for output_batch in train_datagen.flow(x,batch_size=1):
        a = image.img_to_array(output_batch[0])
        img_salida = output_batch[0,:,:]*255
        img_final= cv2.cvtColor(img_salida,cv2.COLOR_BGR2RGB)
        cv2.imwrite(folderDataAugmentation+"/%i%iSandia.jpg"%(contador,t),img_final)
        t+=1
        numero_de_imagenes+=1

        if t>numeroImagenesDA:
            break
    contador+=1
print("Imagenes generadas: ", numero_de_imagenes)        

