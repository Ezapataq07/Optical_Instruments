#Importamos todas las librerias necesarias
from resources.funtions import *

#Importamos la imagen a estudiar
c=cv2.imread('resources/c.jpg',0)

#Importamos la pista (imagen que queremos ver si se encuentra en c)
clue=cv2.imread('resources/c_clue.jpg',0)


#Función para dividir la imagen a estudiar es varias imágenes
#En este caso, se dividirá la imagen en subimágenes de 220x112 píxeles, obteniendo un total de 200 subimágenes
#La idea es tomar manualmente las diferentes columnas (las cuales contienen de a 8 imágenes) y
#hacerles el proceso de correlación mostrado más adelante (pues es inviable imprimir 200 correlaciones al tiempo)
def Img_Divider(Img,column):
    #Img es la imagen a dividir
    #column es la columna a escoger, en este caso iría de 0 a 24
    h=np.hsplit(Img,25) #Se divide la imagen en 25 columnas
    return np.vsplit(h[column],8) #Se divide la columna escogida en 8 imágenes y se retornan

images=Img_Divider(c,10) #Se divide la imagen c y se toma la columna 10 

#Proceso para poner la pista con el mismo tamaño de las subimagenes y que quede centrada
c_clue=251*np.ones(np.shape(images[0]),dtype="uint8")
for i in range(61,159):
    for j in range(7,105):
        c_clue[i][j]=clue[i-61][j-7]

#Proceso de Correlación

for k in images:
    #Primero se crea una imagen (img_joint) que contenga la pista y la subimagen a estudiar
    #Esta imagen estaría en el plano imagen del sistema
    img_joint=np.zeros((220,226),dtype="uint8")
    for i in range(220):
        for j in range(112):
            img_joint[i][j]=k[i][j]
            img_joint[i][j+113]=c_clue[i][j]
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    Complex_Plot(img_joint,'A',0,axs[0])
    #Se hace pasar la imagen unida por la primera parte del sistema
    fft_img_joint=Lenses2F(img_joint,landa=1,f=1,shift=1)
    #Se toma la intensidad en el plano de Fourier, en este caso la Amplitud pues se obtienen mejores 
    #resultados (con la amplitud son muy tenues los picos de correlación)
    i_fft_img_joint=np.abs(fft_img_joint)
    #Se hace pasar la intensidad por la segunda parte del sistema, obteniendo la correlación en el 
    #plano imagen
    corr=np.fft.fftshift(np.fft.fftn(i_fft_img_joint))
    Complex_Plot(corr,'A',1,axs[1])
    plt.show()

    #Se toma la columna 10 pues allí se encuentra la imagen de interés
    #En la segunda imagen impresa se pueden observar picos de correlación alejados del origen, lo que
    #quiere decir que la pista está presente en la imagen de estudio.