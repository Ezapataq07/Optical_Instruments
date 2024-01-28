#Se imortan las librerías relevantes y las unidades
from matplotlib.pyplot import show
from resources.functions import *

#Importamos el holograma
root = "resources/" 
file_name = "Hologram.tiff"
Hologram = cv2.imread(root + file_name, 0)
shape = np.shape(Hologram) #Dimensión del arreglo
dx = (6513/shape[0])*um

#Espectro de fourier
Hologram_Fourier = np.fft.fftshift(np.fft.fftn(Hologram))
#Definimos las condiciones para la primer gráfica
fig, axs = plt.subplots(1, 2, sharey=True)
Complex_Plot(Hologram,'A',0,axs[0], fig)
Complex_Plot(Hologram_Fourier,'A',1,axs[1], fig)
axs[0].set_title("a) Holograma")
axs[0].set_xlabel("{} um/pixel".format(round(dx/um,2)))
axs[1].set_title("b) Espectro de frecuencias \n en escala logaritmica")
axs[1].set_xlabel("{} um-1/pixel".format(round(1/(dx/um),2)))
plt.show()

#Creamos los filtros para eliminar orden cero y orden -1
r = 400 #Radio de los circulos 
coor1=[1726,576] #Coordenadas donde se encuentra la información de interes 
Filter1 = Mask_Circle(shape, r, "A", coor1[0],coor1[1]) #Filtro para eliminación de frecuencias
coor2=[1725,580]
Filter2 = Mask_Circle(shape, r, "A", coor2[0],coor2[1]) #Filtro para combinar métodos
#Obtenemos la imagen filtrada
Hologram_Filter = np.fft.ifftn(Hologram_Fourier*Filter1)

#Eliminamos el orden cero restando la media
Hologram_Mean = Hologram - np.mean(Hologram)

#Combinamos ambos métodos 
Hologram_Mean_Fourier = np.fft.fftshift(np.fft.fftn(Hologram_Mean))
Hologram_Mean_Filter = np.fft.ifftn(Hologram_Mean_Fourier*Filter2)

#Propagamos las imagenes
z = 73*mm
landa = 633*nm
dxf = landa*z/(shape[0]*dx) #Nuevo tamaño de pixel (Por ser difracción de fressnel)
Hologram_Final = Diffraction(Hologram,z,landa,dx) #Reconstrucción sin eliminar orden cero
Hologram_Filter_Final = Diffraction(Hologram_Filter,z,landa,dx) #Reconstrucción por filtrado de frecuencias
Hologram_Mean_Final = Diffraction(Hologram_Mean,z,landa,dx) #Reconstruimos eliminando el orden cero por resta de media
Hologram_Mean_Fourier_Final = Diffraction(Hologram_Mean_Filter,z,landa,dx) #Reconstruimos combinando metodos

#Definimos las condiciones para la primer gráfica
fig, axs = plt.subplots(sharey=True)
Complex_Plot(Hologram_Final, "A",0,axs)
axs.set_title("a) Imagen reconstruida")
axs.set_xlabel("{} um/pixel".format(round(dxf/um,2)))
plt.show()
fig, axs = plt.subplots(sharey=True)
Complex_Plot(Hologram_Filter_Final, "A",0,axs)
axs.set_title("b) Imagen reconstruida \n con filtrado de frecuencias")
axs.set_xlabel("{} um/pixel".format(round(dxf/um,2)))
plt.show()
fig, axs = plt.subplots(sharey=True)
Complex_Plot(Hologram_Filter_Final, "A",0,axs)
axs.set_title("c) Imagen reconstruida \n con resta de media")
axs.set_xlabel("{} um/pixel".format(round(dxf/um,2)))
plt.show()
fig, axs = plt.subplots(sharey=True)
Complex_Plot(Hologram_Mean_Fourier_Final, "A",0,axs)
axs.set_title("d) Imagen reconstruida \n combinando métodos")
axs.set_xlabel("{} um/pixel".format(round(dxf/um,2)))
plt.show()
