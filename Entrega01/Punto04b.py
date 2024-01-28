#Importamos todas las librerias necesarias
from resources.funtions import *

#Importamos la imagen a corregir
b=cv2.imread('resources/puntob.png',0)

#Hacemos pasar la imagen por la primera parte del sistema
fft_b=Lenses2F(b,landa=1,f=1,shift=1)

#Se hace la máscara con la cual se filtrarán las frecuencias en el plano de Fourier.
mask=np.ones(np.shape(b),dtype="uint8")
cv2.circle(mask,(396,367),5,0,-1)
cv2.circle(mask,(372,401),5,0,-1)
cv2.rectangle(mask,(395,374),(396,342),0,-1)
cv2.rectangle(mask,(390,368),(396,365),0,-1)
cv2.rectangle(mask,(368,403),(403,400),0,-1)

#Se hace pasar la información del plano de Fourier por la segunda parte del sistema, aplicando el filtro.
b_filt=Lenses2F(fft_b*mask,landa=1,f=1,shift=0)

#Se muestra la imagen original, La información de Frecuencias con el filtro y la imagen corregida.
fig, axs = plt.subplots(1, 3, figsize=(10, 5), sharey=True)
Complex_Plot(b ,'A',0,axs[0])
axs[0].set_title("Imagen con Ruido")
Complex_Plot(fft_b*mask,'A',1,axs[1])
axs[1].set_title("Plano Fourier con Filtro")
Complex_Plot(b_filt,'A',0,axs[2])
axs[2].set_title("Imagen Corregida")
plt.show()
