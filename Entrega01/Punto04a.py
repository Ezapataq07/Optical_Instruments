#Importamos todas las librerias necesarias
from resources.funtions import *

#Abrimos la matriz compleja desde el txt
a = OpenComplexMat("resources/a.txt")

flat_wave = np.ones(np.shape(a),dtype="uint8") #Definimos una onda plana

interference = a + flat_wave #Sumamos las ondas (interferencia)

fft_sistem=Lenses2F(interference,shift=1) #Pamos por la primera lente

a_image =  Lenses2F(fft_sistem) #Vemos la imagen a la salida del sistema

#Gráficamos
fig, axs = plt.subplots(2, 2, sharey=True)

Complex_Plot(a,"P",0,axs[0][0], fig, True)
axs[0][0].set_title("Imagen a analizar en fase")
Complex_Plot(a,"I",0,axs[0][1], fig, True)
axs[0][1].set_title("Imagen a analizar en intensidad")
Complex_Plot(a_image,"A",0,axs[1][0], fig, True)
axs[1][0].set_title("Imagen final en fase")
Complex_Plot(a_image,"I",0,axs[1][1], fig, True)
axs[1][1].set_title("Imagen final en intensidad")

plt.show()