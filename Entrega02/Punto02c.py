#Se imortan las librerías relevantes y las unidades
from resources.functions import *

#Parámetros del sistema
N = 300 #Tamaño de la imagen
dx = 10*um #Tamaño de pixel
w_length = 633*nm #Longitud de onda
r = 25 #píxeles

#Calculamos la transmitancia
image = Mask_Circle(N, r, kind= "O")

center = []
zs = []
#Hacemos el barrido de la intensidad del campo en el eje óptico para diferentes posiciones axiales
for z in range(0,3000,1):
    zs.append(z/100)
    mat = Diffraction(image,z*dx,w_length,dx)
    center.append(np.abs(mat[int(N/2)][int(N/2)]))
#Gráficamos
fig1, axs = plt.subplots(1,1)
axs.plot(zs,center)
axs.set_xlabel('Distancia de Propagación (mm)')
axs.set_ylabel('Amplitud del Campo Difractado')

#Se grafica la amplitud del campo a lo largo de una línea horizontal que pasa por el centro para diferentes posiciones axiales
#escogidas según la gráfica anterior

fig2, axs = plt.subplots(2,3,sharey=True,sharex=True)
patterns=[]
for j in range(500,1100,100):
    patterns.append(Diffraction(image,j*dx,w_length,dx))
axs[0][0].plot(range(0,300),np.abs(patterns[0][150]))
axs[0][1].plot(range(0,300),np.abs(patterns[1][150]))
axs[0][2].plot(range(0,300),np.abs(patterns[2][150]))
axs[1][0].plot(range(0,300),np.abs(patterns[3][150]))
axs[1][1].plot(range(0,300),np.abs(patterns[4][150]))
axs[1][2].plot(range(0,300),np.abs(patterns[5][150]))
axs[1][1].set_xlabel('0.01mm/píxel')

axs[0][0].set_title('z=5mm')
axs[0][1].set_title('z=6mm')
axs[0][2].set_title('z=7mm')
axs[1][0].set_title('z=8mm')
axs[1][1].set_title('z=9mm')
axs[1][2].set_title('z=10mm')


#Se grafica el patrón de difracción en una posición en la cual es evidente el punto brillante en el centro
fig3, axs = plt.subplots(1,1)
Complex_Plot(Diffraction(image,25*mm,w_length,dx),'A',0,axs,fig3)

plt.show()