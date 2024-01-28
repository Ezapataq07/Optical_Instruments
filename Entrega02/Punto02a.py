#Se imortan las librerías relevantes y las unidades
from resources.functions import *

""" A continuación se presenta el analisis a los casos de difracción planteados """

#Parámetros del sistema
L = 100*um
m = 1
N = 600 #Tamaño de la imagen
dx = 10*um #Tamaño de pixel
w_length = 633*nm #Longitud de onda
z = (L**2)/w_length

#Creamos nuestra cuadrícula 
x = np.arange(-N/2,N/2,1)
y = np.arange(-N/2,N/2,1)
X,Y = np.meshgrid(x,y)

X = X*dx
Y = Y*dx

#Calculamos la transmitancia
image = (1/2)*(1 + m* np.cos(2*np.pi * X/L))

#Gráficamos
#Tamaño del recorte
N_cut = 40

fig, axs = plt.subplots()
Complex_Plot(image[:N_cut,:N_cut],"A",0,axs, fig ,colbar= True)
axs.set_title("Imagen original")
axs.set_xlabel("0.01 mm/pixel")
plt.show()

fig, axs = plt.subplots(1, 2)


Complex_Plot(Diffraction(image,z*4,w_length,dx)[:N_cut,:N_cut],"A",0,axs[0],fig, colbar= False)
Complex_Plot(Diffraction(image,z*5,w_length,dx)[:N_cut,:N_cut],"A",0,axs[1], fig,colbar= False)

axs[0].set_title("a) Difracción con N=4")
axs[1].set_title("b) Difracción con N=5")
axs[0].set_xlabel("0.01 mm/pixel")
axs[1].set_xlabel("0.01 mm/pixel")
plt.show()

fig, axs = plt.subplots()
Complex_Plot(Diffraction(image,z*4.5,w_length,dx)[:N_cut,:N_cut],"A",0,axs,fig, colbar= False)
axs.set_title("Difracción con N=4.5")
axs.set_xlabel("0.01 mm/pixel")
plt.show()

print("4: ",4*z, "5: ",5*z, "4.5: ",4.5*z)