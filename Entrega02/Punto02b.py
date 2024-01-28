#Se imortan las librerías relevantes y las unidades
from resources.functions import *

#Se genera una abertura y un obstáculo circulares, siendo el obstáculo más grande
aperture=Mask_Circle(700,100)
obstacle=Mask_Circle(700,200,'O')

#Se crea un obstáculo en forma de dona a partir de lo anterior
donut=aperture+obstacle

#Se calculan los patrones de difracción de la apertura circular, el obstáculo circular y el obstáculo en forma de dona

U_aperture=Diffraction(aperture,80*mm,633*nm,10*um)
U_obstacle=Diffraction(obstacle,80*mm,633*nm,10*um)
U_donut=Diffraction(donut,80*mm,633*nm,10*um)

#Se calcula el patrón de difracción del obstáculo en forma de dona aplicando el principio de Babinet
U_donut_babinet=U_aperture+U_obstacle

#Se grafican los resultados para los patrones de difracción de la dona y se comprueba su similitud restando
#los campos
fig, axs = plt.subplots(1, 3,figsize=(15,10))

Complex_Plot(U_donut,"A",0,axs[0], fig ,colbar= False)
Complex_Plot(U_donut_babinet,"A",0,axs[1],fig, colbar= False)
Complex_Plot(U_donut-U_donut_babinet,"A",0,axs[2],fig, colbar= True)
axs[0].set_title("Patrón Original")
axs[1].set_title("Patrón Calculado con Babinet")
axs[2].set_title("Resta")

plt.show()