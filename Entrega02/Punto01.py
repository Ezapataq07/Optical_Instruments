#Se imortan las librerías relevantes y las unidades
from resources.functions import *

""" A continuación se presenta un ejemplo del funcionamiento de los diferentes métodos de cálculo 
    para la difracción de una apertura circular de diametro 3mm a una distancia de 60mm para espectro angular 
    y de 300mm para el método de transformada de Fresnel, iluminando
    con luz de 650nm (ir al documento donde se compara el resultado con el resultado real) """

#Se crea computacionalmente la apertura circular (el diámetro se acomoda con las dimensiones de muestreo)
circ_aperture= Mask_Circle(700,100)

#Se definen los parámetros
zF = 300*mm
zAS = 100*mm
w_length = 633*nm
dx = 10*um

#Definimos los patrones de difracción para cada método
U_AS_FFT = Angular_Spectrum_FFT(circ_aperture, zAS, w_length, dx)
U_AS_DFT = Angular_Spectrum_DFT(circ_aperture, zAS, w_length, dx)
U_F_FFT = Fresnel_Transform_FFT(circ_aperture, zF, w_length, dx)
U_F_DFT = Fresnel_Transform_DFT(circ_aperture, zF, w_length, dx)

#Gráficamos
fig, axs = plt.subplots(2, 2,sharey=True)
Complex_Plot(U_AS_FFT,"A",0,axs[0][0], fig, True)
axs[0][0].set_title("a) Espectro ángular FFT a 100mm")
Complex_Plot(U_AS_DFT,"A",0,axs[0][1], fig, True)
axs[0][1].set_title("b) Espectro ángular DFT a 100mm")
Complex_Plot(U_F_FFT,"A",0,axs[1][0], fig, True)
axs[1][0].set_title("c) Transformada de Fresnel FFT a 300mm")
Complex_Plot(U_F_DFT,"A",0,axs[1][1], fig, True)
axs[1][1].set_title("d) Transformada de Fresnel DFT a 300mm")


fig.suptitle('Difracción escalar con 0.01mm/pixel')
fig.tight_layout()
plt.show()

times1_FFT = []
times1_DFT = []
times2_FFT = []
times2_DFT = []
Ns = []
for N in range(300, 700,2):
    ini1_FFT = time.time()
    Angular_Spectrum_FFT(Mask_Circle(N,100), zAS, w_length, dx)
    end1_FFT = time.time()
    times1_FFT.append(end1_FFT-ini1_FFT)
    ini1_DFT = time.time()
    Angular_Spectrum_DFT(Mask_Circle(N,100), zAS, w_length, dx)
    end1_DFT = time.time()
    times1_DFT.append(end1_DFT-ini1_DFT)
    ini2_FFT = time.time()
    Fresnel_Transform_FFT(Mask_Circle(N,100), zF, w_length, dx)
    end2_FFT = time.time()
    times2_FFT.append(end2_FFT-ini2_FFT)
    ini2_DFT = time.time()
    Fresnel_Transform_DFT(Mask_Circle(N,100), zF, w_length, dx)
    end2_DFT = time.time()
    times2_DFT.append(end2_DFT-ini2_DFT)
    Ns.append(N/100)

fig, axs = plt.subplots(1, 2, sharey=True)
axs[0].plot(Ns,times1_FFT, label = "FFT")
axs[0].plot(Ns,times1_DFT, label = "DFT")
axs[0].set_title("a) Espectro angular a 100mm")
axs[0].set_xlabel("Tamaño de la imagen (mm)")
axs[0].set_ylabel("Tiempo de cálculo (s)")
axs[0].legend()

axs[1].plot(Ns,times2_FFT, label = "FFT")
axs[1].plot(Ns,times2_DFT, label = "DFT")
axs[1].set_title("b) Transformada de Fresnel a 300mm")
axs[1].set_xlabel("Tamaño de la imagen (mm)")
axs[1].set_ylabel("Tiempo de cálculo (s)")
axs[1].legend()

plt.show()
