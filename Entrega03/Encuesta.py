#Importamos las librerias 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

#Traemos los datos de la encuesta
root = "resources/"
file_name = "Percepcion_de_Doble_Imagen.csv"

data = pd.read_csv(root+file_name, header=None, skiprows=1)

#Gráficamos con escala de color
fig, axs = plt.subplots(sharey=True)
N, bins, patches = axs.hist(data.iloc[:,1]*10 + 10, bins=17)
fracs = N / N.max()
norm = colors.Normalize(fracs.min(), fracs.max())

for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)
axs.set_xlabel("Distancia al cuadro (m)")
axs.set_ylabel("Frecuencia")
axs.set_title("Resultados encusta de percepción")

plt.show()
