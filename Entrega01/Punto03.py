#Importamos todas las librerias necesarias
from resources.funtions import *

#Función para gráficar todo el sistema 4F
def All_system(image,landa,f,kind):
  #image: Imagen que se ubica en el foco del plano objeto y pasará por la lente 
  #landa: Longitud de onda de la luente de luz que ingresa al sistema
  #f: distancia focal de la lente
  #kind: I=Intensidad, A=Amplitud, P=Fase
  fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
  Complex_Plot(image ,kind,0,axs[0])
  axs[0].set_title("Imagen inicial")
  Complex_Plot(Lenses2F(image,landa,f,1) ,kind,1,axs[1])
  axs[1].set_title("Primera lente escala log")
  Complex_Plot(Lenses4F(image,landa,f) ,kind,0,axs[2])
  axs[2].set_title("Imagen final")
  return

#Definimos unidades
um = 10**(-6)
mm = 10**(-3)
m = 1

#Definimos los parametros y la imagen 
root = "resources/"
filename = "red_heart.jpg"
image = cv2.imread(root + filename, 0)
landa = 0.66*um
f = 2*m

All_system(image, landa, f, "A")
plt.show()