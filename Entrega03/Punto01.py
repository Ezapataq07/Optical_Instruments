#Se imortan las librerías relevantes y las unidades
from resources.functions import *

#Se importa la imagen del lienzo
#Definimos los parametros y la imagen 
root = "Entrega03/resources/"
filename = "gala.jpg"
gala = cv2.imread(root+filename, 0)

#Se grafica el lienzo para determinar el tamaño de los macropixeles
fig1, axs = plt.subplots( sharey=False)
Complex_Plot(gala,'A',0,axs,fig1)
axs.set_xlabel('1.48mm/píxel')
plt.show()

#Se definen los tamaños de píxel de la imagen y del Plano de Fourier
dx=dy=1.48*mm
dfx=1/(np.shape(gala)[1]*dx)
dfy=1/(np.shape(gala)[0]*dy)

#Se define la resolucion que se desea obtener y se calculan los radios de la transmitancia que filtrará las frecuencias
#espaciales en el Plano de Fourier
resolution=237*mm
rx=int(1/(resolution*dfx))
ry=int(1/(resolution*dfy))

#Se genera la transmitancia en forma de elipse
mask=Mask_Ellipse(np.shape(gala),rx,ry,kind='A')

#Se grafica la transmitancia
fig2, axs = plt.subplots(1, 1)
Complex_Plot(mask,'A',0,axs,fig2)
plt.show()

#Se hace la transformada de fourier del lienzo, se multiplica por la transmitancia y luego se hace la transformada inversa
#para obtener la imagen con la resolución definida previamente
gala_fft=(dx**2)*np.fft.fftshift(np.fft.fftn(gala))
gala_filtered=gala_fft*mask
gala_final=(dfx*dfy)*np.fft.ifftn(gala_filtered)

fig3, axs = plt.subplots(1, 1)
Complex_Plot(gala_final,'A',0,axs,fig3)
plt.show()
# A continuación se hará el filtrado espacial de frecuencias simulando diferentes distancias de alejamiento, la resolución
# en cada distancia se calcula teniendo en cuenta que la resolución del ojo humano es de 5 min de arco

def Resolution(distance):
    return 2*distance*np.tan(0.000727)

def Gala_At_Distances(img, rang,dx,dy):
    dfx=1/(np.shape(img)[1]*dx)
    dfy=1/(np.shape(img)[0]*dy)
    images=[]
    for i in rang:
        resolution=Resolution(i)
        rx=int(1/(resolution*dfx))
        ry=int(1/(resolution*dfy))
        mask=Mask_Ellipse(np.shape(img),rx,ry,kind='A')
        img_fft=(dx**2)*np.fft.fftshift(np.fft.fftn(img))
        img_filtered=img_fft*mask
        img_final=(dfx*dfy)*np.fft.ifftn(img_filtered)
        
        plt.figure()
        plt.imshow(np.abs(img_final))
        plt.title(f'{i} metros de alejamiento')
        
    return 
        
#Gala_At_Distances(gala,range(180,10,-10),dx,dy) #Descomentar esta línea de código para generar 17 imágenes con 
                                                #pérdida de resolución progresiva desde los 20m a 180m con saltos de 10m

