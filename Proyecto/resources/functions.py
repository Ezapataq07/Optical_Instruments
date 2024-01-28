import numpy as np
import matplotlib.pyplot as plt
import cv2

#Definimos unidades
nm = 1e-9
um = 1e-6
mm = 1e-3
cm = 1e-2
m = 1

#Función para crear rejillas circulares con diferentes coordenadas y tamaños
def Mask_Circle(shape,r,kind='A',coordx=0,coordy=0):
    #shape dimensiones de la matriz (2-tupla)
    #r es el radio del circulo (en píxeles)
    #kind: 'A' si es una apertura, 'O' si es un obstáculo
    #coordx y coordy son las coordenadas del centro en el eje x y y respectivamente. Definidas en el centro de la imagen por defecto
    if (coordx==0):
        coordx=int(shape[1]/2)
    if (coordy==0):
        coordy=int(shape[0]/2)

    if (kind=='A'):
        circ_aperture=np.zeros(shape,dtype="uint8")
        cv2.circle(circ_aperture,(coordx,coordy),r,1,-1)
        return circ_aperture
    elif (kind=='O'):
        circ_obstacle=np.ones(shape,dtype="uint8")
        cv2.circle(circ_obstacle,(coordx,coordy),r,0,-1)
        return circ_obstacle

#Función para gráficar matrices complejas 
def Complex_Plot(matrix,kind,log,axs,fig = 0, colbar = False):
  #matrix: matriz a gráficar
  #kind: I=Intensidad, A=Amplitud, P=Fase
  #log: 1 si se desea escala logarítmica 0 en caso contrario
  plt.set_cmap('gist_gray')
  if (kind=='I'):
    matrix_to_plot=np.abs(matrix)**2
  elif (kind=='A'):
    matrix_to_plot=np.abs(matrix)
  elif (kind=='P'):
     matrix_to_plot=np.angle(matrix)
  if (log==1):
    image = axs.imshow(np.log(matrix_to_plot + 0.0000001))
  elif (log==0):
    image = axs.imshow(matrix_to_plot)
  if (colbar):
    fig.colorbar(image, ax = axs)
  return 

""" Función de Difracción Discreta por Transformada de Fresnel usando FFTs"""
def Fresnel_Transform(matrix,z,w_length,dx0):
    # matrix es la representación discreta de la imagen o función 
    # z es la distancia de propagación (metros)
    # w_length es la longitud de onda (m)
    # dx0 es el intervalo de muestreo de la función en el plano de entrada

    #Se define la dimensión de la imagen (solo se utiliza N por ser cuadrada)
    N=np.shape(matrix)[0]

    #Se calcula dx, el intervalo de muestreo en el plano de salida
    dx=(w_length*z)/(dx0*N)

    #Se definen las coordenadas del plano de entrada con el fin de aplicar las fases esféricas de entrada
    k=(2*np.pi)/w_length
    x0=np.arange(-int(N/2),int(N/2),1)
    y0=np.arange(-int(N/2),int(N/2),1)
    n0,m0=np.meshgrid(x0,y0)

    #Se aplican las fases esféricas de entrada
    U_prima=matrix*np.exp((1j*k/(2*z))*((dx0*n0)**2+(dx0*m0)**2))

    #Se calcula la transformada de Fourier de la función imediatamente anterior
    U_2prima= np.fft.fftshift(np.fft.fftn(U_prima))
    U_2prima=(dx**2)*U_2prima

    #Se definen las coordenadas del plano de salida con el fin de aplicar las fases esféricas de salida
    x=np.arange(-int(N/2),int(N/2),1)
    y=np.arange(-int(N/2),int(N/2),1)
    n,m=np.meshgrid(x,y)

    #Se aplican las fases esféricas de salida y se obtiene el campo difractado a un distancia z
    U_z=U_2prima*(np.exp(1j*k*z)/(1j*w_length*z))*np.exp((1j*k/(2*z))*((dx*n)**2+(dx*m)**2))
    return U_z,dx




def Inverse_Fresnel_Transform(matrix,z,w_length,dx):
    # matrix es la representación discreta de la imagen o función 
    # z es la distancia de propagación (metros)
    # w_length es la longitud de onda (m)
    # dx0 es el intervalo de muestreo de la función en el plano de entrada

    #Se define la dimensión de la imagen (solo se utiliza N por ser cuadrada)
    N=np.shape(matrix)[0]

    #Se calcula dx, el intervalo de muestreo en el plano de salida
    dx0=(w_length*z)/(dx*N)

    #Se definen las coordenadas del plano de entrada con el fin de aplicar las fases esféricas de entrada
    k=(2*np.pi)/w_length
    x0=np.arange(-int(N/2),int(N/2),1)
    y0=np.arange(-int(N/2),int(N/2),1)
    n0,m0=np.meshgrid(x0,y0)

    #Se aplican las fases esféricas de entrada
    U_prima=matrix*np.exp((-1j*k/(2*z))*((dx*n0)**2+(dx*m0)**2))

    #Se calcula la transformada de Fourier de la función imediatamente anterior
    U_2prima= np.fft.ifftn(U_prima)
    U_2prima=(dx0**2)*U_2prima

    #Se definen las coordenadas del plano de salida con el fin de aplicar las fases esféricas de salida
    x=np.arange(-int(N/2),int(N/2),1)
    y=np.arange(-int(N/2),int(N/2),1)
    n,m=np.meshgrid(x,y)

    #Se aplican las fases esféricas de salida y se obtiene el campo difractado a un distancia z
    U_z=U_2prima*(np.exp(1j*k*z)/(1j*w_length*z))*np.exp((-1j*k/(2*z))*((dx0*n)**2+(dx0*m)**2))
    return U_z

def Random_Phase_Plate(shape):
    #Shape: tupla correspondiente a las dimensiones del plato de fase aleatoria
    random_nums=np.random.uniform(low=0.0, high=1.0, size=shape)
    plate=np.exp(1j*2*np.pi*random_nums)
    return plate

def PlaneWave(M,N,angleX,angleY,dx,dy,w_length):
    Mcenter=int(M/2)
    Ncenter=int(N/2)
    
    x=np.arange(-Mcenter+1,Mcenter+1)
    y=np.arange(-Ncenter+1,Ncenter+1)
    
    X,Y=np.meshgrid(x,y)
    
    k=2*np.pi/w_length
    
    Ax=np.cos(angleX)
    Ay=np.cos(angleY)
    
    wave=np.exp(1j*k*(Ax*X*dx+Ay*Y*dy))
    
    return wave


def Fresnel_Bluestein(matrix,z,w_length,d0,df):
    N=np.shape(matrix)[0]
    k=(2*np.pi)/w_length
    
    x0=np.arange(-int(N/2),int(N/2),1)
    y0=np.arange(-int(N/2),int(N/2),1)
    n0,m0=np.meshgrid(x0,y0)
    
    di=1/(N*d0)
    
    f1=matrix*np.exp((1j*k/(2*z))*(d0*(d0-df)*n0**2+d0*(d0-df)*m0**2))    
    F1=np.fft.fftn(f1)
    
    f2=np.exp((1j*k/(2*z))*(d0*df*n0**2+d0*df*m0**2))
    F2=np.fft.fftn(f2)
    
    x=np.arange(-int(N/2),int(N/2),1)
    y=np.arange(-int(N/2),int(N/2),1)
    n,m=np.meshgrid(x,y)
    
    u1=np.exp(1j*k*z)/(1j*w_length*z)
    u2=np.exp((-1j*k/(2*z))*(df*(d0-df)*n**2+df*(d0-df)*m**2))
    u3=np.fft.ifftn(F1*F2)
    
    U_z=u1*u2*u3
    
    return np.fft.fftshift(U_z)

def Inverse_Fresnel_Bluestein(matrix,z,w_length,d0,df):
    N=np.shape(matrix)[0]
    k=(2*np.pi)/w_length
    
    
    di=1/(N*d0)
    
    x0=np.arange(-int(N/2),int(N/2),1)
    y0=np.arange(-int(N/2),int(N/2),1)
    n0,m0=np.meshgrid(x0,y0)
    
    f1=matrix*np.exp((-1j*k/(2*z))*(df*(df-d0)*n0**2+df*(df-d0)*m0**2))    
    F1=np.fft.ifftn(f1)
    
    f2=np.exp((-1j*k/(2*z))*(df*d0*n0**2+df*d0*m0**2))
    F2=np.fft.ifftn(f2)
    
    x=np.arange(-int(N/2),int(N/2),1)
    y=np.arange(-int(N/2),int(N/2),1)
    n,m=np.meshgrid(x,y)
    
    u1=np.exp(1j*k*z)/(1j*w_length*z)
    u2=np.exp((1j*k/(2*z))*(d0*(df-d0)*n**2+df*(df-d0)*m**2))
    u3=np.fft.fftn(F1*F2)
    
    U_z=u1*u2*u3
    
    return np.fft.fftshift(U_z)