from resources.functions import *

##################### Parámetros del sistema ###########################
w_length=633*nm
dx=5*um #Tamaño de imagen al rededor de los 0.5cm
z1,z2,z3 = 0.1,0.2,0.3 #Distancias del sistema
NDF1, NDF2 = 0.00001,1

##################### Funciones útiles ###########################
#Brazo del interferometro donde se encripta
def Object_Arm(image,r1,r2,z1,z2,w_length,dx,NDF1):
    U1,dx1=Fresnel_Transform(NDF1*image*r1,z1,w_length,dx)
    UCCDo,dx2=Fresnel_Transform(U1*r2,z2,w_length,dx)
    
    return UCCDo,dx1,dx2
#Brazo del interferometro donde se oculta
def Host_Arm(host, phase, z3, w_length, dx0,NDF2):
    imageh = NDF2*host*np.exp(1j*phase)
    UCCD_host, dxf = Fresnel_Transform(imageh,z3,w_length,dx0)

    return UCCD_host, dxf

##################### Campos iniciales ###########################
#Host (Bob esponja)
bob=cv2.imread('Proyecto/resources/bob.jpg',0) #Imagen de Bob esponja (Host)
Complex_Plot(bob,'A',0, plt) 
plt.title("Imagen host (Bob espoja)")
plt.xlabel("dx = 5um")
plt.show()

#Imagen principal (El beso)
beso=cv2.imread('Proyecto/resources/beso.jpg',0)[28:928,200:1100] #Imagen del beso (A encriptar) recortada
Complex_Plot(beso,'A',0, plt)
plt.title("Imagen principal (El beso)")
plt.xlabel("dx = 5um")
plt.show()

#Tamaño de las imagenes
shape=np.shape(bob)

#Creamos los platos de fase aleatoria
r1=Random_Phase_Plate(shape)
r2=Random_Phase_Plate(shape)

##################### Interferencia ###########################
#Hacemos interferir los brazos del interferometro
UCCD,dx1,dx2=Object_Arm(beso,r1,r2,z1,z2,w_length,dx,NDF1)

UCCD_host, dx3 = Host_Arm(bob, np.pi, z3, w_length, dx, NDF2)

Complex_Plot(UCCD+UCCD_host,'I',0, plt)
plt.title("Patrón de interferencia (En intensidad)")
plt.xlabel("dx = "+ str(int(dx3/um))+ "um")
plt.show()

#Capturamos intensidad en el CCD (despues de la lente)
host=Inverse_Fresnel_Transform(UCCD+UCCD_host,z3,w_length,dx3)
Complex_Plot(host,'I',0,plt)
plt.title("Imagén capturada en CCD")
plt.xlabel("dx = 5um")
plt.show()

##################### PHASE SHIFTING ###########################
#Imagen con diferente fase
U_host0,dx_h0= Host_Arm(bob, 0, z3, w_length, dx, NDF2)
U_host90,dx_h90= Host_Arm(bob, np.pi/2, z3, w_length, dx, NDF2)
U_host180,dx_h180= Host_Arm(bob, np.pi, z3, w_length, dx, NDF2)

#Creamos una onda plana 
PW=PlaneWave(shape[0],shape[1],np.pi/2,np.pi/2,dx,dx,w_length)
Complex_Plot(PW,'P',0,plt)
plt.title("Fase de la onda plana")
plt.xlabel("dx= 5um")
plt.show()

#Registros de intensidad a diferentes fases
I0=np.abs(U_host0+PW)**2
I90=np.abs(U_host90+PW)**2
I180=np.abs(U_host180+PW)**2

#Reconstrucción del host 
host_phase=np.angle(PW)+np.arctan2((2*I90-I0-I180),(I180-I0))
host_amplitude=np.sqrt((1/2)*(I0+I180)-np.abs(PW)**2)

#Vemos las imagenes
fig, axs = plt.subplots(1,2)

Complex_Plot(U_host0,'P',0,axs[0])
axs[0].set_title("Host original en fase")

U_host_rec=host_amplitude*np.exp(1j*host_phase)
Complex_Plot(U_host_rec,'P',0,axs[1])
axs[1].set_title("Host reconstruido en fase")
plt.show()

bob_rec=Inverse_Fresnel_Transform(U_host_rec,z3,w_length,dx_h0)
Complex_Plot(bob_rec,'A',0,plt)
plt.title("Host reconstruido")
plt.show()

############################# DECRYPTION ###########################################3
#Aplicamos un phase shifting para obtener los campos
UCCD,dx1,dx2=Object_Arm(beso,r1,r2,z1,z2,w_length,dx,NDF1)

#Registros de intensidad
I0=np.abs(UCCD+U_host0)**2
I90=np.abs(UCCD+U_host90)**2
I180=np.abs(UCCD+U_host180)**2

obj_phase=np.angle(U_host_rec)-np.arctan2((2*I90-I0-I180),(I180-I0))
obj_amplitude=np.sqrt(1/2*(I0+I180)-np.abs(U_host_rec)**2)

object_rec=obj_amplitude*np.exp(1j*obj_phase)

#Hacemos todo el sistema a la inversa
o1=Inverse_Fresnel_Transform(object_rec,z2,w_length,dx2)
o1=o1*np.conjugate(r2)
object_dec=Inverse_Fresnel_Transform(o1,z1,w_length,dx1)
object_dec=object_dec*np.conjugate(r1)

Complex_Plot(np.fft.fftshift(object_dec),'A',0,plt)
plt.title("Imagen obtenida al desencriptar")
plt.show()


#Veamos el registro de intensidad con la modificación
UCCD_End_0 = Inverse_Fresnel_Transform(UCCD+U_host0,z3,w_length,dx_h0)
UCCD_End_90 = Inverse_Fresnel_Transform(UCCD+U_host90,z3,w_length,dx_h90)
UCCD_End_180 = Inverse_Fresnel_Transform(UCCD+U_host180,z3,w_length,dx_h180)

I0=np.abs(UCCD_End_0)**2
I90=np.abs(UCCD_End_90)**2
I180=np.abs(UCCD_End_180)**2

#Ponemos en escala de bits
#I0 = ((I0-I0.min())*255/(I0.max()-I0.min()))//1
#I90 = ((I90-I90.min())*255/(I90.max()-I90.min()))//1
#I180 = ((I180-I180.min())*255/(I180.max()-I180.min()))//1
I0=I0/(np.max(I0))
I90=I90/(np.max(I90))
I180=I180/(np.max(I180))



paquet = cv2.merge([I0,I90,I180])
print(paquet)
plt.imshow(paquet)
plt.title('RGB Codificado')
plt.show()