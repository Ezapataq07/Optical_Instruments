
from resources.functions import *

w_length=633*nm

bob=cv2.imread('Proyecto/resources/elon.jpg',0)[:900,700:1600]
Complex_Plot(bob,'A',0, plt)
plt.show()
beso=cv2.imread('Proyecto/resources/bezos.jpg',0)[:900,700:1600]
Complex_Plot(beso,'A',0, plt)
plt.show()
shape=np.shape(bob)
dx=5*um #tamaño de imagen al rededor de los 0.5cm

r1=Random_Phase_Plate(shape)
r2=Random_Phase_Plate(shape)

def Object_Arm(image,r1,r2,z1,z2,w_length,dx,C1):
    U1=Fresnel_Bluestein(C1*image*r1,z1,w_length,dx,dx)
    UCCDo=Fresnel_Bluestein(U1*r2,z2,w_length,dx,dx)
    
    return UCCDo

def Host_Arm(host, phase, NDF2, z3, w_length, dx0):
    imageh = NDF2*host*np.exp(1j*phase)
    UCCD_host = Fresnel_Bluestein(imageh,z3,w_length,dx,dx)

    return UCCD_host

UCCD=Object_Arm(beso,r1,r2,4*cm,4*cm,w_length,dx,0.00000000001)

UCCD_host = Host_Arm(bob, 0, 1, 5*cm, w_length, dx)


#Phase Shifting

U_host0= Host_Arm(bob, 0, 1, 5*cm, w_length, dx)
U_host90= Host_Arm(bob, np.pi/2, 1,  5*cm, w_length, dx)
U_host180= Host_Arm(bob, np.pi, 1,  5*cm, w_length, dx)

PW=PlaneWave(900,900,np.pi/2,np.pi/2,dx,dx,w_length)

I0=np.abs(U_host0+PW)**2
I90=np.abs(U_host90+PW)**2
I180=np.abs(U_host180+PW)**2

host_phase=np.angle(PW)+np.arctan2((2*I90-I0-I180),(I180-I0))
host_amplitude=np.sqrt((1/2)*(I0+I180)-np.abs(PW)**2)

U_host_rec=host_amplitude*np.exp(1j*host_phase)

bob_rec=Inverse_Fresnel_Bluestein(U_host_rec,5*cm,w_length,dx,dx)


fig3, axs = plt.subplots(1, 2,sharey=True)
Complex_Plot(bob,'A',0,axs[0],fig3)
axs[0].set_title('Imagen Original')
Complex_Plot(bob_rec,'A',0,axs[1],fig3)
axs[1].set_title('Imagen Reconstruida')
fig3.supxlabel(r'5$\mu m$ por píxel')
plt.show()

# Decryption

UCCD=Object_Arm(beso,r1,r2,4*cm,4*cm,w_length,dx,0.00000000001)
I0=np.abs(UCCD+U_host0)**2
I90=np.abs(UCCD+U_host90)**2
I180=np.abs(UCCD+U_host180)**2


fig4, axs = plt.subplots(1, 1,sharey=True)
Complex_Plot(I0,'A',0,axs,fig4)
axs.set_title('Patrón de Difracción con Información Oculta')
fig4.supxlabel(r'11.25$\mu m$ por píxel')
plt.show()


host_with_object=Inverse_Fresnel_Bluestein(UCCD+U_host0, 5*cm,w_length,dx,dx)
fig4, axs = plt.subplots(1, 1,sharey=True)
Complex_Plot(host_with_object,'I',0,axs,fig4)
axs.set_title('Imagen Anfitriona con Información Encriptada y Oculta')
fig4.supxlabel(r'5$\mu m$ por píxel')
plt.show()

obj_phase=np.angle(U_host0)-np.arctan2((2*I90-I0-I180),(I180-I0))
obj_amplitude=np.sqrt(1/2*(I0+I180)-np.abs(U_host0)**2)

object_rec=obj_amplitude*np.exp(1j*obj_phase)
o1=Inverse_Fresnel_Bluestein(object_rec,4*cm,w_length,dx,dx)
o1=o1*np.conjugate(r2)
object_dec=Inverse_Fresnel_Bluestein(o1,4*cm,w_length,dx,dx)
object_dec=object_dec*np.conjugate(r1)

Complex_Plot(object_dec,'A',0,plt)
plt.title('Imagen huésped reconstruida')
plt.xlabel(r'5$\mu m$ por píxel')
plt.show()