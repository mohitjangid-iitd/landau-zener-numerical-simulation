import numpy as np
import matplotlib.pyplot as plt

v=1                               #variables
g=5
ohm=1
phi=np.pi/2*0
(h_bar,time,dt)=(1,100,0.01)      #fixed

psi=np.array([1,0])               #initial state (ground state)

sigma_z=np.array([[1,0],
                  [0,-1]])        #pauli matrices
sigma_x=np.array([[0,1],
                  [1,0]])

def Hmltn (t):
    return (v*t)/2*sigma_z+ g*np.cos(ohm*t + phi)*sigma_x
    #time dependent hamiltonian

t=np.arange(-time,time,dt)        #time interval

'''finding energy eigen values and plotting E vs t graph'''

eigenvalues_list1= []
eigenvalues_list2= []

for i in t:
    d=Hmltn(i)
    eigval, _ = np.linalg.eig(d)                    #finding the eigen values
    eig1 = eigval[0]
    eig2 = eigval[1]
    if eig1 >0 and eig2<0:
        eigenvalues_list1.append(eig1)
        eigenvalues_list2.append(eig2)
    else:
        eigenvalues_list1.append(eig2)
        eigenvalues_list2.append(eig1)
        
eigenvalues_array1 = np.array(eigenvalues_list1)    #array of all +ve eigen values
eigenvalues_array2 = np.array(eigenvalues_list2)    #array of all -ve eigen values

plt.figure(1)                                       #plotting
plt.plot(t,eigenvalues_array1,color='red',lw=1)
plt.plot(t,eigenvalues_array2,color='blue',lw=1)
plt.title(f"Energy vs time graph for v={v}, g={g} and omega ={ohm}")
plt.xlabel("Time")
plt.ylabel("Energy Eigenvalues")
plt.xlim(-20,20)
plt.ylim(-10,10)
plt.text(-15, 7.5, 'ES')
plt.text(15, -7.5, 'ES')
plt.text(15, 7.5, 'GS')
plt.text(-15, -7.5, 'GS')
# plt.show()
'''finding transition probabilities and plotting prob. vs t graph'''


def RK4_step(psi,t,dt):                             #4th order Runge-Kutta numerical method
    def G(psi,t):
        H=Hmltn(t)
        return(-1j/h_bar)*(H@psi)
    k1=G(psi,t)
    k2=G(psi+k1*dt/2,t+dt/2)
    k3=G(psi+k2*dt/2,t+dt/2)
    k4=G(psi+k3*dt,t+dt)
    return dt*(k1+2*k2+2*k3+k4)/6                   #dt*slope

psi_t=[]
for i in t:
    psi=psi+RK4_step(psi, i, dt)                    #psi (n+1) = psi (n) + dt*slope
    psi_t.append(psi.tolist())
    
psi_t_array=np.array(psi_t)                         #array of time dependent wave functions (eigen values)

Nf=[]
for mat in psi_t_array:
    Norm=np.linalg.norm(mat)
    Nf.append(Norm.tolist())
    
Nf_array=np.array(Nf)                               #array of normalization factors

div = [list(x/y) if y != 0 else len(x)*[0,] for x, y in zip(psi_t_array, Nf_array)]
#normalizing the eigen values

div_arr = np.asarray(div)                           #array of normalized time dependent wave functions

c = div_arr[:,0]                                    #extracting prob. amplitudes for ground state

mod_c_sq=[]
for i in c:
    C=np.linalg.norm(i)
    sq = C**2
    mod_c_sq.append(sq.tolist())
     
mod_c_sq_arr = np.array(mod_c_sq)                   #array of probabilities
p=mod_c_sq_arr[-1]
q=round(p,4)
print(q)
print([(round(t[-i],4), round(mod_c_sq_arr[-i], 4)) for i in range(0, 5)])

plt.figure(2)                                       #plotting
plt.plot(t,mod_c_sq_arr)
plt.title(f"Transition Prob. vs time graph for v={v}, g={g} and omega ={ohm}")
plt.xlabel("Time")
plt.ylabel("Transition Prob.")
plt.xlim(-100,100)
plt.text(75, q ,q )
plt.show()