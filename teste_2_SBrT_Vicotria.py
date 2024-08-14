
#%% Definições

# - Beta_k = ganho médio de potência no canal entre o PB e o k-ésimo dispositivo;
# - s_k = sinal transmitido (variável aleatória gaussiana de média zero e variancia unitária);
# - Psi_k = vetor beamforming analógico (Pertence aos complexos com tamanho 1xN);
# - ||Psi_k||² = Potencia de transmissão (Pt);
# - h_k = Vetor de canal (pertence aos complexos com tamanho 1xN);
# - kappa = fator de rice;

# OBS: k = K. j = 1 e vai até j = k-1


#%%

import numpy as np
import matplotlib.pyplot as plt
import random


K = 10                                  # N° dispositivos
N = 4                                  # N° de antenas
R = 10                                  # Raio [m]
f = 915 *(10**6)                        # Freq. da portadora
kappa = 1.5                             # Fator Rice
mu = 10.73 * 10**(-3)                   # Max pot no IoT
a = 0.2308                              # constante circuito a
b = 5.365                               # constante circuito b  
Pt = 3                                  # Max pot. de transmissão
E_min = 200 * (10**-6)                # Min Energia no IoT [J]
alpha = 2.7                             # Coef. perda de percurso
c = 3*(10**(8))                         # Vel. da luz
Omega = 1/(1+np.exp(a*b))               # Constante resposta 0_in-0_out
Pot_k = 0
tau_k_estrela_k_igual_1_valor = 0
tau_k_valor = 0


def dist_euclid_quad (x, y):
    d_k = np.sqrt(((x**2)+(y**2))) 
    return d_k

def Beta_k(d_k):
    Beta_k = (c**2)/(16 * (np.pi**2) * (f**2) * (d_k**alpha))
    return Beta_k

def h_barra_k(N): # Gerador componente LoS
    h_barra_k = np.random.rand(1,N) + 1j*np.random.rand(1,N)
    return h_barra_k

def h_til_k(N): # Gerador de componente NLoS
    h_til_k = (np.random.rand(1,N) + 1j*np.random.rand(1,N))*(1/(np.sqrt(2)))
    return h_til_k

def canal_h_k (kappa, h_barra_k, h_til_k): # Gerador de canal
    canal_h_k = (np.sqrt((kappa)/(1+kappa)) * h_barra_k) + (np.sqrt((1)/(1+kappa)) * h_til_k)
    return canal_h_k

def P_k(Beta_k, Psi_k, canal_h_k_Hermitiano): # Potencia de RF recebida
    P_k = Beta_k * (np.abs((Psi_k) *(canal_h_k_Hermitiano)))**2
    return P_k

def Gamma_k(P_k): # Parâmetro para calcular a Energia coletada
    Gamma_k = mu / (1 + np.exp(-a*(P_k-b)))
    return Gamma_k

def Phi_k(tau_k, Gamma_k): # Energia coletada
    Phi_k = tau_k * ((Gamma_k - (mu*Omega))/(1-Omega))
    return Phi_k

def tau_k_estrela_k_igual_1(Gamma_k):
    tau_k_estrela_k_igual_1 = (E_min * (1 - Omega))/(Gamma_k - (mu*Omega))
    return tau_k_estrela_k_igual_1



#%%
canal_h = np.array([])
h_barra = np.array([])
h_til = np.array([])
Beta = np.array([])
Phi = np.array([])
tau = np.array([])
Psi_k_estrela = np.array([])
Energia = np.array([])
h_barra_dividido_pela_norma_vetor = np.array([])
h_barra_vetor = np.array([])
Pot = np.array([])
Gamma = np.array([])
Gamma_k_j = np.array([])
Phi_NEIG_j = np.array([])
tau_k_estrela_maior_que_1 = np.array([])
tau_k_estrela = np.array([])
Phi_k_vetor = np.array([])
Phi_linha_k = np.array([])
tau_total = np.array([])
tau_total_vetor = np.array([])
H = np.zeros((K,N))



seed = np.random.seed(9)


for k in range (0, K):
    # Canal h
    
    h_barra = h_barra_k(N)
    h_til = h_til_k(N)
    canal_h = canal_h_k(kappa, h_barra, h_til)
    h_k = np.conjugate(np.transpose(canal_h))
    
    # Beta
    x = np.random.randint(1,10)
    y = np.random.randint(1,10)
    Beta = Beta_k(dist_euclid_quad(x,y))
    
    Psi_k_estrela = (np.sqrt(Pt/N)) *(h_barra/np.abs(h_barra))
    
    
    Pot = np.append(Pot, Beta*(np.abs(Psi_k_estrela @ h_k))**2)
    
    
    
    
    