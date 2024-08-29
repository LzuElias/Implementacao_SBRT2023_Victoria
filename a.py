import numpy as np


hb0 = 0.4636 + 0.8532j
hb1 = 0.6732 + 0.8985j
hb2 = 0.4761 + 0.2556j
hb3 = 0.9525 + 0.1564j
hb = np.array([])
hb = np.append(hb, hb0)
hb = np.append(hb, hb1)
hb = np.append(hb, hb2)
hb = np.append(hb, hb3)
hb = hb.reshape(1,4)


ht0 = 0.3372 + 0.0321j
ht1 = 0.5195 + 0.4528j
ht2 = 0.2519 + 0.4816j
ht3 = 0.3991 + 0.2797j
ht = np.array([])
ht = np.append(ht, ht0)
ht = np.append(ht, ht1)
ht = np.append(ht, ht2)
ht = np.append(ht, ht3)
ht = ht.reshape(1,4)

np.shape(ht)

k1 = 0.7745
k2 = 0.6324

h=(k1*hb) + (k2*ht)
hH = np.conjugate(np.transpose(h))

B1 = 2,1163

c = (3*(10**8))
pi = np.pi
f =  (915*(10**6))
x1 = 6
y2 = 6
alfa = 2.7

Beta1  = (c**2) / (16*(pi**2)*(f**2)*((np.sqrt((x1**2)+(y2**2)))**alfa))

hb_norma = np.abs(hb)



P = 3
N = 4
vetor_hb_hb = hb/hb_norma
Energia = np.sqrt(P/N)
Psi = Energia * (vetor_hb_hb)
