
########################### 1a ###################################
import numpy as np
import matplotlib.pyplot as plt

w = np.logspace(-1, 3, num=2000)
s = 1j * w
H1 = 100 / (s + 100)
H2 = s ** 2 / (s ** 2 + 141 * s + 10000)
H = H1 + H2

fig, ax = plt.subplots(2)
ax[0].plot(w, 20 * np.log10(np.abs(H)))
ax[0].set_xscale('log')

P = np.angle(H)
ax[1].plot(w, np.rad2deg(P))
ax[1].set_xscale('log')

D = -np.diff(P) / np.diff(w)
plt.figure()
plt.plot(w[1:], D * 1000)  # ms!!

########################### 1b ###################################
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import helpers

b1 = [100]
a1 = [1, 100]
b2 = [1, 0, 0]
a2 = [1, 141, 10000]
z1, p1, k1 = signal.tf2zpk(b1, a1)  # juste parce que helpers.tf accepte les pôles-zéros-gain plutôt que num-dénum
z2, p2, k2 = signal.tf2zpk(b2, a2)
# voir le code de cette fonction pour comprendre ses opérations
z, p, k = helpers.paratf(z1, p1, k1, z2, p2, k2)
b, a = signal.zpk2tf(z, p, k)
t = np.linspace(0, .2, 2000)
u2 = np.sin(100* t)
t1,y1 = signal.impulse((b,a))
t2,y2,_ = signal.lsim((b,a), u2, t)
fig, ax= plt.subplots(2)
ax[0].plot(t1, y1)
ax[1].plot(t2, y2)


########################### 2a ###################################
import numpy as np
import scipy.signal as signal
import helpers

b1 = [1, 1, 1]  # définition des facteurs du polynome numérateur
a1 = [1, -1, 4]  # définition des facteurs du polynome dénominateur
z2, p2, k2 = signal.tf2zpk(b1, a1)  # passage à la représentation zok (liste de racines + "gain")
helpers.pzmap1(z2, p2, '# 2a')  # voir le contenu de la fonction pour les étapes d'affichage
p2 = [complex(-x.real, x.imag) for x in p2]  # change le signe de la partie réelle
helpers.pzmap1(z2, p2, '# 2b')
b, a = signal.zpk2tf(z2, p2, k2)
w, mag, phdeg = signal.bode((b, a), w=np.logspace(1,3,1000))  # calcul la réponse en fréquence du filtre (H(jw)), fréquence donnée en rad/s
angle = np.deg2rad(phdeg)  # calcul du déphasage en radians
ph = np.unwrap(angle, period=np.pi) if np.__version__ > '1.21' else \
    np.unwrap(2 * angle) / 2
delay = -np.diff(ph) / np.diff(w)
helpers.grpdel1(w, delay, '#2c')


########################### 3a ###################################
import numpy as np
import scipy.signal as signal
import helpers

z = [-2+2j, -2-2j]
p = [-2+1j, -2-1j]
b = np.poly(z)
a = np.poly(p)

########################### 3b ###################################
### LOL je me suis amusé pour l'affichage pas besoin de faire si compliqué
num = ''
for i, val in enumerate(b):
    if num:
        num += ' + '
    if not np.isclose(val, 1.0, atol=1e-6, rtol=1e-6):
        num += f'{val}'
    if i != len(b)-1:
        num += 's'
        if i != len(b) - 2:
            num += f'^{len(b) - i - 1}'
den = ''
for i, val in enumerate(a):
    if den:
        den += ' + '
    if not np.isclose(val, 1.0, atol=1e-6, rtol=1e-6):
        den += f'{val}'
    if i != len(a)-1:
        den += 's'
        if i != len(a) - 2:
            den += f'^{len(a) - i - 1}'
Htext = 'H(s) = '
num = ' ' * len(Htext) + num
den = ' ' * len(Htext) + den
Htext = Htext + '-' * np.amax([len(num), len(den)])
num = ' ' * ((len(Htext) - len(num)) // 2) + num
den = ' ' * ((len(Htext) - len(den)) // 2) + den
print(num)
print(Htext)
print(den)

########################### 3c ###################################
def Hvalue(b, a, w):
    numvalue = np.polyval(b, w*1j)
    denvalue = np.polyval(a, w*1j)
    return np.abs(numvalue/denvalue), np.angle(numvalue/denvalue)
w = 1
mag, ph = Hvalue(b, a, w)
print(f'Valeur de la FT à w = {w} rad/s: gain = {mag} ({20*np.log10(mag)} dB), phase = {ph} rad ({np.rad2deg(ph)} degrés)')



##################################################################
plt.show()

