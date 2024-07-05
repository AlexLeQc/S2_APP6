import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import helpers as hp

#####1a

w = np.logspace(-1, 3, num=2000)
s = 1j*w

H1 = 100 / (s + 100)
H2 = s**2 / (s**2 + 141*s + 100**2)

H = H1 + H2

fig, ax = plt.subplots(2)
ax[0].plot(w, 20*np.log10(np.abs(H)))
ax[0].set_xscale('log')

Phase = np.angle(H)
ax[1].plot(w, np.rad2deg(Phase))
ax[1].set_xscale('log')

delay = -np.diff(Phase) / np.diff(w)
plt.figure()
plt.plot(w[1:], delay * 1000)

#######1b)

b1 = [100]
a1 = [1, 100]

b2 = [1, 0, 0]
a2 = [1, 141, 100**2]

z1, p1, k1 = signal.tf2zpk(b1, a1)
z2, p2, k2 = signal.tf2zpk(b2, a2)

z, p, k = hp.paratf(z1, p1, k1, z2, p2, k2)
b, a = signal.zpk2tf(z, p, k)

t = np.linspace(0, .2, num=2000)
u2 = np.sin(100*t)

t1, y1 = signal.impulse((b, a))
t2, y2, _ = signal.lsim((b, a), u2, t)

fig1, ax = plt.subplots(2)
ax[0].plot(t1, y1)
ax[1].plot(t2, y2)


######2

b3 = [1, 1, 1]
a3 = [1, -1, 4]

z3, p3, k3 = signal.tf2zpk(b3, a3)
hp.pzmap1(z3, p3, '#2a')
p3 = [complex(-x.real, x.imag) for x in p3]
hp.pzmap1(z3, p3, '#2b')


b, a = signal.zpk2tf(z3, p3, k3)

w, mag, phdeg = signal.bode((b, a), w=np.logspace(1, 3, num=2000))
angle = np.deg2rad(phdeg)
ph = np.unwrap(angle, period=np.pi)
delay = -np.diff(ph) / np.diff(w)
hp.grpdel1(w, delay, '#2 c)')
plt.show()



