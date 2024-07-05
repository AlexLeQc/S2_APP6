"fichier pour la problematique de lapp 6"


import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

import helpers as hp


def LieuZero():
    b1 = [1, 0, 0]  # définition des facteurs du polynome numérateur
    a1 = [1, 8823, 39215686]  # définition des facteurs du polynome dénominateur

    (z2, p2, k2) = signal.tf2zpk(b1, a1)  # passage à la représentation zok (liste de racines + "gain")
    print(f'Racine exemple 2 Zéros : {z2}, Pôles: {p2}, Gain: {k2}')
    hp.pzmap1(z2, p2, 'Filtre erroné corrigé (passe-haut 1000 hz)')

    tf1 = signal.TransferFunction(b1, a1)  # définit la fonction de transfert
    # calcul le diagrame de Bode, la magnitude en dB et la phase en degrés la fréquence en rad/s
    w1, mag1, phlin1 = signal.bode(tf1, np.logspace(-1.5, 10, 200))  # on pourrait aussi laisser la fonction générer les w
    # fonction d'affichage
    hp.bode1(w1, mag1, phlin1, 'Filtre erroné corrigé (passe-haut 1000 hz)')


def Bode():
    b1 = [1, 0, 0]  # définition du numérateur de la fonction de transfert
    a1 = [1, 1, 0.5]  # définition du dénominateur de la fonction de transfert

    # méthode 1 avec bode
    tf1 = signal.TransferFunction(b1, a1)  # définit la fonction de transfert
    # calcul le diagrame de Bode, la magnitude en dB et la phase en degrés la fréquence en rad/s
    w1, mag1, phlin1 = signal.bode(tf1, np.logspace(-1.5, 1, 200))  # on pourrait aussi laisser la fonction générer les w
    # fonction d'affichage
    hp.bode1(w1, mag1, phlin1, 'Example 1')

def Butterworth():

    """
    Exemple de génération et affichage pour la FT d'un filtre de butterworth d'ordre 4

    :return:
    """
    #Passe bas 700 Hz
    order = 2
    wn700 = 4398.22   # frequence de coupure = 4398 rad/s
    # définit un filtre passe bas butterworth =>  b1 numerateur, a1 dénominateur
    b700, a700 = signal.butter(order, wn700, 'low', analog=True)
    print(f'Butterworth 700Hz Numérateur {b700}, Dénominateur {a700}')  # affiche les coefficients correspondants au filtre
    print(f'Racine butterwort 700Hz Zéros:{np.roots(b700)}, Pôles:{np.roots(a700)}')  # affichage du resultat dans la console texte

    #Passe haut 7000 Hz
    order = 2
    wn7000 = 2 * np.pi * 7000   # frequence de coupure = 7000 Hz
    # définit un filtre passe bas butterworth =>  b1 numerateur, a1 dénominateur
    b7000, a7000 = signal.butter(order, wn7000, 'high', analog=True)
    print(f'Butterworth 7000Hz Numérateur {b7000}, Dénominateur {a7000}')  # affiche les coefficients correspondants au filtre
    print(f'Racine butterwort 7000Hz Zéros:{np.roots(b7000)}, Pôles:{np.roots(a7000)}')  # affichage du resultat dans la console texte

    #Passe Haut 1000 Hz
    order = 2
    wn1000 = 2 * np.pi * 1000  # frequence de coupure = 7000 Hz
    # définit un filtre passe bas butterworth =>  b1 numerateur, a1 dénominateur
    b1000, a1000 = signal.butter(order, wn1000, 'high', analog=True)
    print(f'Butterworth 1000Hz Numérateur {b1000}, Dénominateur {a1000}')  # affiche les coefficients correspondants au filtre
    print(f'Racine butterwort 1000Hz Zéros:{np.roots(b1000)}, Pôles:{np.roots(a1000)}')  # affichage du resultat dans la console texte

    # Passe bas 5000 Hz
    order = 2
    wn5000 = 2 * np.pi * 5000  # frequence de coupure = 7000 Hz
    # définit un filtre passe bas butterworth =>  b1 numerateur, a1 dénominateur
    b5000, a5000 = signal.butter(order, wn5000, 'low', analog=True)
    print(f'Butterworth 5000Hz Numérateur {b5000}, Dénominateur {a5000}')  # affiche les coefficients correspondants au filtre
    print(
        f'Racine butterwort 5000Hz Zéros:{np.roots(b5000)}, Pôles:{np.roots(a5000)}')  # affichage du resultat dans la console texte

    lowk = 1
    highk = 1
    b7001 = [0,0,-19177110]
    a7001 = [1, 6214, 19177110]
    b50001 = [0,0,-991102673]
    a50001 = [1, 45045, 991102673]
    b10001 = [-1, 0, 0]
    a10001 = [1, 8887.1, 39478212]
    b70001 = [-1, 0, 0]
    a70001 = [1, 61601.64, 1919054290]

    #calcul passe bande en serie
    z1, p1, k1 = signal.tf2zpk(b10001, a10001)
    z2, p2, k2 = signal.tf2zpk(b50001, a50001)

    zPB, pPB, kPB = hp.seriestf(z1, p1, k1, z2, p2, k2)
    kPB = kPB * 0.8


    #calcul passe haut et passe bas parraelele
    z3, p3, k3 = signal.tf2zpk(b70001, a70001)
    z4, p4, k4 = signal.tf2zpk(b7001, a7001)

    zHB, pHB, kHB = hp.paratf(z3, p3, k3, z4, p4, k4)

    #calcul total

    zT, pT, kT = hp.paratf(zPB, pPB, kPB, zHB, pHB, kHB)
    bT, aT = signal.zpk2tf(zT, pT, kT)

    magp, php, wp, fig, ax = hp.bodeplot(bT, aT, 'Egaliseur')
    hp.grpdel1(wp, -np.diff(php) / np.diff(wp), 'Egaliseur')

def testPoleZero():
    b1 = [1, 0, 0]  # définition du numérateur de la fonction de transfert
    a1 = [1, 2, 101]  # définition du dénominateur de la fonction de transfert

    # méthode 1 avec bode
    tf1 = signal.TransferFunction(b1, a1)  # définit la fonction de transfert
    # calcul le diagrame de Bode, la magnitude en dB et la phase en degrés la fréquence en rad/s
    w1, mag1, phlin1 = signal.bode(tf1, np.logspace(0, 10,500))  # on pourrait aussi laisser la fonction générer les w
    # fonction d'affichage
    hp.bode1(w1, mag1, phlin1, 'Example 1')

    (z2, p2, k2) = signal.tf2zpk(b1, a1)  # passage à la représentation zok (liste de racines + "gain")
    print(f'Racine exemple 2 Zéros : {z2}, Pôles: {p2}, Gain: {k2}')
    hp.pzmap1(z2, p2, 'Filtre erroné corrigé (passe-haut 1000 hz)')

    tf1 = signal.TransferFunction(b1, a1)  # définit la fonction de transfert

def main():
    #Butterworth()
    #LieuZero()
    testPoleZero()
    plt.show()



if __name__ == '__main__':
    main()