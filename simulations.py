import copy as cp
import random as rd
import numpy as np
import math
from math import floor
import matplotlib.pyplot as plt
import scipy.optimize as spo
import pandas as pd
from scipy.stats import norm
from scipy.stats import expon
from scipy.stats import beta
from scipy.stats import gamma
from scipy.stats import wald
from scipy.stats import poisson
from scipy.stats import geom
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS
from sklearn.cluster import KMeans


def metropolis(n, q, pi, simul_q, x_0):
    seq = [x_0]
    refus = 0
    while len(seq) < n:
        precedent = seq[len(seq)-1]
        x = simul_q(precedent)
        facteur = min(np.log(pi(x)) - np.log(q(x, precedent)) -
                      np.log(pi(precedent)) + np.log(q(precedent, x)), 1)
        aleat = np.log(rd.random())
        if aleat <= facteur:
            seq.append(x)
        else:
            seq.append(precedent)
            refus = refus + 1
    return [seq, refus]


def simul_q(condit):
    # on simule avec des sauts normaux centrés en la précédente valeure et d'écart-type 0.1
    x = rd.gauss(condit, 0.1)
    return x


def q(x, condit):
    mu = condit
    # on simule avec des sauts normaux centrés en la précédente valeure et d'écart-type 0.1
    return 1/np.sqrt(2*np.pi*0.01)*np.exp(-0.5*(x-mu)**2/0.01)


def gaussian(mu, var, n):
    sigma = np.sqrt(var)

    def pi(x):
        # normale centrée réduite
        return 1/np.sqrt(2*np.pi*var)*np.exp(-0.5*(x-mu)**2/sigma)
    seq, refus = metropolis(n, q, pi, simul_q, 0)
    seq = seq[50000:]
    lx = np.linspace(-5, 5, 1001)  # des bins de taille 0.01 entre -5 et 5
    ly = np.zeros(1001)
    total = 0
    for i in seq:
        if -5 <= i < 5:
            total = total + 1
            place = (math.floor(i) + 5)*100 + math.floor(10*i-10 *
                                                         math.floor(i))*10 + math.floor(100*i-10*math.floor(10*i))  # choisir la bonne bin
            ly[place] = ly[place] + 1
    aire = total*0.01
    ly = ly/aire  # on normalise la courbe
    plt.plot(lx, ly)
    lz = norm.pdf(lx, mu, np.sqrt(var))
    # on compare à la "vraie" loi normale centrée réduite (de l'importance d'avoir normalisé avant)
    plt.plot(lx, lz)
    plt.show()
    plt.clf
    plt.plot(seq)
    plt.show()
    print(refus/n)
    return seq


def verif_gaussian():
    # on teste la simulation de la centrée réduite
    seq = np.array(gaussian(0, 1, 100000))
    moyenne = sum(seq)/50000
    li_moy = np.array([moyenne for i in range(50000)])
    variance = 1/(49999)*sum((seq-li_moy)**2)
    print([moyenne, variance])  # moyenne et variance de l'échantillon simulé


'''environ 3 pourcents de refus en simulant la centrée réduite à partir de sauts de puces normaux centrés sur la valeur précédente d'écart-type' 0.1: idéal pour simuler'''
'''Moyenne et variance empiriques obtenus pour la loi cible: [0.10, 1.07]: proche de [0,1]'''


def phi(x):
    # fonction de répartition de la normale centrée réduite: modèle du probit
    return norm.cdf(x)


def metro_probit(y, X):
    occur = len(X[:, 0])  # nb de lignes

    def pi(beta):
        pi = 1
        for i in range(occur):
            pi = pi*phi(np.dot(X[i, :], beta.T))**y[i] * \
                (1-phi(np.dot(X[i, :], beta.T)))**(1-y[i])
        return pi  # vraisemblance de la donnée
    # on initialise au maximum de vraisemblance la chaîne de metropolis-hastings

    def log_pi(beta):
        return min(-np.log(pi(beta)), 10**99)
    x_0 = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y.T)
    result = spo.minimize(log_pi, x_0, options={
                          'disp': True}, method='Nelder-Mead')
    if result.success:
        x_min = result.x  # estimateur du maximum de vraisemblance
    else:
        print('echec')
        x_min = x_0
    n_col = len(X[0, :])  # nb de colonnes
    Cov = 1/(occur-n_col-1) * \
        np.dot((y.T-np.dot(X, x_0).T), (y.T-np.dot(X, x_0))) * \
        np.linalg.inv(np.dot(
            X.T, X))  # on propose d'utiliser la matrice de covariance de l'estimateur du maximum de vraisemblance dans le cas purement linéaire (et non probit... faiblard)
    print([x_0, x_min])
    det = np.linalg.det(Cov)
    Cov_inv = np.linalg.inv(Cov)
    beta_0 = x_min.T

    def simul_q(condit):
        # simulation du vecteur gaussien centré en la précédente valeur et de matrice de covariance: COV
        value = np.random.multivariate_normal(condit, Cov)
        return value

    def q(beta, condit):
        # densité du vecteur gaussien centré en la précédente valeur et de matrice de covariance: COV
        return 1/(np.sqrt(2*np.pi)**n_col*np.sqrt(abs(det)))*np.exp(-0.5*np.dot(np.dot(beta-condit, Cov_inv), beta.T-condit.T))

    seq, refus = metropolis(1000, q, pi, simul_q, beta_0)
    seq = seq[100:]
    return [seq, refus]


def appli_probit():
    cur = open('bank.txt', 'r')
    li = [[], [], [], [], []]
    for ligne in cur:
        n = 0
        num = ''
        for c in ligne:
            if c == ' ':
                li[n].append(float(num))
                n = n+1
                num = ''
            else:
                num = num + c
        li[n].append(float(num))
    n = len(li[0])
    m = 4
    X = np.zeros((n, m))
    for i in range(4):
        Col = np.array(li[i])
        X[:, i] = Col
    y = np.array(li[4])
    return metro_probit(y, X)


"""
liste, refus = appli_probit()
beta_tot = [[], [], [], []]
for i in range(4):
    for j in range(len(liste)):
        beta_tot[i].append(liste[j][i])
# montrer les composantes générées de beta 2 par 2: montrer l'exploration
plt.scatter(beta_tot[0], beta_tot[1])  # points
plt.show()
matrice_res = np.zeros((len(liste), 4))
for i in range(len(liste)):
    matrice_res[i, :] = liste[i]
moyennes = []
for i in range(4):
    moy = np.sum(matrice_res[:, i])/len(liste)
    moyennes.append(moy)
print(moyennes)
print(refus/900)"""
'''On trouve [-1.2052853651618036, 0.9579336145894043, 0.9546853722751267, 1.116388011464158], taux de refus vers 0.13'''

'''On s'intéresse maintenant aux modèles mixtes. Peut-on retrouver une distribution mixte avec deux régressions linéaires à une dimension?'''

'''Génération d'un échantillon selon un modèle mélangeant deux régressions linéaires unidimensionnelle avec un paramètre de mélange p.'''
'''On choisira pour l'étude param_1 = [-3,2] et param_2 = [5,-1], sigma = 2,p = 0.25. On sait que le premier modèle ne dépend que de x_1 et le second
seulement de x_2'''


def echantillon(n, param_1, param_2, sigma, p):
    # param_1 = [a_1,b_1]; param_2 = [a_2,b_2]; n est le nombre d'éléments générés, p le paramètre du mélange
    x_echantillon = []
    y_echantillon = []
    for i in range(n):
        # on génère des couples de variables explicatives: la première des deux servira à la régression linéaire numéro 1,
        x_echantillon.append([rd.gauss(-10, 5), rd.gauss(10, 5)])
        # la deuxième servira à la régression linéaire numéro 2
    for i in range(n):
        test = rd.random()  # test est généré selon une loi uniforme sur [0,1]
        if test < p:  # dans ce cas, la première régression linéaire est utilisée
            y = rd.gauss(param_1[0]*x_echantillon[i][0] + param_1[1], sigma)
        else:  # dans ce cas, la deuxième régression linéaire est utilisée
            y = rd.gauss(param_2[0]*x_echantillon[i][1] + param_2[1], sigma)
        y_echantillon.append(y)  # enregistrer la variable expliquée générée
    return [x_echantillon, y_echantillon]


'''densité de probabilité de la gaussienne unidimensionnelle'''


def f_gauss(x, mu, sigma):
    return norm.pdf(x, mu, sigma)


'''densité propositionnelle gaussienne d'écart type égal à 1 dans MH'''


def q_nouv(x, condit):
    return f_gauss(x, condit, 1)


'''Simulation d'une donnée selon la densité propositionnelle gaussienne d'écart type égal à 1 dans MH'''


def simul_q_nouv(condit):
    return rd.gauss(condit, 1)


# param_tot est theta_p = [a_1,b_1,a_2,b_2,sigma,p]
def final_prob(param_tot, lx, ly):
    proba = 0
    n = len(lx)
    for i in range(n):
        moy_0 = ly[i] - param_tot[0]*lx[i][0] - \
            param_tot[1]  # ecart à la prevision modèle 0
        moy_1 = ly[i] - param_tot[2]*lx[i][1] - \
            param_tot[3]  # ecart à la prevision modèle 1
        f_0 = norm.pdf(0, moy_0, param_tot[4])
        f_1 = norm.pdf(0, moy_1, param_tot[4])
        proba = proba + np.log(param_tot[5]*f_0 + (1-param_tot[5])*f_1)
        proba = proba + np.log(f_gauss(param_tot[0], 1, 1)*f_gauss(
            param_tot[1], 0, 1)*f_gauss(param_tot[2], 0, 1)*f_gauss(param_tot[3], 1, 1)*expon.pdf(param_tot[4])*beta.pdf(param_tot[5], 1, 1))
    return proba


'''algo Gibbs within MH séparant deux régressions linéaires à une dismension chacune (ax + b). Le paramètre que l'on génère est appelé theta_p. Il s'agit de  [a_1,b_1,a_2,b_2,sigma,p]'''


def Gibbs(T, lx, ly):  # T le nombre d'étapes; lx la liste des variables explicatives, ly la liste des variables expliquées
    # refus liste les refus de MH de chaque paramètre de theta_p dans le même ordre que theta_p i.e. [a_1,b_1,a_2,b_2,sigma,p]
    refus = np.array([0, 0, 0, 0, 0, 0])
    n = len(lx)  # nombre de données d'apprentissage
    # z est la variable qui définit le numéro du modèle utilisé. li_z recense tous les elem_z (définition donnée plus loin)
    li_z = []
    # initialisation; theta_p = [a_1,b_1,a_2,b_2,sigma,p]
    # li_theta_p contient la liste de tous les theta_p acceptés jusqu'à maintenant
    li_theta_p = [np.array([1.0, 0.0, 1.0, 0.0, 2.0, 0.5])]
    for t in range(1, T):
        # p est la probabilité du modèle 0
        a_1, b_1, a_2, b_2, sigma, p = li_theta_p[t-1]
        # elem z décrit à la t-ième itération le numéro "supposé" (0 ou 1) de la régression linéaire utilisée pour générer chaque donnée de l'échantillon: c'est une liste de n entiers dans {0,1}
        elem_z = []
        for i in range(n):
            test = np.log(rd.random())
            value = np.log(p*f_gauss(ly[i], a_1*lx[i][0]+b_1, sigma)) - np.log(p*f_gauss(
                ly[i], a_1*lx[i][0]+b_1, sigma) + (1-p)*f_gauss(ly[i], a_2*lx[i][1]+b_2, sigma))  # on génère z selon \pi(z|p_{t-1}, \theta_{t-1}): on suit Gibbs. Pas besoin de MH ici
            if test < value:
                elem_z.append(0)
            else:
                elem_z.append(1)
        li_z.append(elem_z)
        n_1 = sum(elem_z)  # nombre de cas où le second modèle a été utilisé
        n_0 = n-n_1  # nombre de cas où le premier modèle a été utilisé
        # prochain theta_p que l'on veut créer
        new_theta_p = [0, 0, 0, 0, 0, 0]
        # a priori p suit une dirichlet de params (1,1). Utilisation des lois conjuguées comme dans Bayesian Core:
        # On commence par générer p selon \pi(p|z_t, \theta_{t-1}): Gibbs seul sans MH
        new_theta_p[5] = np.random.beta(1 + n_0, 1 + n_1)

        # theta = [a_1,b_1,a_2,b_2,sigma]; on calcule la log_probabilité \pi(\theta|z_t): utile dans l'étape MH qui suit
        def pi_theta(theta):
            interieur = 0
            count = 0
            for i in elem_z:
                interieur = interieur + \
                    (ly[count] - theta[2*i]*lx[count][i] - theta[2*i + 1])**2
                count = count + 1
            proba = ((-0.5/theta[4]**2)*interieur) - n*np.log(theta[4]) + np.log(f_gauss(theta[0], -2, 1)*f_gauss(
                theta[1], 0.5, 1)*f_gauss(theta[2], 9, 1)*f_gauss(theta[3], -4, 1)*expon.pdf(theta[4]))  # tenir compte des à prioris sur \theta: choisis non symétriquement pour éviter les problèmes d'identification
            return proba  # proportionnel au logarithme de la probabilité en fait

        # simule un saut normal de \theta centré sur condit, utile pour MH: loi propositionnelle
        def simul_q_theta(condit):
            c_1, c_2, c_3, c_4 = np.random.multivariate_normal(
                condit[0:4], np.eye(4))
            c_sigma = np.random.exponential(condit[4])
            return np.array([c_1, c_2, c_3, c_4, c_sigma])

        def q_theta(x, condit):  # densité de probabilité associé au saut: loi propositionnelle
            c_1, c_2, c_3, c_4, c_sigma = condit
            proba = f_gauss(x[0], c_1, 1)*f_gauss(x[1], c_2, 1) * \
                f_gauss(x[2], c_3, 1)*f_gauss(x[3], c_4,
                                              1) * expon.pdf(x[4], scale=c_sigma)
            return proba

        # anciens paramètres pour le moment mais amené à devenir le paramètre propositionnel que l'on va tester
        proposition = np.array([a_1, b_1, a_2, b_2, sigma])
        # anciens paramètres pour le moment mais destiné à conserver les paramètres que l'on va accepter au fur et à mesure
        precedent = np.copy(proposition)
        for i in range(5):
            # génération du nouvel élément (pour le i-ème terme de \theta) par "saut" depuis le précédent
            param_prop = simul_q_theta(np.array(precedent))[i]
            # nouvelle valeur proposée prise en compte puis testée
            proposition[i] = param_prop
            facteur = min(pi_theta(proposition) - np.log(q_theta(proposition, precedent)) -
                          pi_theta(precedent) + np.log(q_theta(precedent, proposition)), 0)  # rappel: pi_theta est déjà logarithmique
            value = np.log(rd.random())
            if value < facteur:  # etape MH sur le i-eme element de \theta: accepter ou non
                new_theta_p[i] = param_prop  # accepter
                # indispensable pour garantir qu'on génère sachant les dernières valeurs dans Gibbs
                precedent[i] = param_prop
            else:
                new_theta_p[i] = precedent[i]  # rejeter
                proposition[i] = precedent[i]
                refus[i] = refus[i] + 1  # augmenter le nombre de refus
        li_theta_p.append(new_theta_p)  # retenir le nouveau theta_p
    # retirer les 5000 premières valeurs
    li_theta_p = np.array(li_theta_p)[5000:, :]
    # clustering
    scaler = StandardScaler()
    scaler.fit(li_theta_p)
    new = scaler.transform(li_theta_p)
    clustering = KMeans(n_clusters=5, random_state=15).fit(new)
    labels = clustering.labels_
    k = max(labels) + 1
    l_tot = [[] for i in range(k)]
    l = len(labels)
    print(refus/10000)
    for i in range(l):
        if labels[i] >= 0:
            l_tot[labels[i]].append(li_theta_p[i, :])
    centroides = [0 for i in range(k)]
    for i in range(k):
        vect = np.array(l_tot[i])
        centroides[i] = [np.mean(vect, axis=0), len(l_tot[i])]
    return [li_theta_p, li_z, centroides]


# le premier chiffre est la taille de l'échantillon d'apprentissage. Vers 1000 ça devient pas mal
'''lx, ly = echantillon(200, [-3, 2], [5, -1], 2, 0.25)
parametres, li_z, centroides = Gibbs(10000, lx, ly)
p = []
a_1 = []
a_2 = []
b_1 = []
b_2 = []
sigma = []
for i in parametres:
    p.append(i[5])
    a_1.append(i[0])
    b_1.append(i[1])
    a_2.append(i[2])
    b_2.append(i[3])
    sigma.append(i[4])
df = pd.DataFrame({'a_1': a_1, 'b_1': b_1, 'a_2': a_2,
                   'b_2': b_2, 'sigma': sigma, 'p': p})
pd.plotting.scatter_matrix(df)
plt.show()
print(centroides)'''

'''Ca marche: [0.9696 0.6975 0.9824 0.7995 0.9419 0.    ]
[[array([-2.95943904,  1.88124505,  5.12652363, -2.23294548,  2.00644419,
        0.28753343]), 893], [array([-2.95369412,  1.92092961,  5.09288814, -1.80341501,  1.84148143,
        0.28854737]), 1147], [array([-3.00031113,  1.38808389,  5.08984444, -1.77991084,  1.98008608,
        0.28347187]), 925], [array([-2.90228964,  2.43738503,  5.09103204, -1.81679442,  1.96929854,
        0.28708016]), 1388], [array([-2.94603764,  2.00010194,  5.04646241, -1.29174479,  1.92983274,
        0.28632984]), 647]]'''


'''Un exemple plus simple'''


def echant_facile(n):
    echant = []
    for i in range(n):
        test = rd.random()
        if test < 0.25:
            x = rd.gauss(0, 1)
        else:
            x = rd.gauss(5, 2)
        echant.append(x)
    return echant


def double_gauss(lx):

    def pi_2(p):
        if 0 <= p <= 1:
            proba = 1
        else:
            proba = 0
        for i in lx:
            proba = proba*(p*f_gauss(i, 0, 1) + (1-p)*f_gauss(i, 5, 2))
        return proba

    '''lp = np.linspace(0, 1, 100)
    pi_2 = np.vectorize(pi_2)
    l_pi = pi_2(lp)
    plt.plot(lp, l_pi)
    plt.show()'''

    def q_2(p, condit):
        return f_gauss(p, condit, 1)

    def simul_q_2(condit):
        return rd.gauss(condit, 1)

    sequence, refus = metropolis(5000, q_2, pi_2, simul_q_2, 0.5)
    sequence = sequence[1000:]
    return [sequence, refus/5000]


'''lx = echant_facile(100)

sequence, taux = double_gauss(lx)
plt.plot(np.array(range(4000)), np.array(sequence))
plt.show()
print(sum(sequence)/4000)
print(taux)'''


'''Essayons une régression linéaire en dimension supérieure'''

'''On a posé:
param_1 = np.array([a_0,a_1,a_2,a_3,...,a_(k-2),sigma]) avec une régession linéaires de moyenne a_0 + a_1*u_1 + a_2*u_2...
param_2 = np.array([b_0,b_1,b_2,b_3,...,b_(l-2),sigma]) avec une régression linéaire de moyenne b_0 + a_1*v_1 + a_2*v_2
est la taille de l'échantillon et p est la probabilité du modèle 0 contre le modèle 1'''


# générer un échantillon selon le mélange de deux régressions linéaires de dimensions quelconques
def echant_superieure(n, param_0, param_1, p):
    k = len(param_0)
    l = len(param_1)
    # Matrice des variables explicatives de la 1ere regression linéaire
    mat_u_echantillon = np.zeros((n, k-1))
    # Matrice des variables explicatives de le seconde regression linéaire
    mat_v_echantillon = np.zeros((n, l-1))
    attila_u = np.array([1 for i in range(k-2)])  # ligne de un
    attila_v = np.array([1 for i in range(l-2)])  # ligne de un
    y_echant = []  # liste des valeurs générées y
    for i in range(n):  # simuler les données
        # astuce de la première colonne de variables explicatives égales à un pour avoir l'ordonnée à l'origine
        mat_u_echantillon[i, 0] = 1
        mat_v_echantillon[i, 0] = 1
        mat_u_echantillon[i,
                          1:] = np.random.multivariate_normal(-3*attila_u, 4*np.eye(k-2))  # générer les lignes de variables expliquées unes à unes
        mat_v_echantillon[i, 1:] = np.random.multivariate_normal(
            3*attila_v, 4*np.eye(l-2))  # générer les lignes de variables expliquées unes à unes
    for i in range(n):
        test = rd.random()
        if test < p:  # utiliser première régression linéaire avec probabilité p
            y = rd.gauss(
                np.dot(mat_u_echantillon[i], param_0[:k-1]), param_0[k-1])  # générer selon une normale d'écart type donné par param_0[k-1] et de moyenne donnée par une combi lin de param_0[:k-1] et des variables explicatives
        else:  # utiliser seconde régression linéaire avec probabilité 1-p
            y = rd.gauss(
                np.dot(mat_v_echantillon[i], param_1[:l-1]), param_1[l-1])  # générer selon une normale d'écart type donné par param_1[l-1] et de moyenne donnée par une combi lin de param_1[:l-1] et des variables explicatives
        y_echant.append(y)  # enregistrer la valeur de y
    return [mat_u_echantillon, mat_v_echantillon, np.array(y_echant)]


# densité de probabilité d'un vecteur gaussien de variables indépendantes d'écart type commun sigma
def f_gauss_multi(x, mu, sigma):
    d = len(x)
    proba = (1/np.sqrt((2*np.pi)**d*sigma**(2*d))) * \
        np.exp(-0.5*np.dot(x-mu, (x-mu).T)/sigma**2)
    return proba

# densité de probabilité d'une v.a suivant une loi de Wald d'espérance mu et de scale s (var = mu**3/scale)


def f_wald(x, mu, s):
    if x >= 0:
        proba = (s/(2*np.pi*x**3))**0.5*np.exp(-0.5*s*(x-mu)**2/(mu**2*x))
    else:
        proba = 0
    return proba


'''Séparer deux régressions linéaires en dimensions quelconquesavec Gibbs within MH'''
'''T: nombre d'itérations du MCMC
 mat_u_échantillon: matrice des variables explicatives de la première régression
 mat_v_échantillon: matrice des variables explicatives de la seconde régression
 ly : échantillon d'apprentissage
 param_0_init: paramètres de départ de Gibbs pour la première régression linéaire
 param_1_init: paramètres de départ pour Gibbs dans la deuxième régression linéaire
 p_init: probabilité à l'étape 0 de Gibbs de la première régression linéaire'''


def sepa_regress(T, mat_u_echantillon, mat_v_echantillon, ly, param_0_init, param_1_init, p_init):
    n = len(ly)  # taille de l'échantillon généré
    k = len(mat_u_echantillon[0]) + 1  # taille du vecteur param_0
    l = len(mat_v_echantillon[0]) + 1  # taille du vecteur param_1
    # enregistrer le nombre de refus des paramètres de param_0 lors des étapes MH
    refus_param_0 = np.zeros(k)
    # enregistrer le nombre de refus des paramètres de param_1 lors des étapes MH
    refus_param_1 = np.zeros(l)
    # li_z va contenir toutes les listes d'attributions supposées (1ere régression : 0 ou deuxième régression: 1) des variables y
    li_z = []
    # li_param_0 va contenir toutes les valeurs de param_0 qui seront généréres par Gibbs within MH
    li_param_0 = [param_0_init]
    # li_param_1 va contenir toutes les valeurs de param_1 qui seront généréres par Gibbs within MH
    li_param_1 = [param_1_init]
    # li_p va contenir toutes les valeurs de p qui seront générées par Gibbs within MH
    li_p = [p_init]
    for t in range(T):  # 1 itération de cette boucle for = une itération de Gibbs within MH
        param_0 = li_param_0[t]  # précédent élément param_0 dans Gibbs
        param_1 = li_param_1[t]  # précédent élément param_1 dans Gibbs
        p = li_p[t]  # précédent élément p dans Gibbs
        # construire pour cette itération de Gibbs within MH une attribution supposée (1ere régression : 0 ou deuxième régression: 1) de chaque y_i
        elem_z = []
        for i in range(n):
            test = np.log(rd.random())
            value = np.log(p*f_gauss(ly[i], np.dot(mat_u_echantillon[i], param_0[:k-1].T), param_0[k-1])) - np.log(p*f_gauss(ly[i], np.dot(
                mat_u_echantillon[i], param_0[:k-1].T), param_0[k-1]) + (1-p)*f_gauss(ly[i], np.dot(mat_v_echantillon[i], param_1[:l-1].T), param_1[l-1]))  # générer elem_z[i] selon \pi(z_i_t|\theta_{t-1}, \p_{t-1})
            if test < value:
                elem_z.append(0)
            else:
                elem_z.append(1)
        li_z.append(elem_z)
        n_1 = sum(elem_z)  # nombre de y_i attribués à la seconde régression
        n_0 = n-n_1  # nombre de y_i attribués à la première régression
        # avec l'à priori en DIrichlet conjugué de bayesian core on trouve l'à postériori de p sachant elem_z suivant:
        new_p = np.random.beta(1 + n_0, 1 + n_1)
        li_p.append(new_p)  # enregistrer la nouvelle valeur de p

        # calculer le logarithme de la densité de probabilité \pi(par_0,par_1|z_t)
        def log_pi(par_0, par_1):
            interieur = 0
            count = 0  # numéro du terme de elem_z parcouru
            for i in elem_z:
                if i == 0:  # première régression linéaire utilisée
                    explicatives = mat_u_echantillon[count]
                    coefficients = par_0
                else:  # seconde régression linéaire utilisée
                    explicatives = mat_v_echantillon[count]
                    coefficients = par_1
                interieur = interieur - 0.5*((ly[count] - np.dot(explicatives, coefficients[:len(
                    coefficients) - 1].T))**2)/coefficients[len(coefficients) - 1]**2 + np.log(1/np.sqrt(2*np.pi*coefficients[len(
                        coefficients) - 1]))  # construction de la likelyhood de param_0 et param_1 sachant z
                count = count + 1  # incrémenter le compte
            log_proba = interieur + np.log(f_gauss_multi(par_0[:k-1], np.zeros(k-1), 10) * f_gauss_multi(
                par_1[:l-1], np.zeros(l-1), 10) * expon.pdf(par_0[k-1]) * expon.pdf(par_1[l-1]))  # tenir compte des à prioris sur les paramètres par_0 et par_1
            return log_proba

        def simul_q_param_gauss(condit):
            return rd.gauss(condit, 1)  # faire des sauts gaussiens de 1

        def simul_q_param_expon(condit):
            # faire des sauts selon une loi de Wald centrée sur condit (pour toujours rester dans les positifs sur l'écart type)
            # espérance  = condit, variance = 1 (ajuster le scale parameter en fonction de condit pour cela)
            return np.random.wald(condit, condit**3)

        def q_proba(par_0, par_1, ex_par_0, ex_par_1):
            proba = f_gauss_multi(par_0[:k-1], ex_par_0[:k-1], 1) * f_gauss_multi(
                par_1[:l-1], ex_par_1[:l-1], 1) * f_wald(par_0[k-1], ex_par_0[k-1], ex_par_0[k-1]**3) * f_wald(par_1[l-1], ex_par_1[l-1], ex_par_1[l-1]**3)  # densité de probabilité de la loi de simulation, conditionnée par ex_par_0 et ex_par_1
            return proba

        # prochaine valeur de param_0, pour l'instant initialisée à un vecteur de 0: va changer
        new_param_0 = np.zeros(k)
        # prochaine valeur de param_1, pour l'instant initialisée à un vecteur de 0: va changer
        new_param_1 = np.zeros(l)
        precedent = np.copy(param_0)  # param_0_{t-1}
        proposition = np.copy(param_0)  # param_0_{t-1}
        for i in range(k-1):
            # étape MH: paramètres dans R: simulé par gaussienne
            param_prop = simul_q_param_gauss(precedent[i])
            # param_prop contient tous les nouveaux paramètres de param_0 jusqu'au (i-1)_ème de l'itération t de Gibbs ainsi que
            proposition[i] = param_prop
            facteur = min(log_pi(proposition, param_1) - np.log(q_proba(proposition, param_1, precedent, param_1)) -
                          log_pi(precedent, param_1) + np.log(q_proba(precedent, param_1, proposition, param_1)), 0)
            value = np.log(rd.random())
            if value < facteur:  # accepter la nouvelle valeur
                new_param_0[i] = param_prop
                precedent[i] = param_prop
            else:  # refuser la nouvelle valeur
                new_param_0[i] = precedent[i]
                proposition[i] = precedent[i]
                refus_param_0[i] = refus_param_0[i] + 1
        # etape MH sur l'écrt type de la première régression
        param_prop = simul_q_param_expon(precedent[k-1])
        proposition[k-1] = param_prop
        facteur = min(log_pi(proposition, param_1) - np.log(q_proba(proposition, param_1, precedent, param_1)) -
                      log_pi(precedent, param_1) + np.log(q_proba(precedent, param_1, proposition, param_1)), 0)
        value = np.log(rd.random())
        if value < facteur:  # accepter la nouvelle valeur
            new_param_0[k-1] = param_prop
            precedent[k-1] = param_prop
        else:  # refuser la nouvelle valeur
            new_param_0[k-1] = precedent[k-1]
            proposition[k-1] = precedent[k-1]
            refus_param_0[k-1] = refus_param_0[k-1] + 1
        li_param_0.append(new_param_0)
        precedent = np.copy(param_1)
        proposition = np.copy(param_1)
        # etape MH sur la seconde régression
        for i in range(l-1):
            param_prop = simul_q_param_gauss(precedent[i])
            proposition[i] = param_prop
            facteur = min(log_pi(new_param_0, proposition) - np.log(q_proba(new_param_0, proposition, new_param_0, precedent)) -
                          log_pi(new_param_0, precedent) + np.log(q_proba(new_param_0, precedent, new_param_0, proposition)), 0)
            value = np.log(rd.random())
            if value < facteur:  # accepter la nouvelle valeur
                new_param_1[i] = param_prop
                precedent[i] = param_prop
            else:  # refuser la nouvelle valeur
                new_param_1[i] = precedent[i]
                proposition[i] = precedent[i]
                refus_param_1[i] = refus_param_1[i] + 1
        # etape MH sur l'écrat type de la seconde régression
        param_prop = simul_q_param_expon(precedent[l-1])
        proposition[l-1] = param_prop
        facteur = min(log_pi(new_param_0, proposition) - np.log(q_proba(new_param_0, proposition, new_param_0, precedent)) -
                      log_pi(new_param_0, precedent) + np.log(q_proba(new_param_0, precedent, new_param_0, proposition)), 0)
        value = np.log(rd.random())
        if value < facteur:  # accepter la nouvelle valeur
            new_param_1[l-1] = param_prop
            precedent[l-1] = param_prop
        else:  # rejeter la nouvelle valeur
            new_param_1[l-1] = precedent[l-1]
            proposition[l-1] = precedent[l-1]
            refus_param_1[l-1] = refus_param_1[l-1] + 1
        # intégrer le nouveau paramètre à la liste
        li_param_1.append(new_param_1)
    return [li_param_0, li_param_1, li_p, li_z, refus_param_0, refus_param_1]


def verification(n, mat_u_echantillon, mat_v_echantillon, param_0, param_1, p):
    k = len(param_0)
    l = len(param_1)
    y_echant = []
    for i in range(n):
        test = rd.random()
        if test < p:
            y = rd.gauss(
                np.dot(mat_u_echantillon[i], param_0[:k-1]), param_0[k-1])
        else:
            y = rd.gauss(
                np.dot(mat_v_echantillon[i], param_1[:l-1]), param_1[l-1])
        y_echant.append(y)
    return np.array(y_echant)


'''mat_u_echantillon, mat_v_echantillon, ly = echant_superieure(
    500, np.array([1, 2, 3, 4, 5]), np.array([-1, -2, -10, 1]), 0.75)  # générer deux matrices de variables explicatives (une par régression pour assurer l'identifiablilité) ainsi que la liste des variables expliquées


li_param_0, li_param_1, li_p, li_z, refus_param_0, refus_param_1 = sepa_regress(5000, mat_u_echantillon, mat_v_echantillon, ly,
                                                                                np.array([0, 0, 0, 0, 2]), np.array([-2, -2, -5, 0.5]), 0.5)  # chercher à séparer les paramètres
# éliminer les 1000 premières valeurs et prendre l'espérance de p
p_guess = sum(li_p[1000:])/4000
param_0_guess = [0, 0, 0, 0, 0]
param_1_guess = [0, 0, 0, 0]
for i in range(4000):
    for j in range(5):
        # éliminer les 1000 premières valeurs et prendre l'espérance de param_0
        param_0_guess[j] = param_0_guess[j] + li_param_0[i+999][j]
    for j in range(4):
        # éliminer les 1000 premières valeurs et prendre l'espérance de param_1
        param_1_guess[j] = param_1_guess[j] + li_param_1[i+999][j]
param_0_guess = np.array(param_0_guess)/4000
param_1_guess = np.array(param_1_guess)/4000
# afficher les paramètres "devinés"
print([param_0_guess, param_1_guess, p_guess])
print([refus_param_0, refus_param_1]) '''  # afficher le nombre de refus'''

'''On trouve:
 [param_0,param_1,p] =  [array([1.42779516, 2.14520455, 2.95794711, 3.87112479, 6.77407718]), array(
     [-1.25562007, -1.93570972, -9.98751438,  1.5526577 ]), 0.7722073230414384]
 []refus_0, refus_1] = [array([3022., 4429., 4369., 4384., 3076.]), array([4073., 4714., 4743., 4049.])] pour 5000 éléments générés
 Bilan: les taux de refus sont cohérents et l'estimation est bonne!'''


'''Comparaison de régression Poisson vs régression géométrique'''

'''Ici, on génère un échantillon selon un mélange de régression de poisson et de régression géométrique de paramètres fixés pour une même matrice de variables explicatives
On le fait pour bien tester que le modèle marche et trouve avec précision le p (ainsi que les paramètres des sous-modèles si possible)
Une seule matrice de variables explicatives car les modèles sont identifiables: on suppose aussi le même nombre de paramètres...'''

'''param_poi = [a_0,...,a_{k-1}], param_geo = [b_0,...,b_{k-1}], p: probabilité du modèle poisson, n: taille du jeu de données; pas de sigma dans ce modèle!'''


# les deux modèles ont la même espérance en théorie.
def mel_poi_geo(param, p, n):
    param = np.array(param)  # utile pour produits scalaires
    k = len(param)  # taille du paramètre
    mat = np.zeros((n, k))  # matrice des variables explicatives
    # colonne de 1 (1e colonne de la matrice pour ordonnée à l'origine de la régression)
    attila_n = np.array([1 for i in range(n)])
    mat[:, 0] = attila_n
    # k-1 à cause de la colonne de 1 (1e colonne de la matrice pour ordonnée à l'origine)
    attila_k = np.array([1 for i in range(k-1)])
    ly = []  # future liste des variables expliquées
    indice_poi = 0
    indice_geo = 0
    for i in range(n):
        mat[i, 1:] = np.random.multivariate_normal(
            0*attila_k, 0.01*np.eye(k-1))
    for i in range(n):
        # calcule l'espérance de y_i
        esp = np.exp(np.dot(mat[i], param))
        value = rd.random()  # réel tiré uniformément dans [0,1]
        while esp > 2.1*10**9:
            mat[i, 1:] = np.random.multivariate_normal(
                0*attila_k, 4*np.eye(k-1))
            # l'espérance est positive (modèle)
            esp = np.exp(np.dot(mat[i], param))
            print('la matrice est refaite')
        # print(esp)
        if value < p:
            y = poisson.rvs(esp)  # simuler selon poisson
            indice_poi = indice_poi + 1
        else:
            # générer selon loi géométrique. Attention, python génère selon une loi géométrique avec k>= 1 et on veut que k puisse valoir 0
            y = geom.rvs(1/(esp + 1)) - 1
            indice_geo = indice_geo + 1
        ly.append(y)
        if y < 0:
            print('erreur')
    ly = np.array(ly)
    return [mat, ly]


'''en sortie, on a la matrice des variables explicatives et la liste de variables expliquées générées par le mélange'''


# un seul paramètre car une seule espérance, pas d'écart-type dans le modèle
def sepa_gp(T, mat, ly, param_init, p_init):
    n = len(mat)  # taille de l'échantillon d'apprentissage
    k = len(param_init)  # taille du vecteur de paramètres
    refus_param = np.zeros(k)  # compter les refus au fur et à mesure
    # li_z va contenir toutes les listes d'attributions supposées (1ere régression : 0 ou deuxième régression: 1) des variables y
    li_z = []
    # li_param_poi va contenir toutes les valeurs de param qui seront généréres par Gibbs within MH
    li_param = [param_init]
    # li_p va contenir toutes les valeurs de p qui seront générées par Gibbs within MH
    li_p = [p_init]
    for t in range(T):
        # dernier paramètre généré
        param = np.array(li_param[t])
        p = li_p[t]  # dernier paramètre généré pour p
        # p = 0.5  # WARNING TESTING!!!!!!!!!!!!!!!

        def log_proba(par):  # par est le vecteur des coefficients pour les deux régressions
            ln_proba = 0
            for i in range(n):
                # calcul de l'espérance pour les i-èmes variables explicatives de la régression poisson
                esp = np.exp(np.dot(mat[i], par))
                # calcul de la probabilité p de réussite pour poisson (esp_géo peut valoir de 0 à +inf), cf convention choisie pour la loi géo
                # log de la probabilité p du succès dans la régression géométrique
                ln_p_geom = -np.log(esp + 1)
                # log de la probabilité q de l'échec dans la régression géométrique
                ln_q_geom = np.log(esp) - np.log(esp + 1)
                #print([ln_p_geom, ln_q_geom, esp])
                if ly[i] > 30:  # alors l'approximation de stirling devient bonne
                    log_fact = ly[i]*np.log(ly[i])-ly[i] + \
                        0.5*np.log(ly[i]) + \
                        np.log(np.sqrt(2*np.pi))  # stirling
                else:
                    # le calcul direct est faisable car la valeur est petite
                    log_fact = np.log(float(np.math.factorial(ly[i])))
                # calcul de la vraisemblance de la donnée y_i pour le modèle poisson
                ln_proba_poisson = -esp - log_fact + ly[i]*np.log(esp)
                # calcul de la vraisemblance de la donnée y_i pour le modèle géométrique
                ln_proba_geo = ln_p_geom + ly[i]*ln_q_geom
                rapport_1 = np.exp(- np.log(p) - ln_proba_poisson +
                                   np.log(1-p) + ln_proba_geo)  # rapport b/a où a = p*\pi(y|poisson,\theta) et b = (1-p)*\pi(y|géométrique,\theta)
                rapport_2 = np.exp(np.log(p) + ln_proba_poisson -
                                   np.log(1-p) - ln_proba_geo)  # avec les mêmes conventions: rapport a/b
                # il est impossible que rapport_1 et rapport_2 soient ensemble écrasés à +inf. Et celui qui est écrasé à 0 donnera un résultat global correct (cf D.L de ln(1+x); x<<1)
                supplement_1 = np.log(
                    p) + ln_proba_poisson + np.log(1 + rapport_1)  # avec les notations précédentes: calcul de ln(a + b) = ln(a) + ln(1 + b/a)
                supplement_2 = np.log(1-p) + ln_proba_geo + \
                    np.log(
                        1 + rapport_2)  # avec les notations précédentes: calcul de ln(a + b) = ln(b) + ln(1 + a/b)
                if rapport_1 <= 1:  # i.e. si rapport_1 pas écrasé à l'infini
                    supplement = supplement_1
                else:  # alors forcément rapport_2 pas écrasé à l'infini
                    supplement = supplement_2
                    if supplement_2 > 0:
                        print('aïe')
                        print([ln_proba_poisson, ln_proba_geo,
                               supplement, esp, ly[i], log_fact])
                ln_proba = ln_proba + supplement
            ln_proba = ln_proba + \
                np.log(f_gauss_multi(par, np.array([3 for i in range(k)]), 10)
                       )  # probabilité des données sachant la théorie
            return ln_proba

        def simul_q_saut(condit):
            # réaliser un saut gaussien centré en condit d'écart-type unitaire: symétrique
            return rd.gauss(condit, 0.5)

        # nouveau paramètre pour les deux régressions
        new_param = np.zeros(k)
        # précédent paramètre , qui tiendra compte des sauts MH progressivement acceptés
        precedent = np.copy(param)
        # précédent paramètre qui tindra compte des sauts MH progressivement acceptés et des supposition de sauts à tester
        proposition = np.copy(param)
        for i in range(k):
            # le i-ème coefficient fait un saut normal d'écart-type 1
            param_prop = simul_q_saut(precedent[i])
            # on tient compte du saut dans proposition
            proposition[i] = param_prop
            # calcul du facteur dans MH avec loi de saut symétrique
            facteur = min(log_proba(proposition) - log_proba(precedent), 0)

            value = np.log(rd.random())
            if value < facteur:  # accepter la nouvelle valeur
                new_param[i] = param_prop
                precedent[i] = param_prop
            else:  # refuser la nouvelle valeur
                new_param[i] = precedent[i]
                proposition[i] = precedent[i]
                refus_param[i] = refus_param[i] + 1
        # ajouter le nouveau paramètre à la liste
        li_param.append(new_param)
        # construire pour cette itération de Gibbs within MH une attribution supposée (1ere régression : 0 ou deuxième régression: 1) de chaque y_i
        # new_param = np.array([1, 2, 3, 4, 5])  # WARNING TESTING!!!!
        p = li_p[t]
        elem_z = []
        for i in range(n):  # etape de Gibbs sampling pour z
            test = np.log(rd.random())  # log d'une v.a. uniforme entre 0 et 1
            # valeur de l'espérance pour cette valeur de variables explicatives
            mu = np.exp(np.dot(mat[i], new_param))
            # calculer le paramètre p de la régression géométrique pour cette valeur de variables explicatives
            ln_p_geo = -np.log(mu + 1)
            ln_q_geo = np.log(mu) - np.log(mu + 1)
            # calculer la likelyhood de y_i dans le modèle de régression de poisson
            if ly[i] > 30:  # alors l'approximation de stirling devient bonne
                log_fac = ly[i]*np.log(ly[i])-ly[i] + 0.5 * \
                    np.log(ly[i]) + np.log(np.sqrt(2*np.pi))
            else:
                # le calcul direct est faisable car la valeur est petite
                log_fac = np.log(float(np.math.factorial(ly[i])))
            # calculer la likelyhood de y_i dans le modèle de régression géométrique, le +1 vient de la définition de la loi géométrique dans la régression sur N et non N\{1}
            log_proba_poi = -mu - log_fac + \
                ly[i]*np.log(mu)  # likelyhood modèle poisson
            log_proba_geo = ln_p_geo + ly[i]*ln_q_geo  # likelyhood modèle geo
            rap_1 = np.exp(- np.log(p) - log_proba_poi +
                           np.log(1-p) + log_proba_geo)
            rap_2 = np.exp(np.log(p) + log_proba_poi -
                           np.log(1-p) - log_proba_geo)
            # avec les notations précédentes: calcul de ln(a + b) = ln(a) + ln(1 + b/a)
            sum_1 = np.log(p) + log_proba_poi + np.log(1 + rap_1)
            sum_2 = np.log(1-p) + log_proba_geo + np.log(1 + rap_2)
            if sum_1 <= 0:
                sumgood = sum_1
            else:
                sumgood = sum_2
                if sumgood > 0:
                    print('ouille')
            value = np.log(p) + log_proba_poi - sumgood
            if test < value:
                elem_z.append(0)
            else:
                elem_z.append(1)
        li_z.append(elem_z)  # enregistrer la nouvelle valeur de z
        # nombre de y_i attribués à la seconde régression (géométrique)
        n_1 = sum(elem_z)
        # nombre de y_i attribués à la première régression (poisson)
        n_0 = n-n_1
        # distribution conjuguée: générer selon \pi(p|z_t,\theta_{t-1})
        new_p = np.random.beta(0.5 + n_0, 0.5 + n_1)
        li_p.append(new_p)  # enregistrer la nouvelle valeur de p
    return [li_param, li_p, li_z, refus_param]


   # générer deux matrices de variables explicatives (une par régression pour assurer l'identifiablilité) ainsi que la liste des variables expliquées
'''mat, ly = mel_poi_geo(np.array([1, 2, 3, 4, 5]), 0, 50)
plt.hist(np.array(ly), bins=50)
plt.show()
li_param, li_p, li_z, refus_param, = sepa_gp(5000, mat, ly, np.array(
    [0, 0, 0, 0, 0]), 0.5)  # chercher à séparer les paramètres
# éliminer les 1000 premières valeurs et prendre l'espérance de p
p_guess = sum(li_p[2000:])/3000
param_guess = [0, 0, 0, 0, 0]
for i in range(3000):
    for j in range(5):
        # éliminer les 1000 premières valeurs et prendre l'espérance de param_0
        param_guess[j] = param_guess[j] + li_param[i+1999][j]
param_guess = np.array(param_guess)/3000
# afficher les paramètres "devinés"
print([param_guess, p_guess])
print(refus_param)
li_p = np.array(li_p)
plt.plot(li_p)
plt.show()
new_li_pasarray = []
for i in range(len(li_param)):
    elem = []
    for j in range(len(li_param[0])):
        elem.append(li_param[i][j])
    elem.append(li_p[i])
    new_li_pasarray.append(elem)
matrix_incomplete = np.array(new_li_pasarray)
df = pd.DataFrame(matrix_incomplete, columns=[
                  'compos_0', 'compos_1', 'compos_2', 'compos_3', 'compos_4', 'proba_poi'])
df = df.assign(p=li_p)
pd.plotting.scatter_matrix(df)
plt.show()'''

'''[array([1.1061218 , 1.99446556, 2.96137141, 3.97702375, 4.95927363]), 0.0953447767013995]
[9106. 9308. 9342. 9270. 9311.]'''

'''Tester si la structure collapsed gibbs fonctionne sur une séparation de régression linéaires'''


def test_collapsed(T, mat_u_echantillon, mat_v_echantillon, ly, param_0_init, param_1_init, p_init):
    n = len(ly)  # taille de l'échantillon généré
    k = len(mat_u_echantillon[0]) + 1  # taille du vecteur param_0
    l = len(mat_v_echantillon[0]) + 1  # taille du vecteur param_1
    # enregistrer le nombre de refus des paramètres de param_0 lors des étapes MH
    refus_param_0 = np.zeros(k)
    # enregistrer le nombre de refus des paramètres de param_1 lors des étapes MH
    refus_param_1 = np.zeros(l)
    # li_z va contenir toutes les listes d'attributions supposées (1ere régression : 0 ou deuxième régression: 1) des variables y
    li_z = []
    # li_param_0 va contenir toutes les valeurs de param_0 qui seront généréres par Gibbs within MH
    li_param_0 = [param_0_init]
    # li_param_1 va contenir toutes les valeurs de param_1 qui seront généréres par Gibbs within MH
    li_param_1 = [param_1_init]
    # li_p va contenir toutes les valeurs de p qui seront générées par Gibbs within MH
    li_p = [p_init]

    def log_pi(par_0, par_1):
        log_proba = 0
        for i in range(n):
            explicatives_u = mat_u_echantillon[i]
            explicatives_v = mat_v_echantillon[i]
            log_proba = log_proba + np.log(p*norm.pdf(ly[i], np.dot(par_0[:k-1], explicatives_u), par_0[k-1]) + (1-p)*norm.pdf(
                ly[i], np.dot(par_1[:l-1], explicatives_v), par_1[l-1]))  # construction de la likelyhood de par_0 et par_1
        log_proba = log_proba + np.log(f_gauss_multi(par_0[:k-1], np.zeros(k-1), 10) * f_gauss_multi(
            par_1[:l-1], np.zeros(l-1), 10) * expon.pdf(par_0[k-1]) * expon.pdf(par_1[l-1]))  # tenir compte des à prioris sur les paramètres par_0 et par_1
        return log_proba

    def simul_q_param_gauss(condit):
        return rd.gauss(condit, 0.3)  # faire des sauts gaussiens de 1

    def simul_q_param_expon(condit):
        # faire des sauts selon une loi de Wald centrée sur condit (pour toujours rester dans les positifs sur l'écart type)
        # espérance  = condit, variance = 1 (ajuster le scale parameter en fonction de condit pour cela)
        return np.random.wald(condit, condit**3)

    def q_proba_wald_reg_u(par_0, ex_par_0):
        proba = f_wald(par_0[k-1], ex_par_0[k-1], ex_par_0[k-1]**3)
        return proba

    def q_proba_wald_reg_v(par_1, ex_par_1):
        proba = f_wald(par_1[l-1], ex_par_1[l-1], ex_par_1[l-1]**3)
        return proba
    for t in range(T):  # 1 itération de cette boucle for = une itération de Gibbs within MH
        param_0 = li_param_0[t]  # précédent élément param_0 dans Gibbs
        param_1 = li_param_1[t]  # précédent élément param_1 dans Gibbs
        p = li_p[t]  # précédent élément p dans Gibbs

        # prochaine valeur de param_0, pour l'instant initialisée à un vecteur de 0: va changer
        new_param_0 = np.zeros(k)
        # prochaine valeur de param_1, pour l'instant initialisée à un vecteur de 0: va changer
        new_param_1 = np.zeros(l)
        precedent = np.copy(param_0)  # param_0_{t-1}
        proposition = np.copy(param_0)  # param_0_{t-1}
        for i in range(k-1):
            # étape MH: paramètres dans R: simulé par gaussienne
            param_prop = simul_q_param_gauss(precedent[i])
            # param_prop contient tous les nouveaux paramètres de param_0 jusqu'au (i-1)_ème de l'itération t de Gibbs ainsi que
            proposition[i] = param_prop
            facteur = min(log_pi(proposition, param_1) -
                          log_pi(precedent, param_1), 0)  # q_proposition est symétrique
            value = np.log(rd.random())
            if value < facteur:  # accepter la nouvelle valeur
                new_param_0[i] = param_prop
                precedent[i] = param_prop
            else:  # refuser la nouvelle valeur
                new_param_0[i] = precedent[i]
                proposition[i] = precedent[i]
                refus_param_0[i] = refus_param_0[i] + 1
        # etape MH sur l'écrt type de la première régression
        param_prop = simul_q_param_expon(precedent[k-1])
        proposition[k-1] = param_prop
        facteur = min(log_pi(proposition, param_1) - np.log(q_proba_wald_reg_u(proposition, precedent)) -
                      log_pi(precedent, param_1) + np.log(q_proba_wald_reg_u(precedent, proposition)), 0)
        value = np.log(rd.random())
        if value < facteur:  # accepter la nouvelle valeur
            new_param_0[k-1] = param_prop
            precedent[k-1] = param_prop
        else:  # refuser la nouvelle valeur
            new_param_0[k-1] = precedent[k-1]
            proposition[k-1] = precedent[k-1]
            refus_param_0[k-1] = refus_param_0[k-1] + 1
        li_param_0.append(new_param_0)
        precedent = np.copy(param_1)
        proposition = np.copy(param_1)
        # etape MH sur la seconde régression
        for i in range(l-1):
            param_prop = simul_q_param_gauss(precedent[i])
            proposition[i] = param_prop
            facteur = min(log_pi(new_param_0, proposition) -
                          log_pi(new_param_0, precedent), 0)
            value = np.log(rd.random())
            if value < facteur:  # accepter la nouvelle valeur
                new_param_1[i] = param_prop
                precedent[i] = param_prop
            else:  # refuser la nouvelle valeur
                new_param_1[i] = precedent[i]
                proposition[i] = precedent[i]
                refus_param_1[i] = refus_param_1[i] + 1
        # etape MH sur l'écrat type de la seconde régression
        param_prop = simul_q_param_expon(precedent[l-1])
        proposition[l-1] = param_prop
        facteur = min(log_pi(new_param_0, proposition) - np.log(q_proba_wald_reg_v(proposition, precedent)) -
                      log_pi(new_param_0, precedent) + np.log(q_proba_wald_reg_v(precedent, proposition)), 0)
        value = np.log(rd.random())
        if value < facteur:  # accepter la nouvelle valeur
            new_param_1[l-1] = param_prop
            precedent[l-1] = param_prop
        else:  # rejeter la nouvelle valeur
            new_param_1[l-1] = precedent[l-1]
            proposition[l-1] = precedent[l-1]
            refus_param_1[l-1] = refus_param_1[l-1] + 1
        # intégrer le nouveau paramètre à la liste
        li_param_1.append(new_param_1)
        elem_z = []
        for i in range(n):
            test = np.log(rd.random())
            value = np.log(p*f_gauss(ly[i], np.dot(mat_u_echantillon[i], new_param_0[:k-1].T), new_param_0[k-1])) - np.log(p*f_gauss(ly[i], np.dot(
                mat_u_echantillon[i], new_param_0[:k-1].T), new_param_0[k-1]) + (1-p)*f_gauss(ly[i], np.dot(mat_v_echantillon[i], new_param_1[:l-1].T), new_param_1[l-1]))  # générer elem_z[i] selon \pi(z_i_t|\theta_{t-1}, \p_{t-1})
            if test < value:
                elem_z.append(0)
            else:
                elem_z.append(1)
        li_z.append(elem_z)
        n_1 = sum(elem_z)  # nombre de y_i attribués à la seconde régression
        n_0 = n-n_1  # nombre de y_i attribués à la première régression
        # avec l'à priori en DIrichlet conjugué de bayesian core on trouve l'à postériori de p sachant elem_z suivant:
        new_p = np.random.beta(1 + n_0, 1 + n_1)
        li_p.append(new_p)  # enregistrer la nouvelle valeur de p
        print('oui')

    return [li_param_0, li_param_1, li_p, li_z, refus_param_0, refus_param_1]


'''mat_u_echantillon, mat_v_echantillon, ly = echant_superieure(
    250, np.array([1, 2, 3, 4, 5]), np.array([-1, -2, -10, 1]), 0.75)  # générer deux matrices de variables explicatives (une par régression pour assurer l'identifiablilité) ainsi que la liste des variables expliquées


li_param_0, li_param_1, li_p, li_z, refus_param_0, refus_param_1 = test_collapsed(5000, mat_u_echantillon, mat_v_echantillon, ly,
                                                                                  np.array([0, 0, 0, 0, 2]), np.array([-2, -2, -5, 0.5]), 0.5)  # chercher à séparer les paramètres
# éliminer les 1000 premières valeurs et prendre l'espérance de p
p_guess = sum(li_p[1000:])/4000
param_0_guess = [0, 0, 0, 0, 0]
param_1_guess = [0, 0, 0, 0]
for i in range(4000):
    for j in range(5):
        # éliminer les 1000 premières valeurs et prendre l'espérance de param_0
        param_0_guess[j] = param_0_guess[j] + li_param_0[i+999][j]
    for j in range(4):
        # éliminer les 1000 premières valeurs et prendre l'espérance de param_1
        param_1_guess[j] = param_1_guess[j] + li_param_1[i+999][j]
param_0_guess = np.array(param_0_guess)/4000
param_1_guess = np.array(param_1_guess)/4000
# afficher les paramètres "devinés"
print([param_0_guess, param_1_guess, p_guess])
print([refus_param_0, refus_param_1])  # afficher le nombre de refus'''

''''[array([2.30909099, 2.03328818, 3.17179405, 4.07081586, 4.55023604]), array([-1.19707473, -1.99299896, -9.96790751,  0.90884179]), 0.7851893867539114]'''

# rajouter une génération d'un aparamètre sigma noninformatif... changer un peu la loi de beta (deja fait), le reste ne doit pas être touché


def sepa_gp_sigma(T, mat, ly, param_init, p_init, ecart_type_init):
    n = len(mat)  # taille de l'échantillon d'apprentissage
    k = len(param_init)  # taille du vecteur de paramètres
    refus_param = np.zeros(k)
    refus_sigma = 0  # compter les refus au fur et à mesure
    # li_z va contenir toutes les listes d'attributions supposées (1ere régression : 0 ou deuxième régression: 1) des variables y
    li_z = []
    # li_param_poi va contenir toutes les valeurs de param qui seront généréres par Gibbs within MH
    li_param = [param_init]
    li_ecart_type = [ecart_type_init]
    # li_p va contenir toutes les valeurs de p qui seront générées par Gibbs within MH
    li_p = [p_init]
    mat_prod_lourd = np.linalg.inv(np.dot(mat.T, mat))

    def simul_sigma(condit):
        return np.random.wald(condit, condit**3)

    def log_q_sigma(x, condit):
        return np.log(f_wald(x, condit, condit**3))

    def log_sigma(ecart):
        return -1.5*ecart

    for t in range(T):
        print(t)
        # dernier paramètre généré
        param = np.array(li_param[t])
        p = li_p[t]  # dernier paramètre généré pour p
        ecart_type = li_ecart_type[t]

        def log_beta_sigma(par, ecart):
            # ecart type de la loi de beta sachant sigma et X
            proba = ecart * \
                multivariate_normal.pdf(par, np.zeros(
                    len(par)), ecart**2*mat_prod_lourd)
            # proba =  ecart * multivariate_normal.pdf(par, np.zeros(len(par)), ecart**2*mat_prod_lourd)  # Plutôt!!!
            return np.log(proba)

        def log_proba(par, sigma):  # par est le vecteur des coefficients pour les deux régressions
            ln_proba = 0
            for i in range(n):
                # calcul de l'espérance pour les i-èmes variables explicatives de la régression poisson
                esp = np.exp(np.dot(mat[i], par))
                # calcul de la probabilité p de réussite pour poisson (esp_géo peut valoir de 0 à +inf), cf convention choisie pour la loi géo
                # log de la probabilité p du succès dans la régression géométrique
                ln_p_geom = -np.log(esp + 1)
                # log de la probabilité q de l'échec dans la régression géométrique
                ln_q_geom = np.log(esp) - np.log(esp + 1)
                #print([ln_p_geom, ln_q_geom, esp])
                if ly[i] > 30:  # alors l'approximation de stirling devient bonne
                    log_fact = ly[i]*np.log(ly[i])-ly[i] + \
                        0.5*np.log(ly[i]) + \
                        np.log(np.sqrt(2*np.pi))  # stirling
                else:
                    # le calcul direct est faisable car la valeur est petite
                    log_fact = np.log(float(np.math.factorial(ly[i])))
                # calcul de la vraisemblance de la donnée y_i pour le modèle poisson
                ln_proba_poisson = -esp - log_fact + ly[i]*np.log(esp)
                # calcul de la vraisemblance de la donnée y_i pour le modèle géométrique
                ln_proba_geo = ln_p_geom + ly[i]*ln_q_geom
                rapport_1 = np.exp(- np.log(p) - ln_proba_poisson +
                                   np.log(1-p) + ln_proba_geo)  # rapport b/a où a = p*\pi(y|poisson,\theta) et b = (1-p)*\pi(y|géométrique,\theta)
                rapport_2 = np.exp(np.log(p) + ln_proba_poisson -
                                   np.log(1-p) - ln_proba_geo)  # avec les mêmes conventions: rapport a/b
                # il est impossible que rapport_1 et rapport_2 soient ensemble écrasés à +inf. Et celui qui est écrasé à 0 donnera un résultat global correct (cf D.L de ln(1+x); x<<1)
                supplement_1 = np.log(
                    p) + ln_proba_poisson + np.log(1 + rapport_1)  # avec les notations précédentes: calcul de ln(a + b) = ln(a) + ln(1 + b/a)
                supplement_2 = np.log(1-p) + ln_proba_geo + \
                    np.log(
                        1 + rapport_2)  # avec les notations précédentes: calcul de ln(a + b) = ln(b) + ln(1 + a/b)
                if rapport_1 <= 1:  # i.e. si rapport_1 pas écrasé à l'infini
                    supplement = supplement_1
                else:  # alors forcément rapport_2 pas écrasé à l'infini
                    supplement = supplement_2
                    if supplement_2 > 0:
                        print('aïe')
                        print([ln_proba_poisson, ln_proba_geo,
                               supplement, esp, ly[i], log_fact])
                ln_proba = ln_proba + supplement
            # probabilité des données sachant la théorie
            # on trouve log(proba(\beta,\sigma|p,X))
            ln_proba = ln_proba + log_beta_sigma(par, sigma) + log_sigma(sigma)
            return ln_proba

        def simul_q_saut(condit):
            # réaliser un saut gaussien centré en condit d'écart-type unitaire: symétrique
            return rd.gauss(condit, 0.5)

        # nouveau paramètre pour les deux régressions
        new_param = np.zeros(k)
        # précédent paramètre , qui tiendra compte des sauts MH progressivement acceptés
        precedent = np.copy(param)
        # précédent paramètre qui tindra compte des sauts MH progressivement acceptés et des supposition de sauts à tester
        proposition = np.copy(param)
        for i in range(k):
            # le i-ème coefficient fait un saut normal d'écart-type 1
            param_prop = simul_q_saut(precedent[i])
            # on tient compte du saut dans proposition
            proposition[i] = param_prop
            # calcul du facteur dans MH avec loi de saut symétrique
            facteur = min(log_proba(proposition, ecart_type) -
                          log_proba(precedent, ecart_type), 0)

            value = np.log(rd.random())
            if value < facteur:  # accepter la nouvelle valeur
                new_param[i] = param_prop
                precedent[i] = param_prop
            else:  # refuser la nouvelle valeur
                new_param[i] = precedent[i]
                proposition[i] = precedent[i]
                refus_param[i] = refus_param[i] + 1

        # ajouter le nouveau paramètre à la liste
        li_param.append(new_param)

        ancient = ecart_type
        proposition = simul_sigma(ancient)
        rapport = log_proba(new_param, proposition) - log_q_sigma(proposition, ancient) - \
            log_proba(new_param, ancient) + log_q_sigma(ancient, proposition)
        facteur = min(1, rapport)
        value = np.log(rd.random())
        if value < facteur:
            new_ecart = proposition
        else:
            new_ecart = ancient
            refus_sigma = refus_sigma + 1
        li_ecart_type.append(new_ecart)

        p = li_p[t]
        elem_z = []
        for i in range(n):  # etape de Gibbs sampling pour z
            test = np.log(rd.random())  # log d'une v.a. uniforme entre 0 et 1
            # valeur de l'espérance pour cette valeur de variables explicatives
            mu = np.exp(np.dot(mat[i], new_param))
            # calculer le paramètre p de la régression géométrique pour cette valeur de variables explicatives
            ln_p_geo = -np.log(mu + 1)
            ln_q_geo = np.log(mu) - np.log(mu + 1)
            # calculer la likelyhood de y_i dans le modèle de régression de poisson
            if ly[i] > 30:  # alors l'approximation de stirling devient bonne
                log_fac = ly[i]*np.log(ly[i])-ly[i] + 0.5 * \
                    np.log(ly[i]) + np.log(np.sqrt(2*np.pi))
            else:
                # le calcul direct est faisable car la valeur est petite
                log_fac = np.log(float(np.math.factorial(ly[i])))
            # calculer la likelyhood de y_i dans le modèle de régression géométrique, le +1 vient de la définition de la loi géométrique dans la régression sur N et non N\{1}
            log_proba_poi = -mu - log_fac + \
                ly[i]*np.log(mu)  # likelyhood modèle poisson
            log_proba_geo = ln_p_geo + ly[i]*ln_q_geo  # likelyhood modèle geo
            rap_1 = np.exp(- np.log(p) - log_proba_poi +
                           np.log(1-p) + log_proba_geo)
            rap_2 = np.exp(np.log(p) + log_proba_poi -
                           np.log(1-p) - log_proba_geo)
            # avec les notations précédentes: calcul de ln(a + b) = ln(a) + ln(1 + b/a)
            sum_1 = np.log(p) + log_proba_poi + np.log(1 + rap_1)
            sum_2 = np.log(1-p) + log_proba_geo + np.log(1 + rap_2)
            if sum_1 <= 0:
                sumgood = sum_1
            else:
                sumgood = sum_2
                if sumgood > 0:
                    print('ouille')
            value = np.log(p) + log_proba_poi - sumgood
            if test < value:
                elem_z.append(0)
            else:
                elem_z.append(1)
        li_z.append(elem_z)  # enregistrer la nouvelle valeur de z
        # nombre de y_i attribués à la seconde régression (géométrique)
        n_1 = sum(elem_z)
        # nombre de y_i attribués à la première régression (poisson)
        n_0 = n-n_1
        # distribution conjuguée: générer selon \pi(p|z_t,\theta_{t-1})
        new_p = np.random.beta(0.5 + n_0, 0.5 + n_1)
        li_p.append(new_p)  # enregistrer la nouvelle valeur de p
    return [li_param, li_p, li_z, li_ecart_type, refus_param, refus_sigma]


def f_esp(serie):
    n = len(serie)
    return sum(serie)/n


def f_var(serie):
    esp = f_esp(serie)
    var = 0
    n = len(serie)
    for elem in serie:
        var = var + (elem - esp)**2
    return var/(n-1)


def autocor(serie, k):
    esp = f_esp(serie)
    var = f_var(serie)
    n = len(serie)
    somme = 0
    for i in range(n-k):
        somme = somme + (serie[i] - esp)*(serie[i + k] - esp)
    return somme/(var*(n-k))


# générer deux matrices de variables explicatives (une par régression pour assurer l'identifiablilité) ainsi que la liste des variables expliquées
def testing(echantillon):
    mat, ly = mel_poi_geo(np.array([1, 2, 3]), 0, 50)
    #plt.hist(np.array(ly), bins=50)
    # plt.show()
    li_param, li_p, li_z, li_ecart_type, refus_param, refus_sigma = sepa_gp_sigma(4000, mat, ly, np.array(
        [0, 0, 0]), 0.5, 1)  # chercher à séparer les paramètres
    # éliminer les 1000 premières valeurs et prendre l'espérance de p
    p_guess = sum(li_p[3000:])/1000
    sigma_guess = sum(li_ecart_type[3000:])/1000
    param_guess = [0, 0, 0]
    for i in range(1000):
        for j in range(3):
            # éliminer les 1000 premières valeurs et prendre l'espérance de param_0
            param_guess[j] = param_guess[j] + li_param[i+2999][j]
    param_guess = np.array(param_guess)/1000
    # afficher les paramètres "devinés"
    print([param_guess, p_guess, sigma_guess, 'guess'])
    print([refus_param/4000, refus_sigma/4000, 'refus'])
    # param_0_moy.append(param_guess[0])
    # param_1_moy.append(param_guess[1])
    # param_2_moy.append(param_guess[2])
    # p_moy.append(p_guess)
    li_p = np.array(li_p)
    li_0 = np.array([li_param[i][0] for i in range(3000, 4000)])
    li_1 = np.array([li_param[i][1] for i in range(3000, 4000)])
    li_2 = np.array([li_param[i][2] for i in range(3000, 4000)])
    print(len(li_0))
    print(len(li_p[3001:]))
    figure, axis = plt.subplots(nrows=4, ncols=2)
    x_vect = np.array(range(len(li_0)))
    axis[0, 0].scatter(x_vect, li_0, s=1)
    axis[1, 0].scatter(x_vect, li_1, s=1)
    axis[0, 1].scatter(x_vect, li_2, s=1)
    axis[1, 1].scatter(x_vect, li_p[3001:], s=1)
    axis[0, 0].set_title("Composante 0")
    axis[1, 0].set_title("Composante 1")
    axis[0, 1].set_title("Composante 2")
    axis[1, 1].set_title("proba")
    number = str(echantillon)
    # plt.savefig("C:\\Users\\33771\\OneDrive\\Documents\\Projet_recherche\\save_files\\li_p{}.jpg".format(number))
    plt.show()
    plt.close('all')
    new_li_pasarray = []
    for i in range(len(li_param)):
        elem = []
        for j in range(len(li_param[0])):
            elem.append(li_param[i][j])
        elem.append(li_p[i])
        elem.append(li_ecart_type[i])
        new_li_pasarray.append(elem)
    matrix_incomplete = np.array(new_li_pasarray)
    df = pd.DataFrame(matrix_incomplete, columns=[
        'compos_0', 'compos_1', 'compos_2', 'proba_poi', 'sigma'])
    df = df.assign(p=li_p)
    pd.plotting.scatter_matrix(df)
    # plt.savefig("C:\\Users\\33771\\OneDrive\\Documents\\Projet_recherche\\save_files\\scatter{}.jpg".format(number))
    plt.show()
    plt.close('all')

    n_auto_cor = 900
    series = [[], [], [], [], []]
    for i in range(1000, len(li_param)):  # offset
        for j in range(len(li_param[0])):
            series[j].append(li_param[i][j])
        series[3].append(li_p[i])
        series[4].append(li_ecart_type[i])
    series = np.array(series)
    auto_cors = np.zeros((n_auto_cor, 7))
    for j in range(5):
        for i in range(n_auto_cor):
            auto_cors[i, j] = autocor(series[j], i)
    figure, axis = plt.subplots(nrows=2, ncols=2)
    x_vect = range(n_auto_cor)
    axis[0, 0].scatter(x_vect, auto_cors[:, 0], s=1)
    axis[1, 0].scatter(x_vect, auto_cors[:, 1], s=1)
    axis[0, 1].scatter(x_vect, auto_cors[:, 2], s=1)
    axis[1, 1].scatter(x_vect, auto_cors[:, 3], s=1)
    #axis[0, 1].scatter(x_vect, auto_cors[:, 4], s=1)
    #axis[1, 1].scatter(x_vect, auto_cors[:, 5], s=1)
    #axis[2, 1].scatter(x_vect, auto_cors[:, 6], s=1)

    axis[0, 0].set_title("Composante 0")
    axis[1, 0].set_title("Composante 1")
    axis[0, 1].set_title("Composante 2")
    axis[1, 1].set_title("proba")
    #axis[0, 1].set_title("Ecart type loi a priori")
    #axis[1, 1].set_title("Probabilité Poisson")
    #axis[2, 1].set_title("Ecart type loi a priori")
    # plt.savefig("C:\\Users\\33771\\OneDrive\\Documents\\Projet_recherche\\save_files\\autocorr{}.jpg".format(number))
    plt.show()
    plt.close('all')


'''testing('str')'''  # lancer le programme

'''résultats: pour une taille 20 avec param = [0.5, 1, 0.7] et p = 1 on trouve:'''
'''param_0_moy = [0.2871, 0.0510, 0.27, 0.347, 0.01, 0.147, 0.08796, 0.38416599, 0.23397627,  0.11849588,
               0.62921017, 0.53093, 0.38232876, 0.50343, 0.13911541, 0.75806737, 0.53405252, 0.3863, 0.20532256, 0.73596023]
param_1_moy = [0.27624597,-0.66, -0.187, 1.620, -0.31046701, -0.77-1.60409722, 0.58219959, -1.553862, 3.23562319, -
               0.461, 0.025, 1.64, 0.26659756, 1.29975, -0.31912167, 1.0762794, -1.8673506, -0.0278, 0.977034, 0.04296297]
param_2_moy = [-0.19176822, -0.530, 0.498, -0.2065, 1.26, 2.78884, 0.5230, -1.55386264, 3.2928, -0.1254557,
               4.00, -3.3634225, 1.08414725, 2.5618, 0.3366, 0.84233579, 0.06880238, 1.72648057, -0.01290041, 1.29925867]
p_moy = [0.86831776, 0.726123, 0.76199478, 0.76845,  0.7523, 0.74152, 0.6492, 0.710128, 0.77298,
         0.7555, 0.891, 0.802, 0.74232, 2.561, 0.662, 0.84233579, 0.7238291, 0.71158, 0.750, 0.8044392]
print([len(p_moy), len(param_0_moy), len(param_1_moy), len(param_2_moy), 'len'])
print(np.array([sum(p_moy), sum(param_0_moy),
                sum(param_1_moy), sum(param_2_moy)])/20)
#figure, axis = plt.subplots(nrows=4, ncols=2)
plt.hist(param_1_moy)
plt.show()'''

'''résultats: pour une taille 50 avec param = [0.5, 1, 0.7] et p = 1 on trouve:'''
'''param_0_moy = [0.19984581, 0.335595, 0.27231634,
               0.25784118, 0.19625746, 0.31255421, 0.485560936, 0.45823965, 0.31607035, 0.34218311, 0.38934441, 0.49945299, 0.3726463, 0.16837356, 0.36074911, 0.15874519, 0.20234849, 0.18247397, 0.26941211, 0.35305541]
param_1_moy = [-0.66096112, 0.6171766, 0.16215013,
               0.58378211,  1.25086189, 1.3591379, 3.13869626, -0.73024301, 1.23687217, 0.2493312, 0.70328718, -0.16062582, 0.52179737, 0.85810341, 1.39478019, -0.78394703, -0.41244197, -0.37092063, 0.76037913, 0.55541037]
param_2_moy = [0.21101548, -0.62938618, 1.63931961,
               1.24350401, -0.79976776, 1.08237896, 0.00222163629, 0.64992396, 0.29077565, -0.03652946, 1.00175523, 0.09984815, 0.3251056, -0.01963333, 0.85372747, 1.98955362, -0.53076909, -1.46669359, -0.3282352, 1.35783267]
p_moy = [0.840, 0.882, 0.9056196, 0.872399902,
         0.9304444, 0.9111665890,  0.872235976650709, 0.9326568424511708, 0.84447120, 0.8160963, 0.798755, 0.896981, 0.909989, 0.83293526, 0.948973155, 0.801581, 0.933469, 0.821932, 0.85470, 0.8976126686094376]
print([len(p_moy), len(param_0_moy), len(param_1_moy), len(param_2_moy), 'len'])
print(np.array([sum(p_moy), sum(param_0_moy),
                sum(param_1_moy), sum(param_2_moy)])/20)
#figure, axis = plt.subplots(nrows=4, ncols=2)
plt.hist(param_1_moy)
plt.show()'''

'''Résultats pour taille avec param = [0.5,1,0.7] et p = 1'''
'''[array([0.41494956, 0.84445105, 0.64213911]), 0.960826828227298, 3.927735190160269, 'guess']
[array([0.90675, 0.39   , 0.379  ]), 0.37725, 'refus']'''

'''param_0_moy = [0.41494956, 0.5988615120551454, 0.5192044655686644, 0.48021795323765165, 0.40072960965096244, 0.5231174950668351, 0.5027670990891783, 0.4566128455360184, 0.45345970595327795, 0.5413873960418923,
               0.5009303787839701, 0.47249200195288993, 0.4639933241993264, 0.45177706584530264, 0.43738621375768666, 0.41203930490705815, 0.4819228082538011, 0.489490931731072, 0.4856397878249028, 0.5002247624049366]
param_1_moy = [0.84445105, 1.0607742473599733, 1.147390569214397, 0.9553022326757238, 0.45105271145561576, 1.4947946180067158, 0.657721263111405, 0.7359294083973686, 1.0554139732358099, 0.7789711822684516,
               1.04156021558582, 0.7983991222879743, 1.1762265916089107, 1.1982180217775469, 1.0446837557988526, 0.9324919250711521, 0.11769754690747417, 0.8475796484262614, 0.651323860674943, 0.7842308914385417]
param_2_moy = [0.64213911, 1.219235228190057, 0.6841464552522283, 0.07478324634602364, 0.988226841155513, 0.4584126894325057, 1.0203350718762345, 0.7795077236476455, 0.9475833137187304, 0.5423179629701612,
               0.43153239191527976, 1.0136771883274762, 1.6050674231380697, 1.0970614895231776, 0.8751943342033957, 0.33807563777774496, 0.26754637836446427, 0.885966160572589, 0.6076449935745385, 0.5205533436329588]
p_moy = [0.9608, 0.9622379977934857, 0.9874857104296312, 0.9770166295767616, 0.9866080587157714, 0.9757294319435107, 0.9856024030931695, 0.9795539253127303, 0.9292439884450721, 0.9811803714036379,
         0.9077425802169742, 0.9869372403556023, 0.9554299783901864, 0.9361543518650999, 0.9905710791544363, 0.9578380487605445, 0.9757218967442488, 0.9871740149195504, 0.9873227506254584, 0.9723031286321048]'''
'''for i in range(1, 20):
    testing(i)
    print([param_0_moy, 'param_0_ite{}'.format(str(i))])
    print([param_1_moy, 'param_1_ite{}'.format(str(i))])
    print([param_2_moy, 'param_2_ite{}'.format(str(i))])
    print([p_moy, 'proba_ite{}'.format(str(i))])'''
'''print(np.array([sum(p_moy), sum(param_0_moy),
                sum(param_1_moy), sum(param_2_moy)])/20)
plt.hist(param_0_moy)
plt.show()
plt.close('all')
plt.hist(param_1_moy)
plt.show()
plt.close('all')
plt.hist(param_2_moy)
plt.show()
plt.close('all')
plt.hist(p_moy)
plt.show()
plt.close('all')'''


'''simuler selon un réseau de neurones 2x2x1 un set de n couples de variables explicatives, et de variable expliquée. Sortie: matrice de variables explicatives X et li_variables expliquées'''
'''poids de la forme dico d'arrays numpy où la clef (j,k) donne la liste des poids menant au j-ème neurone de la couche k (indexes depuis 0) où l'offset est au début de la liste.
La couche 0 est la couche d'entrée: rien ne s'y passe. On commence le travail à la couche 1.'''
'''sigma est l'écart type de chaque perturbation à chaque étape'''
# np.random.multivariate_normal#


def sigmo(x, param):
    return 1/(1 + np.exp(-param*x))


'''On utilisera une fonction d'activation en sigmoïde de paramètre 1, à comparer avec une sigmoïde de paramètre 2'''


def simul_neu_2(n, w, sigma, lambda_sigm_1, lambda_sigm_2, p):
    X_0 = np.zeros((n, 3))
    attila = np.array([1 for i in range(n)])
    X_0[:, 0] = attila
    X_0[:, 1] = np.random.multivariate_normal(
        np.array([0 for i in range(n)]), 0.01*np.eye(n))
    X_0[:, 2] = np.random.multivariate_normal(
        np.array([0 for i in range(n)]), 0.01*np.eye(n))
    X_1 = np.zeros((n, 3))
    X_1[:, 0] = attila
    ly = []
    for i in range(n):
        value = rd.random()
        if value < p:
            X_1[i, 1] = rd.gauss(
                sigmo(lambda_sigm_1, np.dot(w[(0, 1)], X_0[i, :])), sigma)
            X_1[i, 2] = rd.gauss(
                sigmo(lambda_sigm_1, np.dot(w[(1, 1)], X_0[i, :])), sigma)
            y = rd.gauss(sigmo(lambda_sigm_1, np.dot(
                w[(0, 2)], X_1[i, :])), sigma)
        else:
            X_1[i, 1] = rd.gauss(
                sigmo(lambda_sigm_2, np.dot(w[(0, 1)], X_0[i, :])), sigma)
            X_1[i, 2] = rd.gauss(
                sigmo(lambda_sigm_2, np.dot(w[(1, 1)], X_0[i, :])), sigma)
            y = rd.gauss(sigmo(lambda_sigm_2, np.dot(
                w[(0, 2)], X_1[i, :])), sigma)
        ly.append(y)
    return [X_0, ly]


'''w = {(0, 1): [1, 0.7, -0.2], (1, 1): [-1.2, 2, 0.1], (0, 2): [0.6, -0.3, -0.12]}
n = 30
sigma = 0.2
lambda_sigm_1 = 1
lambda_sigm_2 = 2
p = 0.5
X_0, ly = simul_neu_2(n, w, sigma, lambda_sigm_1,lambda_sigm_2,p)
print(X_0)
print(ly)'''


def simul_q_saut_poids(ancient):
    return rd.gauss(ancient, 0.1)


def simul_q_saut_X_1(ancient):
    return rd.gauss(ancient, 0.2)


def simul_q_sigma(ancient):
    return np.random.wald(ancient, ancient**3)


def log_pi_q_sigma(x, ancient):
    return np.log(f_wald(x, ancient, ancient**3))


def sepa_neuro_2(X_0, ly, n_ite, w_1_start, w_2_start, X_1_1_start, X_1_2_start, sigma_start, p_start):
    li_w_1 = [w_1_start]
    refus_w_1 = {key: [0 for i in w_1_start[key]] for key in w_1_start.keys()}
    li_w_2 = [w_2_start]
    refus_w_2 = {key: [0 for i in w_1_start[key]] for key in w_2_start.keys()}
    li_X_1_1 = [X_1_1_start]
    refus_X_1_1 = np.zeros(np.shape(X_1_1_start))
    li_X_1_2 = [X_1_2_start]
    refus_X_1_2 = np.zeros(np.shape(X_1_2_start))
    li_sigma = [sigma_start]
    refus_sigma = 0
    li_p = [p_start]
    li_z = []
    n = len(ly)

    # la vraisemblance de y sachant les w_1,w_2, X_1_1,X_1_2, sigma et p: (c'est le produit qui contient les p et (1-p))
    def log_vrais_ly_sachant_X_1(poids_1, poids_2, Y_1_1, Y_1_2, proba, ecart):
        log_proba = 0
        for i in range(n):
            f_1 = norm.pdf(ly[i], sigmo(
                np.dot(poids_1[(0, 2)], Y_1_1[i, :]), 1), ecart)
            f_2 = norm.pdf(ly[i], sigmo(
                np.dot(poids_2[(0, 2)], Y_1_2[i, :]), 2), ecart)
            log_proba = log_proba + np.log(proba*f_1 + (1-proba)*f_2)
        return log_proba

    # la vraisemblance de X_1_num sachant les poids w_num, sigma et X_0
    def log_vrais_X_1(num, poids, Y_1, ecart):
        log_proba = 0
        for i in range(n):
            log_proba = log_proba + np.log(norm.pdf(Y_1[i, 1], sigmo(np.dot(X_0[i, :], poids[(
                0, 1)]), 1 + num), ecart)) + np.log(norm.pdf(Y_1[i, 2], sigmo(np.dot(X_0[i, :], poids[(1, 1)]), 1 + num), ecart))
        return log_proba

    def log_priori_poids(poids):  # probabilité à priori d'un vecteur poids
        vecteur = []
        for elem in range(3):
            vecteur.append(poids[(0, 1)][elem])
        for elem in range(3):
            vecteur.append(poids[(1, 1)][elem])
        for elem in range(3):
            vecteur.append(poids[(0, 2)][elem])
        vecteur = np.array(vecteur)
        log_proba = np.log(f_gauss_multi(vecteur, 0, 1))
        return log_proba

    def log_priori_sigma(ecart):  # probabilité à priori d'un écart type
        return expon.pdf(ecart)

    def log_proba_beta(num, poids_1, poids_2, Y_1_1, Y_1_2, proba, ecart):
        if num == 0:
            poids = poids_1
        if num == 1:
            poids = poids_2
        log_proba = log_vrais_ly_sachant_X_1(poids_1, poids_2, Y_1_1, Y_1_2, proba, ecart) + log_vrais_X_1(0,
                                                                                                           poids_1, Y_1_1, ecart) + log_vrais_X_1(1, poids_2, Y_1_2, ecart) + log_priori_poids(poids)
        return log_proba

    def log_proba_X1(num, poids_1, poids_2, Y_1_1, Y_1_2, proba, ecart):
        if num == 0:
            Y_1 = Y_1_1
            poids = poids_1
        if num == 1:
            Y_1 = Y_1_2
            poids = poids_2
        log_proba = log_vrais_ly_sachant_X_1(
            poids_1, poids_2, Y_1_1, Y_1_2, proba, ecart) + log_vrais_X_1(num, poids, Y_1, ecart)
        return log_proba

    def log_proba_sigma(poids_1, poids_2, Y_1_1, Y_1_2, proba, ecart):
        log_proba = log_vrais_ly_sachant_X_1(poids_1, poids_2, Y_1_1, Y_1_2, proba, ecart) + log_vrais_X_1(0,
                                                                                                           poids_1, Y_1_1, ecart) + log_vrais_X_1(1, poids_2, Y_1_2, ecart) + log_priori_sigma(ecart)
        return log_proba

    def log_proba_z_i_0(indice, poids_1, poids_2, Y_1_1, Y_1_2, proba, ecart):
        f1 = norm.pdf(ly[indice], sigmo(
            np.dot(poids_1[(0, 2)], Y_1_1[indice, :]), 1), ecart)
        f2 = norm.pdf(ly[indice], sigmo(
            np.dot(poids_2[(0, 2)], Y_1_2[indice, :]), 2), ecart)
        log_proba = np.log(proba) + np.log(f1) - \
            np.log(proba*f1 + (1-proba)*f2)
        return log_proba

    for ite in range(n_ite):
        w_1 = li_w_1[ite]
        w_2 = li_w_2[ite]
        X_1_1 = li_X_1_1[ite]
        X_1_2 = li_X_1_2[ite]
        sigma = li_sigma[ite]
        p = li_p[ite]
        precedent = cp.deepcopy(w_1)
        proposition = cp.deepcopy(w_1)
        new_param = cp.deepcopy(w_1)
        for key in w_1.keys():
            for i in range(len(w_1[key])):
                param_prop = simul_q_saut_poids(precedent[key][i])
                proposition[key][i] = param_prop
                facteur = min(log_proba_beta(0, proposition, w_2, X_1_1, X_1_2, p, sigma) - log_proba_beta(
                    0, precedent, w_2, X_1_1, X_1_2, p, sigma), 0)  # distribution q symétrique
                value = np.log(rd.random())
                if value < facteur:
                    new_param[key][i] = param_prop
                    precedent[key][i] = param_prop
                else:
                    new_param[key][i] = precedent[key][i]
                    proposition[key][i] = precedent[key][i]
                    refus_w_1[key][i] = refus_w_1[key][i] + 1
        w_1 = cp.deepcopy(new_param)
        precedent = cp.deepcopy(w_2)
        proposition = cp.deepcopy(w_2)
        new_param = cp.deepcopy(w_2)
        for key in w_2.keys():
            for i in range(len(w_2[key])):
                param_prop = simul_q_saut_poids(precedent[key][i])
                proposition[key][i] = param_prop
                facteur = min(log_proba_beta(1, w_1, proposition, X_1_1, X_1_2, p, sigma) - log_proba_beta(
                    1, w_1, precedent, X_1_1, X_1_2, p, sigma), 0)  # distribution q symétrique
                value = np.log(rd.random())
                if value < facteur:
                    new_param[key][i] = param_prop
                    precedent[key][i] = param_prop
                else:
                    new_param[key][i] = precedent[key][i]
                    proposition[key][i] = precedent[key][i]
                    refus_w_2[key][i] = refus_w_2[key][i] + 1
        w_2 = cp.deepcopy(new_param)
        precedent = cp.deepcopy(X_1_1)
        proposition = cp.deepcopy(X_1_1)
        new_param = cp.deepcopy(X_1_1)
        for i in range(n):
            for j in range(len(X_1_1[0, 1:])):
                param_prop = simul_q_saut_X_1(precedent[i, j])
                proposition[i, j] = param_prop
                facteur = min(log_proba_X1(0, w_1, w_2, proposition, X_1_2, p, sigma) - log_proba_X1(
                    0, w_1, w_2, precedent, X_1_2, p, sigma), 0)  # distribution q symétrique
                value = np.log(rd.random())
                if value < facteur:
                    new_param[i, j] = param_prop
                    precedent[i, j] = param_prop
                else:
                    new_param[i, j] = precedent[i, j]
                    proposition[i, j] = precedent[i, j]
                    refus_X_1_1[i, j] = refus_X_1_1[i, j] + 1
        X_1_1 = cp.deepcopy(new_param)
        precedent = cp.deepcopy(X_1_2)
        proposition = cp.deepcopy(X_1_2)
        new_param = cp.deepcopy(X_1_2)
        for i in range(n):
            for j in range(len(X_1_2[0, 1:])):
                param_prop = simul_q_saut_X_1(precedent[i, j])
                proposition[i, j] = param_prop
                facteur = min(log_proba_X1(1, w_1, w_2, X_1_1, proposition, p, sigma) - log_proba_X1(
                    1, w_1, w_2, X_1_1, precedent, p, sigma), 0)  # distribution q symétrique
                value = np.log(rd.random())
                if value < facteur:
                    new_param[i, j] = param_prop
                    precedent[i, j] = param_prop
                else:
                    new_param[i, j] = precedent[i, j]
                    proposition[i, j] = precedent[i, j]
                    refus_X_1_2[i, j] = refus_X_1_2[i, j] + 1
        X_1_2 = cp.deepcopy(new_param)
        precedent = sigma
        proposition = simul_q_sigma(precedent)
        facteur = min(log_proba_sigma(w_1, w_2, X_1_1, X_1_2, p, proposition) + log_pi_q_sigma(precedent, proposition) -
                      log_proba_sigma(w_1, w_2, X_1_1, X_1_2, p, precedent) - log_pi_q_sigma(proposition, precedent), 0)
        value = np.log(rd.random())
        if value < facteur:
            sigma = proposition
            precedent = proposition
        else:
            sigma = precedent
            refus_sigma = refus_sigma + 1
        elem = []
        for i in range(n):
            value = np.log(rd.random())
            if value < log_proba_z_i_0(i, w_1, w_2, X_1_1, X_1_2, p, sigma):
                elem.append(0)
            else:
                elem.append(1)
        elem = np.array(elem)
        li_z.append(elem)
        n_1 = sum(elem)
        n_0 = n-n_1
        new_p = np.random.beta(0.5 + n_0, 0.5 + n_1)
        li_w_1.append(cp.deepcopy(w_1))
        li_w_2.append(cp.deepcopy(w_2))
        li_X_1_1.append(cp.deepcopy(X_1_1))
        li_X_1_2.append(cp.deepcopy(X_1_2))
        li_sigma.append(sigma)
        li_p.append(new_p)
        print(new_p)
    return [li_w_1, li_w_2, li_sigma, li_p, refus_X_1_1, refus_X_1_2]


'''w = {(0, 1): [1, 0.7, -0.2], (1, 1): [-1.2, 2, 0.1], (0, 2): [0.6, -0.3, -0.12]}
n = 30
sigma = 0.2
lambda_sigm_1 = 1
lambda_sigm_2 = 2
p = 1
n_ite = 4000
w_1_start = {key: np.array([0 for i in w[key]]) for key in w.keys()}
w_2_start = {key: np.array([0 for i in w[key]]) for key in w.keys()}
X_0, ly = simul_neu_2(n, w, sigma, lambda_sigm_1, lambda_sigm_2, p)
attila = np.array([1 for i in range(n)])
X_1_1_start = np.zeros(np.shape(X_0))
X_1_1_start[:, 0] = attila
X_1_2_start = np.zeros(np.shape(X_0))
X_1_2_start[:, 0] = attila
sigma_start = 2
p_start = 0.5
li_w_1, li_w_2, li_sigma, li_p, refus_X_1_1, refus_X_1_2 = sepa_neuro_2(
    X_0, ly, n_ite, w_1_start, w_2_start, X_1_1_start, X_1_2_start, sigma_start, p_start)
plt.plot(li_p)
plt.show()
print(sum(li_p[1000:])/3000)
print(refus_X_1_1)
print(refus_X_1_2)'''

'''Marche pas: trop lent'''

'''Etude d'un modèle de la régression de poisson avec la méthode hamiltonienne'''

'''générer des données: réutiliser poisson-géométrique: but à terme'''

'''Il faudra corriger cette fonction par un tuning de eps et L'''


def hamilt_poisson_non_corr(X, ly, eps, L, q_init, n_ite_apres_offset, offset):
    n = len(ly)  # nb données
    k = len(X[0, :])  # taille du beta
    # liste des variables positionnelles de la forme [beta sigma] dans une même liste
    li_q = [np.array(q_init)]
    li_p = [np.random.multivariate_normal(np.zeros(k+1), np.eye(k+1))]
    mat_prod_lourd = np.linalg.inv(np.dot(X.T, X))  # (X.T@X)^(-1)
    refus = 0

    # calcule le log de la probabilité de beta_f,sigma_f|X,ly
    def log_proba_goal(beta_f, sigma_f):
        log_pi = 0
        for i in range(n):
            # lambda de la loi de poisson pour le beta_f donné
            param_poi = np.exp(np.dot(X[i, :], beta_f))
            if ly[i] > 30:
                log_fact = ly[i]*np.log(ly[i])-ly[i] + \
                    0.5 * \
                    np.log(ly[i]) + np.log(np.sqrt(2*np.pi)
                                           )  # stirling: calculer factorielle de ly[i]
            else:
                # si faible coût: méthode exacte
                log_fact = np.log(float(np.math.factorial(ly[i])))
            # log de la proba de ly[i] selon la loi de Poisson de paramètre lambda = param_poi
            ln_proba_poisson = -param_poi - log_fact + ly[i]*np.log(param_poi)
            # le produit des vraisemblances de ly[i] devient un somme avec le log
            log_pi = log_pi + ln_proba_poisson
        log_pi = log_pi + np.log(multivariate_normal.pdf(beta_f, np.zeros(
            len(beta_f)), sigma_f**2*mat_prod_lourd)) - 1.5*np.log(sigma_f)  # tenir compte des à prioris: G-prior utilisé...
        return log_pi

    # renvoit le gradient de E au point q = [beta_f sigma_f]
    def calcul_dE(beta_f, sigma_f):
        # futur vecteur gradient (vecteur des dérivées partielles)
        dE = np.zeros(k + 1)
        # cas où on prend la dérivée selon un beta[r] (r<k) et non selon sigma = q[k]
        for r in range(k):
            e_r = np.zeros(k)
            e_r[r] = 1  # vecteur qui vaut zéro partout sauf en r où il vaut 1
            dE[r] = - sum([ly[i]*X[i, r] - X[i, r]*np.exp(np.dot(X[i, :], beta_f))
                           for i in range(n)]) + (1/sigma_f**2)*np.dot(X @ beta_f, X @ e_r)
        dE[k] = k/sigma_f - np.dot(X @ beta_f, X @ beta_f) / \
            sigma_f**3 + 3/(2*sigma_f)  # idem
        return dE

    def calcul_H(q_f, p_f):  # renvoit le hamiltonien en (q_f,p_f)
        E = - log_proba_goal(q_f[:k], q_f[k])  # énergie potentielle
        K = sum([0.5*p_f[i]**2 for i in range(k+1)])  # énergie cinétique
        return E + K  # hamiltonien

    # réalise une étape d'updating en partant de (q_dep,p_dep) (il y en a L dans une "grande étape")
    def step(q_dep, p_dep, eps_step):
        # cf papier: dynamique hamiltonienne
        p_1 = np.copy(p_dep - (eps_step/2)*calcul_dE(q_dep[:k], q_dep[k]))
        q_1 = np.copy(q_dep + eps_step*p_1)  # cf papier
        p_2 = np.copy(p_1 - (eps_step/2) *
                      calcul_dE(q_1[:k], q_1[k]))  # cf papier
        return [q_1, p_2]
    n_ite = n_ite_apres_offset + offset
    for i in range(n_ite-1):  # on va itérer n_ite fois une succession de L (petites) steps dans un sens ou l'autre avec acceptation ou non selon un processus MH
        q_dep = np.copy(li_q[len(li_q) - 1])  # on part d'ici
        q_previous = np.copy(li_q[len(li_q) - 1])  # sauvegarde là d'où on part
        p_dep = np.copy(li_p[len(li_p) - 1])  # on part d'ici
        p_previous = np.copy(li_p[len(li_p) - 1])  # sauvegarde d'où on part
        nu = rd.random()  # Choix du sens dans lequel on ira pour la prochaine grande itération
        if nu > 0.5:
            sens = 1  # direct
        else:
            sens = -1  # à reculon dans le temps
        for l in range(L):
            # on fait une petite étape dans le sens qui convient (donné par sens*eps)
            q_pass, p_pass = np.copy(step(q_dep, p_dep, sens*eps))
            q_dep = np.copy(q_pass)  # sauvegarder le résultat
            p_dep = np.copy(p_pass)  # idem
        H_new = calcul_H(q_dep, p_dep)  # calculer le nouveau hamiltonien
        # calculer l'ancien hamiltonien
        H_previous = calcul_H(q_previous, p_previous)
        # calculer le facteur MH (cf papier)
        facteur = min(1, np.exp(H_previous - H_new))
        mu = rd.random()  # variable uniforme dans[0,1[
        print(i)
        if mu < facteur:  # alors on accepte
            q_pris = np.copy(q_dep)
            p_pris = np.copy(p_dep)
        else:  # alors on refuse
            q_pris = np.copy(q_previous)
            p_pris = np.copy(p_previous)
            if i > offset:
                refus = refus + 1
          # stocker la nouvelle valeur
        li_q.append(q_pris)
        # explorer une région avec un nouveau H (Gibbs Sampling)
        li_p.append(np.random.multivariate_normal(np.zeros(k+1), np.eye(k+1)))
        # li_p.append(p_previous)  # FAUX!!!
        #print([H_new, H_previous])
        # print(q_pris)
        # print(i)
    return [li_q[offset:], refus/n_ite]  # la seule variable d'intérêt est q


def compare_MH(mat, ly, param_init, ecart_type_init, n_ite_apres_offset, offset):
    T = n_ite_apres_offset + offset
    n = len(mat)  # taille de l'échantillon d'apprentissage
    k = len(param_init)  # taille du vecteur de paramètres
    refus_param = np.zeros(k)
    refus_sigma = 0  # compter les refus au fur et à mesure
    # li_param_poi va contenir toutes les valeurs de param qui seront généréres par Gibbs within MH
    li_param = [param_init]
    li_ecart_type = [ecart_type_init]
    mat_prod_lourd = np.linalg.inv(np.dot(mat.T, mat))

    def simul_sigma(condit):
        return np.random.wald(condit, condit**3)

    def log_q_sigma(x, condit):
        return np.log(f_wald(x, condit, condit**3))

    for t in range(T-1):
        # dernier paramètre généré
        param = np.array(li_param[t])
        ecart_type = li_ecart_type[t]

        def log_proba(beta_f, sigma_f):
            log_pi = 0
            for i in range(n):
                # lambda de la loi de poisson pour le beta_f donné
                param_poi = np.exp(np.dot(mat[i, :], beta_f))
                if ly[i] > 30:
                    log_fact = ly[i]*np.log(ly[i])-ly[i] + \
                        0.5 * \
                        np.log(ly[i]) + np.log(np.sqrt(2*np.pi)
                                               )  # stirling: calculer factorielle de ly[i]
                else:
                    # si faible coût: méthode exacte
                    log_fact = np.log(float(np.math.factorial(ly[i])))
                # log de la proba de ly[i] selon la loi de Poisson de paramètre lambda = param_poi
                ln_proba_poisson = -param_poi - \
                    log_fact + ly[i]*np.log(param_poi)
                # le produit des vraisemblances de ly[i] devient un somme avec le log
                log_pi = log_pi + ln_proba_poisson
            log_pi = log_pi + np.log(multivariate_normal.pdf(beta_f, np.zeros(
                len(beta_f)), sigma_f**2*mat_prod_lourd)) - 1.5*np.log(sigma_f)  # tenir compte des à prioris: G-prior utilisé...
            return log_pi

        def simul_q_saut(condit):
            # réaliser un saut gaussien centré en condit d'écart-type unitaire: symétrique
            return rd.gauss(condit, 0.5)

        # nouveau paramètre pour les deux régressions
        new_param = np.zeros(k)
        # précédent paramètre , qui tiendra compte des sauts MH progressivement acceptés
        precedent = np.copy(param)
        # précédent paramètre qui tindra compte des sauts MH progressivement acceptés et des supposition de sauts à tester
        proposition = np.copy(param)
        for i in range(k):
            # le i-ème coefficient fait un saut normal d'écart-type 1
            param_prop = simul_q_saut(precedent[i])
            # on tient compte du saut dans proposition
            proposition[i] = param_prop
            # calcul du facteur dans MH avec loi de saut symétrique
            facteur = min(log_proba(proposition, ecart_type) -
                          log_proba(precedent, ecart_type), 0)

            value = np.log(rd.random())
            if value < facteur:  # accepter la nouvelle valeur
                new_param[i] = param_prop
                precedent[i] = param_prop
            else:  # refuser la nouvelle valeur
                new_param[i] = precedent[i]
                proposition[i] = precedent[i]
                if t > offset:
                    refus_param[i] = refus_param[i] + 1
        li_param.append(new_param)
        ancient = ecart_type
        proposition = simul_sigma(ancient)
        rapport = log_proba(new_param, proposition) - log_q_sigma(proposition, ancient) - \
            log_proba(new_param, ancient) + log_q_sigma(ancient, proposition)
        facteur = min(1, rapport)
        value = np.log(rd.random())
        if value < facteur:
            new_ecart = proposition
        else:
            new_ecart = ancient
            if t > offset:
                refus_sigma = refus_sigma + 1
        li_ecart_type.append(new_ecart)
    return [li_param[offset:], refus_param/offset]


def tuning_poisson(X, ly, eps_dep, L_dep, q_dep, n_offset, n_ite_apres_offset=-1, L_max=5000, moy_obj=0.01, var_obj=0.01):
    if n_ite_apres_offset == -1:
        n_ite = floor(0.2*n_offset)
    else:
        n_ite = n_ite_apres_offset
    eps = eps_dep
    L = L_dep
    seq, refus = hamilt_poisson_non_corr(X, ly,
                                         eps, L, q_dep, n_ite, n_offset)
    if refus > 0.35:
        while refus > 0.01:
            print(['eps trop', eps, refus])
            eps = eps/2
            seq, refus = hamilt_poisson_non_corr(X, ly,
                                                 eps, L, q_dep, n_ite, n_offset)
    else:
        while refus < 0.001:
            print(['eps pas assez', eps, refus])
            eps = eps*1.5
            seq, refus = hamilt_poisson_non_corr(X, ly,
                                                 eps, L, q_dep, n_ite, n_offset)
    print(['eps bon', eps, refus])
    n_autocor = n_ite
    k = len(seq[0])
    li_ordre = [[seq[i][indice]
                 for i in range(len(seq))] for indice in range(k)]
    li_autocor = [[autocor(li_ordre[indice], i)
                   for i in range(20, n_autocor)] for indice in range(k)]
    autocor_moy = 0
    autocor_var = 0
    for indice in range(k):
        autocor_moy = autocor_moy + sum(li_autocor[indice])/n_autocor
        autocor_var = autocor_var + f_var(li_autocor[indice])
    autocor_moy = autocor_moy/4
    autocor_var = autocor_var/4
    n_try = 0
    while (abs(autocor_moy) > moy_obj or abs(autocor_var) > var_obj) and L < L_max:
        n_offset = max(floor(n_offset/2), 500)
        print(['L pas assez', L, autocor_moy, autocor_var])
        L = L*3
        seq, refus = hamilt_poisson_non_corr(X, ly,
                                             eps, L, q_dep, n_ite, n_offset)
        li_ordre = [[seq[i][indice]
                     for i in range(len(seq))] for indice in range(k)]
        li_autocor = [[autocor(li_ordre[indice], i)
                       for i in range(20, n_autocor)] for indice in range(k)]
        autocor_moy = 0
        autocor_var = 0
        for indice in range(k):
            autocor_moy = autocor_moy + sum(li_autocor[indice])/n_autocor
            autocor_var = autocor_var + f_var(li_autocor[indice])
        n_try = n_try + 1
    if abs(autocor_moy) < moy_obj and abs(autocor_var) < var_obj:
        print(['L bon', L, abs(autocor_moy), abs(autocor_var)])
        return [eps, L, seq]
    else:
        print([eps, L, abs(autocor_moy), abs(autocor_var)])
        return ('erreur')


'''
X, ly = mel_poi_geo([1, 2, 3], 1, 200)
#eps, L, li_q = tuning_poisson(X, ly, 0.01, 100, [0, 0, 0, 1], 100, moy_obj=0.05, var_obj=0.05, n_ite_apres_offset=500)
li_q, refus = hamilt_poisson_non_corr(
    X, ly, 0.0225, 25, [0, 0, 0, 1], 100, 10)
#li_q, refus = compare_MH(X, ly, [0, 0, 0], 1, 1000, 5000)
print(refus)
k = len(X[0, :])
li_beta = []
li_sigma = []
for elem in li_q:
    li_beta.append(elem[:k])
    # li_sigma.append(elem[k])
beta = np.zeros(3)
for elem in li_beta:
    beta = beta + elem
beta = beta/len(li_q)
print(beta)  # moyenne des paramètres
beta_plot = []
for i in range(3):
    plot = []
    for elem in li_beta:
        plot.append(elem[i])
    beta_plot.append(plot)
plt.plot(beta_plot[1])  # ploter la composante voulue.
figure, axis = plt.subplots(nrows=2, ncols=2)
x_vect = range(len(beta_plot[0]))
axis[0, 0].scatter(x_vect, beta_plot[0], s=1)
axis[1, 0].scatter(x_vect, beta_plot[1], s=1)
axis[0, 1].scatter(x_vect, beta_plot[2], s=1)
axis[0, 0].set_title("Marche composante 0")
axis[1, 0].set_title("Marche composante 1")
axis[0, 1].set_title("Marche composante 2")
plt.show()
plt.close('all')

figure, axis = plt.subplots(nrows=2, ncols=2)
n_autocor = 300
corel_vect = range(n_autocor)
li_autocor = []
for i in range(3):
    autocorel = np.zeros(n_autocor)
    for j in range(n_autocor):
        autocorel[j] = autocor(beta_plot[i], j)
    li_autocor.append(np.copy(autocorel))
axis[0, 0].scatter(corel_vect, li_autocor[0], s=1)
axis[1, 0].scatter(corel_vect, li_autocor[1], s=1)
axis[0, 1].scatter(corel_vect, li_autocor[2], s=1)
axis[0, 0].set_title("Autocorrélation composante 0")
axis[1, 0].set_title("Autocorrélation composante 1")
axis[0, 1].set_title("Autocorrélation composante 2")
plt.show()
plt.close('all')
'''

# On va faire plus simple et simuler une loi normale centrée réduite avec cet algorithme)


def norm_simple_hamilt(eps, L, q_init, n_ite_apres_offset, offset):
    li_q = [q_init]  # début de la liste des variables de positions
    refus = 0  # compte les refus
    n_ite = n_ite_apres_offset + offset

    def energ(q_f):
        return q_f**2/2  # fonction énergie

    def Denerg(q_f):  # fonction dérivée de l'énergie
        return q_f

    def iter(q_f, p_f, eps_arg):  # un leapfrog
        #eps_sens = np.random.normal(eps_arg, abs(eps_arg)/100)
        eps_sens = eps_arg
        q_1 = q_f + eps_sens*p_f
        p_2 = p_f - eps_sens*Denerg(q_f)
        return [q_1, p_2]

    for ite in range(n_ite - 1):
        p_dep = np.random.normal(0, 1)  # générer nouveau p (Gibbs)
        q_dep = li_q[ite]  # ancien q
        p_previous = p_dep  # garder en mémoire avant leapfrog
        q_previous = q_dep  # garder en mémoire avant leapfrog
        H_previous = energ(q_previous) + p_previous**2 / \
            2  # Hamiltonien avant leapfrog
        nu = rd.random()
        if nu < 0.5:
            sens = 1  # sens du temps vers le futur (réversibilité)
        else:
            sens = -1  # sens du temps vers le passé (réversibilité)
        #p_1 = p_f - 0.5*eps_sens*Denerg(q_f)
        eps_aleat = abs(np.random.normal(eps, eps/10))
        p_dep = p_dep - eps_aleat*sens*0.5*Denerg(q_dep)
        L_aleat = max(floor(L/2), np.random.poisson(L))
        for l in range(L_aleat):
            q_pass, p_pass = iter(q_dep, p_dep, sens*eps_aleat)
            q_dep = q_pass
            p_dep = p_pass
        p_dep = p_dep - eps_aleat*sens*0.5*Denerg(q_dep)
        #p_dep = -p_dep
        H_new = energ(q_dep) + p_dep**2/2
        mu = rd.random()
        fact = min(1, np.exp(H_previous - H_new))
        if mu < fact:
            li_q.append(q_dep)
        else:
            if ite > offset:
                refus = refus + 1
            li_q.append(q_previous)
    return [li_q[offset:], refus/(n_ite-offset)]


'''li_q, refus = norm_simple_hamilt(0.05, 25, 2, 100, 10)
print(refus)
plt.plot(li_q)
plt.show()
plt.close('all')


def f_ker(lis, x, h):
    n = len(lis)
    res = 1/(n*h)*sum([norm.pdf((x-lis[i])/h, 0, 1) for i in range(n)])
    return res


def pi_spé(x):
    return 1/np.sqrt(2*np.pi*1)*np.exp(-0.5*(x-0)**2/1)
'''

'''
seq, refus = metropolis(10000, q, pi_spé, simul_q, 0)
print(refus/10000)
lx = np.linspace(-4, 4, 200)
li_test = []
li_double = []
for i in lx:
    #li_test.append(f_ker(li_q, i, 0.3))
    li_double.append(f_ker(seq[5000:], i, 0.3))
li_ok = norm.pdf(lx, 0, 1)
plt.plot(lx, li_ok, label='Vraie loi')
#plt.plot(lx, li_test, label='Hamiltonien')
plt.plot(lx, li_double, label='MH')
plt.legend(loc='best')
plt.show()
plt.close('all')
n_autocor = 1000
li_autocor = []
for i in range(n_autocor):
    li_autocor.append(autocor(seq[5000:], i))
plt.plot(range(n_autocor), li_autocor)
plt.show()
'''


def tuning_simple(eps_dep, L_dep, q_dep, n_offset, L_max=5000, moy_obj=0.01, var_obj=0.01):
    eps = eps_dep
    L = L_dep
    seq, refus = norm_simple_hamilt(
        eps, L, q_dep, floor(1.2*n_offset), n_offset)
    if refus > 0.35:
        while refus > 0.01:
            print(['eps trop', eps, refus])
            eps = eps/2
            seq, refus = norm_simple_hamilt(
                eps, L, q_dep, floor(1.2*n_offset), n_offset)
    else:
        while refus < 0.001:
            print(['eps pas assez', eps, refus])
            eps = eps*1.5
            seq, refus = norm_simple_hamilt(
                eps, L, q_dep, floor(1.2*n_offset), n_offset)
    print(['eps bon', eps, refus])
    n_autocor = floor(0.2*n_offset/2)
    li_autocor = [autocor(seq, i) for i in range(20, n_autocor)]
    autocor_moy = sum(li_autocor)/n_autocor
    autocor_var = f_var(li_autocor)
    n_try = 0
    while (abs(autocor_moy) > moy_obj or abs(autocor_var) > var_obj) and L < L_max:
        L = L*5
        n_offset = max(floor(n_offset/2), 500)
        print(['L pas assez', L, autocor_moy, autocor_var])
        seq, refus = norm_simple_hamilt(
            eps, L, q_dep, floor(1.2*n_offset), n_offset)
        li_autocor = [autocor(seq, i) for i in range(20, n_autocor)]
        autocor_moy = sum(li_autocor)/n_autocor
        autocor_var = f_var(li_autocor)
        n_try = n_try + 1
    if abs(autocor_moy) < moy_obj and abs(autocor_var) < var_obj:
        print(['L bon', L, abs(autocor_moy), abs(autocor_var)])
        return [eps, L]
    else:
        print([eps, L, abs(autocor_moy), abs(autocor_var)])
        return ('erreur')

# print(tuning_simple(1, 100, 0, 5000, L_max=5000))


def hamilt_sepa_gp(X, ly, eps, L, q_init, proba_init, n_ite_apres_offset, offset):
    n = len(ly)  # nb données
    k = len(X[0, :])  # taille du beta
    # liste des variables positionnelles de la forme [beta sigma] dans une même liste
    li_q = [np.array(q_init)]
    li_p = [np.random.multivariate_normal(np.zeros(k+1), np.eye(k+1))]
    li_proba = [proba_init]
    mat_prod_lourd = np.linalg.inv(np.dot(X.T, X))  # (X.T@X)^(-1)
    refus = 0
    li_log_fact = []
    for i in range(n):
        if ly[i] > 30:  # alors l'approximation de stirling devient bonne
            log_fact = ly[i]*np.log(ly[i])-ly[i] + 0.5 * \
                np.log(ly[i]) + np.log(np.sqrt(2*np.pi))  # stirling
        else:  # le calcul direct est faisable car la valeur est petite
            log_fact = np.log(float(np.math.factorial(ly[i])))
        li_log_fact.append(log_fact)
    li_fact = []
    for i in range(n):
        if ly[i] < 20:
            fact = np.math.factorial(ly[i])
        else:
            fact = (ly[i]/np.exp(1))**ly[i]*np.sqrt(2*ly[i]*np.pi)
        li_fact.append(fact)

    def log_sigma(ecart):
        return -1.5*ecart

    def log_beta_sigma(par, ecart):
        # ecart type de la loi de beta sachant sigma et X
        loga_proba = np.log(multivariate_normal.pdf(par, np.zeros(
            len(par)), ecart**2*mat_prod_lourd))
        # proba =  ecart * multivariate_normal.pdf(par, np.zeros(len(par)), ecart**2*mat_prod_lourd)  # Plutôt!!!
        return loga_proba

    # par est le vecteur des coefficients pour les deux régressions
    def log_proba_goal(par, sigma, proba_f):
        ln_proba = 0
        for i in range(n):
            # calcul de l'espérance pour les i-èmes variables explicatives de la régression poisson
            esp = np.exp(np.dot(X[i], par))
            # calcul de la probabilité p de réussite pour poisson (esp_géo peut valoir de 0 à +inf), cf convention choisie pour la loi géo
            # log de la probabilité p du succès dans la régression géométrique
            ln_p_geom = -np.log(esp + 1)
            # log de la probabilité q de l'échec dans la régression géométrique
            ln_q_geom = np.log(esp) - np.log(esp + 1)
            # calcul de la vraisemblance de la donnée y_i pour le modèle poisson
            ln_proba_poisson = -esp - li_log_fact[i] + ly[i]*np.log(esp)
            # calcul de la vraisemblance de la donnée y_i pour le modèle géométrique
            ln_proba_geo = ln_p_geom + ly[i]*ln_q_geom
            rapport_1 = np.exp(- np.log(proba_f) - ln_proba_poisson +
                               np.log(1-proba_f) + ln_proba_geo)  # rapport b/a où a = p*\pi(y|poisson,\theta) et b = (1-p)*\pi(y|géométrique,\theta)
            rapport_2 = np.exp(np.log(proba_f) + ln_proba_poisson -
                               np.log(1-proba_f) - ln_proba_geo)  # avec les mêmes conventions: rapport a/b
            # il est impossible que rapport_1 et rapport_2 soient ensemble écrasés à +inf. Et celui qui est écrasé à 0 donnera un résultat global correct (cf D.L de ln(1+x); x<<1)
            supplement_1 = np.log(
                proba_f) + ln_proba_poisson + np.log(1 + rapport_1)  # avec les notations précédentes: calcul de ln(a + b) = ln(a) + ln(1 + b/a)
            supplement_2 = np.log(1-proba_f) + ln_proba_geo + \
                np.log(
                    1 + rapport_2)  # avec les notations précédentes: calcul de ln(a + b) = ln(b) + ln(1 + a/b)
            if rapport_1 <= 1:  # i.e. si rapport_1 pas écrasé à l'infini
                supplement = supplement_1
            else:  # alors forcément rapport_2 pas écrasé à l'infini
                supplement = supplement_2
                if supplement_2 > 0:
                    print('aïe')
                    print([ln_proba_poisson, ln_proba_geo,
                           supplement, esp, ly[i], log_fact])
            ln_proba = ln_proba + supplement
        # probabilité des données sachant la théorie
        # on trouve log(proba(\beta,\sigma|p,X))
        ln_proba = ln_proba + log_beta_sigma(par, sigma) + log_sigma(sigma)
        return ln_proba

    # renvoit le gradient de E au point q = [beta_f sigma_f]
    def calcul_dE(beta_f, sigma_f, proba_f):
        # futur vecteur gradient (vecteur des dérivées partielles)
        dE = np.zeros(k + 1)
        # cas où on prend la dérivée selon un beta[r] (r<k) et non selon sigma = q[k]
        for r in range(k):
            e_r = np.zeros(k)
            e_r[r] = 1  # vecteur qui vaut zéro partout sauf en r où il vaut 1
            dE[r] = (1/sigma_f**2)*np.dot(X @ beta_f, X @ e_r)
            for i in range(n):
                lamb = np.exp(np.dot(X[i, :], beta_f))
                numerateur = X[i, r]*(lamb - ly[i])*(proba_f*np.exp(-lamb) /
                                                     li_fact[i] + (1-proba_f)/(1+lamb)**(2 + ly[i]))
                denominateur = proba_f * \
                    np.exp(-lamb)/li_fact[i] + \
                    (1 - proba_f)/(1 + lamb)**(1+ly[i])
                dE[r] = dE[r] + numerateur/denominateur
        dE[k] = k/sigma_f - np.dot(X @ beta_f, X @ beta_f) / \
            sigma_f**3 + 3/(2*sigma_f)  # idem
        return dE

    def calcul_H(q_f, p_f, proba_f):  # renvoit le hamiltonien en (q_f,p_f)
        E = - log_proba_goal(q_f[:k], q_f[k], proba_f)  # énergie potentielle
        K = sum([0.5*p_f[i]**2 for i in range(k+1)])  # énergie cinétique
        return E + K  # hamiltonien

    # réalise une étape d'updating en partant de (q_dep,p_dep) (il y en a L dans une "grande étape")
    def step(q_dep, p_dep, proba_f, eps_step):
        # cf papier: dynamique hamiltonienne
        p_1 = np.copy(p_dep - (eps_step/2) *
                      calcul_dE(q_dep[:k], q_dep[k], proba_f))
        q_1 = np.copy(q_dep + eps_step*p_1)  # cf papier
        p_2 = np.copy(p_1 - (eps_step/2) *
                      calcul_dE(q_1[:k], q_1[k], proba_f))  # cf papier
        return [q_1, p_2]

    n_ite = n_ite_apres_offset + offset
    for ite in range(n_ite-1):  # on va itérer n_ite fois une succession de L (petites) steps dans un sens ou l'autre avec acceptation ou non selon un processus MH
        q_dep = np.copy(li_q[len(li_q) - 1])  # on part d'ici
        q_previous = np.copy(li_q[len(li_q) - 1])  # sauvegarde là d'où on part
        p_dep = np.copy(li_p[len(li_p) - 1])  # on part d'ici
        p_previous = np.copy(li_p[len(li_p) - 1])  # sauvegarde d'où on part
        nu = rd.random()  # Choix du sens dans lequel on ira pour la prochaine grande itération
        proba = li_proba[ite]
        if nu > 0.5:
            sens = 1  # direct
        else:
            sens = -1  # à reculon dans le temps
        for l in range(L):
            # on fait une petite étape dans le sens qui convient (donné par sens*eps)
            q_pass, p_pass = np.copy(step(q_dep, p_dep, proba, sens*eps))
            q_dep = np.copy(q_pass)  # sauvegarder le résultat
            p_dep = np.copy(p_pass)  # idem
        # calculer le nouveau hamiltonien
        H_new = calcul_H(q_dep, p_dep, proba)
        # calculer l'ancien hamiltonien
        H_previous = calcul_H(q_previous, p_previous, proba)
        # calculer le facteur MH (cf papier)
        facteur = min(1, np.exp(H_previous - H_new))
        mu = rd.random()  # variable uniforme dans[0,1[
        if mu < facteur:  # alors on accepte
            q_pris = np.copy(q_dep)
            p_pris = np.copy(p_dep)
        else:  # alors on refuse
            q_pris = np.copy(q_previous)
            p_pris = np.copy(p_previous)
            print('non')
            if ite > offset:
                refus = refus + 1
          # stocker la nouvelle valeur
        li_q.append(q_pris)
        # explorer une région avec un nouveau H (Gibbs Sampling)
        li_p.append(np.random.multivariate_normal(np.zeros(k+1), np.eye(k+1)))
        elem_z = []
        new_param = np.copy(q_pris[:k])
        for i in range(n):  # etape de Gibbs sampling pour z
            test = np.log(rd.random())  # log d'une v.a. uniforme entre 0 et 1
            # valeur de l'espérance pour cette valeur de variables explicatives
            mu = np.exp(np.dot(X[i], new_param))
            # calculer le paramètre p de la régression géométrique pour cette valeur de variables explicatives
            ln_p_geo = -np.log(mu + 1)
            ln_q_geo = np.log(mu) - np.log(mu + 1)
            log_proba_poi = -mu - li_log_fact[i] + \
                ly[i]*np.log(mu)  # likelyhood modèle poisson
            log_proba_geo = ln_p_geo + ly[i]*ln_q_geo  # likelyhood modèle geo
            rap_1 = np.exp(- np.log(proba) - log_proba_poi +
                           np.log(1-proba) + log_proba_geo)
            rap_2 = np.exp(np.log(proba) + log_proba_poi -
                           np.log(1-proba) - log_proba_geo)
            # avec les notations précédentes: calcul de ln(a + b) = ln(a) + ln(1 + b/a)
            sum_1 = np.log(proba) + log_proba_poi + np.log(1 + rap_1)
            sum_2 = np.log(1-proba) + log_proba_geo + np.log(1 + rap_2)
            if sum_1 <= 0:
                sumgood = sum_1
            else:
                sumgood = sum_2
                if sumgood > 0:
                    print('ouille')
            value = np.log(proba) + log_proba_poi - sumgood
            if test < value:
                elem_z.append(0)
            else:
                elem_z.append(1)
        # nombre de y_i attribués à la seconde régression (géométrique)
        n_1 = sum(elem_z)
        # nombre de y_i attribués à la première régression (poisson)
        n_0 = n-n_1
        # distribution conjuguée: générer selon \pi(p|z_t,\theta_{t-1})
        new_proba = np.random.beta(0.5 + n_0, 0.5 + n_1)
        li_proba.append(new_proba)  # enregistrer la nouvelle valeur de p
        print(ite)
    return [li_q[offset:], li_proba[offset:], refus/n_ite_apres_offset]


'''
X, ly = mel_poi_geo([1, 2, 3], 1, 50)
li_q, li_proba, refus = hamilt_sepa_gp(
    X, ly, 0.015, 100, [0, 0, 0, 1], 0.5, 100, 200)
print(refus)
k = len(X[0, :])
li_beta = []
li_sigma = []
for elem in li_q:
    li_beta.append(elem[:k])
beta_p = np.zeros(4)
nume = 0
for elem in li_beta:
    beta_p[:3] = beta_p[:3] + elem
    beta_p[3] = beta_p[3] + li_proba[nume]
    nume = nume + 1
beta_p = beta_p/len(li_q)
print(['[a_0,a_1,a_2,p] =', beta_p])  # moyenne des paramètres
beta_plot = []
for i in range(3):
    plot = []
    for elem in li_beta:
        plot.append(elem[i])
    beta_plot.append(plot)
plot = []
for elem in li_proba:
    plot.append(elem)
beta_plot.append(plot)
figure, axis = plt.subplots(nrows=2, ncols=2)
x_vect = range(len(beta_plot[0]))
axis[0, 0].scatter(x_vect, beta_plot[0], s=1)
axis[1, 0].scatter(x_vect, beta_plot[1], s=1)
axis[0, 1].scatter(x_vect, beta_plot[2], s=1)
axis[1, 1].scatter(x_vect, beta_plot[3], s=1)
axis[0, 0].set_title("Marche composante 0")
axis[1, 0].set_title("Marche composante 1")
axis[0, 1].set_title("Marche composante 2")
axis[1, 1].set_title("Marche p")
plt.show()
plt.close('all')

figure, axis = plt.subplots(nrows=2, ncols=2)
n_autocor = 70
corel_vect = range(n_autocor)
li_autocor = []
for i in range(4):
    autocorel = np.zeros(n_autocor)
    for j in range(n_autocor):
        autocorel[j] = autocor(beta_plot[i], j)
    li_autocor.append(np.copy(autocorel))
axis[0, 0].scatter(corel_vect, li_autocor[0], s=1)
axis[1, 0].scatter(corel_vect, li_autocor[1], s=1)
axis[0, 1].scatter(corel_vect, li_autocor[2], s=1)
axis[1, 1].scatter(corel_vect, li_autocor[3], s=1)
axis[0, 0].set_title("Autocorrélation composante 0")
axis[1, 0].set_title("Autocorrélation composante 1")
axis[0, 1].set_title("Autocorrélation composante 2")
axis[1, 1].set_title("Autocorrélation p")
plt.show()
plt.close('all')
df = pd.DataFrame(np.array(li_beta), columns=[
    'compos_0', 'compos_1', 'compos_2'])
df = df.assign(p=li_proba)
pd.plotting.scatter_matrix(df)
plt.show()
plt.close('all')
'''

'''données réelles'''
array = pd.read_csv('data.CSV')
expliqué = array['Y']
expliqué = expliqué.to_numpy()
explicative = array.loc[:, ["X.I", "X.X1"]]
# c'est quand même une matrice! même à une seule colonne: on a viré les "1" inutiles
vraie_explic = explicative.loc[:, ["X.X1"]].to_numpy()
#explicative = explicative.to_numpy()[:100, :]

faithful = pd.read_csv('faithful.CSV')
explicative_faithful = faithful.loc[:, ['eruptions']].to_numpy()
expliqué_faithful = faithful['waiting'].to_numpy()//5  # trop gros sinon

debut = 0
fin = 200
taille = len(expliqué)
ly = expliqué[debut:fin]  # expliqué
attila = np.array([[1 for i in range(taille)]]).T
print((np.shape(attila), np.shape(vraie_explic)))
X = np.block([[attila, vraie_explic]])[debut:fin, :]

'''
li_q, li_proba, refus = hamilt_sepa_gp(
    X, ly, 0.003, 100, [0, 0, 1], 0.5, 100, 200)
print(refus)
k = len(X[0, :])
li_beta = []
li_sigma = []
for elem in li_q:
    li_beta.append(elem[:k])
beta_p = np.zeros(3)
nume = 0
for elem in li_beta:
    beta_p[:2] = beta_p[:2] + elem
    beta_p[2] = beta_p[2] + li_proba[nume]
    nume = nume + 1
beta_p = beta_p/len(li_q)
print(['[a_0,a_1,p] =', beta_p])  # moyenne des paramètres
beta_plot = []
for i in range(2):
    plot = []
    for elem in li_beta:
        plot.append(elem[i])
    beta_plot.append(plot)
plot = []
for elem in li_proba:
    plot.append(elem)
beta_plot.append(plot)
figure, axis = plt.subplots(nrows=1, ncols=3)
x_vect = range(len(beta_plot[0]))
axis[0].scatter(x_vect, beta_plot[0], s=1)
axis[1].scatter(x_vect, beta_plot[1], s=1)
axis[2].scatter(x_vect, beta_plot[2], s=1)
axis[0].set_title("Marche composante 0")
axis[1].set_title("Marche composante 1")
axis[2].set_title("Marche p")
plt.show()
plt.close('all')

figure, axis = plt.subplots(nrows=1, ncols=3)
n_autocor = 70
corel_vect = range(n_autocor)
li_autocor = []
for i in range(3):
    autocorel = np.zeros(n_autocor)
    for j in range(n_autocor):
        autocorel[j] = autocor(beta_plot[i], j)
    li_autocor.append(np.copy(autocorel))
axis[0].scatter(corel_vect, li_autocor[0], s=1)
axis[1].scatter(corel_vect, li_autocor[1], s=1)
axis[2].scatter(corel_vect, li_autocor[2], s=1)
axis[0].set_title("Autocorrélation composante 0")
axis[1].set_title("Autocorrélation composante 1")
axis[2].set_title("Autocorrélation p")

plt.show()
plt.close('all')
df = pd.DataFrame(np.array(li_beta), columns=[
    'compos_0', 'compos_1'])
df = df.assign(p=li_proba)
pd.plotting.scatter_matrix(df)
plt.show()
plt.close('all')
'''
#['[a_0,a_1,p] =', array([2.1138154 , 0.1437486 , 0.99411225])]


def testing_reel(echantillon):
    mat = X
    # def sepa_gp_sigma(T, mat, ly, param_init, p_init, ecart_type_init):
    li_param, li_p, li_z, li_ecart_type, refus_param, refus_sigma = sepa_gp_sigma(4000, mat, ly, np.array(
        [0, 0]), 0.5, 1)  # chercher à séparer les paramètres
    # éliminer les 1000 premières valeurs et prendre l'espérance de p
    p_guess = sum(li_p[3000:])/1000
    sigma_guess = sum(li_ecart_type[3000:])/1000
    param_guess = [0, 0]
    for i in range(1000):
        for j in range(2):
            # éliminer les 1000 premières valeurs et prendre l'espérance de param_0
            param_guess[j] = param_guess[j] + li_param[i+2999][j]
    param_guess = np.array(param_guess)/1000
    # afficher les paramètres "devinés"
    print([param_guess, p_guess, sigma_guess, 'guess'])
    print([refus_param/4000, refus_sigma/4000, 'refus'])
    # param_0_moy.append(param_guess[0])
    # param_1_moy.append(param_guess[1])
    # param_2_moy.append(param_guess[2])
    # p_moy.append(p_guess)
    li_p = np.array(li_p)
    li_0 = np.array([li_param[i][0] for i in range(3000, 4000)])
    li_1 = np.array([li_param[i][1] for i in range(3000, 4000)])
    figure, axis = plt.subplots(nrows=2, ncols=2)
    x_vect = np.array(range(len(li_0)))
    axis[0, 0].scatter(x_vect, li_0, s=1)
    axis[1, 0].scatter(x_vect, li_1, s=1)
    axis[0, 1].scatter(x_vect, li_p[3001:], s=1)
    axis[0, 0].set_title("Composante 0")
    axis[1, 0].set_title("Composante 1")
    axis[0, 1].set_title("proba")
    number = str(echantillon)
    plt.show()
    plt.close('all')
    new_li_pasarray = []
    for i in range(len(li_param)):
        elem = []
        for j in range(len(li_param[0])):
            elem.append(li_param[i][j])
        elem.append(li_p[i])
        elem.append(li_ecart_type[i])
        new_li_pasarray.append(elem)
    matrix_incomplete = np.array(new_li_pasarray)
    df = pd.DataFrame(matrix_incomplete, columns=[
        'compos_0', 'compos_1', 'proba_poi', 'sigma'])
    df = df.assign(p=li_p)
    pd.plotting.scatter_matrix(df)
    # plt.savefig("C:\\Users\\33771\\OneDrive\\Documents\\Projet_recherche\\save_files\\scatter{}.jpg".format(number))
    plt.show()
    plt.close('all')

    n_auto_cor = 900
    series = [[], [], [], []]
    for i in range(1000, len(li_param)):  # offset
        for j in range(len(li_param[0])):
            series[j].append(li_param[i][j])
        series[2].append(li_p[i])
        series[3].append(li_ecart_type[i])
    series = np.array(series)
    auto_cors = np.zeros((n_auto_cor, 7))
    for j in range(4):
        for i in range(n_auto_cor):
            auto_cors[i, j] = autocor(series[j], i)
    figure, axis = plt.subplots(nrows=2, ncols=2)
    x_vect = range(n_auto_cor)
    axis[0, 0].scatter(x_vect, auto_cors[:, 0], s=1)
    axis[1, 0].scatter(x_vect, auto_cors[:, 1], s=1)
    axis[0, 1].scatter(x_vect, auto_cors[:, 2], s=1)
    #axis[1, 1].scatter(x_vect, auto_cors[:, 3], s=1)
    #axis[0, 1].scatter(x_vect, auto_cors[:, 4], s=1)
    #axis[1, 1].scatter(x_vect, auto_cors[:, 5], s=1)
    #axis[2, 1].scatter(x_vect, auto_cors[:, 6], s=1)

    axis[0, 0].set_title("Composante 0")
    axis[1, 0].set_title("Composante 1")
    axis[0, 1].set_title("proba")
    #axis[1, 1].set_title("proba")
    #axis[0, 1].set_title("Ecart type loi a priori")
    #axis[1, 1].set_title("Probabilité Poisson")
    #axis[2, 1].set_title("Ecart type loi a priori")
    # plt.savefig("C:\\Users\\33771\\OneDrive\\Documents\\Projet_recherche\\save_files\\autocorr{}.jpg".format(number))
    plt.show()
    plt.close('all')


testing_reel('str')
