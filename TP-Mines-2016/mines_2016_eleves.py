#!/usr/bin/env python
# -*- coding: utf-8 -*-

#%%
# Imports de modules externes
import numpy as np
import random as rd


# %%
# Question 1
# faire un tableau d'états

def tri(L:[int])->None:
    """Tri par insertion d'une liste L"""
    n = len(L)
    for i in range(1, n):
        j = i
        x = L[i]
        while 0 < j and x < L[j-1]:
            L[j] = L[j-1]
            j = j-1            
        L[j] = x 

#Tests unitaires
for L in [[], [5, 2, 3, 1, 4], list(range(6)), list(range(5, -1, -1))]:
    tri(L)
    assert L == sorted(L)
print("Test unitaires réussis")

#%%
#Question 9, code à compléter
#Documentation de sqlite3 sur https://docs.python.org/3.8/library/sqlite3.html

import sqlite3                          #module de manipulations de bases sqlite en Python
con = sqlite3.connect('mine-2016.db')  #connexion à la base
cur = con.cursor()                     #création d'un curseur pour interroger la base
requete = '''à compléter'''
cur.execute(requete)                   #exécution d'une requête sur la base par le biais du curseur
con.commit()                           #enregistrement de la requête
deces2010 = cur.fetchall()             #récupération des résultats de la requête dans deces2010
con.close()                            #fermeture de la connexion
deces2010 = [list(t) for t in deces2010] #conversion en liste de tous les tuples résultats
print(deces2010)                       #affichage de deces2010
#trier deces2010 par nombre de décès croissant

#%% 
# Question 11
import numpy as np
from typing import Callable  #pour annoter le type function

def f(X:[float], r:float, a:float, b:float)->np.ndarray:  
    """Fonction définissant l'équation différentielle dX/dt = f(X)
    Paramètres :
        r, a, b : de type flottant, taux de contagion, guérison, mortalité
        X = (S,I,R,D) de type liste de flottants
    Valeur renvoyée : de type array
    """
    #à compléter 


def euler(f:Callable, N:int)->[[float],np.ndarray]:
    """La fonction prend en paramètre une fonction f telle
    que dX/dt = f(X) et un nombre de pas N entier positif
    Renvoie un tableau de temps t et de valeurs XX calculés
    avec le schéma numérique d'Euler pour N pas avec un temps
    max de tmax = 25
    """
    # Parametres
    tmax = 25.
    r = 1.          #taux de contagion
    a = 0.4         #taux de guérison
    b = 0.1         #taux de mortalité
    X0 = np.array([0.95, 0.05, 0., 0.])   #vecteur X=(S,I,R,D)
    
    #initialisation du schéma d'Euler
    dt = tmax/N      #pas de temps du schéma d'Euler    
    t = 0
    X = X0
    tt = [t]
    XX = [X]

    # Schéma  d’Euler à compléter
    
    return tt, np.array(XX)

#Compléter le code ci-dessous pour obtenir la représentation graphique
#de l'évolution des catégories du modèle (S;I,R,D)
import numpy as np
import matplotlib.pyplot as plt
plt.clf()  #nettoyer la figure
# nuages de points  pour un schéma d'Euler à N = 7 pas
N = 7
tt, XX = euler(f, N)
mark = ['o','s','x','*']
for k in range(4):
    plt.scatter(tt, XX[:,k],marker=mark[k])
# courbes  pour un schéma d'Euler à N = 250 pas
#à complét
plt.show()
# %%
# Q12
def f2(X:[float], Itau:float, r:float, a:float, b:float)->np.ndarray:  
    """Fonction définissant l'équation différentielle dX/dt = f(X)
    Paramètres :
        tau, r, a, b : de type flottant, taux de contagion, guérison, mortalité
        Itau est la valeur de I(t - p * dt)
        X = (S,I,R,D) de type liste de flottants
    Valeur renvoyée : de type array
    """
    #à compléter

def euler2(f:Callable, N:int)->[[float],np.ndarray]:
    # Parametres
    tmax = 25.      #temps maximal
    r = 1.          #taux de contagion
    a = 0.4         #taux de guérison
    b = 0.1         #taux de mortalité
    X0 = np.array([0.95, 0.05, 0., 0.])
   
    #initialisation du schéma d'Euler
    dt = tmax/N      #pas de temps du schéma d'Euler    
    p = 50           #nombre de pas de retard
    t = 0
    X = X0
    tt = [t]
    XX = [X]

    # Schéma d’Euler
    for i in range(N):
        t = t + dt
        #à compléter
        tt.append(t)
        XX.append(X)
    return tt, np.array(XX)

#%%

#Q14

import numpy as np

def h(t, mu, sigma):
    """Fonction de densité de la loi normale centrée réduite d'espérance mu
    et d'écart-type sigma"""
    return 1/(sigma * np.sqrt(2*np.pi))*np.exp(-((t -mu) / sigma)**2/2)


def f2(X:[float], Itau:float, r:float, a:float, b:float)->np.ndarray:  
    """Fonction définissant l'équation différentielle dX/dt = f(X)
    Paramètres :
        r, a, b : de type flottant, taux de contagion, guérison, mortalité
        Itau est la valeur de integrale(0,tau, I(t-s)*h(s))
        X = (S,I,R,D) de type liste de flottants
    Valeur renvoyée : de type array
    """
    (S, I, R, D) = X
    return np.array([-r*S*Itau, r*S*Itau - (a + b)*I, a*I, b*I])


def euler3(f:Callable, N:int)->([float],[np.ndarray]):
    # Parametres
    tmax = 25.      #temps maximal
    r = 1.          #taux de contagion
    a = 0.4         #taux de guérison
    b = 0.1         #taux de mortalité
    X0 = np.array([0.95, 0.05, 0., 0.])
 
    #initialisation du schéma d'Euler
    dt = tmax/N      #pas de temps du schéma d'Euler    
    p = 50           #nombre de pas de retard
    t = 0
    X = X0
    tt = [t]
    XX = [X]
    
    #paramètres de la fonction de densité
    mu = (p * dt) / 2  #espérance (moitié de tau = (p *dt) / 2)
    sigma = (p * dt) / 6 #écart-type (tiers  de esperance / 2)
    # Schéma d’Euler
    for i in range(N):
        t = t + dt
        #à compléter
        tt.append(t)
        XX.append(X)
    return tt, XX

#%%

#Partie III Modélisation dans des grilles

#Q15

def grille(n) :
    """Spécifier cette fonction en complétant sa docstring :
    Paramètres(type, sémantique):
    Valeur renvoyée (type, sémantique):
    """"
    M = []
    for i in range(n) :
        L=[]
        for j in range(n): L.append(0)
        M.append(L)
    return M

#%%
#Q16

#à compléter


#%%
#Q18 et Q19

def est_exposee(G, i, j):
    """Retourne un booleen indiquant si une case en ligne i et colonne j
    comporte  au moins une case infectée dans son voisinage """
    n = len(G)
    if i == 0 and j == 0:
        return (G[0][1]-1)*(G[1][1]-1)*(G[1][0]-1) == 0
    elif i == 0 and j == n-1:
        return (G[0][n-2]-1)*(G[1][n-2]-1)*(G[1][n-1]-1) == 0
    elif i == n-1 and j == 0:
        return (G[n-1][1]-1)*(G[n-2][1]-1)*(G[n-2][0]-1) == 0
    elif i == n-1 and j == n-1:
        return (G[n-1][n-2]-1)*(G[n-2][n-2]-1)*(G[n-2][n-1]-1) == 0
    elif i == 0:
        "à compléter"
    elif i == n-1:
        return (G[n-1][j-1]-1)*(G[n-2][j-1]-1)*(G[n-2][j]-1)*(G[n-2][j+1]-1)*(G[n-1][j+1]-1) == 0
    elif j == 0:
        return (G[i-1][0]-1)*(G[i-1][1]-1)*(G[i][1]-1)*(G[i+1][1]-1)*(G[i+1][0]-1) == 0
    elif j == n-1:
        return (G[i-1][n-1]-1)*(G[i-1][n-2]-1)*(G[i][n-2]-1)*(G[i+1][n-2]-1)*(G[i+1][n-1]-1) == 0
    else:
        "à compléter"


#%%

#Q20

import random as rd

def bernoulli(p):
    if rd.random() <= p:
        return 1
    return 0


#%%

#Q21


#%%
#Q22 
#Représenter un graphique similaire à la figure 3 
#(évolution de la proportion de la population atteinte en fonction de p2 pour p1 = 0.5)
#en calculant la moyenne des résultats de plusieurs simulations pour différentes valeurs de p2


#%%
#Q23 Dichotomie
