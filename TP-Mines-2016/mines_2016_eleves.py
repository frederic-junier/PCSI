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

