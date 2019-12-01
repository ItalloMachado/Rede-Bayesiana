# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 00:44:17 2019

@author: Itallo
"""

import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt  # to plot
import matplotlib as mpl
import random
from scipy import optimize       # to compare
import math 
import networkx as nx
import pandas as pd
import csv
#from pgmpy.estimators import BicScore
from pgmpy.models import BayesianModel
import time
from copy import deepcopy as deep_copy
from pgmpy.estimators import BaseEstimator
from math import log
from pgmpy.estimators import BicScore
from pgmpy.readwrite import BIFReader
def vetor_Rede(solucao,nodes):
    G_aux = BayesianModel()
    #G_aux.add_nodes_from(nodes)
    k=0
    aux=1
    for i in range(1,len(nodes)):
        for j in range(aux):
            if solucao[k] == 1:
                if nodes[i] in G_aux.nodes() and nodes[j] in G_aux.nodes() and nx.has_path(G_aux,nodes[j],nodes[i]):
                    return False
                else:
                    G_aux.add_edge(nodes[i], nodes[j])
            elif solucao[k] == 2:
                if nodes[i] in G_aux.nodes() and nodes[j] in G_aux.nodes() and nx.has_path(G_aux,nodes[i],nodes[j]):
                    return False
                else:
                    G_aux.add_edge(nodes[j], nodes[i])
            k=k+1
        aux=aux+1
    for i in nodes:
        if i not in G_aux.nodes():
            return False
    return G_aux


#def mutacao(x,fitness_aux,prob,max_v,min_v,bic_score,nodes,nao_dag):
#    print(min_v)
#    print(max_v)
#    if len(x)*prob<1:
#        for i in range(len(x)):
#            r=random.random()
#            if r<=prob:
#                val=round(random.random()*(len(x)-1))
#                valor_mut=deep_copy(x[val])
#                valor_mut_antigo=deep_copy(x[val])
#                while(valor_mut==x[val]):
#                    valor_mut=min_v+random.randint(min_v, max_v)
#                x[val]=valor_mut
#                if x not in nao_dag:
#                    G=vetor_Rede(x,nodes)
#                    if G:
#                        fitness_aux=abs(bic_score.score(G))
#                    else:
#                        nao_dag.append(x)
#                        x[val]=valor_mut_antigo
#                else:
#                    x[val]=valor_mut_antigo
#    else:
#        numero_mutacao= round(len(x)*prob)
#        print(numero_mutacao)
#        while(numero_mutacao>0):
#            numero_mutacao=numero_mutacao-1
#            val=round(random.random()*(len(x)-1))
#            valor_mut=deep_copy(x[val])
#            valor_mut_antigo=deep_copy(x[val])
#            while(valor_mut==x[val]):
#                valor_mut=min_v+random.randint(min_v, max_v)
#            x[val]=valor_mut
#            if x not in nao_dag:
#                G=vetor_Rede(x,nodes)
#                if G:
#                    fitness_aux=abs(bic_score.score(G))
#                    
#                else:
#                    nao_dag.append(x)
#                    x[val]=valor_mut_antigo
#            else:
#                x[val]=valor_mut_antigo
#
#    return x,fitness_aux

def mutacao(y,fitness_aux,prob,max_v,min_v,bic_score,nodes,nao_dag):
    for val in range(len(y)):
        r=random.random()
        if r<=prob:
            valor_mut=deep_copy(y[val])
            valor_mut_antigo=deep_copy(y[val])
            while(valor_mut==y[val]):
                valor_mut=min_v+random.randint(min_v, max_v)
            y[val]=valor_mut
            if y not in nao_dag:
                G=vetor_Rede(y,nodes)
                if G:
                    fitness_aux=abs(bic_score.score(G))
                else:
                    nao_dag.append(y)
                    y[val]=valor_mut_antigo
            else:
                y[val]=valor_mut_antigo
    return y,fitness_aux

FIGSIZE = (19, 8)  #: Figure size, in inches!
mpl.rcParams['figure.figsize'] = FIGSIZE




interval = (-10, 10)

#def f(x):
#    """ Function to minimize."""
#    return x ** 2

def clip(x):
    """ Force x to be in the interval."""
    a, b = interval
    return max(min(x, b), a)
def random_start():
    """ Random point in the interval."""
    a, b = interval
    print('rand')
    print(rn.random_sample())
    return a + (b - a) * rn.random_sample()
def cost_function(x,bic_score,nodes):
    """ Cost of x = f(x)."""
    G=vetor_Rede(x,nodes)
    return abs(bic_score.score(G))
def random_neighbour(x, fraction=1):
    """Move a little bit x, from the left or the right."""
    amplitude = (max(interval) - min(interval)) * fraction / 10
    delta = (-amplitude/2.) + amplitude * rn.random_sample()
    return clip(x + delta)
def acceptance_probability(cost, new_cost, temperature):
    if new_cost < cost:
        # print("    - Acceptance probabilty = 1 as new_cost = {} < cost = {}...".format(new_cost, cost))
        return 1
    else:
        p = np.exp(- (new_cost - cost) / temperature)
        # print("    - Acceptance probabilty = {:.3g}...".format(p))
        return p
def temperature(fraction):
    """ Example of temperature dicreasing as the process goes on."""
    return max(0.01, min(1, 1 - fraction))

def see_annealing(states, costs):
    plt.figure()
    plt.suptitle("Evolution of states and costs of the simulated annealing")
    plt.subplot(121)
    plt.plot(states, 'r')
    plt.title("States")
    plt.subplot(122)
    plt.plot(costs, 'b')
    plt.title("Costs")
    plt.show()



def pertubacao(y,fitness_aux,prob,max_v,min_v,bic_score,nodes,nao_dag):
    for val in range(len(y)):

        r=random.random()
        if r<=prob:
            if y[val] == 0:
                valor_mut_antigo=0
                maximo=3
                valor_mut=random.randint(1, 2)
                valor_mut2=maximo-valor_mut
                y[val]=valor_mut
                if y not in nao_dag:
                    G=vetor_Rede(y,nodes)
                    if G:
                        fitness_aux=abs(bic_score.score(G))
                    else:
                        nao_dag.append(y)  
                y[val]=valor_mut2
                if y not in nao_dag:
                    G=vetor_Rede(y,nodes)
                    if G:
                        fitness_aux=abs(bic_score.score(G))
                    else:
                        nao_dag.append(y)
                        y[val]=valor_mut_antigo
                else:
                    y[val]=valor_mut_antigo
            else:
                valor_mut_antigo=y[val]
                y[val]=0
                if y not in nao_dag:
                    G=vetor_Rede(y,nodes)
                    if G:
                        fitness_aux=abs(bic_score.score(G))
                    else:
                        nao_dag.append(y)
                        y[val]=valor_mut_antigo
                else:
                    y[val]=valor_mut_antigo
            

    return y,fitness_aux

def annealing(maxsteps=1000,debug=True):
    """ Optimize the black-box function 'cost_function' with the simulated annealing algorithm."""
    #Ler data
    with open('Asia.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        aux = 0
        data =[]
        data1=[ [] for i in range(8)]
        for row in csv_reader:
            data.append(row)
            for i in range(len(row)):
                data1[i].append(row[i])
            aux=aux+1
            if aux == 50001:
                break

    data = {}
    for i in range(len(data1)):
        data[data1[i][0]]=[data1[i][j] for j in range(1,len(data1[i]))]
    data = pd.DataFrame(data)
    print("Data: ")
    print(data) #Dados Retirandos do arquivo
    prob=0.5
    min_valor=0
    max_valor=2
    nao_dag=[]
    nodes=['Pollution','Smoker','Cancer','Xray','Dyspnoea']
    nodes=['asia', 'tub', 'smoke', 'lung', 'bronc', 'either', 'xray', 'dysp']
    ind_size=round((len(nodes)*len(nodes)-len(nodes))/2)
    ind=False
    while ind==False:
        aux=[random.randint(min_valor, max_valor) for i in range(ind_size)]
        if aux not in nao_dag:
            G=vetor_Rede(aux,nodes)
            if G:
                state=deep_copy(aux)
                ind=True
            else:
                nao_dag.append(aux)
    print('state')
    print(state)
    bic_score=BicScore(data)
    print(vetor_Rede(state,nodes))
    cost = cost_function(state,bic_score,nodes)
    states, costs = [state], [cost]
    for step in range(maxsteps):
        print(step)
        fraction = step / float(maxsteps)
        T = temperature(fraction)
        #[new_state,new_cost]=pertubacao(deep_copy(state),deep_copy(cost),prob,max_valor,min_valor,bic_score,nodes,nao_dag)
        [new_state,new_cost] = mutacao(deep_copy(state),deep_copy(cost),prob,max_valor,min_valor,bic_score,nodes,nao_dag)
        #new_cost = cost_function(new_state,bic_score,nodes)
        #if debug: print("Step #{:>2}/{:>2} : T = {:>4.3g}, state = {:>4.3g}, cost = {:>4.3g}, new_state = {:>4.3g}, new_cost = {:>4.3g} ...".format(step, maxsteps, T, state, cost, new_state, new_cost))

        if acceptance_probability(cost, new_cost, T) > random.random():

            state1= new_state.copy()
            cost=deep_copy(new_cost)

            states.append(state1)
            costs.append(cost)
            state=deep_copy(state1)
            # print("  ==> Accept it!")
        # else:
        #    print("  ==> Reject it...")
    return state, cost_function(state,bic_score,nodes), states, costs
state, c, states, costs =annealing(maxsteps=3000,debug=True)
nodes=['asia', 'tub', 'smoke', 'lung', 'bronc', 'either', 'xray', 'dysp']
G=vetor_Rede(state,nodes)
nx.draw(G, with_labels=True)
print(state)
print(c)
with open('Asia.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        aux = 0
        data =[]
        data1=[ [] for i in range(8)]
        for row in csv_reader:
            data.append(row)
            for i in range(len(row)):
                data1[i].append(row[i])
            aux=aux+1
            if aux == 50001:
                break
#22376.39851240954
data = {}
for i in range(len(data1)):
    data[data1[i][0]]=[data1[i][j] for j in range(1,len(data1[i]))]
data = pd.DataFrame(data)
print("Data: ")
print(data) #Dados Retirandos do arquivo
reader = BIFReader('asia.bif') # melhor rede do asia, como esta no bnlearn.com
asia_model = reader.get_model() # lendo esse modelo
print("Score BIC")
print(abs(BicScore(data).score(asia_model)))
#see_annealing(states, costs)
