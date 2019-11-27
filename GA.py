# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 20:08:13 2019

@author: Itallo
"""
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd
import csv
from pgmpy.estimators import BicScore
from pgmpy.models import BayesianModel
from pgmpy.readwrite import BIFReader
from pgmpy.estimators import BayesianEstimator
import time
from copy import deepcopy


#def penalidade(G1,nodes):
#    maior_infuencia=0
#    for i in G1.nodes():
#        aux_nodes = G1.active_trail_nodes(i)
#        size_nodes=int(len(aux_nodes[i]))
#    if maior_infuencia<size_nodes:
#        maior_infuencia=size_nodes
#        
#    return (len(nodes)-maior_infuencia)*0




def cruzamento_binario(x,y,n):
    m=[round(random.random()) for i in range(n)]
    filho1=[]
    filho2=[]
    for i in range(n):
        if m[i] ==0:
            filho1.append(x[i])
            filho2.append(y[i])
        else:
            filho1.append(y[i])
            filho2.append(x[i])
    return filho1,filho2

def mutacao(x,fitness_aux,prob,max_v,min_v):
    if len(x)*len(x[0])*prob<1:
        print("entando")
        for i in range(len(x)):
            for j in range(len(x[i])):
                r=random.random()
                if r<=prob:
                    valor_mut = x[i][j]
                    while (valor_mut == x[i][j]):
                        valor_mut = min_v + random.randint(min_valor, max_valor)
                    x[i][j] = valor_mut
    else:
        numero_mutacao= round(len(x)*len(x[0])*prob)
        while(numero_mutacao>0):
            ind_escolhido=round(random.random()*(len(x)-1))
            val=round(random.random()*(len(x[ind_escolhido])-1))
            valor_mut=deepcopy(x[ind_escolhido][val])
            valor_mut_antigo=deepcopy(x[ind_escolhido][val])
            while(valor_mut==x[ind_escolhido][val]):
                valor_mut=min_v+random.randint(min_valor, max_valor)
            x[ind_escolhido][val]=valor_mut
            if x[ind_escolhido][val] not in nao_dag:
                G=vetor_Rede(x[ind_escolhido],nodes)
                if G:
                    fitness_aux[ind_escolhido]=abs(BicScore(data).score(G))
                    numero_mutacao=numero_mutacao-1
                else:
                    nao_dag.append(x[ind_escolhido])
                    x[ind_escolhido][val]=valor_mut_antigo
            else:
                x[ind_escolhido][val]=valor_mut_antigo
        
        
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



def torneio(fitness,ind):
    k=0.75
    individuo1=round((pop_size-1) * random.random())
    individuo2=round((pop_size-1) * random.random())
    r = random.random()
    if r<=k:
        if fitness[individuo1]<=fitness[individuo2]:
            return fitness[individuo1],ind[individuo1]
        else:
            return fitness[individuo2],ind[individuo2]
    else:
        if fitness[individuo1]>=fitness[individuo2]:
            return fitness[individuo1],ind[individuo1]
        else:
            return fitness[individuo2],ind[individuo2]

def Elitismo(fitness,ind,pop_size):
    prox_ind=[]
    prox_fit=[]
    indice=[i[0] for i in sorted(enumerate(fitness), key=lambda x:x[1])] # sort
    i=0
    while len(prox_ind)<pop_size:
        prox_ind.append(ind[indice[i]])
        prox_fit.append(fitness[indice[i]])
        i=i+1
    return prox_ind,prox_fit
    

#Ler data
with open('cancer100k.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    aux = 0
    data =[]
    data1=[ [] for i in range(5)]
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


ind=[]
fitness=[]
min_valor=0
max_valor=2
pop_size=50
p_mutacao=0.05
p_cruzamento=0.9
#nodes=['asia', 'tub', 'smoke', 'lung', 'bronc', 'either', 'xray', 'dysp']
#nodes=['A','S',	'T','L'	,'B',	'E',	'X',	'D']
nodes=['Pollution','Smoker','Cancer','Xray','Dyspnoea']
nao_dag=[]
ind_size=round((len(nodes)*len(nodes)-len(nodes))/2)
gen_max=50
gen=0
melhor_fit=[]
#populacao inicial
while len(ind)<pop_size:
    aux=[random.randint(min_valor, max_valor) for i in range(ind_size)]
    if aux not in nao_dag:
        G=vetor_Rede(aux,nodes)
        if G:
            ind.append(aux)
            fitness.append(abs(BicScore(data).score(G)))
        else:
            nao_dag.append(aux)
while gen<gen_max:
    print(gen)
    filhos=[]
    filhos_fitness=[]
    while len(filhos)<pop_size:
        [fitness_sel1,ind_sel1]=torneio(fitness,ind)
        [fitness_sel2,ind_sel2]=torneio(fitness,ind)
        r=random.random()
        if r<=p_cruzamento:
            [filho1,filho2]=cruzamento_binario(ind_sel1,ind_sel2,ind_size)
            if filho1 not in nao_dag:
                G=vetor_Rede(filho1,nodes)
                if G:
                    filhos.append(filho1)
                    filhos_fitness.append(abs(BicScore(data).score(G)))
                else:
                    nao_dag.append(filho1)
            if len(filhos)<pop_size:
                if filho2 not in nao_dag:
                    G=vetor_Rede(filho2,nodes)
                    if G:

                        filhos.append(filho2)
                        filhos_fitness.append(abs(BicScore(data).score(G)))

                    else:

                        nao_dag.append(filho2)
        else:
            filhos.append(ind_sel1)
            filhos_fitness.append(fitness_sel1)

    
    
    mutacao(filhos,filhos_fitness,p_mutacao,max_valor, min_valor)
    fitness.clear()
    ind_aux=ind+filhos
    for i in ind_aux:
        G=vetor_Rede(i,nodes)
        fitness.append(abs(BicScore(data).score(G)))

  
    [ind,fitness]=Elitismo(deepcopy(fitness),deepcopy(ind_aux),pop_size)
    melhor_fit.append(fitness[0])
    gen=gen+1

G=vetor_Rede(ind[0],nodes)
best_score=BicScore(data).score(G)
print("melhor score:{}".format(best_score))

nx.draw(G, with_labels=True)
G.fit(data, estimator=BayesianEstimator, prior_type="BDeu")
G.check_model()

