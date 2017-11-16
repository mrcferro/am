
# coding: utf-8

# In[39]:

import scipy as sp
from scipy.spatial.distance import squareform, pdist

# paralelismo..
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool 
from functools import partial


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
import time
import math

# para testes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import zero_one_loss, accuracy_score

# classificador 2.2
import parzenFunction as pz

#graficos
import matplotlib

#grafico distribuicao
import scipy.stats as stats
import pylab as pl
def lerDados(base):
    if (base == 1):
        dfac = pd.read_csv('dados/mfeat-fac', delim_whitespace=True, header=None)
        return dfac, "mfeat-fac"
    elif (base == 2):
        dfou = pd.read_csv('dados/mfeat-fou', delim_whitespace=True, header=None)
        return dfou, "mfeat-fou"
    elif (base == 3):
        dkar = pd.read_csv('dados/mfeat-kar', delim_whitespace=True, header=None)
        return dkar, "mfeat-kar"

      
#importação dos dados
#dfac = pd.read_csv('dados/mfeat-fac', delim_whitespace=True, header=None)
#dfou = pd.read_csv('dados/mfeat-fou', delim_whitespace=True, header=None)
#dkar = pd.read_csv('dados/mfeat-kar', delim_whitespace=True, header=None)



# In[40]:

#cria array com 2000 elementos, onde os 200 primmeiros elementos possuem a classe 0, os proximos 200
#elementos possuem a classe 1, ...
def preparaClasses():
    classes = np.array([])
    for i in range (2000):
        classes = np.append(classes,int(i/200))
    classes[198:203]
    return classes


# In[41]:

# ordena a matriz com os dados do array
# número de linhas da matriz deve ser igual ao número de elementos do array
# baseado no tutorial pandas 
# http://nbviewer.jupyter.org/urls/bitbucket.org/hrojas/learn-pandas/raw/master/lessons/02%20-%20Lesson.ipynb
#def ordenaMatrizPorClasse(matriz,array):
#    c = pd.DataFrame(matriz)
#    c['classe'] = array
#    Sorted = c.sort_values(['classe'], ascending=True)
#    #del Sorted['novacoluna'] # remove a coluna
#    return Sorted


# In[42]:

def classificadorJanelaParzen(dadosTreino, dadosTeste):

    treino = np.array(dadosTreino)
    teste = np.array(dadosTeste)
    colunaClasse = dadosTreino.shape[1] - 1
    
    
    treinoClasse0 = treino[ treino[:,colunaClasse] == 0] 
    treinoClasse1 = treino[ treino[:,colunaClasse] == 1] 
    treinoClasse2 = treino[ treino[:,colunaClasse] == 2] 
    treinoClasse3 = treino[ treino[:,colunaClasse] == 3] 
    treinoClasse4 = treino[ treino[:,colunaClasse] == 4] 
    treinoClasse5 = treino[ treino[:,colunaClasse] == 5] 
    treinoClasse6 = treino[ treino[:,colunaClasse] == 6] 
    treinoClasse7 = treino[ treino[:,colunaClasse] == 7] 
    treinoClasse8 = treino[ treino[:,colunaClasse] == 8] 
    treinoClasse9 = treino[ treino[:,colunaClasse] == 9] 
    
    # eliminar a coluna com as classes
    treinoClasse0 = treinoClasse0[:,:colunaClasse]
    treinoClasse1 = treinoClasse1[:,:colunaClasse]
    treinoClasse2 = treinoClasse2[:,:colunaClasse]
    treinoClasse3 = treinoClasse3[:,:colunaClasse]
    treinoClasse4 = treinoClasse4[:,:colunaClasse]
    treinoClasse5 = treinoClasse5[:,:colunaClasse]
    treinoClasse6 = treinoClasse6[:,:colunaClasse]
    treinoClasse7 = treinoClasse7[:,:colunaClasse]
    treinoClasse8 = treinoClasse8[:,:colunaClasse]
    treinoClasse9 = treinoClasse9[:,:colunaClasse]   

    listaProbabilidadesTeste = []
    for i in range (len(teste)):
        
        classeCorreta = teste[i][colunaClasse]
        
        # tira a ultima coluna, que é a coluna da classe
        linhaTeste = teste[i, :colunaClasse]    
        
        # classficacao...
        probC0 = pz.parzen(linhaTeste, treinoClasse0)
        probC1 = pz.parzen(linhaTeste, treinoClasse1)
        probC2 = pz.parzen(linhaTeste, treinoClasse2)
        probC3 = pz.parzen(linhaTeste, treinoClasse3)
        probC4 = pz.parzen(linhaTeste, treinoClasse4)
        probC5 = pz.parzen(linhaTeste, treinoClasse5)
        probC6 = pz.parzen(linhaTeste, treinoClasse6)
        probC7 = pz.parzen(linhaTeste, treinoClasse7)
        probC8 = pz.parzen(linhaTeste, treinoClasse8)
        probC9 = pz.parzen(linhaTeste, treinoClasse9)
        resultado =  [probC0, probC1, probC2, probC3, probC4, probC5, probC6, probC7, probC8, probC9]
        listaProbabilidadesTeste.append(resultado)

    return listaProbabilidadesTeste


# In[43]:

def classificadorJanelaParzenClassificar(dadosTreino, classesTreino, arrayDadosClassificar):
    treino = np.array(dadosTreino)
    #teste = np.array(dadosTeste)
    colunaClasse = dadosTreino.shape[1] - 1
    
    treinoClasse0 = treino[ treino[:,colunaClasse] == 0] 
    treinoClasse1 = treino[ treino[:,colunaClasse] == 1] 
    treinoClasse2 = treino[ treino[:,colunaClasse] == 2] 
    treinoClasse3 = treino[ treino[:,colunaClasse] == 3] 
    treinoClasse4 = treino[ treino[:,colunaClasse] == 4] 
    treinoClasse5 = treino[ treino[:,colunaClasse] == 5] 
    treinoClasse6 = treino[ treino[:,colunaClasse] == 6] 
    treinoClasse7 = treino[ treino[:,colunaClasse] == 7] 
    treinoClasse8 = treino[ treino[:,colunaClasse] == 8] 
    treinoClasse9 = treino[ treino[:,colunaClasse] == 9] 
    
    # eliminar a coluna com as classes
    treinoClasse0 = treinoClasse0[:,:colunaClasse]
    treinoClasse1 = treinoClasse1[:,:colunaClasse]
    treinoClasse2 = treinoClasse2[:,:colunaClasse]
    treinoClasse3 = treinoClasse3[:,:colunaClasse]
    treinoClasse4 = treinoClasse4[:,:colunaClasse]
    treinoClasse5 = treinoClasse5[:,:colunaClasse]
    treinoClasse6 = treinoClasse6[:,:colunaClasse]
    treinoClasse7 = treinoClasse7[:,:colunaClasse]
    treinoClasse8 = treinoClasse8[:,:colunaClasse]
    treinoClasse9 = treinoClasse9[:,:colunaClasse]   

  
       
    # classficacao...
    probC0 = pz.parzen(arrayDadosClassificar, treinoClasse0)
    probC1 = pz.parzen(arrayDadosClassificar, treinoClasse1)
    probC2 = pz.parzen(arrayDadosClassificar, treinoClasse2)
    probC3 = pz.parzen(arrayDadosClassificar, treinoClasse3)
    probC4 = pz.parzen(arrayDadosClassificar, treinoClasse4)
    probC5 = pz.parzen(arrayDadosClassificar, treinoClasse5)
    probC6 = pz.parzen(arrayDadosClassificar, treinoClasse6)
    probC7 = pz.parzen(arrayDadosClassificar, treinoClasse7)
    probC8 = pz.parzen(arrayDadosClassificar, treinoClasse8)
    probC9 = pz.parzen(arrayDadosClassificar, treinoClasse9)
         
    resultado =  [probC0, probC1, probC2, probC3, probC4, probC5, probC6, probC7, probC8, probC9]
    
    return resultado


# In[44]:

def gerarGraficoResultado(nomeDados, resultado, estimativaPontual):
    resultadoOrdenado = np.sort(resultado)
    plt.figure(figsize=(12,7))
    plt.axis([0,300,0,1])
    plt.plot(resultado)
    plt.ylabel("Precisão (accuracy)")
    plt.xlabel("Folds")
    plt.title("Base de dados: " + nomeDados)
    plt.show()
    


# In[45]:

# adiciona uma coluna em matrizColuna, referente às classes 
def insereColunaClasses(matriz, classes):
    indiceColunaClasses = len(matriz.columns)
    matriz[indiceColunaClasses] = classes
    return matriz, indiceColunaClasses
    


# In[46]:

# embaralha os dados
# encontra 10 folds
# separa treina, classifica e encontra a taxa de acerto 
def core(nomeDados, matrizDados, classes, rodadas=1):

    saida = []
    matrizDados, indiceColunaClasses =  insereColunaClasses(matrizDados.copy(), classes)
    print("[" + nomeDados + "] Número de Rodadas = " + str(rodadas))
    
    
    for i in range(rodadas):
    
        # embaralha a matriz de dados e as classes ao mesmo tempo
        dadosEmbaralhados, classesEmbaralhadas = shuffle(matrizDados, classes, random_state=i)

        # 10 folds
        skf = StratifiedKFold(n_splits=10)
        folds = skf.split(dadosEmbaralhados, classesEmbaralhadas)
        z = 0
        
        for indicesTreino, indicesTeste in folds:        
            
            dadosTreino   = np.array(dadosEmbaralhados.iloc[indicesTreino])
            dadosTeste    = np.array(dadosEmbaralhados.iloc[indicesTeste])          
            classesTreino = np.array(classesEmbaralhadas[indicesTreino])
            classesTeste  = np.array(classesEmbaralhadas[indicesTeste])
            
            # verifica se as classes estao corretas
            for a in range(len(dadosTreino)):
                if (dadosTreino[a][indiceColunaClasses] != classesTreino[a]):
                    print("Erro!")
            for b in range(len(dadosTeste)):
                if (dadosTeste[b][indiceColunaClasses] != classesTeste[b]):
                    print("Erro!")
            
            accuracy = classificadorJanelaParzen(dadosTreino, classesTreino, dadosTeste, classesTeste)
            saida.append(accuracy)          
            print("[" + nomeDados + "] Rodada " + str(i) + ", fold "+ str(z) + " concluido. Accuracy = " + str(accuracy))
            z = z + 1
    
    # ESTIMATIVA PONTUAL
    media = np.mean(saida)       
    return saida, media

# Classifica e mostra o resultado
def preCore(nomeDados, dados, classes, rodadas):
    
    # medir tempo    
    inicio = time.time()
    print("[" + nomeDados + "] Iniciando classificação ")

    # executa a classificação
    resultado, estimativaPontual = core(nomeDados, dados, classes, rodadas)

    # gera o grafico
    gerarGraficoResultado(nomeDados, resultado, estimativaPontual)
    print("[" + str(nomeDados) + "] *** Estimativa Pontual (média) = " + str(estimativaPontual))
    
    # imprime os dados
    print("[" + str(nomeDados) + "] *** Resultados: ")
    print(resultado)
    print("-----------------------")
    
    # calculo do tempo de processamento
    fim = time.time()
    total = fim - inicio
    print("\n[" + nomeDados + "] Fim - Tempo de Execução = " + str(total) + "\n")    
    
    return resultado, estimativaPontual

def iniciaParalelismo(rodadas, base):
    classes = preparaClasses()
    dados, nome = lerDados(base)
    print("inicializando processamento paralelo da matriz '" + str(nome) + "', com dimensões = " + str(dados.shape))
    return preCore(nome, dados, classes, rodadas)

# rodar o treinamento e teste, apresentando o gráfico da precisão
def treinoETeste():
    bases = [1,2,3]
    pool = Pool(4)
    rodadas = 30
    func = partial(iniciaParalelismo, rodadas)
    pool.map(func, bases)
    pool.close()
    pool.join()
    
    


# def classificarUmElemento(base, indiceClassificar):
#     rodadas = 2
#     classes = preparaClasses()
#     resultado = classificadorJanelaParzenClassificar(base,classes,indiceClassificar)
#     print(resultado)
#     print("média")
#     mediaColunas = np.mean(resultado, axis=0)
#     return mediaColunas
#     #print(mediaColunas)
#