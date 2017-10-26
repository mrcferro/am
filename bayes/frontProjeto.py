import pandas
import numpy
import engineBayesClassifier as bayes
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from random import shuffle

#extrai os dados da base e realiza a normalização caso o usuário escolha
#retorna os dados em uma lista
def extrairBase(data, normal):
    # normalizar se for preciso
    if (normal == 1):
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
    listaClasses = []
    aux = 0
    # pegando todas as classes e adicionando em uma lista com 10 posições. Cada posição equivale a uma classe
    for i in range(0, 10):
        listaClasses.append(data[aux:aux + 200, :])
        aux = aux + 200
    return listaClasses

#faz a separacao da quantidade de elementos em cada base de teste
#retorna uma lista com dois elementos, sendo o 1 => base de teste | 2 => base de treino
def prepararBase(listaClasses,tamanhoAmostraTreino):
    # shuffle a lista
    for i in listaClasses:
        shuffle(i)
    tamanhoAmostraTeste = 200 - tamanhoAmostraTreino
    baseTeste = []
    baseTreino = []
    aux = 0
    for i in listaClasses:
        baseTeste.append(i[0:tamanhoAmostraTreino, :])
        baseTreino.append(i[tamanhoAmostraTeste:tamanhoAmostraTreino + tamanhoAmostraTeste, :])
        aux = aux + 200
    return [baseTeste,baseTreino]

#calcula para todos os elementos da base de teste a qual classe pertence e compara com a qual ele realmente é
#retorna a taxa de acerto
def calcularProbabilidade(baseTeste, baseTreino):
    tamanhoAmostraTeste = len(baseTeste[0])
    acertos = 0
    for i in range(0, 10):
        listaProbabilidades = []
        for j in range(0, tamanhoAmostraTeste):
            # calcular para todas as classes as probabilidades do elemento da classe[i][j]
            for l in range(0, 10):
                prob = bayes.probabilidadeCondicional(baseTeste[l], baseTreino[i][j])
                listaProbabilidades.append(prob)
            # melhor classe entre as 10
            classeEncontrada = numpy.argmax(listaProbabilidades)
            listaProbabilidades.clear()
            if (classeEncontrada == i):
                acertos = acertos + 1

    return (acertos / tamanhoAmostraTeste * 10)


def executar(data,normalizada):
    classes = extrairBase(data,normalizada)
    basesPreparadas = prepararBase(classes,180)
    acerto = calcularProbabilidade(basesPreparadas[0],basesPreparadas[1])
    return acerto

#executando o projeto
data = numpy.loadtxt("mfeat-fou.txt")
resultado = executar(data, 0)
print(resultado)
