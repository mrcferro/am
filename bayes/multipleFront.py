import pandas
import numpy
from sklearn import datasets

##carregando os dados
dataFourier = numpy.loadtxt("mfeat-fou.txt")
listaClassesFourier = []
aux = 0
for i in range(0,10):
    listaClassesFourier.append(dataFourier[aux:aux+140,:])
    aux = aux + 200
