import numpy


##calcular o vetor de médias
##parte do treinamento
def calcularMedia(vetor):
    return numpy.mean(vetor,axis=0)

##calcular o vetor Lambda do trabalho
##parte do treinamentos
def calcularVariancia(vetor):
    return numpy.var(vetor,axis=0)

##Parte exponencial definada na equação
def parteExponencial(entrada,vetorMedia,vetorLambda):
    exponencial = numpy.sum(((entrada-vetorMedia)**2)/vetorLambda)
    return exponencial * (-1/2)

##equação da probabilidade (densidade)
def equacaoProbabilidade(entrada,quantidadeElementosTreino,vetorMedia,vetorLamda):
    ##calculo da
    piPotencia = 2*numpy.pi ** (-quantidadeElementosTreino/2)
    produtorio = numpy.prod(vetorLamda) ** -(1/2)

    exponencial = parteExponencial(entrada, vetorMedia, vetorLamda)
    probCondicional = piPotencia * produtorio * numpy.exp(exponencial)
    return probCondicional


def treino(dadosTreino):
    vetorMedia = calcularMedia(dadosTreino)
    vetorLambda = calcularVariancia(dadosTreino)
    quantidadeElementosTreino = len(dadosTreino[0])
    return vetorMedia, vetorLambda, quantidadeElementosTreino

def calcularProbabilidade(vetorMedia, vetorLambda,quantidadeElementosTreino, teste):
    probCondicional = equacaoProbabilidade(teste,quantidadeElementosTreino,vetorMedia,vetorLambda)
    return probCondicional

def probabilidadeCondicional(dadosTreino,entrada):

    ##pegando os vetores de media e lamba - treinamento
    vetorMedia, vetorLambda, quantidadeElementosTreino  = treino(dadosTreino)
    # vetorMedia = calcularMedia(dadosTreino)
    # vetorLambda = calcularVariancia(dadosTreino)
    #calculo da densidade de probabilidade condicional
    probCondicional = equacaoProbabilidade(entrada,quantidadeElementosTreino,vetorMedia,vetorLambda)
    return probCondicional
##calcula a probabilidade de cada classe (Correspondente a frequencia de cada classe no treinamento)


###em construção
def probCondicionalMultiplos(dadosTreino,entrada):
    ##quantidade de dimensões do problema
    quantidadeElementosTreino = len(dadosTreino[0])
    ##pegando os vetores de media e lamba - treinamento
    vetorMedia = calcularMedia(dadosTreino)
    vetorLambda = calcularVariancia(dadosTreino)
    #calculo da densidade de probabilidade condicional
    probCondicional = equacaoProbabilidade(entrada,quantidadeElementosTreino,vetorMedia,vetorLambda)
    return probCondicional
##calcula a probabilidade de cada classe (Correspondente a frequencia de cada classe no treinamento)
