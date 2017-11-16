#
# como usar   
# parzen(features, pdMatriz)
#     onde
#       - features = array contendo os dados que serao classificados
#       - pdMatriz = matriz (pandas) com os dados do dataset (colunas = features)
#

import numpy as np
import pandas as pd
import math


## Arquivo parzenWindow.py. 
#esse codigo está aqui apenas para evitar o problema de cache.. vai sair 


# assumindo que a distribuição é gaussiana, o calculo de H = 1.06 * desvioPadrao * numeroExemplosTreino^-1/5
# slide Francisco, pagina 18
def encontrarH(npColuna):
    coeficiente = -1.0 / 5
    h = 1.06 * np.std(npColuna) * len(npColuna)**(coeficiente)
    return h


# recebe uma matriz de dados e encontra a largura da janela (h) para cada coluna (característica)
# retorna um array contendo os coeficiente, onde cada elemento i representa o H da coluna j da matriz
def encontrarHDimensoes(matrizNumpy):
    hDimensoes = []
    colunas = len(matrizNumpy[0])
    # para cada coluna...
    for i in range(colunas):
        coluna = matrizNumpy[:,i:i+1]
        h = encontrarH(coluna)
        hDimensoes.append(h)
    return hDimensoes



# função kernel univariada normal - slide "Tecnicas nao parametricas", pg 13    
def funcaoKernelUnivariadaNormal(u):
    return (1.0/ math.sqrt(2 * math.pi)) * math.exp(- (u**2)/2)



# features = array com os valores para o qual deseja-se encontrar o valor da função densidade em uma determinada classe.
#            o numero de elementos deve ser igual ao numero de colunas da matriz
# indiceColunaFeature = feature observada (coluna da matriz)
# matriz = matriz que contem todas as features(colunas) das amostras(linhas) de uma determinada classe
# arrayH - array contendo o valor H (janela) para todas as colunas da matriz. A função encontrarHDimensoes() faz isso.
def parzenMultivariadoProduto(features, matrizNumpy, arrayH):
    n = len(matrizNumpy)
    soma = 0
    # p = numero de colunas, dimensoes
    p = len(matrizNumpy[0])
    
    # o numero das linhas podem estar desordenadas, por causa do shurfle
    numLinha = 0
    
    for i in range(n):
        produto = 1
        for j in range(p):            
            h = arrayH[j]
            Xij = matrizNumpy[i][j]
            u = (features[j] - Xij)/h
            #print("i="+str(i)+", j="+str(j)+ ". u= (" + str(features[j]) + " - " + str(Xij) + ")/" + str(h) + " = " + str(u) )
            kernel = funcaoKernelUnivariadaNormal(  u )
            produto = produto * kernel
        soma = soma + produto
    # dividido pela media do H
    resultado = (1.0/n) * (1.0/np.mean(arrayH)) * soma  
    return resultado



# função principal
# parametros:
#      1 - features: array contendo os valores das features que serão analisadas na função densidade
#      2 - matriz de dados contendo em suas colunas as features do dataset
def parzen(features, pdMatriz):
    
    f = np.array(features)
    m = np.array(pdMatriz) 
    
    #print("parzen, features=" + str(f.shape) + ", dados=" + str(m.shape))
    
    arrayH = encontrarHDimensoes(m)
    result = parzenMultivariadoProduto(f, m, arrayH)
    return result


def testeGraficoParzen(indiceColuna, pdMatriz):
    arrayH = encontrarHDimensoes(pdMatriz)
    print("H (janelas) = \n" + str(arrayH))
    print("Média H = " + str(np.mean(arrayH)))
    graficoDensidade(pdMatriz, indiceColuna, arrayH)
    print("---------------------------------------")
    print("Coluna " + str(indiceColuna) + " \n---------------------------------------")
    print("Quantidade de elementos: " + str(len(pdMatriz[:,indiceColuna])))
    print("Dados (primeiros 50 elementos):\n " + str(pdMatriz[:50,indiceColuna]))
    print("Menor Valor = " + str(min(pdMatriz[:,indiceColuna])))
    print("Maior Valor = " + str(max(pdMatriz[:,indiceColuna])))
    print("Média  = " + str(np.mean(pdMatriz[:,indiceColuna])))
    print("Mediana  = " + str(np.median(pdMatriz[:,indiceColuna])))
    print("Janela (parzen) = " + str(arrayH[indiceColuna]))
    print("\n\n")

def graficoDensidade(pdMatriz, coluna, arrayH):
    dados = pdMatriz[:,coluna:coluna+1]    
    janela = arrayH[coluna]

    valor = []
    densidade = []
    
    inicio = min(dados)  
    fim = max(dados)
    amp = fim-inicio
    inicio = inicio - amp
    fim = fim + amp
    intervalo = amp / 20
    
    # prepara o array de features x, inserindo inivialmente o valor da mediana em cada elemento
    f = []
    for i in range (len(pdMatriz[0])):
        f.append(np.median(pdMatriz[:,i:i+1]))
        
    
    ponto = inicio
    while (ponto<=fim):
        # calcula a densidade para todos elementos da dimensao 2
        f[coluna] = ponto
        p = parzenMultivariadoProduto(np.array(f), pdMatriz, arrayH)
        valor.append(ponto)
        densidade.append(p)
        ponto = ponto + intervalo

    f = pd.DataFrame()
    f['x'] = valor
    f['densidade'] = densidade
    texto = "Feature = " + str(coluna)
    f.plot(x='x',y='densidade', label=texto)
