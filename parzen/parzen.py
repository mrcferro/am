# Marcio Roberio, Marcelo, Paulo 
#
# como usar   (x, indiceColunaFeature, pdMatriz, arrayH):
# 1 - armazene em um array numpy o valor das janelas, chamando a função encontrarHDimensoes(pandasMatriz)
# 2 - faça a chamada da função parzenMultivariadoProduto passando 
#      a) o valor X que você deseja usar para calcular o valor na função densidade
#      b) o índice da coluna correspondente à feature observada
#      c) a matriz (pandas)
#      d) o array, resultado da chamada da função encontrarHDimensoes(pandasMatriz) (item 1).


import numpy as np
import pandas as pd
import math


# assumindo que a distribuição é gaussiana, o calculo de H = 1.06 * desvioPadrao * numeroExemplosTreino^-1/5
# slide Francisco, pagina 18
def encontrarH(npColuna):
    coeficiente = -1.0 / 5
    h = 1.06 * np.std(npColuna) * len(npColuna)**(coeficiente)
    return h


# recebe uma matriz de dados e encontra a largura da janela (h) para cada coluna (característica)
# retorna um array contendo os coeficiente, onde cada elemento i representa o H da coluna j da matriz
def encontrarHDimensoes(matriz):
    hDimensoes = []
    colunas = len(matriz[0])
    # para cada coluna...
    for i in range(colunas):
        npColuna = np.array(matriz[:,i:i+1])
        h = encontrarH(npColuna)
        hDimensoes.append(h)
    return hDimensoes


# função kernel univariada normal - slide "Tecnicas nao parametricas", pg 13    
def funcaoKernelUnivariadaNormal(u):
    return (1.0/ math.sqrt(2 * math.pi)) * math.exp(- (u**2)/2)


# features = array com os valores para o qual deseja-se encontrar o valor da função densidade em uma determinada classe.
#            o numero de elementos deve ser igual ao numero de colunas da matriz
# indiceColunaFeature = feature observada (coluna da matriz)
# pdMatriz = matriz (pandas) que contem todas as features(colunas) das amostras(linhas) de uma determinada classe
# arrayH - array contendo o valor H (janela) para todas as colunas da matriz. A função encontrarHDimensoes() faz isso.

def parzenMultivariadoProduto(features, pdMatriz, arrayH):
    n = len(pdMatriz)
    soma = 0
    # p = numero de colunas, dimensoes
    p = len(pdMatriz[0])  
   
    for i in range(n):
        produto = 1
        for j in range(p):
            coluna = np.array(pdMatriz[ : , j:j+1])
            h = arrayH[j]
            u = (features[j] - pdMatriz[i][j])/h 
            kernel = funcaoKernelUnivariadaNormal(  u )
            produto = produto * kernel
        soma = soma + produto
    
    # dividido pela media do H
    
    resultado = (1.0/n) * (1.0/np.mean(arrayH)) * soma  
    return resultado



# gera grafico da densidade
def graficoParzen(pdMatriz, coluna, arrayH):
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


def testeGraficoParzen(indiceColuna, pdMatriz):
    arrayH = encontrarHDimensoes(pdMatriz)
    print("H (janelas) = \n" + str(arrayH))
    print("Média H = " + str(np.mean(arrayH))
    graficoParzen(pdMatriz, indiceColuna, arrayH)
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

