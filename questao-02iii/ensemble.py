import numpy
import pandas

import parzen
import frontKFold as classBay

for j in range(0,30):
    #preparando os dados
    dataFac = pandas.read_csv('bases\\mfeat-fac.txt', delim_whitespace=True, header=None)
    dataFou = pandas.read_csv('bases\\mfeat-Fou.txt', delim_whitespace=True, header=None)
    dataKar = pandas.read_csv('bases\\mfeat-Kar.txt', delim_whitespace=True, header=None)

    #dividindo os dados
    dadosFac, classesFac = classBay.prepararBase(dataFac)
    dadosFou, classesFou = classBay.prepararBase(dataFou)
    dadosKar, classesKar = classBay.prepararBase(dataKar)

    #dividindo os folds
    folds = classBay.dividirTesteTreino(dadosFac, classesFac)

    for index_treino, index_teste in folds:
        testeFac = (dadosFac[index_teste])
        testeFou = (dadosFou[index_teste])
        testeKar = (dadosKar[index_teste])


        #bayesFac
        parametrosTreinamento = classBay.getParametrosTreinados(dadosFac[index_treino])
        probBayesianasFac = classBay.getProbabilidadesTeste(parametrosTreinamento, testeFac)

        # bayesFou
        parametrosTreinamento = classBay.getParametrosTreinados(dadosFou[index_treino])
        probBayesianasFou = classBay.getProbabilidadesTeste(parametrosTreinamento, testeFou)

        # bayesKar
        parametrosTreinamento = classBay.getParametrosTreinados(dadosKar[index_treino])
        probBayesianasKar = classBay.getProbabilidadesTeste(parametrosTreinamento, testeKar)
        probParzenFac = parzen.classificadorJanelaParzen(dadosFac[index_treino], testeFac)
        probParzenFou = parzen.classificadorJanelaParzen(dadosFou[index_treino], testeFou)
        probParzenKar = parzen.classificadorJanelaParzen(dadosKar[index_treino], testeKar)
        acertos = 0
        posicao = 0
        for i in index_teste:
            elemento = dadosFac[i]
            classeTeste = elemento[len(elemento)-1]
            probGeral = probBayesianasFac[posicao] + probBayesianasFou[posicao] + probBayesianasKar[posicao] + probParzenFac[posicao] + probParzenFou[posicao] + probBayesianasKar[posicao]
            classeEncontrada = numpy.argmax(probGeral)
            if (classeEncontrada == classeTeste):
                acertos = acertos + 1
            posicao = posicao + 1
        accuracy = acertos/len(index_teste)
        print(accuracy)