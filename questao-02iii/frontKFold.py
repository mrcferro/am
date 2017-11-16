import pandas, numpy
from sklearn.model_selection import StratifiedKFold

import engineBayes as bayes
from sklearn.preprocessing import StandardScaler

##prepara o array de classes (0-9, a cada 200 elementos)
def prepararArrayClasses():
    classes = numpy.array([])
    for i in range(2000):
        classes = numpy.append(classes,int(i/200))
    return classes


#extrai os dados da base e realiza a normalização caso o usuário escolha
#adiciona a coluna de classes
def prepararBase(data):
    data = numpy.array(data)
    # normaliza os dados
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    #cria um array com 2000 posicoes (a cada 200 posicoes uma classe)
    arrayClasses = prepararArrayClasses()
    #adiciona no vetor de dados uma coluna contendo os valores das classes
    data = numpy.c_[data, arrayClasses]
    return data,arrayClasses



#divide a base em teste e treino
#Retorna uma lista com n posicoes (cada um com 2 elementos) contendo os indices de treino

def dividirTesteTreino(dados,classes):
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    # skf = StratifiedKFold(n_splits=10)
    folds = skf.split(dados, classes)
    return folds

def getParametrosTreinados(dadosTreino):
    #dividindo os vetores de treino em 10 classes de treino
    valorColunaClasse = len(dadosTreino[0]) - 1
    # lista com 3 valores (vetorMedia, vetorLambda, quantidadeElementos)
    valoresTreino = []

    #realizando o treinamento para as classes
    for i in range(0,10):
        #pegar só os elementos da classe i
        dadosTreinoTemp = dadosTreino[numpy.where(dadosTreino[:, valorColunaClasse] == i)]
        #eliminar a coluna das classes
        dadosTreinoTemp = dadosTreinoTemp[:,:valorColunaClasse]
        #treinar a rede só para a classe i
        vetorMedia, vetorLambda, quantidadeElementos = bayes.treino(dadosTreinoTemp)
        temp = [vetorMedia, vetorLambda, quantidadeElementos]
        valoresTreino.append(temp)
    #agora tenho uma lista com 10 elementos que contem (vetorMedia, vetorLambda e quantidadeElementos para cada classe)
    return valoresTreino

def getTaxaAcerto(parametrosTreinamento, dadosTeste):
    #pega todos os elementos da base de teste
    acertos = 0
    for teste in dadosTeste:
        tam = len(teste) - 1
        #qual a classe que ele pertence?
        classeTeste = teste[len(teste) - 1]
        #calculando as probabilidades condicionais para todas as classes
        listaProbabilidades = []
        for parametros in parametrosTreinamento:
            #devo passar o dados de teste sem a última coluna
            prob = bayes.calcularProbabilidade(parametros[0], parametros[1],parametros[2],teste[0:tam])
            #lista com 10 probabilidades para cada classe
            listaProbabilidades.append(prob)
        #soma
        sum = numpy.sum(listaProbabilidades)
        novaListaProbabilidade = []
        for i in listaProbabilidades:
            novaListaProbabilidade.append(i*0.05/sum*0.05)
        classeEncontrada = numpy.argmax(novaListaProbabilidade)
        # classeEncontrada = numpy.argmax(listaProbabilidades)
        if (classeEncontrada == classeTeste):
            acertos = acertos + 1
    tamanho = len(dadosTeste)
    return (acertos/len(dadosTeste))

def getProbabilidadesTeste(parametrosTreinamento, teste):
    # pega todos os elementos da base de teste
    acertos = 0
    # calculando as probabilidades condicionais para todas as classes
    listaProbabilidades = []

    for elementoTeste in teste:
        tam = len(elementoTeste) - 1
        probTemporaria = []
        for parametros in parametrosTreinamento:
            # devo passar o dados de teste sem a última coluna
            prob = bayes.calcularProbabilidade(parametros[0], parametros[1], parametros[2], elementoTeste[0:tam])
            # lista com 10 probabilidades para cada classe
            probTemporaria.append(prob)
        # soma
        sum = numpy.sum(probTemporaria)
        novaListaProbabilidade = []
        for i in probTemporaria:
            novaListaProbabilidade.append(i * 0.05 / sum * 0.05)
        listaProbabilidades.append(novaListaProbabilidade)
    return listaProbabilidades

def treinar (dados, classes):
    folds = dividirTesteTreino(dados,classes)
    # para cada fold, vamos treinar e testar
    listaTaxasAcerto = []
    for index_treino, index_teste in folds:
        parametrosTreinamento = getParametrosTreinados(dados[index_treino])
        taxaAcerto = getTaxaAcerto(parametrosTreinamento, dados[index_teste])
        listaTaxasAcerto.append(taxaAcerto)
    print(listaTaxasAcerto)
# executando o projeto

def executar():
    data = pandas.read_csv('mfeat-kar.txt', delim_whitespace=True, header=None)
    for i in range(30):
        dadosPreparados, vetorClasses = prepararBase(data)
        treinar(dadosPreparados,vetorClasses)

def executarEnsemble(data):
    dadosPreparados, vetorClasses = prepararBase(data)
