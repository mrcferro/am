import engineBayesClassifier as bayes
from sklearn import datasets

#load database
iris = datasets.load_iris()
#getting classes
classes = iris.target

#getting features
features = iris.data
featuresSetosa = features[0:40,:]
featuresVersicolor = features[50:90,:]
featuresVirginica = features[100:140,:]
entrada = features[142,:]





probSetosa = bayes.probabilidadeCondicional(featuresSetosa,entrada)
probVersicolor = bayes.probabilidadeCondicional(featuresVersicolor,entrada)
probVirginica = bayes.probabilidadeCondicional(featuresVirginica,entrada)
print(probSetosa, probVersicolor, probVirginica)
if (probSetosa > probVersicolor and probSetosa > probVirginica):
    print("Setosa")
elif (probVersicolor> probSetosa and probVersicolor> probVirginica):
    print("Versicolor")
else:
    print("Virginica")