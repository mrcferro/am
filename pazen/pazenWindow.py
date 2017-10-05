# algoritmo janela-pazen
# http://www.inf.ufpr.br/lesoliveira/padroes/naoparametricos.pdf
# usando os parametros do slide (p 8)
# http://www.cin.ufpe.br/~fatc/AM/Tecnicas-Nao-Parametricas.pdf 

# FUNÇÃO KERNEL da JANELA DE PAZEN
# kernel (x) = 1/2, se abs(x) < 1
#            = 0,   caso contrario  
 
def kernelPazen(x):
    if (abs(x) < 1):
        return 0.5
    else:
        return 0


# h = janela
# x = valor (dado)
# npDados = vetor de dados
def pazenX(x, h, npDados):
    n = len(npDados)
    soma = 0
    for i in range(n):
        kernel = kernelPazen((x-npDados[i])/h) 
        soma = soma + kernel
    p = (1/n)*(1/h)*soma
    return p


