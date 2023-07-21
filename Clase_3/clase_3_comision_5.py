import numpy as np
#Vector en python puro

vectorA = [1, 2, 3]
vectorB = [4, 5, 6]

print(vectorA+vectorB)

suma = []
for i in range(len(vectorB)):
  suma.append(vectorB[i]+vectorA[i])
print(suma)

#Alternativa
suma = [x + y for x, y in zip(vectorA, vectorB)]
print(suma)

resta = []
for i in range(len(vectorB)):
  resta.append(vectorB[i]-vectorA[i])
print(resta)

n = 10
vector_0 = [0]*n
print(vector_0)

#Vectores con numpy
vector_numpy = np.array([5, 6, 7])
print("suma", vector_numpy+vectorA)
print("suma2", vector_numpy+vector_numpy)

#Multiplicacion por un numero
print(vector_numpy*5)

#Creacion vectores 1 y 0
vector_0_numpy = np.zeros(10)
print(vector_0_numpy)
vector_1_numpy = np.ones(10)
print(vector_1_numpy)
random_array = np.random.rand(3, 3)
print("Array aleatorio: ", random_array)

m_numpy = np.array([[1, 2], [3, 4]])
print(m_numpy)

m_numpy+m_numpy

import matplotlib.pyplot as plt

#Generar datos de ejemplo
x = np.random.randn(100) #Valores aleatoreos
y = np.random.randn(100)

#Plotear los puntos

plt.scatter(x, y)
plt.title("Ploteo de puntos")
plt.xlabel("Eje x")
plt.ylabel("Eje y")
plt.show()

plt.scatter(x, y)
plt.title("Ploteo de puntos")
plt.xlabel("Eje x")
plt.ylabel("Eje y")
plt.xlim(0, 4) #Limita rango eje x
plt.ylim(0, 4)
plt.show()

#ejemplo Pro
n = 100

# Definir el tamaño de la muestra

x1 = np.random.multivariate_normal([0,0], [[1,0],[0,1]], n)
x2 = np.random.multivariate_normal([4,4], [[1,0],[0,1]], n)

# Generar dos conjuntos de puntos aleatorios

data = np.concatenate((x1,x2), axis=0)
labels = np.concatenate(([0]*n, [1]*n))

# Concatenar los conjuntos de puntos y las etiquetas

print(data.shape, labels.shape)

# Imprimir la forma de la matriz de datos y del arreglo de etiquetas

print(data[10], labels[10])
print(data[170], labels[170])

# Imprimir un punto específico de la matriz de datos junto con su etiqueta correspondiente

plt.plot(x1[:, 0], x1[:, 1], 'r*')
plt.plot(x2[:, 0], x2[:, 1], 'b*')

# Graficar los puntos del conjunto x1 en rojo y los puntos del conjunto x2 en azul

plt.show()