import math 
import matplotlib.pyplot as plt 
import numpy as np  



# Sigmoid aktivasyon fonksiyonunu tanımlar
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# -5'ten 5'e kadar 0.1 adım aralığı ile bir dizi oluşturur
x = np.arange(-5., 5., 0.1)

# Sigmoid fonksiyonunu uygular
sigmoid = sigmoid(x)

# Sigmoid fonksiyonunun grafiğini çizer
plt.xlabel("girdiler")
plt.ylabel("scors")
plt.grid(True)
plt.plot(x, sigmoid, label="Step", color="C1", lw=3)
plt.legend()
plt.show()


# Tanh aktivasyon fonksiyonunu tanımlar
def tanh(x):
    return np.tanh(x)

# -5'ten 5'e kadar 0.1 adım aralığı ile bir dizi oluşturur
x = np.arange(-5., 5., 0.1)

# Tanh fonksiyonunu uygular
tanh = tanh(x)

# Tanh fonksiyonunun grafiğini çizer
plt.xlabel("girdiler")
plt.ylabel("scores")
plt.grid(True)
plt.plot(x, tanh, label="Tanh", color="C5", lw=3)
plt.legend()
plt.show()


# ReLU aktivasyon fonksiyonunu tanımlar
def relu(x):
    return np.maximum(0, x)

# -5'ten 5'e kadar 0.1 adım aralığı ile bir dizi oluşturur
x = np.arange(-5., 5., 0.1)

# ReLU fonksiyonunu uygular
relu = relu(x)

# ReLU fonksiyonunun grafiğini çizer
plt.xlabel("girdiler")
plt.ylabel("scores")
plt.grid(True)
plt.plot(x, relu, label="ReLu", color="C2", lw=3)
plt.legend()
plt.show()



def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))
x = np.arange(-5.,5.,0.1)
softmax = softmax(x)

plt.xlabel("girdiler")
plt.ylabel("scores")
plt.grid(True)
plt.plot(x,softmax,label="Softmax",color="C3",lw=3)
plt.legend()
plt.show()



def softplus(x):
    return np.log(1.0+np.exp(x))
x = np.arange(-5.,5.,0.1)

softplus = softplus(x)

plt.xlabel("girdiler")
plt.ylabel("scores")
plt.grid(True)
plt.plot(x,softmax,label="SoftPlus",color="C4",lw=3)
plt.legend()
plt.show()


def elu(x,alpha):
    a = []
    for item in x:
        if item>= 0:
           a.append(item)
        else:
           a.append(alpha*(np.exp(item)-1))
    return a 
x = np.arange(-5.,5.,0.1)
elu = elu(x,1.0)

plt.xlabel("girdiler")
plt.ylabel("Scores")
plt.grid(True)
plt.plot(x,elu,label="Elu",color="C8",lw=3)
plt.legend()
plt.show()

    

import matplotlib.pyplot as plt
import numpy as np


# Aktivasyon fonksiyonlarını tanımlar
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Numerik stabilite için
    return exp_x / np.sum(exp_x)

def softplus(x):
    return np.log(1.0 + np.exp(x))

def elu(x, alpha):
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

# -5'ten 5'e kadar 0.1 adım aralığı ile bir dizi oluşturur
x = np.arange(-5., 5., 0.1)

# Fonksiyonları uygular
sigmoid_values = sigmoid(x)
tanh_values = tanh(x)
relu_values = relu(x)
softmax_values = softmax(x)
softplus_values = softplus(x)
elu_values = elu(x, 1.0)

# Tek bir pencere içinde 6 alt grafik oluşturur
fig, axs = plt.subplots(3, 2, figsize=(12, 15))

# Sigmoid grafiği
axs[0, 0].plot(x, sigmoid_values, label="Sigmoid", color="C1", lw=3)
axs[0, 0].set_xlabel("Girdiler")
axs[0, 0].set_ylabel("Scores")
axs[0, 0].grid(True)
axs[0, 0].legend()

# Tanh grafiği
axs[0, 1].plot(x, tanh_values, label="Tanh", color="C5", lw=3)
axs[0, 1].set_xlabel("Girdiler")
axs[0, 1].set_ylabel("Scores")
axs[0, 1].grid(True)
axs[0, 1].legend()

# ReLU grafiği
axs[1, 0].plot(x, relu_values, label="ReLU", color="C2", lw=3)
axs[1, 0].set_xlabel("Girdiler")
axs[1, 0].set_ylabel("Scores")
axs[1, 0].grid(True)
axs[1, 0].legend()

# Softmax grafiği
axs[1, 1].plot(x, softmax_values, label="Softmax", color="C3", lw=3)
axs[1, 1].set_xlabel("Girdiler")
axs[1, 1].set_ylabel("Scores")
axs[1, 1].grid(True)
axs[1, 1].legend()

# SoftPlus grafiği
axs[2, 0].plot(x, softplus_values, label="SoftPlus", color="C4", lw=3)
axs[2, 0].set_xlabel("Girdiler")
axs[2, 0].set_ylabel("Scores")
axs[2, 0].grid(True)
axs[2, 0].legend()

# ELU grafiği
axs[2, 1].plot(x, elu_values, label="ELU", color="C8", lw=3)
axs[2, 1].set_xlabel("Girdiler")
axs[2, 1].set_ylabel("Scores")
axs[2, 1].grid(True)
axs[2, 1].legend()

# Grafiklerin arasındaki boşlukları ayarlar
plt.tight_layout()

# Grafikleri gösterir
plt.show()