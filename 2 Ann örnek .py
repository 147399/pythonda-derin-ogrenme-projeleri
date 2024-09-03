import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
import tensorflow as tf  # Yapay sinir ağı modelleri oluşturmak için TensorFlow kütüphanesini içe aktarır.

#TensorFlow'un hangi versiyonunun yüklü olduğunu gösterir
print(tf.__version__)

# Veri kümesini CSV dosyasından okur
data = pd.read_csv("C:/Users/ASAF/Desktop/Datasets/Folds5x2_pp.csv")

# Veri kümesinin bir kopyasını oluşturur
df = data.copy()

print(df)

# Hedef değişkeni (bağımlı değişken) olan "PE" sütununu y olarak ayırır
y = df["PE"]

# Girdilerden (bağımsız değişkenler) "PE" sütununu çıkarır ve x olarak tanımlar
x = df.drop(columns=["PE"])

# Veriyi eğitim ve test olarak ayırır (Eğitim için %70, Test için %30)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=20, train_size=0.7)

# Yapay sinir ağı modelini oluşturur
ann = tf.keras.models.Sequential()

# İlk katmanı ekler: 6 birimli, ReLU aktivasyon fonksiyonlu, girdiler için 4 boyutlu
ann.add(tf.keras.layers.Dense(units=6, activation="relu", input_dim=4))

# İkinci katmanı ekler: 6 birimli, ReLU aktivasyon fonksiyonlu
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))

# Çıkış katmanını ekler: 1 birimli (çıkış)
ann.add(tf.keras.layers.Dense(units=1))

# Modeli derler: Optimizer olarak Adam, kayıp fonksiyonu olarak Ortalama Kare Hata
ann.compile(optimizer="adam", loss="mean_squared_error")

# Modeli eğitim verisi üzerinde eğitir, 100 epoch boyunca
ann.fit(x_train, y_train, epochs=100)

# Test verisi üzerinde tahminler yapar
ypred = ann.predict(x_test)

# Gerçek değerleri ekrana yazdırır
for i in y:
    print(i, end="   ")

for i in ypred:
    print(i,end="   ")

    