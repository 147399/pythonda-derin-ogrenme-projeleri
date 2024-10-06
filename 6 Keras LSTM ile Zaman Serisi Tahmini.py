import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

# Veri setini CSV dosyasından okuyoruz
data = pd.read_csv("C:/Users/ASAF/Desktop/Datasets/AirPassengers.csv")
df = data.copy()  # Orijinal veriyi değiştirmemek için bir kopyasını alıyoruz

# Veriyi ekrana yazdırıyoruz, böylece verinin içeriğini görebiliyoruz
print(df)

# Veri çerçevesindeki "#Passengers" sütununu seçip, sadece bu sütunu kullanacağız
df = df["#Passengers"]

# Veriyi NumPy dizisine dönüştürüp, her bir yolcu sayısını ayrı bir satırda gösteriyoruz
df = np.array(df).reshape(-1, 1)

# Yolcu sayısının zaman içindeki değişimini görselleştiriyoruz
plt.plot(df)
plt.show()

# MinMaxScaler nesnesini oluşturup veriyi 0 ile 1 arasına ölçeklendiriyoruz
sc = MinMaxScaler()
df = sc.fit_transform(df)

# Veriyi eğitim ve test setleri olarak ikiye ayırıyoruz
train = df[0:100, :]  # İlk 100 gözlem eğitim seti
test = df[100:, :]    # Geri kalan gözlemler test seti

# LSTM modeli için veriyi hazırlayan bir fonksiyon tanımlıyoruz
def veri(df, steps):
    dfx = []
    dfy = []
    for i in range(len(df) - steps - 1):
        # Her iterasyonda, belirli bir aralığın verisini alıyoruz
        a = df[i:(i + steps), 0]
        dfx.append(a)
        # Sonraki adımın gerçek değerini, hedef değişken olarak alıyoruz
        dfy.append(df[i + steps, 0])
    # Listeyi NumPy dizisine çevirip geri döndürüyoruz
    return np.array(dfx), np.array(dfy)

steps = 3  # Kaç adım geriye gidileceğini belirtiyoruz

# Eğitim ve test verilerini, LSTM modeline uygun şekilde hazırlıyoruz
x_train, y_train = veri(train, steps)
x_test, y_test = veri(test, steps)

# LSTM modeline uygun hale getirmek için veriyi yeniden şekillendiriyoruz
x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

# LSTM modeli oluşturuyoruz
model = Sequential()
# Giriş katmanı olarak LSTM katmanı ekliyoruz, gizli birim sayısı 128
model.add(LSTM(128, input_shape=(1, steps)))
# Birinci yoğun katman (Dense layer), 64 nöronlu
model.add(Dense(64))
# Çıkış katmanı, 1 nöronlu (tek bir değer tahmin ediyor)
model.add(Dense(1))
# Modeli derliyoruz, kayıp fonksiyonu olarak ortalama karesel hata ve optimizasyon algoritması olarak Adam kullanıyoruz
model.compile(loss="mean_squared_error", optimizer="adam")
model.summary()  # Modelin özetini ekrana yazdırıyoruz

# Modeli eğitiyoruz
model.fit(x_train, y_train, epochs=23, batch_size=1)

# Test verisi üzerinde tahminler yapıyoruz
ypred = model.predict(x_test)

# Tahmin edilen değerleri orijinal ölçeğe geri çeviriyoruz
ypred = sc.inverse_transform(ypred)
y_test = y_test.reshape(-1, 1)
y_test = sc.inverse_transform(y_test)

# Gerçek ve tahmin edilen yolcu sayıları arasındaki farkı görselleştiriyoruz
plt.plot(y_test, label="Gerçek Yolcu Sayısı")
plt.plot(ypred, label="Tahmin Edilen Yolcu Sayısı")
plt.xlabel("Hafta")
plt.ylabel("Yolcu Sayısı")
plt.legend()
plt.show()