import tensorflow as tf
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Veri dosyasını okur ve bir DataFrame'e yükler
data = pd.read_csv("C:/Users/ASAF/Desktop/Datasets/Heart_Disease_EDA.csv")
df = data.copy()

# Kullanılmayan sütunları veri setinden çıkarır
df = df.drop(columns=["RestingECG", "ST_Slope", "ChestPainType", "ExerciseAngina"])

# Veri setini erkek ve kadın olarak ikiye ayırır
M = df[df.Sex == "M"]
F = df[df.Sex == "F"]

# Erkek ve kadın veri noktalarını farklı renklerde grafik üzerinde gösterir
plt.scatter(M.Cholesterol, M.MaxHR, label="kotu", color="red", alpha=0.3)
plt.scatter(F.Cholesterol, F.MaxHR, label="iyi", color="green", alpha=0.3)
plt.xlabel("Cholesterol")
plt.ylabel("Max Heart Rate")
plt.legend()
plt.show()

# Cinsiyet sütununu sayısal değerlere çevirir: Erkek = 1, Kadın = 0
df.Sex = [1 if each == "M" else 0 for each in df.Sex]

# Cinsiyet sütununu veri setinden çıkarır
df = df.drop(columns=["Sex"])

# 'HeartDisease' sütununu hedef değişken (y) olarak belirler
y = df["HeartDisease"]

# Geri kalan sütunları özellik değişkenleri (x) olarak belirler
x = df.drop(["HeartDisease"], axis=1)

# Veri setini eğitim ve test seti olarak böler
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=4, train_size=0.7)

# Verileri ölçeklendirmek için Standartlaştırıcı oluşturur
sc = StandardScaler()

# Eğitim ve test verilerini ölçeklendirir
x_train1 = sc.fit_transform(x_train)
x_test1 = sc.transform(x_test)

# Yapay Sinir Ağı (ANN) modeli oluşturur
ann = tf.keras.models.Sequential()
# İlk katman: 6 nöronlu, ReLU aktivasyonlu
ann.add(tf.keras.layers.Dense(1, activation="relu", input_dim=6))
# İkinci katman: 6 nöronlu, ReLU aktivasyonlu
ann.add(tf.keras.layers.Dense(1, activation="relu"))
# Çıktı katmanı: 1 nöronlu, Sigmoid aktivasyonlu
ann.add(tf.keras.layers.Dense(1, activation="sigmoid"))

# Modeli derlerken optimizer, kayıp fonksiyonu ve değerlendirme metriği olarak accuracy kullanır
ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Modeli eğitim verileri üzerinde eğitir
ann.fit(x_train1, y_train, epochs=100)

# Modelin test verileri üzerindeki tahminlerini yapar
ypred = ann.predict(x_test1)

# Sürekli tahminleri (0-1 arası değerler) ikili sınıflandırma değerlerine dönüştürür
ypred_binary = (ypred > 0.5).astype(int)

# Confusion matrix'i oluşturur ve yazdırır
cm = confusion_matrix(y_test, ypred_binary)
print("Confusion Matrix:")
print(cm)

# Modelin doğruluğunu hesaplar ve yazdırır
acs = accuracy_score(y_test, ypred_binary)
print("Accuracy Score:", acs)    
