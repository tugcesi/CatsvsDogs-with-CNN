# 🐱 Cats vs Dogs — CNN Classifier

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?logo=streamlit)](https://streamlit.io/)

> Derin öğrenme ile kedi 🐱 ve köpek 🐶 fotoğraflarını sınıflandıran bir CNN modeli ve etkileşimli Streamlit arayüzü.

---

## 📖 Proje Özeti

Bu proje, **Kaggle Dogs vs Cats** veri seti üzerinde **Evrişimli Sinir Ağı (CNN)** kullanılarak ikili görüntü sınıflandırması yapmaktadır. Notebook'ta ayrıca transfer learning yöntemiyle **VGG16** tabanlı bir model de denenmiştir.

Eğitilen model, **Streamlit** tabanlı bir web arayüzü aracılığıyla gerçek zamanlı tahmin yapabilmektedir.

---

## 🏗️ Model Mimarisi

| Katman | Detay |
|--------|-------|
| Giriş | `128 × 128 × 3` (RGB) |
| Conv Blok 1 | Conv2D(32) → MaxPool |
| Conv Blok 2 | Conv2D(64) → MaxPool |
| Conv Blok 3 | Conv2D(128) → MaxPool |
| Conv Blok 4 | Conv2D(128) → MaxPool |
| Dense | Flatten → Dense(512) → Dropout |
| Çıkış | Dense(1, sigmoid) |

- **Kayıp fonksiyonu:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Aktivasyon (çıkış):** Sigmoid — `< 0.5` → Kedi, `≥ 0.5` → Köpek

---

## 🚀 Kurulum ve Çalıştırma

### 1. Repoyu klonla
```bash
git clone https://github.com/tugcesi/CatsvsDogs-with-CNN.git
cd CatsvsDogs-with-CNN
```

### 2. Bağımlılıkları yükle
```bash
pip install -r requirements.txt
```

### 3. Modeli indir
Eğitilmiş model dosyası (`dogs_vs_cats_cnn.h5`) büyük boyutu nedeniyle repoya eklenmemiştir.  
Aşağıdaki Google Drive bağlantısından indirip **repo kök dizinine** koyun:

📥 **[Google Drive — dogs_vs_cats_cnn.h5](https://drive.google.com/drive/folders/1UfsfkCsKCdfFHnJi0NeT1HvfqQYA5FAP)**

### 4. Uygulamayı başlat
```bash
streamlit run app.py
```

---

## 🖥️ Uygulama Ekran Görüntüsü

> *(Uygulamayı çalıştırdıktan sonra buraya ekran görüntüsü eklenebilir.)*

---

## 🛠️ Kullanılan Teknolojiler

| Teknoloji | Kullanım Amacı |
|-----------|---------------|
| **Python 3.9+** | Ana programlama dili |
| **TensorFlow / Keras** | CNN model eğitimi ve tahmin |
| **Streamlit** | İnteraktif web arayüzü |
| **Pillow (PIL)** | Görsel ön işleme |
| **NumPy** | Dizi işlemleri |

---

## 📓 Notebook Hakkında

`dogs-vs-cats-with-cnn-and-transfer-learning.ipynb` dosyası:

- **Veri Seti:** Kaggle Dogs vs Cats (25.000 görüntü)
- **Yaklaşım 1:** Sıfırdan eğitilen özel CNN modeli
- **Yaklaşım 2:** Transfer Learning — **VGG16** ön-eğitimli ağırlıklar ile ince ayar (fine-tuning)
- Veri artırma (Data Augmentation), erken durdurma (Early Stopping) gibi teknikler uygulanmıştır.

---

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) altında lisanslanmıştır.
