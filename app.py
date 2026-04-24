import streamlit as st
import numpy as np
from PIL import Image
import os

MODEL_PATH = "dogs_vs_cats_cnn.h5"
DRIVE_LINK = "https://drive.google.com/drive/folders/1UfsfkCsKCdfFHnJi0NeT1HvfqQYA5FAP"


@st.cache_resource
def load_model():
    from tensorflow.keras.models import load_model as keras_load_model
    return keras_load_model(MODEL_PATH)


def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((128, 128))
    img_array = np.array(img_resized, dtype="float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 128, 128, 3)
    return img, img_array


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cats vs Dogs — CNN Classifier",
    page_icon="🐾",
    layout="centered",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("ℹ️ Hakkında")
    st.markdown(
        """
        Bu uygulama, **Keras/TensorFlow** ile eğitilmiş bir
        **Evrişimli Sinir Ağı (CNN)** modeli kullanarak yüklediğiniz
        fotoğrafın **kedi mi 🐱 yoksa köpek mi 🐶** olduğunu tahmin eder.

        ---
        **Model Detayları**
        - Giriş boyutu: `128 × 128 × 3`
        - Mimari: 4 × Conv blok + Dense katmanlar
        - Aktivasyon: **sigmoid** (ikili sınıflandırma)
        - Model dosyası: `dogs_vs_cats_cnn.h5`

        ---
        **Model dosyasını indirmek için:**
        [Google Drive →](%s)
        """ % DRIVE_LINK
    )

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🐱 Cats vs 🐶 Dogs — CNN Classifier")
st.write("Bir kedi veya köpek fotoğrafı yükleyin, model sizin için tahmin etsin!")

# Check model file
model = None
if not os.path.exists(MODEL_PATH):
    st.error(
        f"⚠️ **Model dosyası bulunamadı!**\n\n"
        f"`{MODEL_PATH}` dosyası proje kök dizininde mevcut değil.\n\n"
        f"Lütfen modeli aşağıdaki Drive bağlantısından indirip "
        f"repo kök dizinine koyun:\n\n"
        f"👉 [Google Drive — Model İndir]({DRIVE_LINK})"
    )
else:
    with st.spinner("Model yükleniyor…"):
        model = load_model()

# File uploader
uploaded_file = st.file_uploader(
    "Fotoğraf yükleyin (JPG / JPEG / PNG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    original_img, img_array = preprocess_image(uploaded_file)

    st.image(original_img, caption="Yüklenen Görsel", use_container_width=False)

    if st.button("Tahmin Et 🔍", disabled=(model is None)):
        with st.spinner("Tahmin yapılıyor…"):
            pred = model.predict(img_array)[0][0]

        if pred < 0.5:
            label = "🐱 Kedi"
            confidence = 1.0 - float(pred)
            color = "blue"
        else:
            label = "🐶 Köpek"
            confidence = float(pred)
            color = "orange"

        st.markdown(
            f"<h2 style='text-align:center; color:{color};'>{label}</h2>",
            unsafe_allow_html=True,
        )
        st.write(f"**Güven (Confidence):** {confidence * 100:.1f}%")
        st.progress(confidence)
