from pathlib import Path

import joblib
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

# Keep these values exactly the same as in the training notebook.
IMAGE_SIZE = (30, 40)  # PIL uses (width, height)
MODEL_PATH = Path("fashion_ann_model.pkl")
ENCODER_PATH = Path("fashion_label_encoder.pkl")
CLASSES_PATH = Path("fashion_classes.pkl")


st.set_page_config(
    page_title="Fashion Image Classifier (ANN)",
    page_icon="👕",
    layout="centered",
)

@st.cache_resource
def load_artifacts():
    """Load the trained ANN pipeline, label encoder, and class names once."""
    missing = [
        str(path)
        for path in (MODEL_PATH, ENCODER_PATH, CLASSES_PATH)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing trained model file(s): "
            + ", ".join(missing)
            + ". Train the notebook and copy the three saved .pkl files "
              "into the same folder as app.py."
        )

    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    classes = joblib.load(CLASSES_PATH)
    return model, label_encoder, classes


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Match training: RGB, resize to 30x40, flatten, normalize to 0..1."""
    image = image.convert("RGB").resize(IMAGE_SIZE)
    features = np.asarray(image, dtype=np.float32).flatten() / 255.0
    return features.reshape(1, -1)


st.title("👕 Fashion Product Image Classifier")
st.write(
    "Upload a fashion product image to predict its **fashion category** "
    "using a trained Artificial Neural Network (ANN)."
)

st.info(
    "This model predicts categories such as Shirt, Shoes, or Trousers. "
    "It does not identify a unique product name or SKU."
)

try:
    model, label_encoder, classes = load_artifacts()
except (FileNotFoundError, OSError, ValueError) as error:
    st.error(str(error))
    st.stop()

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Predict category", type="primary", use_container_width=True):
            features = preprocess_image(image)
            predicted_encoded = int(model.predict(features)[0])
            predicted_category = str(
                label_encoder.inverse_transform([predicted_encoded])[0]
            )

            st.success(f"Predicted category: **{predicted_category}**")

            # Show confidence only if the trained estimator supports probabilities.
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features)[0]
                best_index = int(np.argmax(probabilities))
                confidence = float(probabilities[best_index])
                st.metric("Model confidence (estimated)", f"{confidence:.1%}")

                ranked = sorted(
                    zip(classes, probabilities),
                    key=lambda item: item[1],
                    reverse=True,
                )[:5]
                st.subheader("Top category scores")
                for class_name, probability in ranked:
                    st.write(f"**{class_name}** — {probability:.1%}")
                    st.progress(float(probability))
            else:
                st.caption(
                    "This model does not provide probability scores; "
                    "only the predicted category is displayed."
                )

    except (UnidentifiedImageError, OSError, ValueError) as error:
        st.error(f"Could not read this image. Please upload a valid JPG or PNG. Details: {error}")

st.divider()
st.caption("Built with Streamlit, scikit-learn, NumPy, Pillow, and joblib.")
