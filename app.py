
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Diabetic Retinopathy Screening",
    page_icon="👁️",
    layout="centered"
)

st.title("👁️ AI Diabetic Retinopathy Screening")
st.write("Upload a retinal fundus image for AI-based screening.")

# -----------------------------
# Class Names
# -----------------------------
CLASS_NAMES = [
    "No Diabetic Retinopathy",
    "Mild Diabetic Retinopathy",
    "Moderate Diabetic Retinopathy",
    "Severe Diabetic Retinopathy",
    "Proliferative Diabetic Retinopathy"
]

IMG_SIZE = 224

# -----------------------------
# Focal Loss
# -----------------------------
def focal_loss(gamma=2.0, alpha=0.25):

    def loss(y_true, y_pred):

        y_pred = tf.clip_by_value(
            y_pred,
            tf.keras.backend.epsilon(),
            1.0 - tf.keras.backend.epsilon()
        )

        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)

        return tf.reduce_sum(
            weight * cross_entropy,
            axis=-1
        )

    return loss


# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "best_focal_model.keras",
        custom_objects={"loss": focal_loss()}
    )


model = load_model()


# -----------------------------
# Find Last Conv Layer
# -----------------------------
backbone = model.layers[0]

last_conv_layer = None

for layer in reversed(backbone.layers):

    try:
        if len(layer.output.shape) == 4:
            last_conv_layer = layer
            break
    except:
        pass


# -----------------------------
# Grad-CAM
# -----------------------------
def make_gradcam_heatmap(img_array, model, backbone, last_conv_layer):

    # Feature model
    feature_model = tf.keras.models.Model(
        inputs=backbone.input,
        outputs=[
            last_conv_layer.output,
            backbone.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, backbone_output = feature_model(
            img_array,
            training=False
        )

        # Pass backbone output through classifier layers
        x = backbone_output

        for layer in model.layers[1:]:
            x = layer(x, training=False)

        predictions = x

        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(
        class_channel,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)

    heatmap /= tf.maximum(
        tf.reduce_max(heatmap),
        1e-8
    )

    return heatmap.numpy()


# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Retinal Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Retinal Image",
        use_container_width=True
    )

    if st.button("🔍 Analyze Image"):

        # Resize
        resized_image = image.resize(
            (IMG_SIZE, IMG_SIZE)
        )

        img_array = np.array(
            resized_image
        ).astype("float32")

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        # -----------------------------
        # Prediction
        # -----------------------------
        predictions = model.predict(
            img_array,
            verbose=0
        )[0]

        predicted_class = int(
            np.argmax(predictions)
        )

        confidence = float(
            predictions[predicted_class] * 100
        )

        # -----------------------------
        # Prediction Result
        # -----------------------------
        st.success(
            f"Prediction: {CLASS_NAMES[predicted_class]}"
        )

        st.subheader("Confidence")

        st.write(
            f"### {confidence:.2f}%"
        )

        # -----------------------------
        # Class Probabilities
        # -----------------------------
        st.subheader("Class Probabilities")

        for i, class_name in enumerate(CLASS_NAMES):

            st.write(
                f"**{class_name}: {predictions[i] * 100:.2f}%**"
            )

        # -----------------------------
        # Grad-CAM
        # -----------------------------
        st.subheader("🔥 Explainable AI - Grad-CAM")

        try:

            heatmap = make_gradcam_heatmap(
                img_array,
                model,
                backbone,
                last_conv_layer
            )

            # Resize heatmap
            heatmap_resized = tf.image.resize(
                heatmap[..., np.newaxis],
                (image.height, image.width)
            ).numpy().squeeze()

            # Create heatmap
            cmap = plt.get_cmap("jet")

            heatmap_color = cmap(
                heatmap_resized
            )[:, :, :3]

            heatmap_color = (
                heatmap_color * 255
            ).astype(np.uint8)

            heatmap_image = Image.fromarray(
                heatmap_color
            )

            # Overlay
            overlay = Image.blend(
                image.convert("RGBA"),
                heatmap_image.convert("RGBA"),
                alpha=0.45
            )

            st.image(
                overlay,
                caption="Grad-CAM: Areas influencing the AI prediction",
                use_container_width=True
            )

        except Exception as e:

            st.warning(
                f"Grad-CAM could not be generated: {e}"
            )

        # -----------------------------
        # Medical Disclaimer
        # -----------------------------
        st.warning(
            "⚠️ This AI system is intended for screening "
            "support and should not replace professional "
            "medical diagnosis."
        )
