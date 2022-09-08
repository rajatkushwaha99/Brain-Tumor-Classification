import numpy as np
import keras
from PIL import Image, ImageOps


import streamlit as st
st.write("""
         # Brain Tumor Classification
         """
         )
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])



def teachable_machine_classification(img, model3):
    # Load the model
    model = keras.models.load_model(model3)

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 128, 128, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (128, 128)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 255.0)

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    return np.argmax(prediction) # return position of the highest probability




if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=False)

    prediction = teachable_machine_classification(image, 'brain_tumor_classifier.hdf5')

    if prediction == 0:
        st.write("Glioma")
    elif prediction == 1:
        st.write("Meningiome")
    elif prediction == 2:
        st.write("No Tumor")
    elif prediction == 3:
        st.write("Pituitary")