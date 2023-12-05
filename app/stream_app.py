# Libraries
import streamlit as st
from inference import classify_new_text

# Page title
st.title("Text classification playground")

# User input for new text
new_text = st.text_area("Enter text for classification:", "Your new text here.")

# Button to trigger classification
if st.button("Classify"):
    # Load the latest model
    output_dir = "./results/"  # Adjust path accordingly
    id2label = {
        0: "MLB-FACIAL_SKIN_CARE_PRODUCTS",
        1: "MLB-MAKEUP",
        2: "MLB-BEAUTY_AND_PERSONAL_CARE_SUPPLIES",
    }  # Map class indices to labels
    predicted_label = classify_new_text(output_dir, new_text, id2label)

    # Display the result
    st.success(f"The predicted class label is: {predicted_label}")
