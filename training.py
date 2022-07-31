from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests
import streamlit as st


def classified_image(url):
    image = Image.open(requests.get(url, stream=True).raw)

    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()
    result = model.config.id2label[predicted_class_idx]
    print("Predicted class:", result)
    return result


def run():
    # Streamlit page title
    st.title("Classify your Images here")
    st.markdown('**Vision Transformer (ViT) model pre-trained on ImageNet-21k (14 million images, 21,843 classes) at resolution 224x224, and fine-tuned on ImageNet 2012 (1 million images, 1,000 classes) at resolution 224x224.**')
    hide_footer_style = """<style>.reportview-container .main footer {visibility: hidden;}"""
    st.markdown(hide_footer_style, unsafe_allow_html=True)

    # Clear input form
    def clearform():
        st.session_state['newtext'] = ''

    # Input form
    with st.form('reviewtext'):
        new_review = st.text_area(label='Give your Image URL',
                                    value = '',
                                    key='newtext')
        b1,b2 = st.columns([1,1])
        with b1:
            submit = st.form_submit_button(label='Submit')
        with b2:
            st.form_submit_button(label='Reset', on_click=clearform)

    if submit and new_review !='':
        # Generate prediction
        result = classified_image(new_review)

        # Display the prediction
        st.markdown('### Predicted Class: {}'.format(result))


if __name__ == "__main__":
    run()
