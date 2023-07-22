import streamlit as st
from io import StringIO
from PIL import Image
import random
import json
from predict_atepc import Predict_ATEPC

def analyze_text(input_text_form):
    ATEPC = Predict_ATEPC()
    ATEPC.predict_single_text(input_text_form)
    ATEPC.list_into_dataframe()
    return ATEPC.dataframe_predict

def analyze_file(input_file_form):
    ATEPC = Predict_ATEPC()
    ATEPC.predict_file_text(input_file_form)
    ATEPC.list_into_dataframe_texts()
    ATEPC.list_into_dataframe()
    ATEPC.make_word_cloud()
    ATEPC.make_bar_chart(type_of_sentiments='positive', palette_color='mako')
    ATEPC.make_bar_chart(type_of_sentiments='negative', palette_color='rocket')
    display_results_file(ATEPC.dataframe_predict_texts)

def display_results_text(results_df, input_form, selected_image):
    if results_df is None or results_df.empty:
        st.error("Sorry the model is still in development ☹️")
    else:
        st.success("Aspect Found!")
        st.subheader('Aspect-Based Sentiment Analysis Results')
        st.markdown(f"Sentence: {input_form}")
        st.dataframe(results_df, hide_index=True)

    if selected_image is not None:
        image_location = Image.open(f'photo/{selected_image}.jpg')
        st.image(image_location, caption=f'{selected_image}')

def display_results_file(dataframe):
    st.success("Aspect Found!")
    st.subheader('Aspect-Based Sentiment Analysis Results')
    st.dataframe(dataframe, hide_index=True)
    image_wordcloud = Image.open('output_file/wordcloud.png')
    image_barchart_positive = Image.open('output_file/bar_chart_positive.png')
    image_barchart_negative = Image.open('output_file/bar_chart_negative.png')
    st.image(image_wordcloud, caption='Wordclouds')
    st.image(image_barchart_positive, caption='Top Aspek Positif')
    st.image(image_barchart_negative, caption='Top Aspek Negatif')

def choose_random_sentence(input_choose):
    with open('datasets/top 10 destination aceh.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    # Extract the relevant comments
    extracted_comment = []
    for item in data:
        place_name = item['place_name']
        if place_name == f'{input_choose}':
            comments = item['comment']
            extracted_comment.extend(comments)
    sentence = random.choice(extracted_comment)
    sentence = sentence.strip()
    return sentence
def clean_input():
    # Code to clean input text
    st.text_input("Enter text here", value="")


def main():
    st.set_page_config(
        page_title="ABSA Aceh Tourism App" ,
        page_icon="🕌",
        initial_sidebar_state="expanded",
        menu_items={
            'About': 'https://www.extremelycoolapp.com/help',
        }
    )
    header_image = Image.open('photo/header_image.jpg')
    st.image(header_image)
    st.title('Aspect-Based Sentiment Analysis for Aceh Tourism')

    st.write('Enter your text by yourself:')
    input_text_area = st.text_area('Input Text')
    st.write('Enter a text-based review on the ten most visited tourism places in Aceh:')
    selected_text = st.selectbox(
        '',
        ('Monument 0 km Indonesia', 'Aceh State Museum', 'Boat at Desa Lampulo',
         'Baiturrahman Grand Mosque', 'Rubiah Island','Mie Razali', 'Freddies Santai Sumurtiga',
         'Mahi Mahi Surf Resort', 'Iboih Beach', 'Aceh Tsunami Museum'), label_visibility="collapsed")

    st.write('You selected:', selected_text)

    # Radio Button
    if selected_text:
        input_text_radio = choose_random_sentence(selected_text)

    with st.spinner('Please Wait...'):
        if st.button('Analyze Text'):
            if len(input_text_area) > 0:
                selected_text = None
                results_df = analyze_text(input_text_area)
                display_results_text(results_df, input_text_area, selected_text)

            else:
                results_df = analyze_text(input_text_radio)
                display_results_text(results_df, input_text_radio, selected_text)

    st.write('---')
    st.write('Upload a file for analysis:')
    uploaded_file = st.file_uploader('Choose a file', type=['txt', 'csv'])

    if uploaded_file is not None:
        if st.button('Analyze File'):
            # To convert to a string based IO:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            string_data = stringio.read()
            sentences = string_data.split('\n')
            analyze_file(sentences)

if __name__ == '__main__':
    main()
