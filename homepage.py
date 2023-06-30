import streamlit as st
from io import StringIO
from PIL import Image
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

def display_results_text(results_df, input_form):
    if results_df.empty:
        st.error("Sorry the model is still development ☹️")
    else:
        st.success("Aspect Found!")
        st.subheader('Aspect-Based Sentiment Analysis Results')
        st.markdown(f"Sentence : {input_form}")
        st.dataframe(results_df, hide_index=True)

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


def main():
    st.title('Aspect-Based Sentiment Analysis App')
    st.write('Enter your text below:')
    input_text = st.text_area('Input Text')

    if st.button('Analyze Text'):
        results_df = analyze_text(input_text)
        display_results_text(results_df, input_text)

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
