import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from pyabsa import AspectTermExtraction as ATEPC

class Predict_ATEPC:
    def __init__(self):
        self.results_predict = []
        self.dataframe_predict = None
        self.dataframe_predict_texts = None
        self.aspect_extractor = ATEPC.AspectExtractor(checkpoint="A2_I_Max seq len_200")

    def predict_single_text(self, input_text):
        result = self.aspect_extractor.predict(
            text=input_text,
            print_result=False,
            save_result=False,
            ignore_error=True,
            eval_batch_size=32,
        )
        self.results_predict.append(result)

    def predict_file_text(self, input_file):
        for sentence in input_file:
            result = self.aspect_extractor.predict(
                text=sentence,
                print_result=False,
                save_result=False,
                ignore_error=True,
                eval_batch_size=32,
            )
            self.results_predict.append(result)

    def list_into_dataframe(self):
        self.dataframe_predict = pd.DataFrame(self.results_predict)
        self.dataframe_predict.drop(['IOB', 'tokens', 'position', 'probs', 'sentence'], axis=1, inplace=True)
        self.dataframe_predict = self.dataframe_predict.explode(['aspect', 'sentiment', 'confidence']).reset_index(drop=True)
        if type(self.dataframe_predict['aspect'][0]) != str:
            self.dataframe_predict = None

    def list_into_dataframe_texts(self):
        self.dataframe_predict_texts = pd.DataFrame(self.results_predict)
        self.dataframe_predict_texts.drop(['IOB', 'tokens', 'position', 'probs'], axis=1, inplace=True)

    def make_word_cloud(self):
        self.clean_text_df(self.dataframe_predict, 'aspect')
        text = ' '.join(self.dataframe_predict['aspect'].astype(str))
        # Create a WordCloud object
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

        # Display the word cloud using matplotlib
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')

        # Save the word cloud visualization as an image file
        wordcloud_image_path = 'output_file/wordcloud.png'  # Replace with your desired file path
        plt.savefig(wordcloud_image_path)

    def make_bar_chart(self, type_of_sentiments, palette_color):
        new_df = self.dataframe_predict[(self.dataframe_predict['sentiment'] == f'{type_of_sentiments}')]
        self.clean_text_df(new_df, 'aspect')
        # Count the frequency of each aspect
        aspect_counts = new_df['aspect'].value_counts().head(20)

        # Create a bar chart using seaborn with the specified color
        plt.figure(figsize=(10, 6))
        sns.barplot(x=aspect_counts.index, y=aspect_counts.values, palette=f'{palette_color}')
        plt.xlabel('Aspect')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the bar chart visualization as an image file
        bar_chart_image_path = f'output_file/bar_chart_{type_of_sentiments}.png'  # Replace with your desired file path
        plt.savefig(bar_chart_image_path)

    def clean_text_df(df, text_field):
        df[text_field] = df[text_field].astype(str)
        df[text_field] = df[text_field].str.lower()
        df[text_field] = df[text_field].apply(lambda elem: re.sub(r"([^a-zA-Z0-9 ]+)", "", elem))
        df = df.dropna(subset=[text_field])  # Remove rows with NaN values in the specified text field
        return df