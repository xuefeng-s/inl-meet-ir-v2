
import re
import sklearn
import string
import nltk
import transformers as ppb
import pandas as pd
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.pipeline import Pipeline
from transformers import pipeline
from datetime import datetime
from typing import List
import joblib
import re

nltk.download('stopwords')


print(pipeline('sentiment-analysis')('we love you'))



# Man kann auch mehrere hintereinander machen

class TextToSentenceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_to_transform, new_column_name):
        self.column_to_transform = column_to_transform
        self.new_column_name = new_column_name

    def set_column_to_transform(self, column_to_transform):
        self.column_to_transform = column_to_transform

    def set_new_column_name(self, new_column_name):
        self.new_column_name = new_column_name

    def fit(self, X, y=None):
        return self

    def transform(self, data):
        if self.column_to_transform in data.columns:
            # data[self.new_column_name] = self.split_text_in_sentences(data)
            # return data
            return pd.DataFrame({self.new_column_name: self.split_text_in_sentences(data)})
        else:
            self.log_error(
                f'no column with name {self.column_to_transform} in dataframe.\nGiving back original dataframe.')
            return data

    def split_text_in_sentences(self, data) -> List[str]:
        texts = data[self.column_to_transform].tolist()
        # delimiter = ['.', '\?', '!']
        delimiter = '[?.!]'
        sentences = list()
        for text in texts:
            # sentences_in_text = [e + delimiter for e in text.split(delimiter) if e]
            sentences_in_text = [e for e in re.split(delimiter, text) if e]
            sentences += sentences_in_text
        return sentences

    def log_error(self, description, filename='text_to_sentence_transformer.error'):
        with open(filename, 'a', encoding='utf-8') as file:
            file.write('#------------------------------------------------------------------------------------------\n')
            file.write(f'{datetime.now().strftime("%b-%d-%Y %H:%M:%S")}\n')
            file.write(f'\t{description}\n')
            file.write(
                '#------------------------------------------------------------------------------------------\n\n\n')


class BertTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, column):
        self.column = column
        model_class, tokenizer_class, pretrained_weights = (
            ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

        # Load pretrained model/tokenizer

        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.model = model_class.from_pretrained(pretrained_weights)

    def set_column(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, data):
        # Sätze zerstückeln lassen
        dataList = data[self.column].tolist()
        dataList = list((str(s) for s in dataList))
        tokenized = []

        for s in dataList:
            tokenized.append(self.tokenizer.encode(s, add_special_tokens=True))

        # Padding hinzufügen
        max_len = 0
        for i in tokenized:
            if len(i) > max_len:
                max_len = len(i)

        padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized])

        # Maske erstellen, um das Padding bei der Verarbeitung zu filtern
        mask = np.where(padded != 0, 1, 0)
        mask.shape

        # mache padded Array und Maske zu einem Tensor
        # Tensor = mehrdimensionale Matrix mit einheitlichem Datentyp
        input = torch.tensor(padded).to(torch.long).long()
        mask = torch.tensor(mask).to(torch.long).long()

        # gib unser Zeug an BERT
        # no_grad = Angabe zur Simplifikation des Rechenvorgangs
        with torch.no_grad():
            output = self.model(input, attention_mask=mask)

        # nur die erste Spalte auslesen = von BERT geschriebene Kennwerte
        features = output[0][:, 0, :].numpy()

        return (data, features)


class PreprocessorTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, column):
        self.column = column

    def set_column(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, data_and_features, y=None):
        data, features = data_and_features
        sentences = data[self.column].tolist()
        sentences = list((str(s) for s in sentences))

        # muss vom generator object zurück zur liste gemacht werden
        sentences = list((s.lower() for s in sentences))

        table = str.maketrans('', '', string.punctuation)
        sentences = [s.translate(table) for s in sentences]

        sentences = [re.sub(r'\d+', 'num', s) for s in sentences]

        stopwords = set(nltk.corpus.stopwords.words('english'))
        sentences = [[word for word in s.split() if word not in stopwords] for s in sentences]
        return (sentences, features)


class SentimentOpinionValueCalculatorSingleValueTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, dict_name):
        self.dict_name = dict_name
        df = pd.read_csv(dict_name, sep=';')
        self.value_dict = pd.Series(df.value.values, index=df.word).to_dict()

    def fit(self, X, y=None):
        return self

    def transform(self, sentences_and_features):
        sentiment_opinion_scores = []
        sentences, features = sentences_and_features
        for sentence in sentences:
            word_count = len(sentence)
            # print(f'length of sentence {sentence} = {word_count}')
            sentiment_opinion_score = 0
            if word_count > 0:
                for word in sentence:
                    if word in self.value_dict:
                        sentiment_opinion_score = sentiment_opinion_score + self.value_dict[word]
                sentiment_opinion_score = sentiment_opinion_score / word_count
            sentiment_opinion_scores.append([sentiment_opinion_score])
        for i in range(len(sentiment_opinion_scores)):
            features[i] = features[i] + (sentiment_opinion_scores[i][0])
        return features


class SentimentOpinionValueCounterTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, value_file_name):
        df = pd.read_csv(value_file_name, sep=';')
        self.value_dict = pd.Series(df.value.values, index=df.word).to_dict()

    def fit(self, X, y=None):
        return self

    def transform(self, sentences_and_features):
        sentiment_opinion_scores = []
        sentences, features = sentences_and_features
        for sentence in sentences:
            word_count = len(sentence)
            # print(f'length of sentence {sentence} = {word_count}')
            sentiment_opinion_score = 0
            if word_count > 0:
                for word in sentence:
                    if word in self.value_dict:
                        sentiment_opinion_score += 1
                # sentiment_opinion_score = sentiment_opinion_score / word_count
            sentiment_opinion_scores.append([sentiment_opinion_score])
        for i in range(len(sentiment_opinion_scores)):
            features[i] = features[i] + (sentiment_opinion_scores[i][0])
        return features


class PipelineRunner:

    def __init__(self, dict_file, training_file, test_file, log_file='results_with_correct_input.log'):
        self.dict_file = dict_file
        self.log_file = log_file
        self.data_training = pd.read_excel(training_file, sheet_name='sentences')
        self.data_training.drop(
            ['SUBJindl', 'SUBJsrce', 'SUBJrhet', 'SUBJster', 'SUBJspee', 'SUBJinspe', 'SUBJprop', 'SUBJpolit'],
            axis=1,
            inplace=True)
        self.data_test = pd.read_excel(test_file, sheet_name='sentences')
        self.data_test.drop(
            ['SUBJindl', 'SUBJsrce', 'SUBJrhet', 'SUBJster', 'SUBJspee', 'SUBJinspe', 'SUBJprop', 'SUBJpolit'],
            axis=1,
            inplace=True)
        self.Cs = np.logspace(-6, 6, 200)
        self.pipeline_to_use = None
        self.clf = None
        self.text_to_sentence_transformer = None
        self.bert = None
        self.preprocessor = None

    def start_all_pipelines(self, data_column):
        self.bert = BertTransformer('Sentence')
        self.preprocessor = PreprocessorTransformer('Sentence')
        self.text_to_sentence_transformer = TextToSentenceTransformer('text', 'Sentence')
        transformer_list = [self.text_to_sentence_transformer,
                            self.bert,
                            self.preprocessor,
                            SentimentOpinionValueCalculatorSingleValueTransformer(self.dict_file)]

        description = f'Bert und Sentiment Durchschnittswert (mit langem Dictonary). Spalte {data_column}'

        print('Starting with Logistic Regression')

        self.pipeline_to_use = self.make_pipeline(transformer_list, LogisticRegression(max_iter=500), dict(C=self.Cs))
        log_reg_typ = "Logistic Regression"
        accuracy = self.fit_and_predict_and_calculate_accuracy_pipe(self.pipeline_to_use, data_column)
        self.write_result_to_file(accuracy, log_reg_typ, description)

        # print('Starting with Gaussian Naive Bayes')
        #
        # gau_nb_typ = "Gaussian Naive Bayes"
        #
        # pipeline_to_use = self.make_pipeline(transformer_list, GaussianNB(), dict(var_smoothing=self.Cs))
        # accuracy = self.fit_and_predict_and_calculate_accuracy_pipe(pipeline_to_use, data_column)
        # self.write_result_to_file(accuracy, gau_nb_typ, description)
        #
        #
        # print('Starting with Bernoulli Naive Bayes')
        #
        # bernoulli_nb_typ = "Bernoulli Naive Bayes"
        #
        # pipeline_to_use = self.make_pipeline(transformer_list, BernoulliNB(), dict(alpha=self.Cs, binarize=self.Cs))
        # accuracy = self.fit_and_predict_and_calculate_accuracy_pipe(pipeline_to_use, data_column)
        # self.write_result_to_file(accuracy, bernoulli_nb_typ, description)

    def make_pipeline(self, transformer_list, estimator, param_gird):
        self.clf = GridSearchCV(estimator=estimator, param_grid=param_gird, n_jobs=-1, scoring='accuracy')

        return sklearn.pipeline.Pipeline(
            [(f'stage: {index}', transformer_list[index]) for index in range(len(transformer_list))] + [
                ('clf', self.clf)]
        )

    def fit_and_predict_and_calculate_accuracy_pipe(self, pipe, data_column):
        pipe.fit(self.data_training, self.data_training[data_column].to_numpy())

        y_pred_pipe = pipe.predict(self.data_test)

        return accuracy_score(self.data_test[data_column].to_numpy(), y_pred_pipe)

    def write_result_to_file(self, accuracy, type, description):
        with open(self.log_file, 'a', encoding='utf-8') as file:
            file.write('#------------------------------------------------------------------------------------------\n')
            file.write(f'{datetime.now().strftime("%b-%d-%Y %H:%M:%S")}\n')
            file.write(f'\t{description}\n')
            file.write(f'\t\tAccuracy for classifier {type}: {accuracy}\n')
            file.write('#------------------------------------------------------------------------------------------\n')

    def predict_data(self, data_file_name, column_to_transform=None, new_column_name=None, result_for_column=''):
        data_validation = pd.read_csv(data_file_name)

        if column_to_transform is not None:
            self.text_to_sentence_transformer.set_column_to_transform(column_to_transform)

        if new_column_name is not None:
            self.text_to_sentence_transformer.set_new_column_name(new_column_name)
            self.bert.set_column(new_column_name)
            self.preprocessor.set_column(new_column_name)
        # bert = BertTransformer('text')
        # preprocessor = PreprocessorTransformer('text')
        # transformer_list = [bert,
        #                     preprocessor,
        #                     SentimentOpinionValueCalculatorSingleValueTransformer(dict_file)]
        #
        # pipe = sklearn.pipeline.Pipeline(
        #     [(f'stage: {index}', transformer_list[index]) for index in range(len(transformer_list))] + [
        #         ('clf', self.clf)]
        # )
        with open('mbfc_results.log', 'a', encoding='utf-8') as f:
            f.write('#-------------------------------------------------')
            f.write(f'result for column {result_for_column}:\n')

        row_count = data_validation.shape[0]
        batch_size = 15
        counter = 0

        while counter * batch_size < row_count:

            output = self.pipeline_to_use.predict(data_validation.loc[counter * batch_size : (counter + 1) * batch_size])
            with open('mbfc_results.log', 'a', encoding='utf-8') as f:
                # f.write('#-------------------------------------------------')
                # f.write(f'result for column {result_for_column}:\n')
                f.write(f'{output.tolist()}')
                f.write('\n')
            counter += 1


        with open('mbfc_results.log', 'a', encoding='utf-8') as f:
            # f.write('#-------------------------------------------------')
            # f.write(f'result for column {result_for_column}:\n')
            # f.write(output)
            f.write('#-------------------------------------------------\n\n\n\n\n')


def fit_and_predict_and_calculate_accuracy_pipe(pipe, train_input, train_ouput, test_input, test_output):
    pipe.fit(train_input, train_ouput)

    y_pred_pipe = pipe.predict(test_input)

    return accuracy_score(test_output, y_pred_pipe)


def write_result_to_file(accuracy, type, description):
    result_file = 'results_with_correct_input.log'
    with open(result_file, 'a', encoding='utf-8') as file:
        file.write('#------------------------------------------------------------------------------------------\n')
        file.write(f'{datetime.now().strftime("%b-%d-%Y %H:%M:%S")}\n')
        file.write(f'\t{description}\n')
        file.write(f'\t\tAccuracy for classifier {type}: {accuracy}\n')
        file.write('#------------------------------------------------------------------------------------------\n')


if __name__ == '__main__':
    dir_path = ''
    # dict_file = dir_path + 'AFINN-both-abs.csv'
    dict_file = dir_path + 'sentiment_dict.csv'

    training_file = dir_path + 'Trainingdata_train.xlsx'

    test_file = dir_path + 'Trainingdata_test.xlsx'
    # print(pd.read_csv('MBFC_Dataset_Sample.csv').shape[0])
    # print(TextToSentenceTransformer('text', 'Sentence').transform(pd.read_csv('MBFC_Dataset_Sample.csv')))

    pipeline_runner = PipelineRunner(dict_file, training_file, test_file)
    pipeline_runner.start_all_pipelines('SUBJopin01')
    pipeline_runner.predict_data(data_file_name='MBFC_Dataset_Sample.csv', result_for_column='SUBJopin')
    pipeline_runner.start_all_pipelines('SUBJlang01')
    pipeline_runner.predict_data(data_file_name='MBFC_Dataset_Sample.csv', result_for_column='SUBJlang')
