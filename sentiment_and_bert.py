
import re
import sklearn
import string
import nltk
import transformers as ppb
import pandas as pd
import numpy as np
import torch
import joblib
import re
import os
import time
from enum import Enum
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.pipeline import Pipeline
from transformers import pipeline
from datetime import datetime
from typing import List

nltk.download('stopwords')


print(pipeline('sentiment-analysis')('we love you'))


# Man kann auch mehrere hintereinander machen

class ColumnUser:
    def set_column_to_use(self, column_name):
        pass


class ColumnTransformer(ColumnUser):
    def set_column_to_transform(self, column_to_transform):
        pass


class TextToSentenceTransformer(BaseEstimator, TransformerMixin, ColumnTransformer):
    def __init__(self, column_to_transform, new_column_name):
        self.column_to_transform = column_to_transform
        self.new_column_name = new_column_name

    def set_column_to_transform(self, column_to_transform):
        self.column_to_transform = column_to_transform

    def set_column_to_use(self, column_name):
        self.new_column_name = column_name

    def fit(self, X, y=None):
        return self

    def transform(self, data):
        if self.column_to_transform in data.columns:
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

    def log_error(self, description, filename='logs/text_to_sentence_transformer.error'):
        with open(filename, 'a', encoding='utf-8') as file:
            file.write('#------------------------------------------------------------------------------------------\n')
            file.write(f'{datetime.now().strftime("%b-%d-%Y %H:%M:%S")}\n')
            file.write(f'\t{description}\n')
            file.write(
                '#------------------------------------------------------------------------------------------\n\n\n')


class BertTransformer(BaseEstimator, TransformerMixin, ColumnUser):

    def __init__(self, column):
        self.column = column
        model_class, tokenizer_class, pretrained_weights = (
            ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

        # Load pretrained model/tokenizer

        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.model = model_class.from_pretrained(pretrained_weights)

    def set_column_to_use(self, column_name):
        self.column = column_name

    def fit(self, X, y=None):
        return self

    def transform(self, data):
        # Sätze zerstückeln lassen
        dataList = data[self.column].tolist()
        dataList = list((str(s) for s in dataList))
        tokenized = []

        for s in dataList:
            t = self.tokenizer.encode(s, add_special_tokens=True)
            tokenized.append(t)
            if len(t) > 500:
                print('wtf oO')
                print(s)
                print(t)

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


class PreprocessorTransformer(BaseEstimator, TransformerMixin, ColumnUser):

    def __init__(self, column):
        self.column = column

    def set_column_to_use(self, column_name):
        self.column = column_name

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
            sentiment_opinion_score = 0
            if word_count > 0:
                for word in sentence:
                    if word in self.value_dict:
                        sentiment_opinion_score += 1
            sentiment_opinion_scores.append([sentiment_opinion_score])
        for i in range(len(sentiment_opinion_scores)):
            features[i] = features[i] + (sentiment_opinion_scores[i][0])
        return features


class PipelineRunner:

    def __init__(self, dict_file, training_file, test_file, log_file='results/results_with_correct_input.log'):
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
        self.pipeline = None
        self.transformer_name_dict = {
                                      TextToSentenceTransformer.__name__: "ts",
                                      BertTransformer.__name__: "bert",
                                      PreprocessorTransformer.__name__: "prepro",
                                      SentimentOpinionValueCalculatorSingleValueTransformer.__name__: "sentval",
                                      SentimentOpinionValueCounterTransformer.__name__: "sentcount"
                                      }
        self.estimator_name_dict = {
            LogisticRegression.__name__: 'log_reg',
            GaussianNB.__name__: 'gau_nb',
            BernoulliNB.__name__: 'ber_nb'
        }

    def prepare_pipeline(self, data_column, estimator_type, transformer_types_list):

        if not self.pipeline:
            raise RuntimeError("Don't call this function directly use make_pipeline")

        print(f'Starting fitting: {estimator_type}')
        accuracy, f1, recall, precision = self.fit_and_predict_and_calculate_accuracy_pipe(data_column)
        description = f'\tUsed estimator: {estimator_type}\n'
        description += f'\tUsed transformers: {", ".join(transformer_types_list)}\n'
        description += f'\tColumn: {data_column}\n'
        self.write_result_to_file(accuracy f1, recall, precision, description)

    def make_pipeline(self, transformer_list, estimator,
                      data_column, param_gird, classifier_description='',
                      dir_path='', force_fitting=False):

        classifier_file = self.create_pipe_line_name(transformer_list, estimator, classifier_description, data_column, dir_path)
        if force_fitting or not os.path.exists(classifier_file):
            print('No classfier found for your configuration or force_fitting=True. Creating new one and saving it.')
            clf = GridSearchCV(estimator=estimator, param_grid=param_gird, n_jobs=-1, scoring='accuracy')

            self.pipeline = sklearn.pipeline.Pipeline(
                [(f'stage: {self.transformer_name_dict[type(transformer_list[index]).__name__]}',
                  transformer_list[index]) for index in range(len(transformer_list))] +
                [('clf', clf)]
            )
            self.prepare_pipeline(data_column, type(estimator).__name__, [type(t).__name__ for t in transformer_list])
            print(f'Saving classifier to file {classifier_file}')
            self.save_classifier(classifier_file)
        else:
            print(f'Classifier exists. Loaded from file {classifier_file}')
            self.load_classifier(classifier_file)

        return self.pipeline


    def create_pipe_line_name(self, transformer_list, estimator, classifier_description, data_column, dir_path=''):
        names = [self.transformer_name_dict[type(transformer).__name__] for transformer in transformer_list]
        description = ''
        if classifier_description:
            description = '_' + classifier_description
        if dir_path and not dir_path.endswith('/') and not dir_path.endswith('\\'):
            dir_path += '/'
        return dir_path + f'classifier/{self.estimator_name_dict[type(estimator).__name__]}_with_{"_".join(names)}{description}_{data_column}_pipeline.joblib.plk'

    def load_classifier(self, filename):
        self.pipeline = joblib.load(filename)

    def save_classifier(self, filename):
        joblib.dump(self.pipeline, filename)

    def fit_and_predict_and_calculate_accuracy_pipe(self, data_column):
        print('Start fiting')
        self.pipeline.fit(self.data_training, self.data_training[data_column].to_numpy())
        print('Start prediction')
        start = time.time()
        y_pred_pipe = self.pipeline.predict(self.data_test)
        end = time.time()
        print(f'time needed: {end - start}')

        #return accuracy_score(self.data_test[data_column].to_numpy(), y_pred_pipe)

        acc = accuracy_score(self.data_test[data_column].to_numpy(), y_pred_pipe)
        f1 = f1_score(self.data_test[data_column].to_numpy(), y_pred_pipe, average='weighted')
        rec = recall_score(self.data_test[data_column].to_numpy(), y_pred_pipe, average='weighted')
        precision = precision_score(self.data_test[data_column].to_numpy(), y_pred_pipe, average='weighted')
        return acc, f1, rec, precision

    def write_result_to_file(self, accuracy, f1, recall, precision, description):
        with open(self.log_file, 'a', encoding='utf-8') as file:
            file.write('#------------------------------------------------------------------------------------------\n')
            file.write(f'{datetime.now().strftime("%b-%d-%Y %H:%M:%S")}\n')
            file.write(f'{description}\n')
            file.write(f'\t\tAccuracy for classifier {type}: {accuracy}\n')
            file.write(f'\t\tF1 Score {type}: {f1}\n')
            file.write(f'\t\tRecall Score {type}: {recall}\n')
            file.write(f'\t\tPrecision Score {type}: {precision}\n')
            file.write('#------------------------------------------------------------------------------------------\n')

    def predict_data(self, data_file_name, column_to_transform=None, new_column_name=None, batch_size=1, result_for_column='', log_file=f'results/results_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'):
        data_validation = pd.read_csv(data_file_name)

        if column_to_transform is not None:
            for obj in self.pipeline.named_steps.values():
                if issubclass(type(obj), ColumnTransformer):
                    obj.set_column_to_transform(column_to_transform)

        if new_column_name is not None:
            for obj in self.pipeline.named_steps.values():
                if issubclass(type(obj), ColumnUser):
                    obj.set_column_to_use(new_column_name)

        with open(log_file, 'a', encoding='utf-8') as f:
            f.write('#-------------------------------------------------')
            f.write(f'result for column {result_for_column}:\n')

        row_count = data_validation.shape[0]
        counter = 0

        while counter * batch_size < row_count:

            output = self.pipeline.predict(data_validation.loc[counter * batch_size: (counter + 1) * batch_size])
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f'{output.tolist()}')
                f.write('\n')
            counter += 1


        with open(log_file, 'a', encoding='utf-8') as f:
            f.write('#-------------------------------------------------\n\n\n\n\n')


def fit_and_predict_and_calculate_accuracy_pipe(pipe, train_input, train_ouput, test_input, test_output):
    pipe.fit(train_input, train_ouput)

    y_pred_pipe = pipe.predict(test_input)

    acc = accuracy_score(test_output, y_pred_pipe)
    f1 = f1_score(test_output, y_pred_pipe, average='weighted')
    rec = recall_score(test_output, y_pred_pipe, average='weighted')
    precision = precision_score(test_output, y_pred_pipe, average='weighted')

    #return accuracy_score(test_output, y_pred_pipe)
    return acc, f1, rec, precision

def write_result_to_file(accuracy, f1, recall, precision, type, description):
    result_file = 'results/results_with_correct_input.log'
    with open(result_file, 'a', encoding='utf-8') as file:
        file.write('#------------------------------------------------------------------------------------------\n')
        file.write(f'{datetime.now().strftime("%b-%d-%Y %H:%M:%S")}\n')
        file.write(f'\t{description}\n')
        file.write(f'\t\tAccuracy for classifier {type}: {accuracy}\n')
        file.write(f'\t\tF1 Score {type}: {f1}\n')
        file.write(f'\t\tRecall Score {type}: {recall}\n')
        file.write(f'\t\tPrecision Score {type}: {precision}\n')
        file.write('#------------------------------------------------------------------------------------------\n')


if __name__ == '__main__':
    dir_path = ''
    # dict_file = dir_path + 'AFINN-both-abs.csv'
    dict_file = dir_path + 'sentiment_dict.csv'

    training_file = dir_path + 'Trainingdata_train.xlsx'

    test_file = dir_path + 'Trainingdata_test.xlsx'

    transformers_list = [TextToSentenceTransformer('text', 'Sentence'),
                         BertTransformer('Sentence'),
                         PreprocessorTransformer('Sentence'),
                         SentimentOpinionValueCalculatorSingleValueTransformer(dict_file)]

    pipeline_runner = PipelineRunner(dict_file, training_file, test_file, log_file=dir_path + 'results/results_with_dict_only.log')
    Cs = np.logspace(-6, 6, 200)
    max_iter = 500
    log_reg_subjopin = LogisticRegression(max_iter=max_iter)
    pipeline_runner.make_pipeline(transformers_list, log_reg_subjopin, 'SUBJopin01', dict(C=Cs), dir_path=dir_path)

    pipeline_runner.predict_data(data_file_name=dir_path + 'MBFC_Dataset_Sample.csv',
                                 result_for_column='SUBJopin',
                                 log_file=dir_path + f'results/mbfc_results_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_SUBJopin.log')

    log_reg_subjlang = LogisticRegression(max_iter=max_iter)
    pipeline_runner.make_pipeline(transformers_list, log_reg_subjlang, 'SUBJlang01', dict(C=Cs), dir_path=dir_path)
    pipeline_runner.predict_data(data_file_name=dir_path + 'MBFC_Dataset_Sample.csv',
                                 result_for_column='SUBJlang',
                                 log_file=dir_path + f'results/mbfc_results_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_SUBJlang.log')
