

import re
import sklearn
import string
import nltk
from nltk import tokenize as tk
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
from scipy import stats as stats

nltk.download('stopwords')
nltk.download('punkt')


print(pipeline('sentiment-analysis')('we love you'))


# Man kann auch mehrere hintereinander machen

class ColumnUser:
    def set_column_to_use(self, column_name):
        pass


class ColumnTransformer(ColumnUser):
    def set_column_to_transform(self, column_to_transform):
        pass


class TextToSentenceTransformer(BaseEstimator, TransformerMixin, ColumnTransformer):
    def __init__(self, column_to_transform, new_column_name, filename='data/logs/text_to_sentence_transformer.error'):
        self.column_to_transform = column_to_transform
        self.new_column_name = new_column_name
        self.log_file = filename

    def set_column_to_transform(self, column_to_transform):
        self.column_to_transform = column_to_transform

    def set_column_to_use(self, column_name):
        self.new_column_name = column_name

    def fit(self, X, y=None):
        return self

    def transform(self, data):
        if self.column_to_transform and self.column_to_transform in data.columns:
            return pd.DataFrame({self.new_column_name: self.split_text_in_sentences(data)})
        else:
            self.log_error(
                f'no column with name {self.column_to_transform} in dataframe.\nGiving back original dataframe.')
            return data

    def split_text_in_sentences(self, data) -> List[str]:
        texts = data[self.column_to_transform].tolist()
        sentences = list()
        for text in texts:
            # sentences_in_text = [e + delimiter for e in text.split(delimiter) if e]
            sentences_in_text = [e for e in tk.sent_tokenize(str(text)) if e]
            sentences += sentences_in_text

        print(sentences[0])
        return sentences

    def log_error(self, description):
        with open(self.log_file, 'a', encoding='utf-8') as file:
            file.write('#------------------------------------------------------------------------------------------\n')
            file.write(f'{datetime.now().strftime("%b-%d-%Y %H:%M:%S")}\n')
            file.write(f'\t{description}\n')
            file.write(
                '#------------------------------------------------------------------------------------------\n\n\n')


class BertTransformer(BaseEstimator, TransformerMixin, ColumnUser):

    def __init__(self, column, batchsize=10):
        self.column = column
        self.batch_size = batchsize
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

        features = list()
        row_count = data.shape[0]
        counter = 0
        start_index = 0
        while start_index < row_count:
            d = data.loc[start_index: start_index + self.batch_size]
            feature = self.embedding(d)
            features.extend(feature)
            counter += 1
            start_index += self.batch_size + 1 # das plus 1 kommt daher, dass bei den pandas Dataframes start und end index inklusive sind
            if counter % 10 == 0:
                print(f'{min(100.0, round(((start_index / row_count) * 100), 2))}% done')

        return (data, features)

    def embedding(self, data):
        dataList = data[self.column].tolist()
        dataList = list((str(s) for s in dataList))
        tokenized = []

        for s in dataList:
            t = self.tokenizer.encode(s, add_special_tokens=True)
            tokenized.append(t)

        # Padding hinzufügen
        max_len = 0
        for i in tokenized:
            if len(i) > max_len:
                max_len = len(i)

        padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized])

        # Maske erstellen, um das Padding bei der Verarbeitung zu filtern
        mask = np.where(padded != 0, 1, 0)

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

        return features


class PreprocessorTransformer(BaseEstimator, TransformerMixin, ColumnUser):

    def __init__(self, column):
        self.column = column

    def set_column_to_use(self, column_name):
        self.column = column_name

    def fit(self, X, y=None):
        return self

    def transform(self, data_and_features, y=None):
        print('Starting with preprocessing')
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
        print('Starting with sentiment value calculation')
        sentiment_opinion_scores = []
        sentences, features = sentences_and_features
        counter = 0
        count_of_sentences = len(sentences)
        for sentence in sentences:
            word_count = len(sentence)
            sentiment_opinion_score = 0
            if word_count > 0:
                for word in sentence:
                    if word in self.value_dict:
                        sentiment_opinion_score = sentiment_opinion_score + self.value_dict[word]
                sentiment_opinion_score = sentiment_opinion_score / word_count
            else:
                sentiment_opinion_score = 0
            sentiment_opinion_scores.append([sentiment_opinion_score])
            counter += 1
            if counter % 10 == 0:
                print(f'{min(100.0, round(((counter / count_of_sentences) * 100), 2))}% of sentences done')

        # TEST
        print(len(sentiment_opinion_scores))
        print(len(features))
        for i in range(len(sentiment_opinion_scores)):
            features[i] = features[i] + (sentiment_opinion_scores[i])
        print('starting with classification')
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

    def __init__(self, dict_file, training_file, test_file, log_file='data/results/results_with_correct_input.log'):
        self.dict_file = dict_file
        self.log_file = log_file
        self.data_training = pd.read_excel(training_file, sheet_name='sentences')
        self.data_test = pd.read_excel(test_file, sheet_name='sentences')
        self.pipeline = None
        self.estimator = None
        self.classifier_file = None
        self.transformer_name_dict = {
                                      TextToSentenceTransformer.__name__: "ts",
                                      BertTransformer.__name__: "bert",
                                      PreprocessorTransformer.__name__: "prepro",
                                      SentimentOpinionValueCalculatorSingleValueTransformer.__name__: "sentval",
                                      SentimentOpinionValueCounterTransformer.__name__: "sentcount",
                                      sklearn.preprocessing.StandardScaler.__name__: "std-scaler",
                                      sklearn.preprocessing.MinMaxScaler.__name__: "minmax"
                                      }
        self.estimator_name_dict = {
            LogisticRegression.__name__: 'log_reg',
            GaussianNB.__name__: 'gau_nb',
            BernoulliNB.__name__: 'ber_nb',
            sklearn.svm.SVC.__name__: 'svc'
        }

    def prepare_pipeline(self, data_column, estimator_type, transformer_types_list):

        if not self.pipeline:
            raise RuntimeError("Don't call this function directly use make_pipeline")

        print(f'Starting fitting: {estimator_type}')
        accuracy, f1, recall, precision = self.fit_and_predict_and_calculate_accuracy_pipe(data_column)
        description = f'\tUsed estimator: {estimator_type}\n'
        description += f'\tUsed transformers: {", ".join(transformer_types_list)}\n'
        description += f'\tColumn: {data_column}\n'
        self.write_result_to_file(accuracy, f1, recall, precision, description)

    def prepare_pipeline_confidence(self, data_column, estimator_type, transformer_types_list):

        if not self.pipeline:
            raise RuntimeError("Don't call this function directly use make_pipeline")

        print(f'Starting fitting: {estimator_type}')
        result = self.fit_and_predict_and_calculate_confidence_pipe(data_column)
        for threshold, accuracy, f1, recall, precision, num_all_entries, num_entries in result:
            description = f'\tUsed estimator: {estimator_type}\n'
            description += f'\tUsed transformers: {", ".join(transformer_types_list)}\n'
            # description += f'\tonly 60percent confidence \n'
            description += f'\tColumn: {data_column}\n'
            self.write_result_to_file_confidence(threshold, accuracy, f1, recall, precision, description, num_all_entries, num_entries)

    def make_pipeline(self, transformer_list, estimator,
                      data_column, param_gird, classifier_description='',
                      dir_path='', force_fitting=False):

        self.classifier_file = self.create_pipe_line_name(transformer_list, estimator, classifier_description, data_column, dir_path)
        if force_fitting or not os.path.exists(self.classifier_file):
            print('No classfier found for your configuration or force_fitting=True. Creating new one and saving it.')
            self.estimator = GridSearchCV(estimator=estimator, param_grid=param_gird, n_jobs=-1, scoring='accuracy')
            self.create_pipe_line(transformer_list)
            self.prepare_pipeline(data_column, type(estimator).__name__, [type(t).__name__ for t in transformer_list])
        else:
            print(f'Classifier exists. Loaded from file {self.classifier_file}')
            self.load_classifier(self.classifier_file)
            self.create_pipe_line(transformer_list)

        return self.pipeline

    def make_pipeline_confidence(self, transformer_list, estimator,
                                data_column, param_gird, classifier_description='',
                                dir_path='', force_fitting=False):

        self.classifier_file = self.create_pipe_line_name(transformer_list, estimator, classifier_description, data_column, dir_path)
        if force_fitting or not os.path.exists(self.classifier_file):
            print('No classfier found for your configuration or force_fitting=True. Creating new one and saving it.')
            self.estimator = GridSearchCV(estimator=estimator, param_grid=param_gird, n_jobs=-1, scoring='accuracy')
            self.create_pipe_line(transformer_list)
            self.prepare_pipeline_confidence(data_column, type(estimator).__name__, [type(t).__name__ for t in transformer_list])
        else:
            print(f'Classifier exists. Loaded from file {self.classifier_file}')
            self.load_classifier(self.classifier_file)
            self.create_pipe_line(transformer_list)

        return self.pipeline

    def create_pipe_line(self, transformer_list):
        self.pipeline = sklearn.pipeline.Pipeline(
            [(f'stage: {self.transformer_name_dict[type(transformer_list[index]).__name__]}',
              transformer_list[index]) for index in range(len(transformer_list))] +
            [('clf', self.estimator)]
        )

    def create_pipe_line_name(self, transformer_list, estimator, classifier_description, data_column, dir_path=''):
        names = [self.transformer_name_dict[type(transformer).__name__] for transformer in transformer_list]
        description = ''
        if classifier_description:
            description = '_' + classifier_description
        if dir_path and not dir_path.endswith('/') and not dir_path.endswith('\\'):
            dir_path += '/'
        return dir_path + f'classifier/{self.estimator_name_dict[type(estimator).__name__]}_using_{"_".join(names)}{description}_{data_column}_pipeline.joblib.plk'

    def load_classifier(self, filename):
        self.estimator = joblib.load(filename)

    def save_classifier(self, filename):
        joblib.dump(self.estimator, filename)

    def conduct_ttest(self, predicted, validation, baseline_acc=0.5347):
        print("Preparing for t-test.")
        print("Collating prediction results.")

        #results = right/wrong
        prediction_results = []
        acc = 0
        for i in range(0, len(predicted)):
            if predicted[i] == validation[i]:
                prediction_results.append(1)
                acc = acc + 1
            else:
                prediction_results.append(0)
        acc = acc / len(prediction_results)
        print("Prediction accuracy calculated as " + str(acc))

        #DAS IST DIE MAGIC LINE
        print("Starting test")
        tstatistic, pvalue = stats.ttest_1samp(prediction_results, baseline_acc, alternative='greater')
        print("Calculated pvalue: "+ str(pvalue))
        if pvalue < 0.05:
           print("Success: Classifier significantly outperforms baseline")
           return 1
        else:
            print("Failure: Null-Hypothesis not rejectable")
            return 0

    def fit_and_predict_and_calculate_accuracy_pipe(self, data_column):
        print('Start fiting')
        self.pipeline.fit(self.data_training, self.data_training[data_column].to_numpy())
        print(f'Saving classifier to file {self.classifier_file}')
        self.save_classifier(self.classifier_file)
        return self.predict_test_data(data_column)

    def predict_test_data(self, data_column):
        print('Start test prediction')
        start = time.time()
        y_pred_pipe = self.pipeline.predict(self.data_test)
        # self.safe_prediction(y_pred_pipe, data_column)
        end = time.time()
        print(f'time needed: {end - start}')

        acc = accuracy_score(self.data_test[data_column].to_numpy(), y_pred_pipe)
        f1 = f1_score(self.data_test[data_column].to_numpy(), y_pred_pipe, average='weighted')
        rec = recall_score(self.data_test[data_column].to_numpy(), y_pred_pipe, average='weighted')
        precision = precision_score(self.data_test[data_column].to_numpy(), y_pred_pipe, average='weighted')
        return acc, f1, rec, precision

    def fit_and_predict_and_calculate_confidence_pipe(self, data_column):
        print('Start fiting')
        self.pipeline.fit(self.data_training, self.data_training[data_column].to_numpy())
        print(f'Saving classifier to file {self.classifier_file}')
        self.save_classifier(self.classifier_file)
        print(f'best gridsearch params:\n{self.estimator.best_params_}')
        return self.predict_test_data_confidence(data_column)


    def predict_test_data_confidence(self, data_column):
        print('Start test prediction')

        # predicte die confidence für jedes Item
        y_pred_pipe = self.pipeline.predict_proba(self.data_test)
        df = pd.DataFrame(data=y_pred_pipe, columns=['confidence class 0', 'confidence class 1'])

        # predicte die Klasse für jedes Item
        df['prediction'] = self.classify_prediction(df)

        # schreibe die confidence der predicted class in die Spalte max confidence
        df['max confidence'] = df[['confidence class 0', 'confidence class 1']].max(axis=1)

        # lösche spalten confidence class 0 und confidence class 1
        df.drop(['confidence class 0', 'confidence class 1'], axis=1)

        # neuen dataframe erstellen mit den richtigen Klassen, den predicted classes und den dazugehörigen confidences
        df2 = pd.DataFrame(data=self.data_test[data_column].to_numpy(), columns=['real sentiment label'])
        df2['prediction'] = df['prediction']
        df2['confidence'] = df['max confidence']

        # df und df2 in die excel speichern
        df.to_excel(f'data/results/max_confidence_{data_column}.xlsx', index=False)
        df2.to_excel(f'data/results/classes_and_confidence_{data_column}.xlsx', index=False)

        result_list = list()
        print("Entries before filtering for confidence: " + str(df2.shape[0]))
        thresholds = [0.6, 0.7, 0.8]
        for threshold in thresholds:
            df3 = self.filter_threshold(df2, threshold)
                    # dummy Zeile wieder löschen
            df3.drop(df3.head(1).index, inplace=True)

            print("Filtered for confidence " + str(threshold) + "+. Remaining Entries: " + str(df3.shape[0]))

            #die jeweiligen df3 speichern
            df3.to_excel(f'data/results/confidence{threshold}_{data_column}.xlsx', index=False)

            acc = accuracy_score(df3['real sentiment label'], df3['prediction'])
            f1 = f1_score(df3['real sentiment label'], df3['prediction'], average='weighted')
            rec = recall_score(df3['real sentiment label'], df3['prediction'], average='weighted')
            precision = precision_score(df3['real sentiment label'], df3['prediction'], average='weighted')
            result_list.append((threshold, acc, f1, rec, precision, df2.shape[0], df3.shape[0]))

        return result_list

    def filter_threshold(self, dataframe, threshold):
        # alle Zeilen aussortieren, die eine confidence unter dem threshold haben
        df_new = pd.DataFrame({'real sentiment label': [100], 'prediction': [100], 'confidence': [100]})
        for index, row in dataframe.iterrows():
            if (row['confidence']) > threshold:
                item = pd.DataFrame({'real sentiment label': [row['real sentiment label']],
                                     'prediction': [row['prediction']],
                                     'confidence': [row['confidence']]})
                df_new = df_new.append(item, ignore_index=True)
        return df_new

    def classify_prediction(self, df):
        class_predictions = list()
        for index, row in df.iterrows():
            if row['confidence class 0'] >= row['confidence class 1']:
                class_predictions.append(0)
            else:
                class_predictions.append(1)
        return class_predictions


    def safe_prediction(self, prediction, data_column):
        pred_data_frame = self.data_test.copy(deep=True)
        pred_data_frame[f'prediction_{data_column}'] = prediction
        pred_data_frame.to_excel(f'data/results/prediction_for_{data_column}.xlsx', index=False)
        pass

    def write_result_to_file(self, accuracy, f1, recall, precision, description):
        with open(self.log_file, 'a', encoding='utf-8') as file:
            file.write('#------------------------------------------------------------------------------------------\n')
            file.write(f'{datetime.now().strftime("%b-%d-%Y %H:%M:%S")}\n')
            file.write(f'{description}\n')
            file.write(f'\t\tAccuracy: {accuracy}\n')
            file.write(f'\t\tF1 Score: {f1}\n')
            file.write(f'\t\tRecall Score: {recall}\n')
            file.write(f'\t\tPrecision Score: {precision}\n')
            file.write('#------------------------------------------------------------------------------------------\n')

    def write_result_to_file_confidence(self, threshhold, accuracy, f1, recall, precision, description, num_all_entries, num_entries):
        with open(self.log_file, 'a', encoding='utf-8') as file:
            file.write('#------------------------------------------------------------------------------------------\n')
            file.write(f'{datetime.now().strftime("%b-%d-%Y %H:%M:%S")}\n')
            file.write(f'for threshhold: {threshhold}\n')
            file.write(f'Dropped Entries: {num_all_entries - num_entries}\n')
            file.write(f'Remaining Entries: {num_entries} of {num_all_entries}\n')
            file.write(f'{description}\n')
            file.write(f'\t\tAccuracy: {accuracy}\n')
            file.write(f'\t\tF1 Score: {f1}\n')
            file.write(f'\t\tRecall Score: {recall}\n')
            file.write(f'\t\tPrecision Score: {precision}\n')
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

        output = self.pipeline.predict(data_validation)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write('#-------------------------------------------------')
            f.write(f'result for column {result_for_column}:\n')
            f.write(f'{output.tolist()}')
            f.write('\n')
            f.write('#-------------------------------------------------\n\n\n\n\n')


if __name__ == '__main__':
    dir_path = 'data/'
    # dict_file = dir_path + 'AFINN-both-abs.csv'
    dict_file = dir_path + 'sentiment_dict.csv'

    # training_file = dir_path + 'datasetSentimentSRF_train.xlsx'
    training_file = dir_path + 'TrainingdataNew_train.xlsx'
    #
    # test_file = dir_path + 'datasetSentimentSRF_test.xlsx'
    test_file = dir_path + 'TrainingdataNew_test.xlsx'

    transformers_list = [TextToSentenceTransformer('text', 'Sentence'),
                         BertTransformer('Sentence', batchsize=10),
                         PreprocessorTransformer('Sentence'),
                         SentimentOpinionValueCalculatorSingleValueTransformer(dict_file)]

    pipeline_runner = PipelineRunner(dict_file, training_file, test_file, log_file=dir_path + 'results/results_for_different_threshholds.log')
    Cs = np.logspace(0.0001, 6, 100)
    max_iter = 500
    # bert = BertTransformer('Sentence', batchsize=100)
    # bert.transform(pd.read_excel('data/datasetSingleSentences.xlsx', sheet_name='Sheet1'))
    # log_reg_subjopin = LogisticRegression(max_iter=max_iter)
    # pipeline_runner.make_pipeline(transformers_list, log_reg_subjopin, 'SUBJopin01', dict(C=Cs), dir_path=dir_path, classifier_description='probability')

    # pipeline_runner.predict_data(data_file_name=dir_path + 'MBFC-sentences-Dataset.csv',
    #                              result_for_column='SUBJopin',
    #                              log_file=dir_path + f'results/mbfc_sentences_results_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_SUBJopin.log',
    #                              new_column_name='sentences')

    log_reg_subjlang = LogisticRegression(max_iter=max_iter)
    pipeline_runner.make_pipeline_confidence(transformers_list, log_reg_subjlang, 'SUBJlang01', dict(C=Cs), dir_path=dir_path, classifier_description='probability')

    #gnb_subjlang = GaussianNB()
    #pipeline_runner.make_pipeline_confidence(transformers_list, gnb_subjlang, 'SUBJlang01', dict(var_smoothing=Cs), dir_path=dir_path, classifier_description='probability')

    # bnb_subjlang = BernoulliNB()
    # pipeline_runner.make_pipeline_confidence(transformers_list, bnb_subjlang, 'SUBJlang01', dict(alpha=Cs),
    #                                          dir_path=dir_path, classifier_description='probability')

    # transformers_list_svm = [TextToSentenceTransformer('text', 'Sentence'),
    #                      BertTransformer('Sentence', batchsize=100),
    #                      PreprocessorTransformer('Sentence'),
    #                      SentimentOpinionValueCalculatorSingleValueTransformer(dict_file),
    #                      sklearn.preprocessing.StandardScaler(),
    #                      sklearn.preprocessing.MinMaxScaler()]
    #
    # svm_subjlang = sklearn.svm.SVC(kernel='linear')
    # pipeline_runner.make_pipeline_confidence(transformers_list_svm, svm_subjlang, 'SUBJlang01', dict(C=Cs),
    #                                          dir_path=dir_path, classifier_description='probability')
    #
    # pipeline_runner.predict_data(data_file_name=dir_path + 'MBFC-sentences-Dataset.csv',
    #                              result_for_column='SUBJlang',
    #                              log_file=dir_path + f'results/mbfc_sentences_results_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_SUBJlang.log',
    #                              new_column_name='sentences')
