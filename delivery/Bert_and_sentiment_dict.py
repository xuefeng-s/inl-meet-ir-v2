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
import spacy
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
import sys

nltk.download('stopwords')
nltk.download('punkt')
#spacy.cli.download('en_core_web_sm')

print(pipeline('sentiment-analysis')('we love you'))


# Man kann auch mehrere hintereinander machen

class FileReader:
    """
    Reads file to pandas dataframe depending on the file extension
    """
    def read_file(self, file_name):
        file_extension = file_name[file_name.rfind('.')+1:].lower()
        if file_extension == 'csv':
            return pd.read_csv(file_name)
        elif file_extension == 'xls' or file_extension == 'xlsx':
            return pd.read_excel(file_name)
        else:
            raise RuntimeError(f"Can't identify file extension {file_extension}. Abort try to read it")


class ColumnUser:
    """
    Indicates if a transformer uses a column. Is an interface
    """
    def set_column_to_use(self, column_name):
        pass


class ColumnTransformer(ColumnUser):
    """
    Indicates if a transformer uses and transforms/produce a column. Is an interface
    """
    def set_column_to_transform(self, column_to_transform):
        pass


class TextToSentenceTransformer(BaseEstimator, TransformerMixin, ColumnTransformer):
    """
    Transformer which naive transform text to sentences. Split by nltk's sentence tokenizer
    """
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



class PreprocessorBeforeBertTransformer(BaseEstimator, TransformerMixin, ColumnUser):

    """
    Transformer which makes every word lower case, removes punctuation, numbers and stopwords.
    """
    def __init__(self, column):
        self.column = column

    def set_column_to_use(self, column_name):
        self.column = column_name

    def fit(self, X, y=None):
        return self

    def transform(self, data, y=None):
        print('Start preprocessing')
        sentences = data[self.column].tolist()
        sentences = list((str(s) for s in sentences))

        # muss vom generator object zurück zur liste gemacht werden
        sentences = list((s.lower() for s in sentences))

        table = str.maketrans('', '', string.punctuation)
        sentences = [s.translate(table) for s in sentences]

        sentences = [re.sub(r'\d+', 'num', s) for s in sentences]

        stopwords = set(nltk.corpus.stopwords.words('english'))
        sentences = [[word for word in s.split() if word not in stopwords] for s in sentences]
        return pd.DataFrame({self.column: sentences})




class BertTransformer(BaseEstimator, TransformerMixin, ColumnUser):
    """
    Lets bert calculate the sentence embeddings
    """
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
            t = self.tokenizer.encode(s, add_special_tokens=True, truncation=True)
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

        # nur die erste Spalte auslesen = sentence embeddings
        features = output[0][:, 0, :].numpy()

        return features


class PreprocessorTransformer(BaseEstimator, TransformerMixin, ColumnUser):
    """
    Transformer which makes every word lower case, removes punctuation, numbers and stopwords.
    """
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

        # make lower case
        sentences = list((s.lower() for s in sentences))

        # remove punctuation
        table = str.maketrans('', '', string.punctuation)
        sentences = [s.translate(table) for s in sentences]

        sentences = [re.sub(r'\d+', 'num', s) for s in sentences]

        stopwords = set(nltk.corpus.stopwords.words('english'))
        sentences = [[word for word in s.split() if word not in stopwords] for s in sentences]
        return (sentences, features)


class AdvancedPreprocessorTransformer(BaseEstimator, TransformerMixin, ColumnUser):
    """
    Transformer which is an advanced version of PreprocessorTransformer.
    This one also remove names
    """
    def __init__(self, column):
        self.column = column

    def set_column_to_use(self, column_name):
        self.column = column_name

    def fit(self, X, y=None):
        return self

    def transform(self, data_and_features, y=None):
        print('Starting with advanced preprocessing')
        data, features = data_and_features
        sentences = data[self.column].tolist()
        sentences = list((str(s) for s in sentences))

        # make lower case
        sentences = list((s.lower() for s in sentences))

        # remove punctuation
        table = str.maketrans('', '', string.punctuation)
        sentences = [s.translate(table) for s in sentences]

        # replace numbers
        sentences = [re.sub(r'\d+', 'num', s) for s in sentences]

        # remove stopwords
        stopwords = set(nltk.corpus.stopwords.words('english'))
        sentences = [[word for word in s.split() if word not in stopwords] for s in sentences]

        # remove named entities
        nlp = spacy.load("en_core_web_sm")
        sentences2 = []
        for s in sentences:
            text_no_namedentities = []
            document = nlp(str(s))
            for item in document:
                if item.ent_type:  # falls es ein name ist
                    # text_no_namedentities.append('ne')        # durch ne ersetzen
                    pass  # oder ganz auslassen
                else:
                    text_no_namedentities.append(item.text)
            sentences2.append(" ".join(text_no_namedentities))
        return sentences2, features


class SentimentOpinionValueCalculatorTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer which calculates the average sentiment/opinion value of the sentence
    """
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
    """
    Transformer which counts how many words from the dictionary occurs in the sentence
    """
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
    """
    Encapsulates how to use the pipeline.
    Creates or load the classifier. fits classifier if needed. saves statistics.
    """
    def __init__(self, dict_file, training_file, test_file, log_file, result_dir):
        self.dict_file = dict_file
        self.log_file = log_file
        self.file_reader = FileReader()
        self.data_training = self.file_reader.read_file(training_file)
        self.data_test = self.file_reader.read_file(test_file)
        self.pipeline = None
        self.estimator = None
        self.classifier_file = None
        # short name of possible transformers
        # must be updated if a new one is introduced
        self.transformer_name_dict = {
                                      TextToSentenceTransformer.__name__: "ts",
                                      PreprocessorBeforeBertTransformer.__name__: "preprob",
                                      BertTransformer.__name__: "bert",
                                      PreprocessorTransformer.__name__: "prepro",
                                      AdvancedPreprocessorTransformer.__name__: "advprepro",
                                      SentimentOpinionValueCalculatorTransformer.__name__: "valcalc",
                                      SentimentOpinionValueCounterTransformer.__name__: "count",
                                      sklearn.preprocessing.StandardScaler.__name__: "std-scaler",
                                      sklearn.preprocessing.MinMaxScaler.__name__: "minmax"
                                      }
        # short name of possible estimators
        # must be updated if a new one is introduced
        self.estimator_name_dict = {
            LogisticRegression.__name__: 'log_reg',
            GaussianNB.__name__: 'gau_nb',
            BernoulliNB.__name__: 'ber_nb',
            sklearn.svm.SVC.__name__: 'svc'
        }
        self.result_dir = result_dir
        self.classifier_dir = 'data/classifier/'

    def prepare_pipeline(self, data_column, estimator_type, transformer_types_list):
        """
        Fits new classifier and writes results to file
        :param data_column:
        :param estimator_type:
        :param transformer_types_list:
        :return: nothing
        """
        if not self.pipeline:
            raise RuntimeError("Don't call this function directly use make_pipeline")

        print(f'Starting fitting: {estimator_type}')
        accuracy, f1, recall, precision = self.fit_and_predict_and_calculate_accuracy_pipe(data_column)
        description = f'\tUsed estimator: {estimator_type}\n'
        description += f'\tUsed transformers: {", ".join(transformer_types_list)}\n'
        description += f'\tColumn: {data_column}\n'
        self.write_result_to_file(accuracy, f1, recall, precision, description)

    def prepare_pipeline_confidence(self, data_column, estimator_type, transformer_types_list):
        """
        Fits new classifier and writes results to file
        :param data_column:
        :param estimator_type:
        :param transformer_types_list:
        :return: nothing
        """
        if not self.pipeline:
            raise RuntimeError("Don't call this function directly use make_pipeline")

        print(f'Starting fitting: {estimator_type}')
        result = self.fit_and_predict_and_calculate_confidence_pipe(data_column)
        for threshold, accuracy, f1, recall, precision, num_all_entries, num_entries in result:
            description = f'\tUsed estimator: {estimator_type}\n'
            description += f'\tUsed transformers: {", ".join(transformer_types_list)}\n'
            description += f'\tColumn: {data_column}\n'
            self.write_result_to_file_confidence(threshold, accuracy, f1, recall, precision, description, num_all_entries, num_entries)

    def make_pipeline(self, transformer_list, estimator,
                      data_column, param_gird, classifier_description='', force_fitting=False):

        """
        Creates or load a pipeline. if it is created it will be tested without confidence
        :param transformer_list: list of transformers
        :param estimator: which estimator is used in GridSearchCV
        :param data_column: opinion or sentiment (lang)
        :param param_gird: parameter for GridSearchCV
        :param classifier_description: advanced description of classifier
        :param force_fitting: true if a new classifier should be created even it exists one.
        :return: fitted or loaded pipeline
        """
        self.classifier_file = self.create_pipe_line_name(transformer_list, estimator, classifier_description, data_column)
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
                                 data_column, param_gird, classifier_description='', force_fitting=False):
        """
        Creates or load a pipeline. if it is created it will be tested with confidence
        :param transformer_list: list of transformers
        :param estimator: which estimator is used in GridSearchCV
        :param data_column: opinion or sentiment (lang)
        :param param_gird: parameter for GridSearchCV
        :param classifier_description: advanced description of classifier
        :param force_fitting: true if a new classifier should be created even it exists one.
        :return: fitted or loaded pipeline
        """
        self.classifier_file = self.create_pipe_line_name(transformer_list, estimator, classifier_description, data_column)
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

    def create_pipe_line_name(self, transformer_list, estimator, classifier_description, data_column):
        names = [self.transformer_name_dict[type(transformer).__name__] for transformer in transformer_list]
        description = ''
        if classifier_description:
            description = '_' + classifier_description
        return self.classifier_dir + f'{self.estimator_name_dict[type(estimator).__name__]}_using_{"_".join(names)}{description}_{data_column}_pipeline.joblib.plk'

    def load_classifier(self, filename):
        self.estimator = joblib.load(filename)

    def save_classifier(self, filename):
        joblib.dump(self.estimator, filename)

    def conduct_ttest(self, predicted, validation, baseline_acc=0.5347):
        """
        Calculates results of conduct test
        :param predicted:
        :param validation:
        :param baseline_acc:
        :return:
        """
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
        """
        Fits pipeline and calculates accuracy
        :param data_column:
        :return:
        """
        print('Start fiting')
        self.pipeline.fit(self.data_training, self.data_training[data_column].to_numpy())
        print(f'Saving classifier to file {self.classifier_file}')
        self.save_classifier(self.classifier_file)
        print(f'best gridsearch params:\n{self.estimator.best_params_}')
        return self.predict_test_data(data_column)

    def predict_test_data(self, data_column):
        """
        Predicts test_data
        :param data_column: which column (opinion or sentiemnt [lang]) is it
        :return:
        """
        print('Start test prediction')
        start = time.time()
        y_pred_pipe = self.pipeline.predict(self.data_test)
        end = time.time()
        print(f'time needed: {end - start}')

        acc = accuracy_score(self.data_test[data_column].to_numpy(), y_pred_pipe)
        f1 = f1_score(self.data_test[data_column].to_numpy(), y_pred_pipe, average='weighted')
        rec = recall_score(self.data_test[data_column].to_numpy(), y_pred_pipe, average='weighted')
        precision = precision_score(self.data_test[data_column].to_numpy(), y_pred_pipe, average='weighted')
        return acc, f1, rec, precision

    def fit_and_predict_and_calculate_confidence_pipe(self, data_column):
        """
        Fits pipeline and calculates confidence and different scores
        :param data_column:
        :return:
        """
        print('Start fiting')
        self.pipeline.fit(self.data_training, self.data_training[data_column].to_numpy())
        print(f'Saving classifier to file {self.classifier_file}')
        self.save_classifier(self.classifier_file)
        print(f'best gridsearch params:\n{self.estimator.best_params_}')
        return self.predict_test_data_confidence(data_column)


    def predict_test_data_confidence(self, data_column):
        """
        Calculates f1-Score, recall, precision and accruacy for 60%, 70% and 80% confidence for test data.
        :param data_column: name of the data column in test data
        :return: list of f1-Score, recall, precision and accruacy for 60%, 70% and 80% confidence
        """
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
        df.to_excel(f'{self.result_dir}max_confidence_{data_column}.xlsx', index=False)
        df2.to_excel(f'{self.result_dir}classes_and_confidence_{data_column}.xlsx', index=False)

        result_list = list()
        print("Entries before filtering for confidence: " + str(df2.shape[0]))
        thresholds = [0.6, 0.7, 0.8]
        for threshold in thresholds:
            df3 = self.filter_threshold(df2, threshold)
                    # dummy Zeile wieder löschen
            df3.drop(df3.head(1).index, inplace=True)

            print("Filtered for confidence " + str(threshold) + "+. Remaining Entries: " + str(df3.shape[0]))

            #die jeweiligen df3 speichern
            df3.to_excel(f'{self.result_dir}confidence{threshold}_{data_column}.xlsx', index=False)

            acc = accuracy_score(df3['real sentiment label'], df3['prediction'])
            f1 = f1_score(df3['real sentiment label'], df3['prediction'], average='weighted')
            rec = recall_score(df3['real sentiment label'], df3['prediction'], average='weighted')
            precision = precision_score(df3['real sentiment label'], df3['prediction'], average='weighted')
            result_list.append((threshold, acc, f1, rec, precision, df2.shape[0], df3.shape[0]))

        return result_list

    def filter_threshold(self, dataframe, threshold):
        """
        Filters entries of the dataframe by threshold
        :param dataframe: dataframe to filter
        :param threshold: threshold
        :return: filterd dataframe
        """
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
        """
        predicts data. class with higher confidence is predicted
        :param df: dataframe with confidence
        :return: list with prediction
        """
        class_predictions = list()
        for index, row in df.iterrows():
            if row['confidence class 0'] >= row['confidence class 1']:
                class_predictions.append(0)
            else:
                class_predictions.append(1)
        return class_predictions


    def save_prediction(self, prediction, data_column, data):
        """
        Saves the original data with the prediciton as extra column
        :param prediction: prediction
        :param data_column: which column is predicted
        :param data: original data
        :return: nothing
        """
        pred_data_frame = data.copy(deep=True)
        pred_data_frame[f'prediction_{data_column}'] = prediction
        pred_data_frame.to_excel(f'{self.result_dir}prediction_for_{data_column}.xlsx', index=False)
        pass


    def save_prediction_confidence(self, prediction, data_column, data):
        """
        Saves the original data with the prediciton as extra column
        :param prediction: prediction
        :param data_column: which column is predicted
        :param data: original data
        :return: nothing
        """
        pred_data_frame = data.copy(deep=True)
        thresholds = [0.5, 0.6, 0.7, 0.8]
        pred_data_frame[f'prediction_{data_column}_confidence_class0'] = [p[0] for p in prediction]
        pred_data_frame[f'prediction_{data_column}_confidence_class1'] = [p[1] for p in prediction]
        for threshold in thresholds:
            pred_data_frame[f'prediction_{data_column}_{threshold}'] = [self.filter_prediction(p, threshold) for p in prediction]
        pred_data_frame.to_excel(f'{self.result_dir}prediction_confidence_for_{data_column}.xlsx', index=False)
        pass

    def filter_prediction(self, p, threshold):
        """
        Filters prediction
        :param p: confidence of classes
        :param threshold: threshold
        :return: 0 if class_0-confidence >= threshold, 1 if class_1-confidence >= threshold, else n.a.
        """
        if p[0] >= threshold:
            return 0
        elif p[1] >= threshold:
            return 1
        else:
            return 'n.a.'
        pass

    def write_result_to_file(self, accuracy, f1, recall, precision, description):
        """
        Writes accuracy, f1, recall and precision to log file.
        :param accuracy:
        :param f1:
        :param recall:
        :param precision:
        :param description: description of classifier
        :return:
        """
        with open(self.log_file, 'a', encoding='utf-8') as file:
            file.write('#------------------------------------------------------------------------------------------\n')
            file.write(f'{datetime.now().strftime("%b-%d-%Y %H:%M:%S")}\n')
            file.write(f'{description}\n')
            file.write(f'\t\tAccuracy: {accuracy}\n')
            file.write(f'\t\tF1 Score: {f1}\n')
            file.write(f'\t\tRecall Score: {recall}\n')
            file.write(f'\t\tPrecision Score: {precision}\n')
            file.write('#------------------------------------------------------------------------------------------\n')

    def write_result_to_file_confidence(self, threshold, accuracy, f1, recall, precision, description, num_all_entries, num_entries):
        """
        Writes accuracy, f1, recall and precision to log file with a given threshold
        :param threshold
        :param accuracy:
        :param f1:
        :param recall:
        :param precision:
        :param description: description of classifier
        :return:
        """
        with open(self.log_file, 'a', encoding='utf-8') as file:
            file.write('#------------------------------------------------------------------------------------------\n')
            file.write(f'{datetime.now().strftime("%b-%d-%Y %H:%M:%S")}\n')
            file.write(f'for threshhold: {threshold}\n')
            file.write(f'Dropped Entries: {num_all_entries - num_entries}\n')
            file.write(f'Remaining Entries: {num_entries} of {num_all_entries}\n')
            file.write(f'{description}\n')
            file.write(f'\t\tAccuracy: {accuracy}\n')
            file.write(f'\t\tF1 Score: {f1}\n')
            file.write(f'\t\tRecall Score: {recall}\n')
            file.write(f'\t\tPrecision Score: {precision}\n')
            file.write('#------------------------------------------------------------------------------------------\n')

    def predict_data(self, data_file_name, result_for_column, column_to_transform=None, new_column_name=None):
        """
        Predicts given data without confidence
        :param data_file_name: file name of data file
        :param column_to_transform: column name which contains data if it is text
        :param new_column_name: if it is text rename it to this if not it will look for sentence or sentences (case insensitive)
        :param result_for_column:
        :return: nothing
        """
        data_validation = self.file_reader.read_file(data_file_name)

        if not new_column_name in data_validation.columns:
            print('Data does not contain a ' + new_column_name + ' column')
            for column in ['sentence', 'Sentence', 'sentences', 'Sentences']:
                if column in data_validation.columns:
                    new_column_name = column
                    print('Found a column named ' + column + '. Proceeding with found column instead.')

        if column_to_transform is not None:
            for obj in self.pipeline.named_steps.values():
                if issubclass(type(obj), ColumnTransformer):
                    obj.set_column_to_transform(column_to_transform)

        if new_column_name is not None:
            for obj in self.pipeline.named_steps.values():
                if issubclass(type(obj), ColumnUser):
                    obj.set_column_to_use(new_column_name)

        output = self.pipeline.predict(data_validation)
        if result_for_column != '':
            self.save_prediction(output, result_for_column, data_validation)

    def predict_data_confidence(self, data_file_name, result_for_column, column_to_transform=None, new_column_name=None):
        """
        Predicts given data with confidence
        :param data_file_name: file name of data file
        :param column_to_transform: column name which contains data if it is text
        :param new_column_name: if it is text rename it to this if not it will look for sentence or sentences (case insensitive)
        :param result_for_column:
        :return: nothing
        """
        data_validation = self.file_reader.read_file(data_file_name)

        if not new_column_name in data_validation.columns:
            print('Data does not contain a ' + new_column_name + ' column')
            for column in ['sentence', 'Sentence', 'sentences', 'Sentences']:
                if column in data_validation.columns:
                    new_column_name = column
                    print('Found a column named ' + column + '. Proceeding with found column instead.')

        if column_to_transform is not None:
            for obj in self.pipeline.named_steps.values():
                if issubclass(type(obj), ColumnTransformer):
                    obj.set_column_to_transform(column_to_transform)

        if new_column_name is not None:
            for obj in self.pipeline.named_steps.values():
                if issubclass(type(obj), ColumnUser):
                    obj.set_column_to_use(new_column_name)

        output = self.pipeline.predict_proba(data_validation)
        if result_for_column != '':
            self.save_prediction_confidence(output, result_for_column, data_validation)


def init_and_run_pipeline():
    # default values
    default_log_file_name_opinion = f'data/results/result_opinion_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'
    default_log_file_name_sentiment = f'data/results/result_sentiment_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'

    default_training_file_opinion = 'AllOpinionDatasetRF_train.xlsx'
    default_test_file_opinion = 'AllOpinionDatasetRF_test.xlsx'
    default_result_dir_opinion = 'data/results/'
    default_training_file_sentiment = 'data/datasetSentimentSRF_train.xlsx'
    default_test_file_sentiment = 'data/datasetSentimentSRF_test.xlsx'
    default_result_dir_sentiment = 'data/results/'

    if len(sys.argv) > 1 and (sys.argv[1] == '-i' or sys.argv[1] == '--interactive'):
        # get parameter for opinion
        training_file_opinion = input(
            f'enter file path to training file for opinion [default {default_training_file_opinion}]: ')
        if not training_file_opinion:
            training_file_opinion = default_training_file_opinion
        test_file_opinion = input(f'enter file path to test file for opinion [default {default_test_file_opinion}]: ')
        if not test_file_opinion:
            test_file_opinion = default_test_file_opinion
        log_file_opinion = input(f'enter file path to log file for opinion [default {default_log_file_name_opinion}]: ')
        if not log_file_opinion:
            log_file_opinion = default_log_file_name_opinion
        result_dir_opinion = input(
            f'enter file path to result directory for opinion [default {default_result_dir_opinion}]: ')
        if not result_dir_opinion:
            result_dir_opinion = default_result_dir_opinion
        fitting = input('Do you want to force new fitting of classifier for opinion. This will overwrite [y, [n]]: ')
        force_fitting_opinion = (fitting == 'y')

        # get parameter for sentiment
        training_file_sentiment = input(
            f'enter file path to training file for sentiment [default {default_training_file_sentiment}]: ')
        if not training_file_sentiment:
            training_file_sentiment = default_training_file_sentiment
        test_file_sentiment = input(
            f'enter file path to test file for sentiment [default {default_test_file_sentiment}]: ')
        if not test_file_sentiment:
            test_file_sentiment = default_test_file_sentiment
        log_file_sentiment = input(
            f'enter file path to log file for sentiment [default {default_log_file_name_sentiment}]: ')
        if not log_file_sentiment:
            log_file_sentiment = default_log_file_name_sentiment
        result_dir_sentiment = input(
            f'enter file path to result directory for sentiment [default {default_result_dir_sentiment}]: ')
        if not result_dir_sentiment:
            result_dir_sentiment = default_result_dir_sentiment
        fitting = input('Do you want to force new fitting of classifier for sentiment. This will overwrite [y, [n]]: ')
        force_fitting_sentiment = (fitting == 'y')

        use_confidence = input('Do you want confidence values? [[y], n]')
        data_file = input('enter file path to data file: ')
    else:
        # parameters for opinion
        training_file_opinion = default_training_file_opinion
        test_file_opinion = default_test_file_opinion
        log_file_opinion = default_log_file_name_opinion
        result_dir_opinion = default_result_dir_opinion

        # parameters for sentiment
        training_file_sentiment = default_training_file_sentiment
        test_file_sentiment = default_test_file_sentiment
        log_file_sentiment = default_log_file_name_sentiment
        result_dir_sentiment = default_result_dir_sentiment
        use_confidence = 'y'
        force_fitting_opinion = False
        force_fitting_sentiment = False

        data_file = ''
        if not data_file:
            raise RuntimeError(
                'use script in interacitve mode (-i or --interactive) or modify this script with a data file')

    # advanced description of classifier
    # change in script could be used to use different versions
    classifier_description = 'from_2021-08-22'

    # file path to dictionary
    dict_file = 'data/dict.csv'

    transformers_list = [TextToSentenceTransformer('text', 'Sentence'),
                         BertTransformer('Sentence', batchsize=10),
                         PreprocessorTransformer('Sentence'),
                         SentimentOpinionValueCalculatorTransformer(dict_file)]

    Cs = np.logspace(-6, 6, 200)
    max_iter = 500

    param_grid_log_reg = [
        # Standard Konfiguration
        {
            'solver': ['lbfgs'],
            'C': Cs
        },
    ]

    if use_confidence == 'n':
        # SUBJopin
        pipeline_runner = PipelineRunner(dict_file, training_file_opinion, test_file_opinion,
                                         log_file=log_file_opinion, result_dir=result_dir_opinion)
        log_reg = LogisticRegression(max_iter=max_iter)
        pipeline_runner.make_pipeline(transformers_list, log_reg, 'SUBJopin01', param_grid_log_reg,
                                      classifier_description='from_2021-08-22', force_fitting=force_fitting_opinion)

        print("Running prediction for opinion...")
        pipeline_runner.predict_data(data_file_name=data_file,
                                     result_for_column='SUBJopin01',
                                     new_column_name='Sentence')

        # SUBJlang
        pipeline_runner = PipelineRunner(dict_file, training_file_sentiment, test_file_sentiment,
                                         log_file=log_file_sentiment, result_dir=result_dir_sentiment)
        log_reg = LogisticRegression(max_iter=max_iter)
        pipeline_runner.make_pipeline(transformers_list, log_reg, 'SUBJlang01', param_grid_log_reg,
                                      classifier_description='from_2021-08-22', force_fitting=force_fitting_sentiment)

        print("Running prediction for sentiment...")
        pipeline_runner.predict_data(data_file_name=data_file,
                                     result_for_column='SUBJlang01',
                                     new_column_name='Sentence')

    else:
        # SUBJopin
        pipeline_runner = PipelineRunner(dict_file, training_file_opinion, test_file_opinion,
                                         log_file=log_file_opinion, result_dir=result_dir_opinion)
        log_reg = LogisticRegression(max_iter=max_iter)
        pipeline_runner.make_pipeline_confidence(transformers_list, log_reg, 'SUBJopin01', param_grid_log_reg,
                                                 classifier_description='from_2021-08-22', force_fitting=force_fitting_opinion)

        print("Running prediction for opinion...")
        pipeline_runner.predict_data_confidence(data_file_name=data_file,
                                                result_for_column='SUBJopin01',
                                                new_column_name='Sentence')

        # SUBJlang
        pipeline_runner = PipelineRunner(dict_file, training_file_sentiment, test_file_sentiment,
                                         log_file=log_file_sentiment, result_dir=result_dir_sentiment)
        log_reg = LogisticRegression(max_iter=max_iter)
        pipeline_runner.make_pipeline_confidence(transformers_list, log_reg, 'SUBJlang01', param_grid_log_reg,
                                                 classifier_description='from_2021-08-22', force_fitting=force_fitting_sentiment)

        print("Running prediction for sentiment...")
        pipeline_runner.predict_data_confidence(data_file_name=data_file,
                                                result_for_column='SUBJlang01',
                                                new_column_name='Sentence')


if __name__ == '__main__':
    init_and_run_pipeline()

