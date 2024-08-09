import copy
import gc
import string
from collections import defaultdict, Counter

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm

from l0_settings import Settings

global_settings = Settings()

import re
import warnings

import pandas as pd
import numpy as np

from scipy.stats import kurtosis

warnings.filterwarnings("ignore")


def processingInputs(currTextInput):
    essayText = ""
    for Input in currTextInput.values:
        # If activity = Replace
        if Input[0] == 'Replace':
            # splits text_change at ' => '
            replaceTxt = Input[2].split(' => ')
            essayText = essayText[:Input[1] - len(replaceTxt[1])] + replaceTxt[1] + essayText[
                                                                                    Input[1] - len(replaceTxt[1]) + len(
                                                                                        replaceTxt[0]):]
            continue

        # If activity = Paste
        if Input[0] == 'Paste':
            essayText = essayText[:Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]):]
            continue

        # If activity = Remove/Cut
        if Input[0] == 'Remove/Cut':
            essayText = essayText[:Input[1]] + essayText[Input[1] + len(Input[2]):]
            continue

        # If activity = Move...
        if "M" in Input[0]:
            # Gets rid of the "Move from to" text
            croppedTxt = Input[0][10:]
            # Splits cropped text by ' To '
            splitTxt = croppedTxt.split(' To ')
            # Splits split text again by ', ' for each item
            valueArr = [item.split(', ') for item in splitTxt]
            # Move from [2, 4] To [5, 7] = (2, 4, 5, 7)
            moveData = (
                int(valueArr[0][0][1:]), int(valueArr[0][1][:-1]), int(valueArr[1][0][1:]), int(valueArr[1][1][:-1]))
            # Skip if someone manages to activiate this by moving to same place
            if moveData[0] != moveData[2]:
                # Check if they move text forward in essay (they are different)
                if moveData[0] < moveData[2]:
                    essayText = essayText[:moveData[0]] + essayText[moveData[1]:moveData[3]] + \
                                essayText[moveData[0]:moveData[1]] + essayText[moveData[3]:]
                else:
                    essayText = essayText[:moveData[2]] + essayText[moveData[0]:moveData[1]] + \
                                essayText[moveData[2]:moveData[0]] + essayText[moveData[1]:]
            continue

        # If activity = input
        essayText = essayText[:Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]):]
    return essayText


def getEssays(df):
    # Copy required columns
    textInputDf = copy.deepcopy(df[['id', 'activity', 'cursor_position', 'text_change']])
    # Get rid of text inputs that make no change
    textInputDf = textInputDf[textInputDf.activity != 'Nonproduction']
    # construct essay, fast
    tqdm.pandas()
    essay = textInputDf.groupby('id')[['activity', 'cursor_position', 'text_change']].progress_apply(
        lambda x: processingInputs(x))
    # to dataframe
    essayFrame = essay.to_frame().reset_index()
    essayFrame.columns = ['id', 'essay']
    # Returns the essay series
    return essayFrame


# Helper functions for feature engineering
def q1(x):
    return x.quantile(0.25)


def q3(x):
    return x.quantile(0.75)


AGGREGATIONS = ['count', 'mean', 'std', 'min', 'max', 'first', 'last', 'sem', q1, 'median', q3, 'skew', kurtosis, 'sum']


def split_essays_into_words(df):
    essay_df = df
    essay_df['word'] = essay_df['essay'].apply(lambda x: re.split(' |\\n|\\.|\\?|\\!', x))
    essay_df = essay_df.explode('word')
    essay_df['word_len'] = essay_df['word'].apply(lambda x: len(x))
    essay_df = essay_df[essay_df['word_len'] != 0]
    return essay_df


def compute_word_aggregations(word_df):
    word_agg_df = word_df[['id', 'word_len']].groupby(['id']).agg(AGGREGATIONS)
    word_agg_df.columns = ['_'.join(x) for x in word_agg_df.columns]
    word_agg_df['id'] = word_agg_df.index
    for word_l in [5, 6, 7, 8, 9, 10, 11, 12]:
        word_agg_df[f'word_len_ge_{word_l}_count'] = word_df[word_df['word_len'] >= word_l].groupby(
            ['id']).count().iloc[:, 0]
        word_agg_df[f'word_len_ge_{word_l}_count'] = word_agg_df[f'word_len_ge_{word_l}_count'].fillna(0)
    word_agg_df = word_agg_df.reset_index(drop=True)
    return word_agg_df


def split_essays_into_sentences(df):
    essay_df = df
    # essay_df['id'] = essay_df.index
    essay_df['sent'] = essay_df['essay'].apply(lambda x: re.split('\\.|\\?|\\!', x))
    essay_df = essay_df.explode('sent')
    essay_df['sent'] = essay_df['sent'].apply(lambda x: x.replace('\n', '').strip())
    # Number of characters in sentences
    essay_df['sent_len'] = essay_df['sent'].apply(lambda x: len(x))
    # Number of words in sentences
    essay_df['sent_word_count'] = essay_df['sent'].apply(lambda x: len(x.split(' ')))
    essay_df = essay_df[essay_df.sent_len != 0].reset_index(drop=True)
    return essay_df


def compute_sentence_aggregations(df):
    sent_agg_df = pd.concat(
        [df[['id', 'sent_len']].groupby(['id']).agg(AGGREGATIONS),
         df[['id', 'sent_word_count']].groupby(['id']).agg(AGGREGATIONS)], axis=1
    )
    sent_agg_df.columns = ['_'.join(x) for x in sent_agg_df.columns]
    sent_agg_df['id'] = sent_agg_df.index

    # New features intoduced here: https://www.kaggle.com/code/mcpenguin/writing-processes-to-quality-baseline-v2
    for sent_l in [50, 60, 75, 100]:
        sent_agg_df[f'sent_len_ge_{sent_l}_count'] = df[df['sent_len'] >= sent_l].groupby(['id']).count().iloc[:, 0]
        sent_agg_df[f'sent_len_ge_{sent_l}_count'] = sent_agg_df[f'sent_len_ge_{sent_l}_count'].fillna(0)

    sent_agg_df = sent_agg_df.reset_index(drop=True)
    sent_agg_df.drop(columns=["sent_word_count_count"], inplace=True)
    sent_agg_df = sent_agg_df.rename(columns={"sent_len_count": "sent_count"})
    return sent_agg_df


def split_essays_into_paragraphs(df):
    essay_df = df
    # essay_df['id'] = essay_df.index
    essay_df['paragraph'] = essay_df['essay'].apply(lambda x: x.split('\n'))
    essay_df = essay_df.explode('paragraph')
    # Number of characters in paragraphs
    essay_df['paragraph_len'] = essay_df['paragraph'].apply(lambda x: len(x))
    # Number of words in paragraphs
    essay_df['paragraph_word_count'] = essay_df['paragraph'].apply(lambda x: len(x.split(' ')))
    essay_df = essay_df[essay_df.paragraph_len != 0].reset_index(drop=True)
    return essay_df


def compute_paragraph_aggregations(df):
    paragraph_agg_df = pd.concat(
        [df[['id', 'paragraph_len']].groupby(['id']).agg(AGGREGATIONS),
         df[['id', 'paragraph_word_count']].groupby(['id']).agg(AGGREGATIONS)], axis=1
    )
    paragraph_agg_df.columns = ['_'.join(x) for x in paragraph_agg_df.columns]
    paragraph_agg_df['id'] = paragraph_agg_df.index
    paragraph_agg_df = paragraph_agg_df.reset_index(drop=True)
    paragraph_agg_df.drop(columns=["paragraph_word_count_count"], inplace=True)
    paragraph_agg_df = paragraph_agg_df.rename(columns={"paragraph_len_count": "paragraph_count"})
    return paragraph_agg_df


train_logs = pd.read_csv(global_settings.get_train_logs_path())

train_essays = getEssays(train_logs)

# Word features for train dataset
train_word_df = split_essays_into_words(train_essays)
train_word_agg_df = compute_word_aggregations(train_word_df)

# Sentence features for train dataset
train_sent_df = split_essays_into_sentences(train_essays)
train_sent_agg_df = compute_sentence_aggregations(train_sent_df)

# Paragraph features for train dataset
train_paragraph_df = split_essays_into_paragraphs(train_essays)
train_paragraph_agg_df = compute_paragraph_aggregations(train_paragraph_df)

del train_essays, train_word_df, train_sent_df, train_paragraph_df
gc.collect()

test_logs = pd.read_csv(global_settings.get_test_logs_path())

# Features for test dataset
test_essays = getEssays(test_logs)
test_word_agg_df = compute_word_aggregations(split_essays_into_words(test_essays))
test_sent_agg_df = compute_sentence_aggregations(split_essays_into_sentences(test_essays))
test_paragraph_agg_df = compute_paragraph_aggregations(split_essays_into_paragraphs(test_essays))

del test_essays
gc.collect()

# Keeping the states of these objects global to reuse them in the test data
count_vect = CountVectorizer()
tfidf_vect = TfidfVectorizer()

# Define the n-gram range
ngram_range = (1, 2)  # Example: Use unigrams and bigrams
count_vect_ngram = CountVectorizer(ngram_range=ngram_range)
tfidf_vect_ngram = TfidfVectorizer(ngram_range=ngram_range)


def make_text_features(df, name="Train Logs"):
    def count_encoding_ngram(essays_as_string, name=name):
        """Applies Count Encoding to the essay data and returns a DataFrame with prefixed column names."""

        if name == "Train Logs":
            features = count_vect_ngram.fit_transform(essays_as_string)
        else:
            features = count_vect_ngram.transform(essays_as_string)

        feature_names = [f'bow-{ngram}' for ngram in count_vect_ngram.get_feature_names_out()]
        return pd.DataFrame(features.toarray(), columns=feature_names)

    def tfidf_encoding_ngram(essays_as_string, name=name):
        """Applies TF-IDF Encoding to the essay data and returns a DataFrame with prefixed column names."""

        if name == "Train Logs":
            features = tfidf_vect_ngram.fit_transform(essays_as_string)
        else:
            features = tfidf_vect_ngram.transform(essays_as_string)

        feature_names = [f'tfidf-{ngram}' for ngram in tfidf_vect_ngram.get_feature_names_out()]
        return pd.DataFrame(features.toarray(), columns=feature_names)

    def count_encoding(essays_as_string, name=name):
        """Applies Count Encoding to the essay data and returns a DataFrame with prefixed column names."""

        if name == "Train Logs":
            features = count_vect.fit_transform(essays_as_string)
        else:
            features = count_vect.transform(essays_as_string)

        feature_names = [f'bow_{name}' for name in count_vect.get_feature_names_out()]
        return pd.DataFrame(features.toarray(), columns=feature_names)

    def tfidf_encoding(essays_as_string, name=name):
        """Applies TF-IDF Encoding to the essay data and returns a DataFrame with prefixed column names."""

        if name == "Train Logs":
            features = tfidf_vect.fit_transform(essays_as_string)
        else:
            features = tfidf_vect.transform(essays_as_string)

        feature_names = [f'tfidf_{name}' for name in tfidf_vect.get_feature_names_out()]
        return pd.DataFrame(features.toarray(), columns=feature_names)

    def custom_feature_engineering(essays):
        """Example custom feature: calculates the length of each essay with prefixed column name."""

        custom_text_features = {
            'custom_length': [len(essay) for essay in essays],
            'custom_word_count': [len(essay.split()) for essay in essays],
            'custom_unique_word_count': [len(set(essay.split())) for essay in essays],
            'custom_punctuation_count': [sum([1 for char in essay if char in string.punctuation]) for essay in essays],
            'custom_paragraph_count': [essay.count('\n') + 1 for essay in essays],
            'custom_sentence_count': [essay.count('.') + 1 for essay in essays],
            'custom_comma_count': [essay.count(',') for essay in essays],
            'custom_question_mark_count': [essay.count('?') for essay in essays],
            'custom_exclamation_mark_count': [essay.count('!') for essay in essays],
            'custom_colon_count': [essay.count(':') for essay in essays],
            'custom_semicolon_count': [essay.count(';') for essay in essays],
            'custom_dash_count': [essay.count('-') for essay in essays],
            'custom_quote_count': [essay.count('"') for essay in essays],
            'custom_apostrophe_count': [essay.count("'") for essay in essays],
            'custom_parenthesis_count': [essay.count('(') + essay.count(')') for essay in essays],
            'custom_bracket_count': [essay.count('[') + essay.count(']') for essay in essays],
            'custom_brace_count': [essay.count('{') + essay.count('}') for essay in essays],
            'custom_mathematical_symbol_count': [
                essay.count('+') + essay.count('-') + essay.count('*') + essay.count('/') for essay in essays],
            'custom_digit_count': [sum([1 for char in essay if char.isdigit()]) for essay in essays],
            'custom_average_word_length': [np.mean([len(word) for word in essay.split()]) for essay in essays],
            'custom_average_sentence_length': [np.mean([len(sentence) for sentence in essay.split('.')]) for essay in
                                               essays],
            'custom_average_paragraph_length': [np.mean([len(paragraph) for paragraph in essay.split('\n')]) for essay
                                                in essays],
            'custom_average_word_count_per_sentence': [np.mean([len(sentence.split()) for sentence in essay.split('.')])
                                                       for essay in essays],
            'custom_average_word_count_per_paragraph': [
                np.mean([len(paragraph.split()) for paragraph in essay.split('\n')]) for essay in essays],
            'custom_average_sentence_count_per_paragraph': [
                np.mean([paragraph.count('.') + 1 for paragraph in essay.split('\n')]) for essay in essays],
            'custom_average_punctuation_count_per_sentence': [
                np.mean([sum([1 for char in sentence if char in string.punctuation]) for sentence in essay.split('.')])
                for essay in essays],
            'custom_average_punctuation_count_per_paragraph': [np.mean(
                [sum([1 for char in paragraph if char in string.punctuation]) for paragraph in essay.split('\n')]) for
                essay in essays],
            'custom_average_digit_count_per_sentence': [
                np.mean([sum([1 for char in sentence if char.isdigit()]) for sentence in essay.split('.')]) for essay in
                essays],
            'custom_average_digit_count_per_paragraph': [
                np.mean([sum([1 for char in paragraph if char.isdigit()]) for paragraph in essay.split('\n')]) for essay
                in essays],
            'custom_average_mathematical_symbol_count_per_sentence': [np.mean(
                [sum([1 for char in sentence if char in ['+', '-', '*', '/']]) for sentence in essay.split('.')]) for
                essay in essays],
            'custom_average_mathematical_symbol_count_per_paragraph': [np.mean(
                [sum([1 for char in paragraph if char in ['+', '-', '*', '/']]) for paragraph in essay.split('\n')]) for
                essay in essays],
        }
        return pd.DataFrame(custom_text_features)

    def merge_features(data, name):
        """Merges features from different methods into one DataFrame with the id column."""
        essays_as_string = data['essay']

        # Extract features
        bow_df = count_encoding(essays_as_string, name)
        tfidf_df = tfidf_encoding(essays_as_string, name)
        custom_features_df = custom_feature_engineering(data['essay'])

        count_encoding_ngram_features = count_encoding_ngram(essays_as_string, name=name)
        # tfidf_encoding_ngram_features = tfidf_encoding_ngram(essays_as_string, name=name)

        # Merge all features
        merged_features = pd.concat([data[['id']], bow_df, tfidf_df, custom_features_df,
                                     # count_encoding_ngram_features,
                                     # tfidf_encoding_ngram_features
                                     ], axis=1)
        return merged_features

    return merge_features(getEssays(df), name)


class FeatureMaker:

    def __init__(self, seed):
        self.seed = seed

        self.activities = ['Input', 'Remove/Cut', 'Nonproduction', 'Replace', 'Paste']
        self.events = ['q', 'Space', 'Backspace', 'Shift', 'ArrowRight', 'Leftclick', 'ArrowLeft', '.', ',',
                       'ArrowDown', 'ArrowUp', 'Enter', 'CapsLock', "'", 'Delete', 'Unidentified']
        self.text_changes = ['q', ' ', 'NoChange', '.', ',', '\n', "'", '"', '-', '?', ';', '=', '/', '\\', ':']
        self.punctuations = ['"', '.', ',', "'", '-', ';', ':', '?', '!', '<', '>', '/',
                             '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+']
        self.gaps = [1, 2, 3, 5, 10, 20, 50, 100]

        self.idf = defaultdict(float)

    def activity_counts(self, df):
        tmp_df = df.groupby('id').agg({'activity': list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df['activity'].values):
            items = list(Counter(li).items())
            di = dict()
            for k in self.activities:
                di[k] = 0
            for item in items:
                k, v = item[0], item[1]
                if k in di:
                    di[k] = v
            ret.append(di)
        ret = pd.DataFrame(ret)
        cols = [f'activity_{i}_count' for i in range(len(ret.columns))]
        ret.columns = cols

        cnts = ret.sum(1)

        for col in cols:
            if col in self.idf.keys():
                idf = self.idf[col]
            else:
                idf = df.shape[0] / (ret[col].sum() + 1)
                idf = np.log(idf)
                self.idf[col] = idf

            ret[col] = 1 + np.log(ret[col] / cnts)
            ret[col] *= idf

        return ret

    def event_counts(self, df, colname):
        tmp_df = df.groupby('id').agg({colname: list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df[colname].values):
            items = list(Counter(li).items())
            di = dict()
            for k in self.events:
                di[k] = 0
            for item in items:
                k, v = item[0], item[1]
                if k in di:
                    di[k] = v
            ret.append(di)
        ret = pd.DataFrame(ret)
        cols = [f'{colname}_{i}_count' for i in range(len(ret.columns))]
        ret.columns = cols

        cnts = ret.sum(1)

        for col in cols:
            if col in self.idf.keys():
                idf = self.idf[col]
            else:
                idf = df.shape[0] / (ret[col].sum() + 1)
                idf = np.log(idf)
                self.idf[col] = idf

            ret[col] = 1 + np.log(ret[col] / cnts)
            ret[col] *= idf

        return ret

    def text_change_counts(self, df):
        tmp_df = df.groupby('id').agg({'text_change': list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df['text_change'].values):
            items = list(Counter(li).items())
            di = dict()
            for k in self.text_changes:
                di[k] = 0
            for item in items:
                k, v = item[0], item[1]
                if k in di:
                    di[k] = v
            ret.append(di)
        ret = pd.DataFrame(ret)
        cols = [f'text_change_{i}_count' for i in range(len(ret.columns))]
        ret.columns = cols

        cnts = ret.sum(1)

        for col in cols:
            if col in self.idf.keys():
                idf = self.idf[col]
            else:
                idf = df.shape[0] / (ret[col].sum() + 1)
                idf = np.log(idf)
                self.idf[col] = idf

            ret[col] = 1 + np.log(ret[col] / cnts)
            ret[col] *= idf

        return ret

    def match_punctuations(self, df):
        tmp_df = df.groupby('id').agg({'down_event': list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df['down_event'].values):
            cnt = 0
            items = list(Counter(li).items())
            for item in items:
                k, v = item[0], item[1]
                if k in self.punctuations:
                    cnt += v
            ret.append(cnt)
        ret = pd.DataFrame({'punct_cnt': ret})
        return ret

    def get_input_words(self, df):
        tmp_df = df[(~df['text_change'].str.contains('=>')) & (df['text_change'] != 'NoChange')].reset_index(drop=True)
        tmp_df = tmp_df.groupby('id').agg({'text_change': list}).reset_index()
        tmp_df['text_change'] = tmp_df['text_change'].apply(lambda x: ''.join(x))
        tmp_df['text_change'] = tmp_df['text_change'].apply(lambda x: re.findall(r'q+', x))
        tmp_df['input_word_count'] = tmp_df['text_change'].apply(len)
        tmp_df['input_word_length_mean'] = tmp_df['text_change'].apply(
            lambda x: np.mean([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df['input_word_length_max'] = tmp_df['text_change'].apply(
            lambda x: np.max([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df['input_word_length_std'] = tmp_df['text_change'].apply(
            lambda x: np.std([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df.drop(['text_change'], axis=1, inplace=True)
        return tmp_df

    def make_feats(self, df, name="Train Logs"):

        feats = pd.DataFrame({'id': df['id'].unique().tolist()})

        print("Engineering time data")
        for gap in self.gaps:
            df[f'up_time_shift{gap}'] = df.groupby('id')['up_time'].shift(gap)
            df[f'action_time_gap{gap}'] = df['down_time'] - df[f'up_time_shift{gap}']
        df.drop(columns=[f'up_time_shift{gap}' for gap in self.gaps], inplace=True)

        print("Engineering cursor position data")
        for gap in self.gaps:
            df[f'cursor_position_shift{gap}'] = df.groupby('id')['cursor_position'].shift(gap)
            df[f'cursor_position_change{gap}'] = df['cursor_position'] - df[f'cursor_position_shift{gap}']
            df[f'cursor_position_abs_change{gap}'] = np.abs(df[f'cursor_position_change{gap}'])
        df.drop(columns=[f'cursor_position_shift{gap}' for gap in self.gaps], inplace=True)

        print("Engineering word count data")
        for gap in self.gaps:
            df[f'word_count_shift{gap}'] = df.groupby('id')['word_count'].shift(gap)
            df[f'word_count_change{gap}'] = df['word_count'] - df[f'word_count_shift{gap}']
            df[f'word_count_abs_change{gap}'] = np.abs(df[f'word_count_change{gap}'])
        df.drop(columns=[f'word_count_shift{gap}' for gap in self.gaps], inplace=True)

        print("Engineering statistical summaries for features")
        feats_stat = [
            ('event_id', ['max']),
            ('up_time', ['max']),
            ('action_time', ['max', 'min', 'mean', 'std', 'quantile', 'sem', 'sum', 'skew', kurtosis]),
            ('activity', ['nunique']),
            ('down_event', ['nunique']),
            ('up_event', ['nunique']),
            ('text_change', ['nunique']),
            ('cursor_position', ['nunique', 'max', 'quantile', 'sem', 'mean']),
            ('word_count', ['nunique', 'max', 'quantile', 'sem', 'mean'])]
        for gap in self.gaps:
            feats_stat.extend([
                (f'action_time_gap{gap}', ['max', 'min', 'mean', 'std', 'quantile', 'sem', 'sum', 'skew', kurtosis]),
                (f'cursor_position_change{gap}', ['max', 'mean', 'std', 'quantile', 'sem', 'sum', 'skew', kurtosis]),
                (f'word_count_change{gap}', ['max', 'mean', 'std', 'quantile', 'sem', 'sum', 'skew', kurtosis])
            ])

        pbar = tqdm(feats_stat)
        for item in pbar:
            colname, methods = item[0], item[1]
            for method in methods:
                pbar.set_postfix()
                if isinstance(method, str):
                    method_name = method
                else:
                    method_name = method.__name__
                pbar.set_postfix(column=colname, method=method_name)
                tmp_df = df.groupby(['id']).agg({colname: method}).reset_index().rename(
                    columns={colname: f'{colname}_{method_name}'})
                feats = feats.merge(tmp_df, on='id', how='left')

        print("Engineering activity counts data")
        tmp_df = self.activity_counts(df)
        feats = pd.concat([feats, tmp_df], axis=1)

        print("Engineering event counts data")
        tmp_df = self.event_counts(df, 'down_event')
        feats = pd.concat([feats, tmp_df], axis=1)
        tmp_df = self.event_counts(df, 'up_event')
        feats = pd.concat([feats, tmp_df], axis=1)

        print("Engineering text change counts data")
        tmp_df = self.text_change_counts(df)
        feats = pd.concat([feats, tmp_df], axis=1)

        print("Engineering punctuation counts data")
        tmp_df = self.match_punctuations(df)
        feats = pd.concat([feats, tmp_df], axis=1)

        print("Engineering input words data")
        tmp_df = self.get_input_words(df)
        feats = pd.merge(feats, tmp_df, on='id', how='left')

        print("Engineering ratios data")
        feats['word_time_ratio'] = feats['word_count_max'] / feats['up_time_max']
        feats['word_event_ratio'] = feats['word_count_max'] / feats['event_id_max']
        feats['event_time_ratio'] = feats['event_id_max'] / feats['up_time_max']
        feats['idle_time_ratio'] = feats['action_time_gap1_sum'] / feats['up_time_max']

        # print("Engineering text features")
        tmp_df = make_text_features(df, name)
        feats = pd.merge(feats, tmp_df, on='id', how='left')

        return feats


feature_maker = FeatureMaker(seed=global_settings.global_seed)

# train_logs = pd.read_csv(global_settings.get_train_logs_path())
# test_logs = pd.read_csv(global_settings.get_test_logs_path())
train_scores = pd.read_csv(global_settings.get_train_scores_path())

train_feats = feature_maker.make_feats(train_logs, name="Train Logs")
test_feats = feature_maker.make_feats(test_logs, name="Test Logs")

nan_cols = train_feats.columns[train_feats.isna().any()].tolist()

train_feats = train_feats.drop(columns=nan_cols)
test_feats = test_feats.drop(columns=nan_cols)

train_agg_fe_df = train_logs.groupby("id")[
    ['down_time', 'up_time', 'action_time', 'cursor_position', 'word_count']].agg(
    ['mean', 'std', 'min', 'max', 'last', 'first', 'sem', 'median', 'sum'])
train_agg_fe_df.columns = ['_'.join(x) for x in train_agg_fe_df.columns]
train_agg_fe_df = train_agg_fe_df.add_prefix("tmp_")
train_agg_fe_df.reset_index(inplace=True)

test_agg_fe_df = test_logs.groupby("id")[['down_time', 'up_time', 'action_time', 'cursor_position', 'word_count']].agg(
    ['mean', 'std', 'min', 'max', 'last', 'first', 'sem', 'median', 'sum'])
test_agg_fe_df.columns = ['_'.join(x) for x in test_agg_fe_df.columns]
test_agg_fe_df = test_agg_fe_df.add_prefix("tmp_")
test_agg_fe_df.reset_index(inplace=True)

train_feats = train_feats.merge(train_agg_fe_df, on='id', how='left')
test_feats = test_feats.merge(test_agg_fe_df, on='id', how='left')

del train_agg_fe_df, test_agg_fe_df
gc.collect()

data = []

for logs in [train_logs, test_logs]:
    logs['up_time_lagged'] = logs.groupby('id')['up_time'].shift(1).fillna(logs['down_time'])
    logs['time_diff'] = abs(logs['down_time'] - logs['up_time_lagged']) / 1000

    group = logs.groupby('id')['time_diff']
    largest_lantency = group.max()
    smallest_lantency = group.min()
    median_lantency = group.median()
    initial_pause = logs.groupby('id')['down_time'].first() / 1000
    pauses_half_sec = group.apply(lambda x: ((x > 0.5) & (x < 1)).sum())
    pauses_1_sec = group.apply(lambda x: ((x > 1) & (x < 1.5)).sum())
    pauses_1_half_sec = group.apply(lambda x: ((x > 1.5) & (x < 2)).sum())
    pauses_2_sec = group.apply(lambda x: ((x > 2) & (x < 3)).sum())
    pauses_3_sec = group.apply(lambda x: (x > 3).sum())

    data.append(pd.DataFrame({
        'id': logs['id'].unique(),
        'largest_lantency': largest_lantency,
        'smallest_lantency': smallest_lantency,
        'median_lantency': median_lantency,
        'initial_pause': initial_pause,
        'pauses_half_sec': pauses_half_sec,
        'pauses_1_sec': pauses_1_sec,
        'pauses_1_half_sec': pauses_1_half_sec,
        'pauses_2_sec': pauses_2_sec,
        'pauses_3_sec': pauses_3_sec,
    }).reset_index(drop=True))

train_eD592674, test_eD592674 = data

train_feats = train_feats.merge(train_eD592674, on='id', how='left')
test_feats = test_feats.merge(test_eD592674, on='id', how='left')
# train_feats = train_feats.merge(train_scores, on='id', how='left')

del train_eD592674, test_eD592674, train_logs, test_logs
gc.collect()

# Adding the additional features to the original feature set

train_feats = train_feats.merge(train_word_agg_df, on='id', how='left')
train_feats = train_feats.merge(train_sent_agg_df, on='id', how='left')
train_feats = train_feats.merge(train_paragraph_agg_df, on='id', how='left')

test_feats = test_feats.merge(test_word_agg_df, on='id', how='left')
test_feats = test_feats.merge(test_sent_agg_df, on='id', how='left')
test_feats = test_feats.merge(test_paragraph_agg_df, on='id', how='left')

del train_word_agg_df, train_sent_agg_df, train_paragraph_agg_df, test_word_agg_df
del test_sent_agg_df, test_paragraph_agg_df
gc.collect()

print('Writing to csv the raw features in the output directory')

# Create output directory if it doesn't exist
global_settings.raw_features_dir.mkdir(parents=True, exist_ok=True)

train_feats_with_scores = train_feats.merge(train_scores, on=global_settings.id_col, how='left')

# replace inf with nan
train_feats_with_scores.replace([np.inf, -np.inf], np.nan, inplace=True)
test_feats.replace([np.inf, -np.inf], np.nan, inplace=True)

# Write to csv files
train_feats_with_scores.to_csv(global_settings.raw_features_dir / 'raw_train_features_with_scores_v2.csv.gz',
                               index=False)
test_feats.to_csv(global_settings.raw_features_dir / 'raw_test_features_v2.csv.gz', index=False)

print('Done!')
