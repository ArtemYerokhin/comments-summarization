import sys
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from pymystem3 import Mystem
import re


TOPIC_THRESHOLD = 0.04
POST_THRESHOLD = 0.2

m = Mystem()
p = re.compile(r'[^а-яА-ЯёЁ]')
stop_words = stopwords.words('russian')
model_name = "ArtemYerokhin/ru_t5_comments_sum"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)


def clear_text(text):
    text = re.sub(p, ' ', text)
    text = text.split()
    return ' '.join(text)


def compare_texts(row):
    text1 = ''.join(m.lemmatize(clear_text(row['text_x'])))
    text2 = ''.join(m.lemmatize(clear_text(row['text_y'])))
    vectorized = CountVectorizer(stop_words=stop_words, ngram_range=(1, 1)).fit_transform([text1, text2])
    vectors = vectorized.toarray()
    similarity = cosine_similarity(vectors)[0, 1]
    return similarity


def summarize(texts):
    all_texts = ''.join(texts)
    input_ids = tokenizer(
        [all_texts],
        max_length=1024,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )["input_ids"]

    output_ids = model.generate(
        input_ids=input_ids,
        no_repeat_ngram_size=4
    )[0]

    summary = tokenizer.decode(output_ids, skip_special_tokens=True)
    return summary


def process(mode, input_file, output_file):
    df = pd.read_json(input_file, lines=True)
    res = df[['id', 'text', 'hash']].merge(df[['root_id', 'text', 'hash']], left_on='id', right_on='root_id')
    del df
    if mode != 'all_comments':
        res['similarity'] = res.apply(compare_texts, axis=1)
        if mode == 'topic_comments':
            res = res[res['similarity'] > TOPIC_THRESHOLD]
        elif mode == 'post_comments':
            res = res[res['similarity'] > POST_THRESHOLD]
    df_grouped = res.groupby(by=['id', 'text_x', 'hash_x'], as_index=False)[['text_y', 'hash_y']].agg(list)
    del res
    df_grouped.rename(columns={
        'text_x': 'post_text',
        'hash_x': 'post_hash',
        'text_y': 'comment_texts',
        'hash_y': 'comments_hash'},
        inplace=True)
    df_grouped['summary'] = df_grouped['comment_texts'].apply(summarize)
    result = df_grouped[['summary', 'post_hash', 'comments_hash']]
    del df_grouped
    result.to_json(output_file, orient="records", lines=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3]

    process(mode, input_file, output_file)
