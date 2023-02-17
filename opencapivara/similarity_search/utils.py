import unidecode
import pandas as pd
from opencapivara.similarity_search import match_rank

def text2canonical(text: str) -> str:
    if type(text) is not str:
        text = str(text)
    _text = text.lower()
    _text = _text\
        .replace('-', ' ')\
        .replace('.', ' ')\
        .replace(',', ' ')\
        .replace('!', ' ')\
        .replace('?', ' ')\
        .replace('_', ' ')\
        .replace('\n', ' ')\
        .replace('\r', ' ')\
        .replace('\t', ' ')\
        .replace('*', ' ')\
        .strip()
    _text = unidecode.unidecode(_text)
    return _text

def give_suggestions(df_queries, df_docs, n):
    df_queries['suggested_documents'] = None
    for index, row in df_queries.iterrows():
        l = match_rank(
            desired = row['signature'],
            candidates = df_docs['signature'].to_list(),
        )
        df_queries.at[index, 'suggested_documents'] = [int(signature.name) for signature, distance in l[:n]]
    return df_queries

def give_suggestions_from_itself(df, n):
    df['suggested_documents'] = None
    for index, row in df.iterrows():
        df_without_current = df[df.index != index]
        l = match_rank(
            desired = row['signature'],
            candidates = df_without_current['signature'].to_list(),
        )
        df.at[index, 'suggested_documents'] = [int(signature.name) for signature, distance in l[:n]]
    return df

def enrich_df_with_signature(df:pd.DataFrame, signature_generator) -> pd.DataFrame:
    df['signature'] = None
    for index, row in df.iterrows():
        # I'm using `at` instead of `loc` because of the array
        # See: https://stackoverflow.com/a/53299945
        df.at[index, 'signature'] = signature_generator.predict_one(
            text=row['text'],
            name=str(index),
        )
    return df
