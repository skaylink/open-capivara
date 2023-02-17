from typing import Callable, Optional
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import SpectralClustering
from opencapivara.similarity_search.signature_generators import SignatureGenerator, SignatureGeneratorRandom, SignatureGeneratorRuleBased, SignatureGeneratorTFIDF, SignatureGeneratorLDA, SignatureGeneratorWord2vec, SignatureGeneratorWord2VecRetrained, SignatureGeneratorDoc2VecRetrained, SignatureGeneratorBertMultilingual, SignatureGeneratorSentenceBertMultilingual, SignatureGeneratorSentenceBertEnglish, SignatureGeneratorSentenceBertRetrained
from opencapivara.similarity_search.metrics import accuracy_at_least_one, precision
from opencapivara.similarity_search.utils import text2canonical, enrich_df_with_signature, give_suggestions_from_itself
import pandas as pd


# BM25
from rank_bm25 import BM25Okapi
def give_suggestions_bm25(df, n):
    corpus = df['text'].tolist()
    corpus = [text2canonical(c) for c in corpus]
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    df['suggested_documents'] = None
    for index, row in df.iterrows():
        doc_scores = bm25.get_scores(row['text'].split(' '))
        suggested_documents_idx = np.argpartition(doc_scores, -n-1)[-n-1:].tolist()
        suggested_documents = [df.iloc[[d]].index[0] for d in suggested_documents_idx]
        if index in suggested_documents:
            suggested_documents.remove(index)
        else:
            suggested_documents = suggested_documents[:-1]
        df.at[index, 'suggested_documents'] = suggested_documents
    return df

models = [
    {
        'name': 'Seleção aleatória',
        'signature_class': SignatureGeneratorRandom
    },
    {
        'name': 'Sistema especialista',
        'signature_class': SignatureGeneratorRuleBased
    },

    {
        'name': 'TF-IDF',
        'signature_class': SignatureGeneratorTFIDF
    },
    {
        'name': 'BM25',
        'give_suggestions_func': give_suggestions_bm25
    },
    {
        'name': 'LDA',
        'signature_class': SignatureGeneratorLDA
    },
    
    
    {
        'name': 'Word2Vec inglês',
        'signature_class': SignatureGeneratorWord2vec
    },
    {
        'name': 'Word2Vec retreinado',
        'signature_class': SignatureGeneratorWord2VecRetrained
    },
    {
        'name': 'Doc2Vec retreinado',
        'signature_class': SignatureGeneratorDoc2VecRetrained
    },

    {
        'name': 'Bert multi-idioma',
        'signature_class': SignatureGeneratorBertMultilingual
    },    
    
    {
        'name': 'Sentence Bert multi-idioma',
        'signature_class': SignatureGeneratorSentenceBertMultilingual
    },
    {
        'name': 'Sentence Bert inglês',
        'signature_class': SignatureGeneratorSentenceBertEnglish
    },
    {
        'name': 'Sentence Bert retreinado',
        'signature_class': SignatureGeneratorSentenceBertRetrained
    },
]

metrics = [
    {
        'name': 'Acurácia pelo-menos-um',
        'eval_func': accuracy_at_least_one
    },
    {
        'name': 'Precisão',
        'eval_func': precision
    },
]



def consolidate_labeling(path_labeled:str, path_filtered:str, n: int) -> pd.DataFrame:
    '''
    returns the default annotated queries from Miro labeling
    deprecated?
    '''
    df1 = pd.read_csv(path_labeled)
    df1.set_index('id', inplace=True)

    df2 = pd.read_csv(path_filtered)
    df2.set_index('id', inplace=True)
    df2.fillna('', inplace=True)
    df2['text'] = df2['title'] + ' ' + df2['description']
    df2.drop(['title', 'description', 'category_truth'], axis=1, inplace=True)
    df = df1.join(df2)

    # Calculate grount truth
    nbrs = NearestNeighbors(n_neighbors=n+1, algorithm='ball_tree', metric='euclidean').fit(df[['x', 'y']].to_numpy())
    df['relevant_documents'] = None
    for index, row in df.iterrows():
        distances, indices = nbrs.kneighbors([row[['x', 'y']].to_numpy()]) 
        #df.at[index, 'groud_truth'] = df.index[indices[0, 1]]
        df.at[index, 'relevant_documents'] = [int(df.index[int(i)]) for i in indices[0][1:]]
    df.drop(['x', 'y'], axis=1, inplace=True)
    return df

def evaluate_models(models: list, metrics: list, df: pd.DataFrame, n: int, col_name: str = 'Name') -> pd.DataFrame:
    print(f'Number of eval sentences: {df.shape[0]}')
    results = []
    for model in models:
        print(f"Starting for model {model['name']}")
        if 'signature_class' in model:
            signature_generator = model['signature_class']()
            df = enrich_df_with_signature(df, signature_generator)
            df = give_suggestions_from_itself(df, n)
            del signature_generator
        elif 'give_suggestions_func' in model:
            df = model['give_suggestions_func'](df, n)
        else:
            raise ValueError('either signature_class or give_suggestions_func should be provided')

        results.append([model['name']]+[df.apply(metric['eval_func'], axis=1).mean()*100 for metric in metrics])

    df_results = pd.DataFrame(results, columns=[col_name]+[metric['name'] for metric in metrics])
    return df_results

def calculate_relevant_documets(df: pd.DataFrame, n: int, method: str = 'nearest_neighbors') -> pd.DataFrame:
    df['relevant_documents'] = None
    if method=='nearest_neighbors':
        nbrs = NearestNeighbors(
            n_neighbors=n+1, 
            algorithm='ball_tree', 
            #metric='cosine',
        ).fit(df[['x', 'y']].to_numpy())
        
        for index, row in df.iterrows():
            distances, indices = nbrs.kneighbors([row[['x', 'y']].to_numpy()]) 
            #df.at[index, 'groud_truth'] = df.index[indices[0, 1]]
            df.at[index, 'relevant_documents'] = [int(df.index[int(i)]) for i in indices[0][1:]]
    elif method=='clustering':
        model = SpectralClustering(
            n_clusters=int(df.shape[0]/n),
            random_state=0,
        ).fit(df[['x', 'y']]/1000)
        df['cluster'] = model.labels_
        for index, row in df.iterrows():
            df_relevants = df[(df['cluster']==row['cluster']) & (df.index != index)]
            df.at[index, 'relevant_documents'] = df_relevants.index.tolist()
    else:
        raise NotImplementedError(f'Unsuported method: {method}')
    return df

def load_data(path_labeled: str, path_filtered: str) -> pd.DataFrame:
    df1 = pd.read_csv(path_labeled)
    df1.set_index('id', inplace=True)

    df2 = pd.read_csv(path_filtered)
    df2.set_index('id', inplace=True)
    df2.fillna('', inplace=True)
    df2['text'] = df2['title'] + ' ' + df2['description']
    df = df1.join(df2)
    df['text'] = df['text'].astype(str)
    return df

def analyze_all(path_labeled: str, path_filtered: str, models: list, metrics: list, n: int, filter_function: Optional[Callable] = None, relevance_method: str = 'nearest_neighbors', groups=['1', '2', '3'], col_name:str = 'Nome') -> pd.DataFrame:
    '''
    give_suggestions_func should receive `df, n` and return `df`
    
    filter_function should receive `df` and return `df`
    '''

    #df.columns
    #df[['category_truth']].value_counts()
    #df[['language']].value_counts()
    results = []
    results_size_control = []
    for group in groups:
        df = load_data(
            path_labeled=path_labeled.replace('{{group}}', group), 
            path_filtered=path_filtered.replace('{{group}}', group),
        )
        #print('>>>>>>>>>> DEBUG:')
        #print(df.to_markdown())
        #print('>>>>>>>>>> END DEBUG')

        if filter_function is not None:
            df_to_evaluate = filter_function(df.copy())
            df_to_evaluate = calculate_relevant_documets(df_to_evaluate, n=n, method=relevance_method)
            
            df_to_evaluate_reference = df.sample(n=df_to_evaluate.shape[0]).copy()
            df_to_evaluate_reference = calculate_relevant_documets(df_to_evaluate_reference, n=n, method=relevance_method)
            result_size_control = evaluate_models(models=models, metrics=metrics, df=df_to_evaluate_reference, n=n, col_name=col_name)
            result_size_control = result_size_control.sort_values(col_name)
            results_size_control.append(result_size_control)

        else:
            df = calculate_relevant_documets(df, n=n, method=relevance_method)
            df_to_evaluate = df
        result = evaluate_models(models=models, metrics=metrics, df=df_to_evaluate.copy(), n=n, col_name=col_name)
        result = result.sort_values(col_name)
        results.append(result)

    if filter_function is not None:
        print('Size control results (averaged among all groups):')
        df_results_size_control = pd.concat(results_size_control).groupby(col_name).mean().reset_index()
        print(df_results_size_control.to_markdown(index=False))
        print('\n--------------\n')

    print('Evaluation results (averaged among all groups):')
    df_results = pd.concat(results).groupby(col_name).mean().reset_index()
    print(df_results.to_markdown(index=False))

    return df_results
