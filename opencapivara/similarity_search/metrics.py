def precision(row) -> float:
    total_relevant = 0
    for doc in row['suggested_documents']:
        if doc in row['relevant_documents']:
            total_relevant += 1
    return total_relevant/len(row['suggested_documents'])

def accuracy_at_least_one_perfect(row) -> bool:
    return row['relevant_documents'][0] in row['suggested_documents']

def accuracy_at_least_one(row) -> bool:
    for doc in row['suggested_documents']:
        if doc in row['relevant_documents']:
            return True
    return False

def mean_sum_of_scores(row) -> float:
    maximum_relevance = sum([row['relevant_documents'][k] for k in row['relevant_documents']])
    total_relevance = 0.0
    for doc in row['suggested_documents']:
        if doc in row['relevant_documents']:
            total_relevance += row['relevant_documents'][doc]
    print(f"{total_relevance} / {maximum_relevance}")
    return total_relevance / maximum_relevance