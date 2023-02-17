from opencapivara.schema import Signature, EntityText
from opencapivara.similarity_search.signature_generators import SignatureGenerator
from typing import List, Tuple
import numpy as np


def match(desired:Signature, candidates:List[Signature]) -> Tuple[Signature, float]:
    '''
    Matching algorithm
    Formulação: argmax_i similarity(desired, candidates[i])
    Brief: given a set of cantidates, select the one that is most similar to the desired
    '''
    assert len(candidates) > 0
    highest_similarity = -np.inf
    highest_similarity_candidate:Signature
    #print(desired.name, '->', desired.tags)
    for candidate in candidates:
        # TODO: check if versions are compatible
        #print(candidate.name, '->', candidate.tags)
        similarity = calculate_similarity(desired, candidate)
        if similarity > highest_similarity:
            highest_similarity = similarity
            highest_similarity_candidate = candidate
    return highest_similarity_candidate, highest_similarity

def match_rank(desired:Signature, candidates:List[Signature]) -> List[Tuple[Signature, float]]:
    assert len(candidates) > 0
    response = []
    for candidate in candidates:
        similarity = calculate_similarity(desired, candidate)
        response.append((candidate, similarity))
    return sorted(response, key=lambda pair: pair[1], reverse=True)

def calculate_similarity(a:Signature, b:Signature) -> float:
    similarity:float = 0
    similarity += similarity_jaccard(a.tags, b.tags)
    similarity += similarity_cosine(a.embedding, b.embedding)
    # TODO: weight factor? whatever, for now I'll either have tags or embedding
    return similarity

def similarity_cosine(a: list, b: list) -> float:
    assert len(a) == len(b)
    if len(a) == 0:
        return 0
    
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    den = (np.linalg.norm(a) * np.linalg.norm(b))
    if den == 0: # due to the warning "RuntimeWarning: invalid value encountered in double_scalars"
        return 0
    r = np.dot(a, b)/den
    return float(r)

def similarity_jaccard(a:dict, b:dict) -> float:
    #TODO: should I actually name this "weighted jaccard"?
    if len(a)==0: return 0

    similarity = 0
    for key in a:
        if key in b:
            similarity = similarity + a[key]*b[key]
    return similarity/len(a)

def match_from_text(desired:EntityText, candidates:List[EntityText], signature_generator:SignatureGenerator) -> Tuple[EntityText, float]:
    '''
    Will regenate the signatures even if they already exists
    '''
    desired_signatures = signature_generator.predict_one(desired.text)
    candidates_signatures = [signature_generator.predict_one(candidate.text) for candidate in candidates]
    chosen_signature, similarity = match(desired_signatures, candidates_signatures)
    
    for i in range(len(candidates)):
        if candidates_signatures[i] is chosen_signature:
            return candidates[i], similarity
    raise RuntimeError('Internal error: the matching algorithm returned an inexistend signature')