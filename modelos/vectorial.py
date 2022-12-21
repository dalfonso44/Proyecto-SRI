from fileinput import filename
import glob
import math
from operator import invert
import os
from collections import defaultdict
from pydoc import doc
from random import triangular
from nlp.languaje_procesing import normalize_text
from functools import reduce

class VectorSpaceModel(object):
    def __init__(self, Corpus,d) -> None:

        self.Corpus = Corpus


        self.documents = defaultdict(dict)


        self.documents_vector = defaultdict(list)

        self.query_vector = list()

        
        self.vocabulary = set()

        # ahora se supone que cuando no tenga nada ya tiene un cero
        def default_value_for_postings():
            return [0 for i in range(d)]
        
        self.postings = defaultdict(default_value_for_postings)

        self.query_postings = defaultdict(int)


        self.global_terms_frequency = defaultdict(int)

        
        self.documents_norm = defaultdict(float)

        def preprocesing_corpus():
            index = 0
            for i,file in enumerate(self.Corpus) :
                id,title,text,au,bib = file
                self.documents[i]['id'] = id
                self.documents[i]['title'] = title
                self.documents[i]['text'] = text
                self.documents[i]['author'] = au
                self.documents[i]['biblio'] = bib
                self.documents[i]['rel'] = 0

                normalized_text = normalize_text(text) 
                unique_terms = set(normalized_text)

                self.documents_vector[index] = unique_terms
                self.vocabulary = self.vocabulary.union(unique_terms)

                for term in normalized_text:
                    self.postings[term][index] = normalized_text.count(term) # frecuencia del termino en el doc
                    self.global_terms_frequency[term] += 1

            #    self.documents[index] = os.path.basename(filename)
                index += 1

        preprocesing_corpus()


    def proces_query(self,query):
        self.query_vector = self.lexer(query)
        scores = defaultdict(float)
        for id in range(len(self.documents)):
            scores[id] = self.similarity(id)

        import operator
        scores_sorted = sorted(scores.items(),key = operator.itemgetter(1),reverse=True)

        docs = []
        for t in scores_sorted:
            i,value = t
            self.documents[i]['rel'] = value
            docs.append(self.documents[i])

        d = docs[:20]    



        return docs


    def similarity(self, doc):
        similarity = 0.0
        query_norm = 0
        doc_norm = 0

        for term in self.query_vector:
            weight_in_query = self.weight_in_query(term)
            query_norm += weight_in_query**2
            weigth_in_document = self.weight_in_document(term, doc)
            similarity += (weight_in_query * weigth_in_document) # Dj * Q



        query_norm = math.sqrt(query_norm)
        for term in self.documents_vector[doc]:
            doc_norm += self.weight_in_document(term,doc)**2

        doc_norm = math.sqrt(doc_norm)    

        if similarity == 0:
            return 0
        similarity = similarity / (query_norm * doc_norm)   
        return similarity 


    def weight_in_document(self, term,doc):
        tf_ij = self.normalized_term_frequency(term,doc)
        idf_i = self.inverse_document_frequecy(term)
        weight = tf_ij * idf_i
        return weight

    def weight_in_query(self,term):
        alpha = 0.4 # termino de suavizado
        tf_iq = self.normalized_term_frequency(term,-1)
        idf_i = self.inverse_document_frequecy(term)

        weight = (alpha + (1+alpha)*tf_iq) * idf_i
        return weight


    def normalized_term_frequency(self,term, doc): # si doc = -1 -> el documento es la consulta
        frequency = 0
        if doc == -1 :   
            frequency = self.query_postings[term] # el documento es la consulta
        else :
            frequency = self.postings[term][doc]    

        if frequency == 0:
            return 0
        max_frequency = 1 # no es cero pa que no de mal la division

        if doc == -1:
            for term in self.query_vector:
                freq = self.query_postings[term]
                if max_frequency < freq:
                    max_frequency = freq
        else :
            for terms in self.documents_vector[doc]:
                freq = self.postings[terms][doc]
                if max_frequency < freq:
                    max_frequency = freq    
        
        normalized_frequency = frequency / max_frequency # ver que pasa si esto es cero
        return normalized_frequency

    def inverse_document_frequecy(self,term):
        N = len(self.documents)    
        n_i = self.global_terms_frequency[term]     

        #revisar esto, sumar 1 arriba y abajo pa que nunca de cero
        idf = math.log(((N+1)/(1+n_i)),2)   
        return idf



    def lexer(self,query):
        normalized_query = normalize_text(query)
        for term in normalized_query:
            self.query_postings[term] = normalized_query.count(term)

        normalized_query = list(set(normalized_query))    

        return normalized_query


