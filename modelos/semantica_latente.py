

from statistics import variance
from typing import List
from nlp import normalize_text
from numpy.linalg import svd, norm
from collections import defaultdict, Counter
import numpy as np
import os
import glob
import math



class LSI_Model(object):
    def __init__(self, corpus, rank_aproximation,cant_docs) -> None:
        self.corpus = corpus
        self.M = 0 # numero de terminos indexados
        self.N = 0 # numero de coumentos

        self.rank_aproximation = rank_aproximation

        self.C = None # matriz termino-documento MxN

        # A cada elemento Cij se le asigna un peso

        # matrices de la descomposicion SVD : C = UZV_t
        self.U = None 
        self.Z = None
        self.V_t = None

        #query en las nuevas matrices reducidas
        self.q = None
        #documentos reducidos
        self.d = None


        self.documents = defaultdict(dict)
        self.index = defaultdict(dict)

        self.documents_vector = defaultdict(list)
        self.query_vector = None

        
        self.vocabulary = set()


       # self.postings = defaultdict([0 for i in range(glob.glob(self.corpus))])

       # self.postings = defaultdict(dict) 
        def default_value_for_postings():
            return [0 for i in range(cant_docs)]
        
        self.postings = defaultdict(default_value_for_postings)

        self.query_postings = defaultdict(int)


        self.global_terms_frequency = defaultdict(int)

        
        self.documents_norm = defaultdict(float)


        


        self.terms_representation = np.array(None) # reduced representation of terms after svd
        self.docs_representation = np.array(None) # reduced representation of docs after svd

       # self.__update_index = True # update flag to rebuild index
        self.term_index_in_A = {}
        self.doc_index_in_A = {}

        self.k = None # reduced dimension after SVD
        self.variance = 0.9

        def preprocesing_corpus():
            index = 0
            for i,file in enumerate(self.corpus) :
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
                #self.M += len(unique_terms)
                

                for term in normalized_text:
                    self.postings[term][index] = normalized_text.count(term) # frecuencia del termino en el doc
                    a = self.postings[term]
                    self.global_terms_frequency[term] += 1

                #self.documents[index] = os.path.basename(filename)
                index += 1
            self.N = index 
            self.M=len(self.vocabulary)
           # print(self.N)
            #print(self.M)

        preprocesing_corpus()
        self.C = self.build_term_doc_matrix()        


    def build_term_doc_matrix(self):
        c = np.zeros((self.M,self.N), dtype=int)
        for i,word in enumerate(self.vocabulary):
            for j in range(self.N):
                if word in self.documents_vector[j]:
                    l_ij = math.log((self.postings[word][j] + 1)) # LOG(TF +1)
                    gi = math.log((self.N/ (1 + self.global_terms_frequency[word])),2) # gi = log2 n/1+dfi
                    c[i,j] = l_ij * gi
                    
        return c


    def svd_with_dimensionality_reduction(self):
        u, s, v = np.linalg.svd(self.C,full_matrices=False)
        s = np.diag(s)
        k = self.rank_aproximation
        return u[:, :k], s[:k, :k], v[:, :k]


    def proces_query(self, query, top = 5 ):
        query = self.lexer(query)
        self.query_vector = self.make_query_vector(query)
        
        self.U, self.Z, self.V_t = self.svd_with_dimensionality_reduction()

        #query_trasp = np.transpose(self.query_vector)
       

        q = self.query_vector.T @ self.U @ np.linalg.pinv(self.Z)
        d = self.C.T @ self.U @ np.linalg.pinv(self.Z)

        #self.q = np.dot(np.dot(np.linalg.inv(self.Z),np.transpose(self.U)),query_trasp)
        #C_transp=np.transpose(self.C)
        #self.d = np.dot(np.dot(np.linalg.inv(self.Z),np.transpose(self.U)),C_transp)


        res = np.apply_along_axis(lambda row: self.sim(q, row), axis=1, arr=d)
        #print(res)
        ranking = np.argsort(res)
        result = []
        for i,id in enumerate(self.documents.keys()) :
            self.documents[id]['rel'] = res[i]
            result.append(self.documents[id])
        return result    


        #similitudes = []
        #for i in range(self.d.shape[1]):
        #    similitudes.append(self.d[:,i],self.q)

        #return similitudes

    
    def rank(self,similitudes,doc):
        for i in range(len(similitudes)):
            for j in range(i + 1, len(similitudes)):
                if similitudes[i] < similitudes[j]:
                    similitudes[i], similitudes[j] = similitudes[j], similitudes[i]
                    doc[i], doc[j] = doc[j], doc[i]
        return doc
                

    def sim(self,doc,q):
        p = np.dot(doc,q)
        n= (np.linalg.norm(doc) * np.linalg.norm(q))
        if n == 0: return 0
        simility=p/n
        return simility
    
    def lexer(self,query):
        normalized_query = normalize_text(query)
        for term in normalized_query:
            self.query_postings[term] = normalized_query.count(term)

        normalized_query = list(set(normalized_query))    

        return normalized_query

    def make_query_vector(self,query: List):
        query_vector = np.zeros(len(self.vocabulary))
        for i,term in enumerate(self.vocabulary):
            query_vector[i] += query.count(term) # x ahora va a ser la cantidad de veces que aparece 
        return query_vector

