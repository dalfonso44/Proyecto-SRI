from statistics import variance
from ..nlp.languaje_procesing import normalize_text
from numpy.linalg import svd, norm
from collections import defaultdict, Counter
import numpy as np
import os
import glob



class LSI_Model(object):
    def __init__(self, corpus) -> None:
        self.corpus = corpus
        self.M = 0 # numero de terminos indexados
        self.N = 0 # numero de coumentos

        self.C = None # matriz termino-documento MxN

        # A cada elemento Cij se le asigna un peso

        # matrices de la descomposicion SVD : C = UZV_t
        self.U = None 
        self.Z = None
        self.V_t = None











        self.documents = dict()
        self.index = defaultdict(dict)

        self.documents_vector = defaultdict(list)
        self.query_vector = list()

        
        self.vocabulary = set()


        self.postings = defaultdict([0 for i in range(glob.glob(self.Corpus))])

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

        def preprocesing_corpus(self):
            index = 1
            for filename in glob.glob(self.corpus):
                with open(filename,"r") as file:
                    text = file.read()
                normalized_text = normalize_text(text) 

                unique_terms = set(normalized_text)

                self.documents_vector[index] = unique_terms
                self.vocabulary = self.vocabulary.union(unique_terms)


                for term in unique_terms:
                    self.postings[term][index] = text.count(term) # frecuencia del termino en el doc
                    self.global_terms_frequency[term] += 1

                self.documents[index] = os.path.basename(filename)
                index += 1

        preprocesing_corpus()        


    def build_index(self):
        pass    


    def svd_C(self):
        self.U, self.Z, self.V_t = svd(self.C)


    def proces_query(self, query, top = 5 ):
        pass    

        
