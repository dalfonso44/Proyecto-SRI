from nlp import normalize_text
import os
from collections import defaultdict
import glob
import numpy as np

class BooleanModel(object):
    def __init__(self, Corpus) -> None:

        #path to the documents
        self.Corpus = Corpus 

        # matrix for indexed docs
        self.inverted_matrix = defaultdict(list)

        #
        self.terms = []

        #
        self.documents = dict()

        def preprocesing_corpus():
            index = 1
            for filename in glob.glob(self.Corpus):
                with open(filename,"r") as file:
                    text = file.read()
                normalized_text = normalize_text(text)  

                # erase repeted words
                normalized_text = list(set(normalized_text))  

                # create inverted matrix for index terms
                for word in normalized_text:
                    self.inverted_matrix[word].append(index) 
                    self.terms.append(word)
                self.documents[index] = (os.path.basename(filename),"resumen") 
                index += 1  

        preprocesing_corpus()    



    # E -> BX
    # X -> OR BX | epsilon
    # B -> AND CY 
    # Y -> AND CY |epsilon
    # C -> D | not D 
    # D -> term | (A)         


    def proces_query(self,query):
        query_tokens = self.lexer(query)
        vector = self._evaluate_query(query_tokens)
        relevant_docs = dict()
        for i in range (len(self.documents)):
            if vector[i] == True:
                yield self.documents[i+1]        


    def lexer(self,query):
        tokens = [] 
        for item in query.split():
            tokens.append(item)
        return tokens    


    def _evaluate_query(self,query_tokens):
        i, vector= self._parse_expression(query_tokens,0)
        if i!=len(query_tokens):
            print("La expresion no es correcta")
            return np.zeros(len(self.documents), dtype=bool)
        return vector   

    def _parse_expression(self,tokens,i):
        i,term = self._parse_B(tokens,i) 
        return self._parse_X(tokens,i,term)

    def _parse_X(self,tokens,i,value):
        if i < len(tokens):
            if tokens[i] == 'OR':
                i,term2 = self._parse_B(tokens,i+1)
                value = value | term2
                return self._parse_X(tokens,i,value)
        return i,value        

    def _parse_B(self,tokens,i):
        i,factor = self._parse_C(tokens,i)
        return self._parse_Y(tokens,i,factor)

    def _parse_Y(self,tokens,i,value):
        if i<len(tokens):
            if tokens[i] == 'AND':
                i,factor2 = self._parse_C(tokens,i+1)
                value = value & factor2
                return self._parse_Y(tokens,i,value)
        return i,value        

    def _parse_C(self,tokens,i) :
        if i < len(tokens):
            if tokens [i] == 'not':
                i, vector = self._parse_C(tokens,i+1)            
                return i, ~vector
            elif self._is_term(tokens[i]):
                vector = self._vector(tokens[i])
                return i + 1, vector
            elif tokens[i] == '(':
                i, vector = self._parse_expression(tokens,i+1) 
                if tokens[i] != ')':
                  print ("Expresion mal formada")  
                return i + 1,vector   
            else:
                print("Expresion mal formada")  
                return -1,np.zeros(len(self.documents),dtype = bool)
        else:
            print("Expresion mal formada")
            return -1,np.zeros(len(self.documents),dtype = bool)            

        

    def _is_term(self,token):
        if token == 'AND' or token == 'OR' or token == '(' or token ==')' or token == 'not':
            return False
        return True         

    def _vector(self,term):
        docs_count = len(self.documents)
        if term in self.terms:
            vector = np.zeros(docs_count,dtype=bool)

            postings = self.inverted_matrix[term]
            for doc_id in postings:
                vector[doc_id-1] = True
            return vector
        else:
            print("The term"+ term + " Was not found in the corpus")                
            return np.zeros(docs_count,dtype=bool)    