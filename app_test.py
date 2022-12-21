from itertools import count
from modelos import VectorSpaceModel
from modelos import LSI_Model
from modelos import BooleanModel
from nlp.languaje_procesing import normalize_text
import eval as mt
import ir_datasets
import ir_measures
from ir_measures import *

import eval

dataset = ir_datasets.load("cranfield")
n_d = dataset.docs_iter()
q_rels = dataset.qrels_iter()
run = eval.eval_model('lsi',n_d, dataset.queries_iter())

results = []
c = 0

for x in ir_measures.iter_calc([P@5, R@5, SetF], q_rels, run):
    results.append(x)

for item in results:
    id,mesure, value =item
    if value > 0:
        print(item) 


# n_d = dataset.docs_iter()[:10]
# #print(n_d)
# m = VectorSpaceModel(n_d,10)    
# result = m.proces_query("result friend document file")
# print(result)



# Medidas de evaluacion para cranfield    

# cableado
