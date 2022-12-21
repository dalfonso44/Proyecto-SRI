
from pydoc import doc
import ir_datasets
#import ir_measures
import sys
import os
from eval import *
import numpy as np
from modelos import BooleanModel
from modelos import VectorSpaceModel
from modelos import LSI_Model


#dataset = ir_datasets.load("cranfield")

def eval_model(model,dataset,querys):
    run = {}
    results = []
    count = 0
    if model == 'vectorial':
        m = VectorSpaceModel(dataset,1400)
    elif model == 'lsi':
        m = LSI_Model(dataset,2,1400)
    
    for query in querys:
        query_text = query[1]
        results = m.proces_query(query_text) 
        create_run(results,query,run)
        print(count)
        count += 1
    print(run)
    return run    


def create_run(results, query, run):
    run[query[0]] = {}
    for item in results:
        doc_id = item['id']
        #print(doc_id)
        if item['rel'] > 0.1:
            run[query[0]][doc_id] = item['rel']

            


