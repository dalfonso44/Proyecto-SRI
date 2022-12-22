from itertools import count
from modelos import BooleanModel, VectorSpaceModel,LSI_Model
import ir_datasets


def main():
    dataset = ir_datasets.load("cranfield")
    corpus = dataset.docs_iter()
    m = input("Escriba el nombre del modelo a utilizar (booleano, vectorial o lsi)\n >>>")
    print("")
    while(m != 'booleano' and m!='vectorial' and m!= 'lsi'):
        print("Ese modelo no lo tenemos, vuelvelo a intentar")
        m = input("Escriba el nombre del modelo a utilizar (booleano, vectorial o lsi)\n >>>")

    if m == 'booleano':
        model = BooleanModel(corpus)
    elif m == 'vectorial':
        model = VectorSpaceModel(corpus,1400)
    elif m == 'lsi':
        model = LSI_Model(corpus,2,1400) 

    else: 
        print("lo siento, no tengo ese modelo")
        return
    

    query = input("Ahora haga su consulta\n >>>")
    print("")
   # print(">>>")

    print("")
    print("Resultados")
    print(".................................")
    print("")


    results = []
    count = 0
    docs = model.proces_query(query)
    for i,doc in enumerate(docs) :
        if count == 50:
            break
        count += 1
        #d = doc
        #id = doc['id']
        title = doc['title']
        #id,title,text,au,bib,rel = docs[i]
        results.append(title)
        

    for i,result in enumerate(results) :
        print(i,result)
        print("")    

            

            

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("EXIT")                