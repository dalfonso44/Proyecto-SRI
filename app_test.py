from modelos import VectorSpaceModel
from modelos import LSI_Model
from modelos import BooleanModel
from nlp.languaje_procesing import normalize_text

a = VectorSpaceModel("modelos/corpus/*")
print(a.proces_query("man manager main"))




