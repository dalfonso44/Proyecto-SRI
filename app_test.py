from modelos import VectorSpaceModel
from nlp.languaje_procesing import normalize_text

a = VectorSpaceModel("modelos/corpus/*")
a.proces_query("friends twice week")

