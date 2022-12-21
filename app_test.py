from modelos import VectorSpaceModel
from modelos import LSI_Model
from modelos import BooleanModel
from nlp.languaje_procesing import normalize_text
import metrics as mt


path = "colecciones/cranfieldDocs/*"

short_path = "modelos/corpus/*"

model = LSI_Model(path,2) # 4 11 19
a = model.proces_query("shear flow of incompressible fluid is considered supersonic aerodynamics conical flow fields")

print(a)


# Medidas de evaluacion para cranfield    

# cableado
