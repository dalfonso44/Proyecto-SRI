from modelos import LSI_Model

a = LSI_Model("modelos/corpus/*",2)
print(a.proces_query("friends twice week"))

