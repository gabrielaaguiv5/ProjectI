import requests

# Ejemplo: crea un diccionario con los valores para cada categoría
user_data = {
    'Avg_Churches': 4, 
    'Avg_Churches': 4,
    'Avg_Resorts': 4,
    'Avg_Beaches': 4,
    'Avg_Parks': 4,
    'Avg_Theatres': 4,
    'Avg_Museums': 4,
    'Avg_Malls': 4,
    'Avg_Zoo': 4,
    'Avg_Restaurants': 4,
    'Avg_Pubs/bars': 4,
    'Avg_Local services': 4,
    'Avg_Burger/pizza shops': 4,
    'Avg_Hotels/other lodgings': 4,
    'Avg_Juice bars': 4,
    'Avg_Art galleries': 4,
    'Avg_Dance clubs': 4,
    'Avg_Swimming pools': 4,
    'Avg_Gyms': 4,
    'Avg_Bakeries': 4,
    'Avg_Beauty & spas': 4,
    'Avg_Cafes': 4,
    'Avg_View points': 4,
    'Avg_Monuments': 4,
    'Avg_Gardens': 4
}

response = requests.post('http://127.0.0.1:5000/predict', json=user_data)
print(response.json())  # {'cluster': X}
