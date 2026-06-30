import os
import django
import pandas as pd

# 1. Configurar el entorno de Django para poder usar los modelos desde este script
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'PronosticosDeportivos.settings')
django.setup()

# Ahora sí podemos importar nuestros modelos y funciones de BD
from django.db import transaction
from Pronosticador.models import Equipo, Partido

def cargarPartidos():
    # URL oficial del repositorio usado en casi todos los proyectos de data science de fútbol
    urlCsv = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    
    print(f"Descargando dataset desde GitHub... Por favor espera.")
    
    # Usamos pandas para leer el CSV directamente desde internet
    df = pd.read_csv(urlCsv)
    
    # Limpieza inicial
    # Descartamos partidos con datos nulos en los goles (a veces hay partidos suspendidos)
    df = df.dropna(subset=['home_score', 'away_score'])
    
    # Convertimos la columna de fecha a formato datetime de pandas
    df['date'] = pd.to_datetime(df['date'])
    
    # EL FILTRO DEL VIDEO: Solo partidos desde el 2018
    # Esto es crucial porque el fútbol cambia y los datos de 1950 arruinarían el modelo
    dfFiltrado = df[df['date'] >= '2018-01-01']
    
    totalPartidos = len(dfFiltrado)
    print(f"Dataset descargado. Partidos válidos desde 2018 a importar: {totalPartidos}")
    
    # Limpiar la base de datos antes de insertar (por si corres el script varias veces)
    print("Limpiando base de datos antigua...")
    Partido.objects.all().delete()
    Equipo.objects.all().delete()
    
    print("Iniciando la inserción en la base de datos (esto tomará unos segundos)...")
    
    # transaction.atomic() hace que la inserción sea muchísimo más rápida 
    # y si hay un error, deshace todo para no dejar la BD a medias.
    with transaction.atomic():
        contador = 0
        for index, row in dfFiltrado.iterrows():
            # Buscamos el equipo en la BD, si no existe, lo crea automáticamente
            equipoLocal, creadoL = Equipo.objects.get_or_create(nombre=row['home_team'])
            equipoVisita, creadoV = Equipo.objects.get_or_create(nombre=row['away_team'])
            
            # Guardamos el partido
            # (Los nombres de la izquierda mantienen snake_case porque son columnas de la base de datos)
            Partido.objects.create(
                fecha=row['date'].date(),
                equipo_local=equipoLocal,
                equipo_visitante=equipoVisita,
                goles_local=int(row['home_score']),
                goles_visitante=int(row['away_score']),
                torneo=row['tournament'],
                neutral=row['neutral']
            )
            
            contador += 1
            if contador % 1000 == 0:
                print(f"  -> Insertados {contador} de {totalPartidos} partidos...")

    print("¡IMPORTACIÓN FINALIZADA CON ÉXITO! La base de datos está lista para XGBoost.")

if __name__ == "__main__":
    cargarPartidos()