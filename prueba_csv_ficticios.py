import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_test_data(num_rows=200):
    start_date = datetime.now() - timedelta(days=3*365)
    end_date = datetime.now()
    date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days)]

    tickers = ["NVDA", "MSFT", "AAPL", "GOOGL", "AMZN", "TSLA", "META", "UNH", "ASML", "BRK-B", "TSM", "GOOG", "0P0000IKFS.F"]
    
    # Definir rangos de precios aproximados para algunos tickers (basados en tu CSV y datos históricos)
    price_ranges = {
        "NVDA": (10, 100),
        "MSFT": (200, 400),
        "AAPL": (100, 200),
        "GOOGL": (80, 150),
        "AMZN": (80, 180),
        "TSLA": (150, 300),
        "META": (150, 400),
        "UNH": (350, 550),
        "ASML": (500, 700),
        "BRK-B": (200, 400),
        "TSM": (80, 120),
        "GOOG": (80, 150),
        "0P0000IKFS.F": (500, 700)
    }

    data = []
    for _ in range(num_rows):
        date = random.choice(date_range).strftime('%d/%m/%Y')
        ticker = random.choice(tickers)
        operation_type = random.choice(["BUY", "BUY", "BUY", "Dividendo", "Split"]) # Mayor probabilidad de compras

        if operation_type == "BUY":
            volume = round(random.uniform(0.5, 50), 2)
            if ticker == '0P0000IKFS.F':
                volume = round(random.uniform(0.5, 5), 3) # Volúmenes más bajos para el fondo
            min_price, max_price = price_ranges.get(ticker, (10, 100))
            price_per_share = round(random.uniform(min_price, max_price), 2)
            operation_price = round(volume * price_per_share, 2)
            commission = round(random.uniform(0, 0.1), 2)
            comment = ""
        elif operation_type == "Dividendo":
            volume = ""
            price_per_share = ""
            operation_price = round(random.uniform(0.1, 50), 2)  # Valores típicos de dividendos
            commission = 0
            comment = ""
        elif operation_type == "Split":
            volume = ""
            price_per_share = ""
            operation_price = 0
            commission = 0
            comment = f"1 A {random.choice([2, 5, 10, 20])}"

        data.append([date, "", operation_type, ticker, volume, price_per_share, operation_price, commission, comment])

    df = pd.DataFrame(data, columns=['FECHA', 'Fecha Liquidación', 'TIPO_OP', 'TICKER', 'VOLUMEN', 'PRECIO_ACCION', 'PRECIO_OPERACION_EUR', 'COMISION', 'COMENTARIO'])
    return df

# Generar el DataFrame de prueba
df_test = generate_test_data()

# Guardar el DataFrame en un archivo CSV
df_test.to_csv("inversiones_test.csv", index=False, sep=";")