import streamlit as st
import yfinance as yf
import pandas as pd

def calculate_annualized_return(ticker, years=10):
    """
    Calcula la revalorización anualizada de una empresa en los últimos 'years' años.

    Args:
        ticker (str): El ticker de la empresa.
        years (int): El número de años a considerar.

    Returns:
        float: La revalorización anualizada en porcentaje, o None si hay un error.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=f"{years}y")

        if hist.empty:
            print(f"No se encontraron datos históricos para {ticker} en los últimos {years} años.")
            return None

        start_price = hist["Close"].iloc[0]
        end_price = hist["Close"].iloc[-1]

        total_return = (end_price / start_price) - 1
        annualized_return = (1 + total_return) ** (1 / years) - 1

        return annualized_return * 100  # Convertir a porcentaje

    except Exception as e:
        print(f"Error al obtener datos para {ticker}: {e}")
        return None

def main():
    st.title("Calculadora de Revalorización Anualizada")

    ticker = st.text_input("Introduce el ticker de la empresa:", "AAPL").upper()
    years = st.number_input("Número de años a considerar:", min_value=1, max_value=50, value=10)

    if st.button("Calcular"):
        annualized_return = calculate_annualized_return(ticker, years)

        if annualized_return is not None:
            st.write(f"### Revalorización anualizada de {ticker} en los últimos {years} años:")
            st.write(f"<span style='font-size:24px; color: {'green' if annualized_return >= 0 else 'red'};'>{annualized_return:.2f}%</span>", unsafe_allow_html=True)
        else:
            st.error("No se pudo calcular la revalorización. Revisa el ticker y el número de años.")

if __name__ == "__main__":
    main()