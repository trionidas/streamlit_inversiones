import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io
import re
import numpy as np
from typing import Optional


def clean_number(x):
    if isinstance(x, str):
        x = x.replace('\xa0', '').replace(' ', '')
        x = x.replace('.', '')
        x = x.replace(',', '.')
        x = re.sub(r'[^\d.-]', '', x)
    return x

def load_data(uploaded_file):
    content = uploaded_file.getvalue().decode('utf-8')
    df = pd.read_csv(io.StringIO(content), sep=';')
    df['FECHA'] = pd.to_datetime(df['FECHA'], format='%d/%m/%Y')
    
    numeric_columns = ['VOLUMEN', 'PRECIO_ACCION', 'PRECIO_OPERACION_EUR', 'COMISION']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].apply(clean_number).astype(float)
    
    return df

def get_exchange_rate():
    try:
        # Priorizar la consulta del tipo de cambio directo USDEUR=X
        usd_eur = yf.Ticker("USDEUR=X")
        rate = usd_eur.fast_info['last_price']  # fast_info es más rápido para datos básicos
        
        if rate is not None:
            return rate
        
        # Respaldo: Calcular tipo de cambio inverso a partir de EURUSD=X
        eur_usd = yf.Ticker("EURUSD=X")
        rate = eur_usd.fast_info['last_price']
        if rate is not None:
            return 1 / rate
       
        # Valor por defecto si todas las fuentes fallan
        return 0.93
    except Exception as e:
        st.warning(f"Error al obtener el tipo de cambio: {str(e)}. Usando 1 USD = 0.93 EUR")
        return 0.93

def get_current_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        for price_field in ['regularMarketPrice', 'currentPrice', 'previousClose', 'ask', 'bid', 'open', 'dayHigh', 'dayLow']:
            if price_field in info and info[price_field] is not None:
                return info[price_field]
        
        hist = stock.history(period="1d")
        if not hist.empty and 'Close' in hist.columns:
            return hist['Close'].iloc[-1]
        
        st.warning(f"No se pudo obtener el precio actual para {ticker}. Info disponible: {info.keys()}")
        return None
    except Exception as e:
        st.warning(f"Error al obtener datos para {ticker}: {str(e)}")
        return None
    

# Función para obtener el precio actual (recuperamos para Intrinsec value)
def get_current_price2(info: dict) -> float:
    """
    Obtiene el precio actual del mercado desde la información del stock.
    """
    return info.get('currentPrice', None)

@st.cache_data(ttl=3600)
def get_historical_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date, end=end_date)
    return hist

def analyze_investments(df):
    results = []
    exchange_rate = get_exchange_rate()
    
    for ticker in df['TICKER'].unique():
        ticker_df = df[df['TICKER'] == ticker]
        buy_df = ticker_df[ticker_df['TIPO_OP'] == 'BUY']
        dividend_df = ticker_df[ticker_df['TIPO_OP'] == 'Dividendo']
        split_df = ticker_df[ticker_df['TIPO_OP'] == 'Split']
        
        total_invested = buy_df['PRECIO_OPERACION_EUR'].sum()
        total_shares = buy_df['VOLUMEN'].sum()
        avg_price = total_invested / total_shares if total_shares != 0 else 0
        
        # Definir total_dividends por defecto
        total_dividends = dividend_df['PRECIO_OPERACION_EUR'].sum() if not dividend_df.empty else 0
        
        current_price = get_current_price(ticker)
        previous_close = None
        
        try:
            # Obtener el precio de cierre del día anterior
            hist = get_historical_data(ticker, start_date=datetime.now() - timedelta(days=5), end_date=datetime.now())
            if not hist.empty and 'Close' in hist.columns:
                previous_close = hist['Close'].iloc[-2]  # Penúltimo día disponible
        except Exception as e:
            st.warning(f"Error obteniendo el cierre anterior para {ticker}: {e}")
        
        if ticker != '0P0000IKFS.F' and current_price is not None:
            current_price *= exchange_rate
        if previous_close is not None:
            previous_close *= exchange_rate
        
        if current_price is not None:
            current_value = current_price * total_shares
            profit_loss = current_value - total_invested + total_dividends
            profit_loss_percentage = (profit_loss / total_invested) * 100 if total_invested != 0 else 0
            
            # Calcular variación diaria
            daily_change = current_price - previous_close if previous_close is not None else None
            daily_change_percentage = (daily_change / previous_close) * 100 if previous_close else None
        else:
            current_value = None
            profit_loss = None
            profit_loss_percentage = None
            daily_change = None
            daily_change_percentage = None
        
        splits = split_df['VOLUMEN'].sum()
        
        results.append({
            'Ticker': ticker,
            'Total Invertido (EUR)': total_invested,
            'Acciones': total_shares,
            'Precio Promedio (EUR)': avg_price,
            'Precio Actual (EUR)': current_price,
            'Valor Actual (EUR)': current_value,
            'Dividendos Recibidos (EUR)': total_dividends,
            'Ganancia/Pérdida (EUR)': profit_loss,
            'Ganancia/Pérdida %': profit_loss_percentage,
            'Variación Diaria (EUR)': daily_change,
            'Variación Diaria %': daily_change_percentage,
            'Splits': splits
        })
    
    return pd.DataFrame(results)

def plot_portfolio_distribution_bars(results):
    # Calcular el porcentaje de cada ticker
    distribution_data = results.groupby('Ticker')['Valor Actual (EUR)'].sum().reset_index()
    distribution_data['Porcentaje'] = (distribution_data['Valor Actual (EUR)'] / 
                                       distribution_data['Valor Actual (EUR)'].sum()) * 100
    # Ordenar por porcentaje descendente
    distribution_data = distribution_data.sort_values('Porcentaje', ascending=True)  # Ascendente para que el mayor quede arriba

    # Crear el gráfico de barras horizontales
    fig = px.bar(
        distribution_data,
        x='Porcentaje',
        y='Ticker',
        orientation='h',
        text='Porcentaje',
        color='Porcentaje',
        color_continuous_scale=px.colors.sequential.Blues,
        labels={'Ticker': 'Ticker', 'Porcentaje': 'Porcentaje (%)'}
    )
    # Estilizar el gráfico
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='inside')
    fig.update_layout(
        xaxis_title="Porcentaje (%)",
        yaxis_title="",
        coloraxis_showscale=False,  # Ocultar barra de colores
        title=dict(text=None),  # Asegurar que el título esté completamente desactivado
        plot_bgcolor='rgba(0,0,0,0)',  # Fondo transparente
        xaxis=dict(gridcolor='rgba(200,200,200,0.5)')  # Líneas de la cuadrícula suaves
    )
    return fig

def plot_portfolio_distribution(results):
    fig = px.pie(
        results,
        values='Valor Actual (EUR)',
        names='Ticker',
        title=' ',
        color_discrete_sequence=px.colors.sequential.Blues_r  # Colores en tonos de azul suave
    )
    
    # Ajustes de estilo para el gráfico
    fig.update_traces(
        textposition='inside',  # Etiquetas dentro de cada sección
        textinfo='percent+label',  # Mostrar etiquetas y porcentaje
        pull=[0.05 if v > results['Valor Actual (EUR)'].sum() * 0.1 else 0 for v in results['Valor Actual (EUR)']]  # Resaltar secciones mayores al 10%
    )
    
    # Configuración de diseño
    fig.update_layout(
        showlegend=True,
        legend=dict(
            title='Ticker',
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=40, b=40, l=0, r=0),
        paper_bgcolor='white',  # Fondo blanco para todo el gráfico
        plot_bgcolor='white',
        title_font=dict(size=24, color='#4a90e2', family='Arial'),  # Estilo del título
    )
    
    return fig

@st.cache_data(ttl=3600)
def plot_ticker_performance(ticker, start_date, end_date):
    historical_data = get_historical_data(ticker, start_date, end_date)
    
    fig = px.line(
        historical_data, 
        x=historical_data.index, 
        y='Close',
        title=f'Rendimiento de {ticker} a lo Largo del Tiempo',
        labels={'Close': 'Precio de Cierre', 'Date': 'Fecha'}
    )
    
    fig.update_layout(
        xaxis_title='Fecha',
        yaxis_title='Precio (USD)',
        hovermode='x unified'
    )
    
    return fig

def plot_performance_over_time(df):
    fig = px.line(
        df, 
        x='FECHA', 
        y='PRECIO_OPERACION_EUR', 
        color='TICKER',
        title='Rendimiento de las Inversiones a lo Largo del Tiempo',
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    return fig

def get_monthly_prices(tickers, start_date, end_date):
    monthly_prices = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        
        # Usar el método history con intervalo mensual
        hist = stock.history(start=start_date, end=end_date, interval="1mo")
        if not hist.empty and 'Close' in hist.columns:
            # Asegurarse de que las fechas estén en el último día del mes
            hist.index = hist.index + pd.offsets.MonthEnd(0)
            monthly_prices[ticker] = hist['Close']

        else:
            st.write(f"No se pudieron obtener datos para {ticker}")

    df_monthly_prices = pd.DataFrame(monthly_prices)
    
    return df_monthly_prices

def calculate_investment_value_over_time(df, results):
    exchange_rate = get_exchange_rate()
    
    df_sorted = df.sort_values('FECHA')
    date_range = pd.date_range(start=df_sorted['FECHA'].min(), end=datetime.now(), freq='M')
    monthly_data = pd.DataFrame(index=date_range)
    
    # Corregir el cálculo de la inversión acumulada para no incluir dividendos
    buy_df = df_sorted[df_sorted['TIPO_OP'] == 'BUY']
    monthly_data['Inversión Acumulada'] = buy_df.set_index('FECHA').resample('M')['PRECIO_OPERACION_EUR'].sum().cumsum()
    
    tickers = df['TICKER'].unique()
    monthly_prices = get_monthly_prices(tickers, df_sorted['FECHA'].min(), datetime.now())
    # Convertir el índice a fin de mes
    monthly_prices.index = monthly_prices.index.to_period('M').to_timestamp('M')
    monthly_data.index = monthly_data.index.to_period('M').to_timestamp('M')
    
    shares_accumulated = {}
    for ticker in tickers:
        ticker_df = df_sorted[df_sorted['TICKER'] == ticker]
        shares = ticker_df.set_index('FECHA')['VOLUMEN'].cumsum()
        shares_accumulated[ticker] = shares.resample('M').last().ffill()
        shares_accumulated[ticker].index = shares_accumulated[ticker].index.to_period('M').to_timestamp('M')
    
    monthly_data['Valor de la Inversión'] = 0.0
    
    for date in monthly_data.index:
        
        ticker_values = []
        for ticker in tickers:
            shares = shares_accumulated[ticker].loc[:date].iloc[-1] if not shares_accumulated[ticker].loc[:date].empty else 0
            
            price = None
            if date in monthly_prices.index and ticker in monthly_prices.columns:
                price_series = monthly_prices.loc[date, ticker]
                if isinstance(price_series, pd.Series):
                    price = price_series.dropna().iloc[-1] if not price_series.dropna().empty else None
                else:
                    price = price_series
                
                if pd.isna(price):
                    # Si el precio es NaN, buscar el último precio válido en el mes
                    last_valid_price = monthly_prices.loc[:date, ticker].last_valid_index()
                    if last_valid_price is not None:
                        price_series = monthly_prices.loc[last_valid_price, ticker]
                        if isinstance(price_series, pd.Series):
                            price = price_series.dropna().iloc[-1] if not price_series.dropna().empty else None
                        else:
                            price = price_series
                        st.write(f"Usando precio de {last_valid_price} para {ticker}")
            
            if price is not None and shares > 0:
                if ticker != '0P0000IKFS.F':
                    price *= exchange_rate
                value = shares * price
                ticker_values.append(value)
      
        total_value = np.nansum(ticker_values)
        
        monthly_data.loc[date, 'Valor de la Inversión'] = total_value

    monthly_data['Inversión Acumulada'] = monthly_data['Inversión Acumulada'].clip(lower=0)
    
    return monthly_data

# Version táctil
def plot_investment_over_time(df, results):
    monthly_data = calculate_investment_value_over_time(df, results)
   
    # Crear el gráfico
    fig = go.Figure()
    
    # Añadir la línea de inversión acumulada
    fig.add_trace(
        go.Scatter(x=monthly_data.index, y=monthly_data['Inversión Acumulada'], name="Inversión Acumulada", line=dict(color='blue'))
    )
    
    # Añadir la línea de valor de la inversión
    fig.add_trace(
        go.Scatter(x=monthly_data.index, y=monthly_data['Valor de la Inversión'], name="Valor de la Inversión", line=dict(color='red'))
    )
    
    # Actualizar el diseño
    fig.update_layout(
        title="Inversión Acumulada vs Valor de la Inversión",
        xaxis_title="Fecha",
        yaxis_title="EUR",
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.5)'),
        hovermode="x unified"
    )
    
    return fig

def get_earnings_date(ticker):
    try:
        stock = yf.Ticker(ticker)
        earnings_date = stock.calendar.get('Earnings Date')
        if isinstance(earnings_date, pd.Timestamp):
            return earnings_date.strftime('%Y-%m-%d')
        elif isinstance(earnings_date, list) and len(earnings_date) > 0:
            return earnings_date[0].strftime('%Y-%m-%d')
        else:
            return "No disponible"
    except Exception as e:
        return f"Error: {str(e)}"

def get_stock_splits(ticker, start_date):
    try:
        stock = yf.Ticker(ticker)
        all_splits = stock.splits
        
        # Convertir start_date a timestamp UTC
        if isinstance(start_date, str):
            start_date = pd.Timestamp(start_date).tz_localize('UTC')
        elif isinstance(start_date, pd.Timestamp) and start_date.tz is None:
            start_date = start_date.tz_localize('UTC')
        
        # Asegurar que el índice de all_splits esté en UTC
        all_splits.index = all_splits.index.tz_convert('UTC')
        
        # Filtrar los splits
        relevant_splits = all_splits[all_splits.index >= start_date]
        
        return relevant_splits
    except Exception as e:
        st.warning(f"Error al obtener splits para {ticker}: {str(e)}")
        return pd.Series([], dtype=float)

def analizar_sp500():
    sp500 = yf.Ticker("^GSPC")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=14*30)  # Aumentamos a 14 meses para asegurar 13 meses completos
    data = sp500.history(start=start_date, end=end_date, interval="1mo")
    monthly_closes = data['Close'].resample('M').last().tail(13)  # Ahora tomamos los últimos 13 meses
    
    contador = 0
    cambios = []
    
    for i in range(1, len(monthly_closes)):
        if monthly_closes.iloc[i] > monthly_closes.iloc[i-1]:
            contador += 1
            cambios.append(1)
        elif monthly_closes.iloc[i] < monthly_closes.iloc[i-1]:
            contador -= 1
            cambios.append(-1)
        else:
            cambios.append(0)
    
    df_analysis = pd.DataFrame({
        'Fecha': monthly_closes.index,
        'Precio de Cierre': monthly_closes.values,
        'Cambio': [0] + cambios  # Añadimos un 0 al principio para el primer mes
    })
    
    recomendacion = "Quédate en el Fondo" if contador > 0 else "Cambia a Bonobos"
    return contador, recomendacion, df_analysis

def calculate_intrinsic_value2(
    ticker: str,
    discount_rate: float = 0.10,
    growth_stage_years: int = 10,
    terminal_stage_years: int = 10,
    terminal_growth_rate: float = 0.04
) -> Optional[float]:
    """
    Calcula el valor intrínseco usando trailingEPS de yfinance con ajustes para evitar divisores problemáticos.
    """
    try:
        # Validar tasas para evitar divisiones por cero o problemas de precisión
        if discount_rate <= 0:
            raise ValueError("La tasa de descuento debe ser positiva.")
        if terminal_growth_rate >= discount_rate:
            terminal_growth_rate = discount_rate - 0.01  # Ajustar para evitar problemas

        # Obtener datos de la acción
        stock = yf.Ticker(ticker)
        info = stock.info

        # Obtener trailingEPS
        trailing_eps = info.get('trailingEps', None)
        if trailing_eps is None or trailing_eps <= 0:
            print(f"No se encontró un trailingEPS válido para {ticker}. Usando valor predeterminado.")
            trailing_eps = 5.0  # EPS predeterminado si no está disponible

        # Suponer una tasa de crecimiento promedio (ajustada para evitar problemas)
        growth_rate = min(0.10, discount_rate - 0.01)  # Limitar crecimiento para que sea razonable

        # Calcular factores x e y según el modelo DCF
        x = (1 + growth_rate) / (1 + discount_rate)
        y = (1 + terminal_growth_rate) / (1 + discount_rate)

        # Validar que x y y no generen divisiones por cero o resultados cercanos a 1
        if abs(1 - x) < 1e-3 or abs(1 - y) < 1e-3:
            print(f"Tasas conflictivas para {ticker}. Ajustando factores x e y.")
            x = 0.95  # Valor ajustado
            y = 0.90  # Valor ajustado

        # Calcular valor presente de ganancias futuras en etapa de crecimiento
        growth_stage_value = trailing_eps * x * (1 - x**growth_stage_years) / (1 - x)

        # Calcular valor presente del valor terminal
        terminal_value = (trailing_eps *
                          (x**growth_stage_years) *
                          y *
                          (1 - y**terminal_stage_years) /
                          (1 - y))

        # Valor intrínseco total
        intrinsic_value = growth_stage_value + terminal_value
        print(f"Valor intrínseco calculado para {ticker}: {intrinsic_value}")

        return float(intrinsic_value)

    except ZeroDivisionError as e:
        print(f"Error: División por cero en el cálculo para {ticker}. Detalles: {str(e)}")
        return None
    except ValueError as ve:
        print(f"Error calculando el valor intrínseco para {ticker}: {str(ve)}")
        return None
    except Exception as e:
        print(f"Error calculando el valor intrínseco para {ticker}: {str(e)}")
        return None

# Función para calcular el valor intrínseco
def calculate_intrinsic_value(
    eps: float,
    discount_rate: float,
    growth_rate: float,
    growth_stage_years: int,
    terminal_growth_rate: float
) -> dict:
    """
    Calcula el valor intrínseco basado en EPS y tasas proporcionadas.
    Retorna un diccionario con detalles del cálculo.
    """
    try:
        if discount_rate <= terminal_growth_rate:
            raise ValueError("La tasa de descuento debe ser mayor que la tasa de crecimiento terminal.")
        
        # Detalles anuales de crecimiento
        growth_values = []
        for i in range(1, growth_stage_years + 1):
            growth_value = eps * ((1 + growth_rate) ** i) / ((1 + discount_rate) ** i)
            growth_values.append({
                "Año": i,
                "Crecimiento EPS (estimado)": eps * ((1 + growth_rate) ** i),
                "Valor presente (descuento aplicado)": growth_value
            })

        # Cálculo del valor terminal limitado a 10 años
        terminal_values = []
        eps_final = eps * ((1 + growth_rate) ** growth_stage_years)  # EPS final al final del crecimiento
        for i in range(growth_stage_years + 1, growth_stage_years + 11):  # 10 años adicionales
            terminal_eps = eps_final * ((1 + terminal_growth_rate) ** (i - growth_stage_years))
            terminal_value = terminal_eps / ((1 + discount_rate) ** i)
            terminal_values.append({
                "Año": i,
                "Valor Terminal (estimado)": terminal_eps,
                "Valor presente (descuento aplicado)": terminal_value
            })

        # Sumar ambos componentes
        total_growth_value = sum([row["Valor presente (descuento aplicado)"] for row in growth_values])
        total_terminal_value = sum([row["Valor presente (descuento aplicado)"] for row in terminal_values])

        intrinsic_value = total_growth_value + total_terminal_value

        return {
            "intrinsic_value": intrinsic_value,
            "growth_values": growth_values,
            "terminal_values": terminal_values,
            "total_growth_value": total_growth_value,
            "total_terminal_value": total_terminal_value,
        }
    except Exception as e:
        st.warning(f"Error al calcular el valor intrínseco: {str(e)}")
        return None

# Función para obtener datos del stock
def get_stock_info(ticker: str) -> dict:

    try:
        stock = yf.Ticker(ticker)
        return stock.info
    except Exception as e:
        st.error(f"Error al obtener la información del stock: {str(e)}")
        return {}

# Function to determine background color based on thresholds, útil para el análisis de stocks (PE PBV, etc)
def get_bg_color(value, thresholds):
    if value is None or value == "N/A":
        return "background-color: gray;"
    elif value < thresholds[0]:
        return "background-color: green; color: white;"
    elif thresholds[0] <= value <= thresholds[1]:
        return "background-color: yellow; color: black;"
    else:
        return "background-color: red; color: white;"

# Function to display a styled subheader
def styled_subheader(text):
    st.markdown(
        f"""
        <div class="subheader-container">
            <span class="subheader-icon"></span> {text}
        </div>
        """,
        unsafe_allow_html=True
    )

# Configuración de la página
st.set_page_config(page_title="Análisis de Inversiones", page_icon="📊", layout="wide")



st.markdown(
    """
    <style>
    .subheader-container {
        font-family: 'Roboto', sans-serif;
        font-size: 18px; /* Tamaño moderado */
        font-weight: 600; /* Peso ligeramente más fuerte para destacar */
        color: #2C3E50; /* Azul oscuro para el texto */
        margin: 10px 0; /* Espaciado superior e inferior */
        padding: 5px 0; /* Menor relleno */
        position: relative; /* Para posicionar la línea azul */
    }
    .subheader-container::after {
        content: "";
        display: block;
        width: 100%; /* Línea que ocupa todo el ancho */
        height: 2px; /* Grosor más fino */
        background-color: #4A90E2; /* Azul elegante */
        margin-top: 5px; /* Separación de la línea respecto al texto */
        border-radius: 1px; /* Bordes suaves de la línea */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Get current date in the specified format
current_date = datetime.now().strftime("%Y-%m-%d")

# Custom CSS for a title with a horizontal line underneath
st.markdown(
    f"""
    <style>
    .title-container {{
        display: inline-block; /* Limita el ancho del subrayado al contenido */
        text-align: left; /* Alinear título a la izquierda */
        padding: 5px 0;
        font-family: 'Roboto', sans-serif;
        font-size: 24px; /* Tamaño ligeramente mayor */
        font-weight: bold;
        color: #2C3E50; /* Azul oscuro para el texto */
        border-bottom: 3px solid #4A90E2; /* Subrayado azul elegante */
    }}
    .date {{
        color: #4A90E2; /* Azul elegante para la fecha */
        font-weight: normal; /* Peso más ligero para diferenciar del título */
        font-size: 20px; /* Tamaño ligeramente menor */
    }}
    </style>
    <div class="title-container">
        🐸 Intelligent Investor 📅 <span class="date">({current_date})</span>
    </div>
    """,
    unsafe_allow_html=True
)


# Inicializar el estado de la sesión si es necesario
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False

# Mostrar el cargador solo si no se ha cargado un archivo
if not st.session_state.file_uploaded:
    uploaded_file = st.file_uploader("", type="csv")
    if uploaded_file is not None:
        try:
            # Validación básica del archivo
            df = pd.read_csv(uploaded_file)  # Intenta cargar el archivo
            st.session_state.uploaded_file = uploaded_file
            st.session_state.file_uploaded = True

        except Exception as e:
            st.error(f"Error al cargar el archivo: {e}")

# Configuración de las pestañas
if st.session_state.file_uploaded and hasattr(st.session_state, 'uploaded_file'):
    # Si el archivo está cargado, mostrar todas las pestañas
    tabs = st.tabs(["Resumen", "Visualizaciones", "Datos Cargados", "Análisis Empresas", "Análisis SP500"])
    tab1, tab2, tab3, tab4, tab5 = tabs
    # Cargar los datos del archivo CSV
    df = load_data(st.session_state.uploaded_file)
    df['FECHA'] = pd.to_datetime(df['FECHA'])
else:
    # Si no hay archivo cargado, mostrar solo las pestañas 4 y 5
    tabs = st.tabs(["Análisis Empresas", "Análisis SP500"])
    tab4, tab5 = tabs

# Condiciones para las pestañas
if st.session_state.file_uploaded and hasattr(st.session_state, 'uploaded_file'):
    with tab1:

            # styled_subheader('Resumen Total de la Cartera')

            results = analyze_investments(df)

            # CSS personalizado para asegurar que el tamaño de fuente se aplique correctamente
            st.markdown("""
                <style>
                    .resumen-cartera p {
                        font-size: 16px !important;
                        line-height: 1.5;
                    }
                </style>
            """, unsafe_allow_html=True)

            # Cálculos de resumen total
            total_invested = results['Total Invertido (EUR)'].sum()
            total_current_value = results['Valor Actual (EUR)'].sum()
            total_dividends = results['Dividendos Recibidos (EUR)'].sum()
            total_profit_loss = total_current_value - total_invested + total_dividends
            total_profit_loss_percentage = (total_profit_loss / total_invested) * 100 if total_invested != 0 else 0

            # Mostrar datos en tarjetas
            col1, col2, col3, col4 = st.columns(4)

            # Tarjeta para el Capital Invertido
            with col1:
                st.markdown("""
                    <div style="background-color:#f9f9f9; padding:15px; border-radius:10px; text-align:center; box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);">
                        <h4 style="margin:0; color:#333; font-size:14px;">💰 Invertido</h4>
                        <p style="font-size:20px; font-weight:bold; margin:5px 0;">{0:,.2f} €</p>
                    </div>
                """.format(total_invested), unsafe_allow_html=True)

            # Tarjeta para el Valor Actual
            with col2:
                st.markdown("""
                    <div style="background-color:#f9f9f9; padding:15px; border-radius:10px; text-align:center; box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);">
                        <h4 style="margin:0; color:#333; font-size:14px;">📈 Valor Actual</h4>
                        <p style="font-size:20px; font-weight:bold; margin:5px 0;">{0:,.2f} €</p>
                    </div>
                """.format(total_current_value), unsafe_allow_html=True)

            # Tarjeta para el Rendimiento
            rendimiento_color = "green" if total_profit_loss >= 0 else "red"
            with col3:
                st.markdown("""
                    <div style="background-color:#f9f9f9; padding:15px; border-radius:10px; text-align:center; box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);">
                        <h4 style="margin:0; color:#333; font-size:14px;">📊 Rendimiento</h4>
                        <p style="font-size:20px; font-weight:bold; margin:5px 0; color:{1};">
                            {0:+,.2f} € ({2:+.2f}%)
                        </p>
                    </div>
                """.format(total_profit_loss, rendimiento_color, total_profit_loss_percentage), unsafe_allow_html=True)

            # Tarjeta para los Dividendos
            with col4:
                st.markdown("""
                    <div style="background-color:#f9f9f9; padding:15px; border-radius:10px; text-align:center; box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);">
                        <h4 style="margin:0; color:#333; font-size:14px;">💸 Dividendos</h4>
                        <p style="font-size:20px; font-weight:bold; margin:5px 0;">{0:,.2f} €</p>
                    </div>
                """.format(total_dividends), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            exchange_rate = get_exchange_rate()
            st.info(f"Tipo de cambio: 1 USD = {exchange_rate:.4f} EUR", icon="💱")

            # styled_subheader('Detalle de Inversiones por Ticker')
            st.markdown("<br>", unsafe_allow_html=True)  # Añadir un espacio

            # Crear el DataFrame con detalles de inversión por ticker

            # Crear el DataFrame con detalles de inversión por ticker
            data = []
            for ticker in results['Ticker'].unique():
                ticker_results = results[results['Ticker'] == ticker]
                ticker_invested = ticker_results['Total Invertido (EUR)'].values[0]
                ticker_current_value = ticker_results['Valor Actual (EUR)'].values[0]
                ticker_profit_loss = ticker_current_value - ticker_invested
                ticker_profit_loss_percentage = (ticker_profit_loss / ticker_invested) * 100 if ticker_invested != 0 else 0
                
                # Obtener valores existentes
                ticker_daily_change = ticker_results['Variación Diaria (EUR)'].values[0] if 'Variación Diaria (EUR)' in ticker_results.columns else 0
                ticker_daily_change_percentage = ticker_results['Variación Diaria %'].values[0] if 'Variación Diaria %' in ticker_results.columns else 0
                ticker_shares = ticker_results['Acciones'].values[0] if 'Acciones' in ticker_results.columns else 0
                
                # Calcular Ganancia/Pérdida Diaria en dinero
                ticker_daily_profit_loss = ticker_daily_change * ticker_shares

                data.append({
                    'Ticker': ticker,
                    'Capital Invertido': ticker_invested,
                    'Valor Actual': ticker_current_value,
                    'Ganancia/Pérdida': ticker_profit_loss,
                    'Ganancia/Pérdida %': ticker_profit_loss_percentage,
                    'Variación Diaria %': ticker_daily_change_percentage,
                    'Ganancia/Pérdida Diaria (€)': ticker_daily_profit_loss
                })

            ticker_details_df = pd.DataFrame(data)
            ticker_details_df = ticker_details_df.sort_values('Ganancia/Pérdida %', ascending=False)

            # Visualización de los resultados con las nuevas columnas unidas
            col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 2])

            # Fila inicial con nombres de las columnas
            with col1:
                st.markdown("**Ticker**", unsafe_allow_html=True)
            with col2:
                st.markdown("**💰 Capital Invertido**", unsafe_allow_html=True)
            with col3:
                st.markdown("**📈 Valor Actual**", unsafe_allow_html=True)
            with col4:
                st.markdown("**Ganancia/Pérdida**", unsafe_allow_html=True)
            with col5:
                st.markdown("**Variación Diaria**", unsafe_allow_html=True)

            st.markdown("<hr style='margin:5px 0;'>", unsafe_allow_html=True)


            # Visualización de los resultados con las nuevas columnas unidas
            for index, row in ticker_details_df.iterrows():
                col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 2])
                
                with col1:
                    st.markdown(f"**<span style='color:blue'>{row['Ticker']}</span>**", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"{row['Capital Invertido']:,.2f} €")
                with col3:
                    st.markdown(f"{row['Valor Actual']:,.2f} €")
                
                # Mostrar Ganancia/Pérdida con color según positivo/negativo
                profit_loss_color = "green" if row['Ganancia/Pérdida'] >= 0 else "red"
                with col4:
                    st.markdown(
                        f"<span style='color:{profit_loss_color}'> {row['Ganancia/Pérdida']:+,.2f} € ({row['Ganancia/Pérdida %']:+.2f}%)</span>", 
                        unsafe_allow_html=True
                    )
                
                # Unir Ganancia/Pérdida Diaria (€) y Variación Diaria % en una sola columna con formato correcto
                try:
                    daily_profit_loss = float(row['Ganancia/Pérdida Diaria (€)'])
                    daily_profit_loss_color = "green" if daily_profit_loss > 0 else "red"
                    daily_change_percentage = float(row['Variación Diaria %'])  # Convertimos a float para evitar errores
                except (ValueError, TypeError):
                    daily_profit_loss = 0
                    daily_profit_loss_color = "black"
                    daily_change_percentage = 0

                with col5:
                    st.markdown(
                        f"<span style='color:{daily_profit_loss_color}'> {daily_profit_loss:+,.2f} € ({daily_change_percentage:+.2f}%)</span>", 
                        unsafe_allow_html=True
                    )
                
                st.markdown("<hr style='margin:5px 0;'>", unsafe_allow_html=True)

    with tab2:
    
        styled_subheader('Distribución de la Cartera')

        portfolio_distribution_fig = plot_portfolio_distribution_bars(results)
        if portfolio_distribution_fig is not None:
            st.plotly_chart(portfolio_distribution_fig, use_container_width=True)
        else:
            st.warning("No se pudo generar el gráfico de distribución de la cartera.")

        styled_subheader('Evolución de la Inversión')  
            
        try:
            # Calcular los datos de inversión a lo largo del tiempo
            monthly_data = calculate_investment_value_over_time(df, results)
                            
            # Crear y mostrar el gráfico
            investment_over_time_fig = plot_investment_over_time(df, results)
            st.plotly_chart(investment_over_time_fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error al generar el gráfico de evolución de la inversión: {str(e)}")
            
    
        
        styled_subheader('Rendimiento de Ticker Específico')

        # Selector de ticker
        tickers = df['TICKER'].unique()
        selected_ticker = st.selectbox('Selecciona un ticker:', tickers)
        
        if selected_ticker:
            ticker_df = df[df['TICKER'] == selected_ticker]
            start_date = ticker_df['FECHA'].min()
            end_date = datetime.now()
            
            # Plotear el rendimiento del ticker seleccionado
            ticker_performance_fig = plot_ticker_performance(selected_ticker, start_date, end_date)
            st.plotly_chart(ticker_performance_fig, use_container_width=True)
        else:
            st.info("Selecciona un ticker para ver su rendimiento.")

    with tab3:
        
    # Información de empresas con formato visual mejorado
                
        styled_subheader('Información de Empresas')
        company_data = []
        for ticker in df['TICKER'].unique():
            # Excluir el ticker "NVDA"
            if ticker == "0P0000IKFS.F":
                continue

            ticker_df = df[df['TICKER'] == ticker]
            first_purchase_date = ticker_df['FECHA'].min()
            earnings_date = get_earnings_date(ticker)
            splits = get_stock_splits(ticker, first_purchase_date)
            
            split_details = []
            if not splits.empty:
                for date, ratio in splits.items():
                    split_details.append(f"{date.strftime('%Y-%m-%d')}: {ratio:.2f}-for-1")
            
            company_data.append({
                'Ticker': ticker,
                'Resultados': earnings_date,
                'Nº splits': len(splits),
                'Info splits': ', '.join(split_details) if split_details else 'Ninguno',
            })

        company_info_df = pd.DataFrame(company_data)

        # Ordenar por 'Próxima presentación de resultados' en orden ascendente
        company_info_df = company_info_df.sort_values(by='Resultados', ascending=True)

        # Dar formato a la columna 'Ticker' para que sea azul
        company_info_df['Ticker'] = company_info_df['Ticker'].apply(lambda x: f'<span style="color:blue;">{x}</span>')

        # Ajustar la alineación de la columna 'Detalles de splits' a la izquierda
        styled_table = company_info_df.style.format({
            'Ticker': lambda x: f'<span style="color:blue;">{x}</span>'
        }).set_properties(subset=['Info splits'], **{'text-align': 'left'})

        # Mostrar la tabla en Streamlit usando HTML
        st.write(styled_table.to_html(escape=False, index=False), unsafe_allow_html=True)

        styled_subheader('Datos Cargados')
        st.markdown("<br>", unsafe_allow_html=True)  # Añadir un espacio

        # Seleccionar solo las columnas relevantes y reemplazar NaN con cadena vacía
        df_to_display = df[['FECHA', 'TIPO_OP', 'TICKER', 'VOLUMEN', 'PRECIO_ACCION', 'PRECIO_OPERACION_EUR', 'COMENTARIO']].copy()
        df_to_display.fillna("", inplace=True) # Reemplaza NaN con cadena vacía
        df_to_display.reset_index(drop=True, inplace=True)

        # Aplicar estilo al DataFrame y solo formatear celdas numéricas
        styled_html = df_to_display.style.format({
            'FECHA': lambda x: x.strftime('%Y-%m-%d') if isinstance(x, pd.Timestamp) else x,
            'VOLUMEN': lambda x: "{:.2f}".format(x) if isinstance(x, (int, float)) else x,
            'PRECIO_ACCION': lambda x: "{:,.2f} €".format(x) if isinstance(x, (int, float)) else x,
            'PRECIO_OPERACION_EUR': lambda x: "{:,.2f} €".format(x) if isinstance(x, (int, float)) else x
        }).map(lambda x: 'color: green; font-weight: bold;' if x == 'BUY' else 'color: red; font-weight: bold;', subset=['TIPO_OP']) \
        .map(lambda x: 'color: blue; font-weight: bold;' if isinstance(x, str) else '', subset=['TICKER']) \
        .map(lambda x: 'color: blue; font-weight: bold;' if isinstance(x, str) and x.startswith("TRANSFERENCIA") else '', subset=['COMENTARIO']).to_html(index=False)

        # Mostrar el DataFrame estilizado como HTML en Streamlit
        st.markdown(styled_html, unsafe_allow_html=True)# En la Tab 2

# Custom CSS para mejorar la apariencia
st.markdown("""
<style>
    /* Estilos generales */
    .stApp {
        background-color: #f8fafc;
    }
    
    /* Estilo para las métricas */
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        margin: 0.5rem 0;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-label {
        color: #64748b;
        font-size: 0.875rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    
    /* Estilos para la tabla */
    .styled-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        margin: 1rem 0;
        background-color: white;
        border-radius: 0.5rem;
        overflow: hidden;
    }
    .styled-table thead th {
        background-color: #f1f5f9;
        color: #475569;
        font-weight: 600;
        padding: 0.75rem 1rem;
        text-align: left;
    }
    .styled-table tbody td {
        padding: 0.75rem 1rem;
        border-top: 1px solid #e2e8f0;
    }
    
    /* Estilos para los indicadores */
    .indicator-green {
        background-color: #dcfce7;
        color: #166534;
    }
    .indicator-yellow {
        background-color: #fef9c3;
        color: #854d0e;
    }
    .indicator-red {
        background-color: #fee2e2;
        color: #991b1b;
    }
</style>
""", unsafe_allow_html=True)

with tab4:
    styled_subheader("📒 Elección de empresa")

    # Preload S&P 500 tickers
    sp500_tickers = ['NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA']
    
    selected_ticker = st.selectbox("Select a Stock Ticker", options=sp500_tickers)

    if selected_ticker:
        try:
            # Fetch stock data
            stock = yf.Ticker(selected_ticker)
            info = stock.info
            # Recuperar EPS y EPS without NRI
            trailing_eps = info.get('trailingEps', None)

            # Extract and process metrics
            metrics = {
                "Market Cap": {
                    "value": round(info.get('marketCap', 0) / 1e9, 2),
                    "format": lambda x: f"${x}B",
                    "thresholds": {"green": 10, "yellow": 5}
                },
                "ROE": {
                    "value": round(info.get('returnOnEquity', 0) * 100, 2),
                    "format": lambda x: f"{x}%",
                    "thresholds": {"green": 15, "yellow": 8}
                },
                "Debt/Equity": {
                    "value": round(info.get('debtToEquity', 0), 2),
                    "format": lambda x: f"{x}",
                    "thresholds": {"green": 0.6, "yellow": 1.5},
                    "inverse": True
                },
                "Current Ratio": {
                    "value": round(info.get('currentRatio', 0), 2),
                    "format": lambda x: f"{x}",
                    "thresholds": {"green": 2, "yellow": 1}
                },
                "PE Ratio": {
                    "value": round(info.get('trailingPE', 0), 2),
                    "format": lambda x: f"{x}",
                    "thresholds": {"green": 15, "yellow": 30},
                    "inverse": True
                },
                "PBV Ratio": {
                    "value": round(info.get('priceToBook', 0), 2),
                    "format": lambda x: f"{x}",
                    "thresholds": {"green": 1.5, "yellow": 4.5},
                    "inverse": True
                }
            }
            
            cols = st.columns(3)
            for idx, (metric_name, metric_data) in enumerate(metrics.items()):
                with cols[idx % 3]:
                    value = metric_data["value"]
                    formatted_value = metric_data["format"](value)
                    
                    if metric_data.get("inverse", False):
                        color = ("indicator-green" if value < metric_data["thresholds"]["green"] else
                                "indicator-yellow" if value < metric_data["thresholds"]["yellow"] else
                                "indicator-red")
                    else:
                        color = ("indicator-green" if value > metric_data["thresholds"]["green"] else
                                "indicator-yellow" if value > metric_data["thresholds"]["yellow"] else
                                "indicator-red")
                    
                    st.markdown(f"""
                        <div class="metric-card {color}">
                            <div class="metric-label">{metric_name}</div>
                            <div class="metric-value">{formatted_value}</div>
                        </div>
                    """, unsafe_allow_html=True)

            styled_subheader("💰 Análisis de Flujo de Efectivo")
            try:
                cashflow = stock.quarterly_cashflow
                if not cashflow.empty:
                    main_cashflows = ['Operating Cash Flow', 'Investing Cash Flow', 'Financing Cash Flow', 'Free Cash Flow']
                    cashflow_data = cashflow.loc[main_cashflows].T
                    cashflow_data.index = pd.to_datetime(cashflow_data.index)
                    cashflow_data = cashflow_data.sort_index()
                    cashflow_data = cashflow_data.div(1e6)
                    cashflow_data = cashflow_data.tail(5)
                    
                    fig = go.Figure()
                    colors = {
                        'Operating Cash Flow': '#2563eb',
                        'Investing Cash Flow': '#16a34a',
                        'Financing Cash Flow': '#dc2626',
                        'Free Cash Flow': '#9333ea'
                    }
                    
                    for column in cashflow_data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=cashflow_data.index,
                                y=cashflow_data[column],
                                name=column,
                                mode='lines+markers',
                                line=dict(color=colors[column], width=3),
                                marker=dict(size=8),
                                hovertemplate='<b>%{x|%Y-%m-%d}</b><br>' + f'{column}: $%{{y:.0f}}M<br>'
                            )
                        )
                    
                    fig.update_layout(
                        template='plotly_white',
                        height=500,
                        margin=dict(t=30, r=10, b=10, l=10),
                        hovermode='x unified',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#f1f5f9'),
                        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#f1f5f9', title=dict(text="Millones USD", standoff=10))
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    styled_subheader("📈 Decisión sobre Cashflows")
                    metric_cols = st.columns(len(main_cashflows))
                    for idx, column in enumerate(cashflow_data.columns):
                        with metric_cols[idx]:
                            last_value = cashflow_data[column].iloc[-1]
                            prev_value = cashflow_data[column].iloc[-2]
                            change = ((last_value - prev_value) / abs(prev_value)) * 100 if prev_value != 0 else 0

                            # Determinar el color de la tarjeta
                            if column == 'Operating Cash Flow':
                                card_color = "background-color: #dcfce7; color: #166534;" if change > 0 else "background-color: #fee2e2; color: #991b1b;"
                            elif column in ['Investing Cash Flow', 'Financing Cash Flow']:
                                card_color = "background-color: #dcfce7; color: #166534;" if change < 0 else "background-color: #fee2e2; color: #991b1b;"
                            else:
                                card_color = "background-color: #f1f5f9; color: #333;"  # Default color for Free Cash Flow

                            # Crear tarjeta con porcentaje más grande
                            st.markdown(f"""
                                <div style="{card_color} padding: 10px; border-radius: 5px; text-align: center;">
                                    <h4 style="margin: 0;">{column.replace("Cash Flow", "CF")}</h4>
                                    <p style="margin: 0; font-size: 32px; font-weight: bold;">{change:+.1f}%</p>
                                    <p style="margin: 0; font-size: 20px;">${last_value:,.0f}M</p>
                                </div>
                            """, unsafe_allow_html=True)

                else:
                    st.warning("No hay datos de flujo de efectivo disponibles para este stock.")
            except Exception as e:
                st.error(f"Error al procesar datos de flujo de efectivo: {str(e)}")

        except Exception as e:
            st.error(f"Error al obtener datos para {selected_ticker}: {str(e)}")

        st.markdown("<br>", unsafe_allow_html=True)  # Añadir un espacio
        styled_subheader("Valores Intrínsecos de Inversiones")

    cols = st.columns(2)  # Crear 2 columnas para agrupar los inputs
    with cols[0]:
        discount_rate = st.number_input(
            "Tasa de Descuento (%)", min_value=0.01, value=10.0, step=0.1, key="discount_rate"
        ) / 100
        growth_rate = st.number_input(
            "Tasa de Crecimiento Anual (%)", min_value=0.01, value=10.0, step=0.1, key="growth_rate"
        ) / 100

    with cols[1]:
        growth_stage_years = st.number_input(
            "Años de Crecimiento", min_value=1, value=10, step=1, key="growth_stage_years"
        )
        terminal_growth_rate = st.number_input(
            "Tasa de Crecimiento Terminal (%)", min_value=0.01, value=4.0, step=0.1, key="terminal_growth_rate"
        ) / 100

        # Inicializar variables en session_state
        if "iv_result" not in st.session_state:
            st.session_state.iv_result = None
            st.session_state.current_price = None
            st.session_state.mos = None
            st.session_state.calculate = False  # Control de cálculo

        # Botón para calcular el valor intrínseco
        if st.button("Calcular Valor Intrínseco"):
            st.session_state.calculate = True  # Marcar que se debe calcular
            try:
                # Obtener datos del stock
                stock_info = get_stock_info(selected_ticker)
                trailing_eps = stock_info.get("trailingEps", None)

                if trailing_eps is None:
                    st.error("El EPS (trailingEps) no está disponible para este ticker.")
                    st.session_state.calculate = False  # Desactivar cálculo
                    st.stop()

                # Calcular el valor intrínseco
                iv_result = calculate_intrinsic_value(
                    eps=trailing_eps,
                    discount_rate=discount_rate,
                    growth_rate=growth_rate,
                    growth_stage_years=growth_stage_years,
                    terminal_growth_rate=terminal_growth_rate
                )

                if iv_result:
                    current_price = get_current_price2(stock_info)
                    mos = ((iv_result["intrinsic_value"] - current_price) / iv_result["intrinsic_value"]) * 100

                    # Guardar resultados en el estado
                    st.session_state.iv_result = iv_result
                    st.session_state.current_price = current_price
                    st.session_state.mos = mos
                else:
                    st.warning("No se pudo calcular el valor intrínseco.")
                    st.session_state.calculate = False  # Desactivar cálculo
            except Exception as e:
                st.error(f"Error durante el cálculo: {str(e)}")
                st.session_state.calculate = False  # Desactivar cálculo

        # Mostrar los resultados solo si se presionó el botón y se calcularon los resultados
        if st.session_state.calculate and st.session_state.iv_result:
            iv_result = st.session_state.iv_result
            current_price = st.session_state.current_price
            mos = st.session_state.mos

            st.subheader(f"Resultados para {selected_ticker}")

            # Diseño con columnas para alinear las tarjetas
            col1, col2, col3 = st.columns(3)

            with col1:
                # Tarjeta para el Precio Actual
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Precio Actual</div>
                        <div class="metric-value">${current_price:,.2f}</div>
                    </div>
                """, unsafe_allow_html=True)

            with col2:
                # Tarjeta para el Valor Intrínseco
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Valor Intrínseco</div>
                        <div class="metric-value">${iv_result['intrinsic_value']:,.2f}</div>
                    </div>
                """, unsafe_allow_html=True)

            with col3:
                # Tarjeta para el Margen de Seguridad (MoS)
                mos_color = "indicator-green" if mos > 0 else "indicator-red"
                st.markdown(f"""
                    <div class="metric-card {mos_color}">
                        <div class="metric-label">Margen de Seguridad (MoS)</div>
                        <div class="metric-value">{mos:.2f}%</div>
                    </div>
                """, unsafe_allow_html=True)

with tab5:

# Título de la sección
    styled_subheader('📈 Análisis del S&P 500')
    
    # Resultado y recomendación del análisis
    resultado, recomendacion, df_analysis = analizar_sp500()

    # Determinar el color basado en el resultado
    resultado_color = "#a3d977" if resultado > 0 else "#ffcdd2"
    
    # Formatear la fecha para el gráfico
    df_analysis['Fecha_Formato'] = df_analysis['Fecha'].dt.strftime('%b %Y')
    
    # Gráfico de barras con colores basados en el cambio
    fig = go.Figure()
    
    for i in range(1, len(df_analysis)):
        color = "#a3d977" if df_analysis.iloc[i]['Precio de Cierre'] > df_analysis.iloc[i-1]['Precio de Cierre'] else "#ffcdd2"
        fig.add_trace(go.Bar(
            x=[df_analysis.iloc[i]['Fecha_Formato']],
            y=[df_analysis.iloc[i]['Precio de Cierre']],
            marker_color=color
        ))
    
    fig.update_layout(
        xaxis_title='Fecha',
        yaxis_title='Valor de Cierre ($)',
        yaxis_tickformat='$,.0f',
        showlegend=False

    )
    
    st.plotly_chart(fig)

    # Mostrar análisis y recomendación con HTML para el color
    st.markdown(
        f"""
        <div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px; border: 1px solid #cce7ff;">
            📈 Resultado del análisis: <span style="color:{resultado_color}; font-weight:normal;">{resultado}</span> |
             <span style="font-weight:normal;">{recomendacion}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

