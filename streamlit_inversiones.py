import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pypfopt import EfficientFrontier, risk_models, expected_returns
import io
import re
import numpy as np

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
        usd_eur = yf.Ticker("USDEUR=X")
        rate = usd_eur.info.get('regularMarketPrice')
        
        if rate is not None:
            st.write(f"Tipo de cambio obtenido: 1 USD = {rate} EUR")
            return rate
        
        eur_usd = yf.Ticker("EURUSD=X")
        rate = eur_usd.info.get('regularMarketPrice')
        
        if rate is not None:
            inverse_rate = 1 / rate
            st.write(f"Tipo de cambio obtenido (inverso): 1 USD = {inverse_rate} EUR")
            return inverse_rate
        
        hist = usd_eur.history(period="1d")
        if not hist.empty and 'Close' in hist.columns:
            rate = hist['Close'].iloc[-1]
            # st.write(f"Tipo de cambio obtenido de datos históricos: 1 USD = {rate} EUR")
            return rate
        
        st.warning("No se pudo obtener el tipo de cambio actual. Usando 1 USD = 0.85 EUR")
        return 0.85
    
    except Exception as e:
        st.warning(f"Error al obtener el tipo de cambio: {str(e)}. Usando 1 USD = 0.85 EUR")
        return 0.85

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
        current_price = get_current_price(ticker)
        
        total_dividends = dividend_df['PRECIO_OPERACION_EUR'].sum()
        
        if ticker != '0P0000IKFS.F' and current_price is not None:
            current_price *= exchange_rate
        
        if current_price is not None:
            current_value = current_price * total_shares
            profit_loss = current_value - total_invested + total_dividends
            profit_loss_percentage = (profit_loss / total_invested) * 100 if total_invested != 0 else 0
        else:
            current_value = None
            profit_loss = None
            profit_loss_percentage = None
        
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
            'Splits': splits
        })
    
    return pd.DataFrame(results)

def plot_portfolio_distribution(results):
    fig = px.pie(
        results, 
        values='Valor Actual (EUR)', 
        names='Ticker', 
        title='Distribución de la Cartera por Valor Actual',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
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
        shares_accumulated[ticker] = shares.resample('M').last().fillna(method='ffill')
        shares_accumulated[ticker].index = shares_accumulated[ticker].index.to_period('M').to_timestamp('M')
    
    monthly_data['Valor de la Inversión'] = 0
    
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

import numpy as np

import numpy as np

def optimize_portfolio(df, results, exclude_ticker='0P0000IKFS.F'):
    tickers = [ticker for ticker in df['TICKER'].unique() if ticker != exclude_ticker]
    start_date = df['FECHA'].min()
    end_date = datetime.now()
    
    prices = {}
    ticker_data_counts = {}
    for ticker in tickers:
        hist = get_historical_data(ticker, start_date, end_date)
        hist.index = hist.index.normalize()
        prices[ticker] = hist['Close']
        ticker_data_counts[ticker] = hist['Close'].count()
    
    st.write("Información de datos por ticker:")
    for ticker, count in ticker_data_counts.items():
        st.write(f"{ticker}: {count} datos")
    
    common_dates = set.intersection(*[set(prices[ticker].index) for ticker in tickers])
    
    st.write(f"Número de fechas comunes a todos los tickers: {len(common_dates)}")
    
    if len(common_dates) < 126:
        raise ValueError(f"No hay suficientes fechas comunes para todos los tickers. Se necesitan al menos 126 días, pero solo hay {len(common_dates)}.")
    
    aligned_prices = pd.DataFrame({ticker: prices[ticker][prices[ticker].index.isin(common_dates)] for ticker in tickers})
    
    st.write(f"Filas de precios después de alinear: {len(aligned_prices)}")
    st.write(f"Primera fecha después de alinear: {aligned_prices.index.min()}")
    st.write(f"Última fecha después de alinear: {aligned_prices.index.max()}")
    
    mu = expected_returns.mean_historical_return(aligned_prices)
    S = risk_models.sample_cov(aligned_prices)
    
    S = (S + S.T) / 2
    
    if not np.all(np.linalg.eigvals(S) > 0):
        S = risk_models.cov_nearest(S)
    
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    
    portfolio_allocation = pd.DataFrame({
        'Ticker': cleaned_weights.keys(),
        'Asignación Óptima (%)': [f"{weight*100:.2f}%" for weight in cleaned_weights.values()]
    })
    
    return portfolio_allocation

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
    start_date = end_date - timedelta(days=14*30) # Aumentamos a 14 meses para asegurar 13 meses completos
    data = sp500.history(start=start_date, end=end_date, interval="1mo")
    monthly_closes = data['Close'].resample('M').last().tail(13) # Ahora tomamos los últimos 13 meses
    
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

    # Añadir un 0 al principio de la lista de cambios para alinear con monthly_closes
    cambios = [0] + cambios
    
    df_analysis = pd.DataFrame({
        'Fecha': monthly_closes.index,
        'Precio de Cierre': monthly_closes.values,
        'Cambio': cambios
    })
    
    recomendacion = "QUEDATE EN FONDO" if contador > 0 else "SALTA A BONOS"
    return contador, recomendacion, df_analysis


def company_info_tab(df, results):
    st.subheader('Información de Empresas')
    
    company_data = []
    for ticker in df['TICKER'].unique():
        ticker_df = df[df['TICKER'] == ticker]
        first_purchase_date = ticker_df['FECHA'].min()
        
        earnings_date = get_earnings_date(ticker)
        splits = get_stock_splits(ticker, first_purchase_date)
        
        split_details = []
        if not splits.empty:
            for date, ratio in splits.items():
                split_details.append(f"{date.strftime('%Y-%m-%d')}: {ratio:.2f}-for-1")
        
        dividends = ticker_df[ticker_df['TIPO_OP'] == 'Dividendo']
        total_dividends = dividends['PRECIO_OPERACION_EUR'].sum()
        
        company_data.append({
            'Ticker': ticker,
            'Próxima presentación de resultados': earnings_date,
            'Splits desde la primera compra': len(splits),
            'Detalles de splits': ', '.join(split_details) if split_details else 'Ninguno',
            'Total Dividendos (EUR)': total_dividends
        })
    
    company_info_df = pd.DataFrame(company_data)
    st.dataframe(company_info_df.style.format({
        'Splits desde la primera compra': '{:d}',
        'Total Dividendos (EUR)': '{:.2f}'
    }))
    

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
    
    recomendacion = "QUEDATE EN FONDO" if contador > 0 else "SALTA A BONOS"
    return contador, recomendacion, df_analysis


def company_info_tab(df, results):
    st.subheader('Información de Empresas')
    
    company_data = []
    for ticker in df['TICKER'].unique():
        ticker_df = df[df['TICKER'] == ticker]
        first_purchase_date = ticker_df['FECHA'].min()
        
        earnings_date = get_earnings_date(ticker)
        splits = get_stock_splits(ticker, first_purchase_date)
        
        split_details = []
        if not splits.empty:
            for date, ratio in splits.items():
                split_details.append(f"{date.strftime('%Y-%m-%d')}: {ratio:.2f}-for-1")
        
        dividends = ticker_df[ticker_df['TIPO_OP'] == 'Dividendo']
        total_dividends = dividends['PRECIO_OPERACION_EUR'].sum()
        
        company_data.append({
            'Ticker': ticker,
            'Próxima presentación de resultados': earnings_date,
            'Splits desde la primera compra': len(splits),
            'Detalles de splits': ', '.join(split_details) if split_details else 'Ninguno',
            'Total Dividendos (EUR)': total_dividends
        })
    
    company_info_df = pd.DataFrame(company_data)
    st.dataframe(company_info_df.style.format({
        'Splits desde la primera compra': '{:d}',
        'Total Dividendos (EUR)': '{:.2f}'
    }))
    
    st.subheader('Análisis del S&P 500')
    resultado, recomendacion, df_analysis = analizar_sp500()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Resultado del análisis", resultado)
    with col2:
        st.metric("Recomendación", recomendacion)
    
    st.subheader('Precios de cierre mensuales del S&P 500 (últimos 13 meses)')
    
    # Crear una nueva columna con el formato de fecha deseado
    df_analysis['Fecha_Formato'] = df_analysis['Fecha'].dt.strftime('%b %Y')
    
    # Crear el gráfico con Plotly Express
    fig = px.line(df_analysis, x='Fecha_Formato', y='Precio de Cierre', 
                  labels={'Fecha_Formato': 'Fecha', 'Precio de Cierre': 'Precio de Cierre ($)'},
                  title='S&P 500 - Últimos 13 meses')
    fig.update_layout(xaxis_title='Fecha', yaxis_title='Precio de Cierre ($)')
    
    # Mostrar el gráfico
    st.plotly_chart(fig)
    
    # Formatear el DataFrame para la presentación en la tabla
    df_display = df_analysis.copy()
    df_display['Fecha'] = df_display['Fecha'].dt.strftime('%Y-%m-%d')
    df_display['Precio de Cierre'] = df_display['Precio de Cierre'].round(2)
    df_display['Cambio'] = df_display['Cambio'].map({1: '+1', -1: '-1', 0: '0'})
    
    # Añadir una columna que indique si es el mes base
    df_display['Mes Base'] = ['Sí' if i == 0 else 'No' for i in range(len(df_display))]
    
    # Reordenar las columnas para que 'Mes Base' aparezca después de 'Fecha'
    df_display = df_display[['Fecha',  'Precio de Cierre', 'Cambio']]
    
    st.table(df_display)
    
    st.info("Nota: El 'Mes Base' es el punto de partida para la comparación. Los cambios se calculan respecto al mes anterior, empezando desde el segundo mes.")
st.set_page_config(layout="wide")
st.title('Análisis de Inversiones')

uploaded_file = st.file_uploader("Sube tu archivo CSV", type="csv")

if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)
        df['FECHA'] = pd.to_datetime(df['FECHA'])
        
        tab1, tab2, tab3 = st.tabs(["Resumen", "Visualizaciones", "Información de Empresas"])


        with tab1:
            st.subheader('Resumen Total de la Cartera')
            
            results = analyze_investments(df)
            
            total_invested = results['Total Invertido (EUR)'].sum()
            total_current_value = results['Valor Actual (EUR)'].sum(skipna=True)
            total_dividends = results['Dividendos Recibidos (EUR)'].sum()
            total_profit_loss = total_current_value - total_invested + total_dividends if pd.notnull(total_current_value) else None
            total_profit_loss_percentage = (total_profit_loss / total_invested) * 100 if pd.notnull(total_profit_loss) and total_invested != 0 else None

            col1, col2, col3 = st.columns(3)
            col1.metric("Capital Invertido", f"{total_invested:,.2f} €", delta=None)
            col2.metric("Valor Actual", f"{total_current_value:,.2f} €" if pd.notnull(total_current_value) else "N/A", 
                        delta=f"{total_profit_loss:+,.2f} €" if pd.notnull(total_profit_loss) else None)
            col3.metric("Dividendos Recibidos", f"{total_dividends:,.2f} €", delta=None)
            
            st.markdown("<br>", unsafe_allow_html=True)  # Añadir un espacio
            
            col4, col5 = st.columns(2)
            
            if total_profit_loss is not None:
                color = "green" if total_profit_loss > 0 else "red"
                col4.metric("Rendimiento Absoluto", f"{total_profit_loss:+,.2f} €", delta=None)
            else:
                col4.metric("Rendimiento Absoluto", "N/A", delta=None)
            
            if total_profit_loss_percentage is not None:
                color = "green" if total_profit_loss_percentage > 0 else "red"
                col5.metric("Rendimiento Porcentual", f"{total_profit_loss_percentage:+.2f}%", delta=None)
            else:
                col5.metric("Rendimiento Porcentual", "N/A", delta=None)

            st.markdown("<br>", unsafe_allow_html=True)  # Añadir un espacio

            exchange_rate = get_exchange_rate()
            st.info(f"Tipo de cambio utilizado: 1 USD = {exchange_rate:.4f} EUR", icon="💱")

            st.subheader('Detalle de Inversiones')
            
            # Opciones de ordenación y agrupación
            sort_options = list(results.columns)
            sort_by = st.selectbox('Ordenar por:', sort_options)
            ascending = st.checkbox('Orden ascendente', value=False)
            
            # Aplicar ordenación
            sorted_results = results.sort_values(by=sort_by, ascending=ascending)
            
            # Mostrar tabla ordenada
            st.dataframe(sorted_results.style.format({
                'Total Invertido (EUR)': '{:,.2f}',
                'Precio Promedio (EUR)': '{:,.2f}',
                'Precio Actual (EUR)': '{:,.2f}',
                'Valor Actual (EUR)': '{:,.2f}',
                'Dividendos Recibidos (EUR)': '{:,.2f}',
                'Ganancia/Pérdida (EUR)': '{:+,.2f}',
                'Ganancia/Pérdida %': '{:+.2f}%',
                'Splits': '{:.0f}'
            }).bar(subset=['Valor Actual (EUR)'], color='#5fba7d'))

            st.markdown("<br>", unsafe_allow_html=True)  # Añadir un espacio

            st.subheader('Datos Cargados')
            st.dataframe(df)

        with tab2:
            st.subheader('Rendimiento de Ticker Específico')

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

            st.subheader('Evolución de la Inversión')  
              
            try:
                # Calcular los datos de inversión a lo largo del tiempo
                monthly_data = calculate_investment_value_over_time(df, results)
                                
                # Crear y mostrar el gráfico
                investment_over_time_fig = plot_investment_over_time(df, results)
                st.plotly_chart(investment_over_time_fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error al generar el gráfico de evolución de la inversión: {str(e)}")
                
                
            st.subheader('Distribución de la Cartera')
            portfolio_distribution_fig = plot_portfolio_distribution(results)
            if portfolio_distribution_fig is not None:
                st.plotly_chart(portfolio_distribution_fig, use_container_width=True)
            else:
                st.warning("No se pudo generar el gráfico de distribución de la cartera.")

            st.subheader('Asignación Óptima de Activos')
            try:
                optimal_allocation = optimize_portfolio(df, results)
                st.dataframe(optimal_allocation)
                
                # Crear un gráfico de barras para la asignación óptima
                fig = px.bar(optimal_allocation, x='Ticker', y='Asignación Óptima (%)',
                            title='Asignación Óptima de Activos',
                            labels={'Asignación Óptima (%)': 'Asignación (%)'})
                fig.update_layout(xaxis_title='Ticker', yaxis_title='Asignación (%)')
                st.plotly_chart(fig, use_container_width=True)
                
                st.warning("Nota: La optimización se realizó con datos alineados, lo que puede resultar en un período más corto de lo ideal. Considera esto al interpretar los resultados.")
                
            except ValueError as ve:
                st.warning(f"Advertencia en la optimización de la cartera: {str(ve)}")
            except Exception as e:
                st.error(f"Error al calcular la asignación óptima de activos: {str(e)}")
                st.error("Detalles del error:", exc_info=True)

        with tab3:
            company_info_tab(df, results)

    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo: {str(e)}")