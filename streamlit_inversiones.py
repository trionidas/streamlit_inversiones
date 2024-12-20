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
from streamlit_extras.app_logo import add_logo
from st_aggrid import AgGrid, GridOptionsBuilder
import numpy_financial as npf
from pyxirr import xirr

sp500_ticker_to_name = {
    '0P0000IKFS.F': 'MSCI North America',
    "MMM": "3M Company",
    "AOS": "A. O. Smith Corporation",
    "ABT": "Abbott Laboratories",
    "ABBV": "AbbVie Inc.",
    "ACN": "Accenture plc",
    "ATVI": "Activision Blizzard, Inc.",
    "ADM": "Archer-Daniels-Midland Company",
    "ADBE": "Adobe Inc.",
    "ADP": "Automatic Data Processing, Inc.",
    "AAP": "Advance Auto Parts, Inc.",
    "AES": "AES Corporation",
    "AFL": "Aflac Incorporated",
    "A": "Agilent Technologies, Inc.",
    "APD": "Air Products and Chemicals, Inc.",
    "AKAM": "Akamai Technologies, Inc.",
    "ALK": "Alaska Air Group, Inc.",
    "ALB": "Albemarle Corporation",
    "ARE": "Alexandria Real Estate Equities, Inc.",
    "ALGN": "Align Technology, Inc.",
    "ALLE": "Allegion plc",
    "LNT": "Alliant Energy Corporation",
    "ALL": "Allstate Corporation",
    "GOOGL": "Alphabet Inc.",
    "GOOG": "Alphabet Inc.",
    "MO": "Altria Group, Inc.",
    "AMZN": "Amazon.com, Inc.",
    "AMCR": "Amcor plc",
    "AMD": "Advanced Micro Devices, Inc.",
    "AME": "AMETEK, Inc.",
    "AMGN": "Amgen Inc.",
    "APH": "Amphenol Corporation",
    "ADI": "Analog Devices, Inc.",
    "ANSS": "ANSYS, Inc.",
    "AON": "Aon plc",
    "APA": "APA Corporation",
    "AAPL": "Apple Inc.",
    "AMAT": "Applied Materials, Inc.",
    "APTV": "Aptiv PLC",
    "ACGL": "Arch Capital Group Ltd.",
    "ANET": "Arista Networks, Inc.",
    "AJG": "Arthur J. Gallagher & Co.",
    "AIZ": "Assurant, Inc.",
    "T": "AT&T Inc.",
    "ATO": "Atmos Energy Corporation",
    "ADSK": "Autodesk, Inc.",
    "AEE": "Ameren Corporation",
    "AVB": "AvalonBay Communities, Inc.",
    "AVY": "Avery Dennison Corporation",
    "AXON": "Axon Enterprise, Inc.",
    "AXP": "American Express Company",
    "AVGO": "Broadcom Inc.",
    "BKR": "Baker Hughes Company",
    "BALL": "Ball Corporation",
    "BAC": "Bank of America Corporation",
    "BBWI": "Bath & Body Works, Inc.",
    "BAX": "Baxter International Inc.",
    "BDX": "Becton, Dickinson and Company",
    "BRK.B": "Berkshire Hathaway Inc.",
    "BBY": "Best Buy Co., Inc.",
    "BIO": "Bio-Rad Laboratories, Inc.",
    "TECH": "Bio-Techne Corporation",
    "BIIB": "Biogen Inc.",
    "BLK": "BlackRock, Inc.",
    "BK": "The Bank of New York Mellon Corporation",
    "BA": "Boeing Company",
    "BKNG": "Booking Holdings Inc.",
    "BWA": "BorgWarner Inc.",
    "BXP": "Boston Properties, Inc.",
    "BSX": "Boston Scientific Corporation",
    "BMY": "Bristol-Myers Squibb Company",
    "AVGO": "Broadcom Inc.",
    "BR": "Broadridge Financial Solutions, Inc.",
    "BF.B": "Brown-Forman Corporation",
    "CHRW": "C.H. Robinson Worldwide, Inc.",
    "CDNS": "Cadence Design Systems, Inc.",
    "CZR": "Caesars Entertainment, Inc.",
    "CPT": "Camden Property Trust",
    "CPB": "Campbell Soup Company",
    "COF": "Capital One Financial Corporation",
    "CAH": "Cardinal Health, Inc.",
    "KMX": "CarMax, Inc.",
    "CCL": "Carnival Corporation & plc",
    "CARR": "Carrier Global Corporation",
    "CTLT": "Catalent, Inc.",
    "CAT": "Caterpillar Inc.",
    "CBOE": "Cboe Global Markets, Inc.",
    "CBRE": "CBRE Group, Inc.",
    "CDW": "CDW Corporation",
    "CE": "Celanese Corporation",
    "COR": "Cencora, Inc.",
    "CNC": "Centene Corporation",
    "CNP": "CenterPoint Energy, Inc.",
    "CDAY": "Ceridian HCM Holding Inc.",
    "CF": "CF Industries Holdings, Inc.",
    "CRL": "Charles River Laboratories International, Inc.",
    "SCHW": "Charles Schwab Corporation",
    "CHTR": "Charter Communications, Inc.",
    "CVX": "Chevron Corporation",
    "CMG": "Chipotle Mexican Grill, Inc.",
    "CB": "Chubb Limited",
    "CHD": "Church & Dwight Co., Inc.",
    "CI": "Cigna Corporation",
    "CINF": "Cincinnati Financial Corporation",
    "CTAS": "Cintas Corporation",
    "CSCO": "Cisco Systems, Inc.",
    "C": "Citigroup Inc.",
    "CFG": "Citizens Financial Group, Inc.",
    "CLX": "Clorox Company",
    "CME": "CME Group Inc.",
    "CMS": "CMS Energy Corporation",
    "KO": "Coca-Cola Company",
    "CTSH": "Cognizant Technology Solutions Corporation",
    "CL": "Colgate-Palmolive Company",
    "CMCSA": "Comcast Corporation",
    "CMA": "Comerica Incorporated",
    "CAG": "Conagra Brands, Inc.",
    "COP": "ConocoPhillips",
    "ED": "Consolidated Edison, Inc.",
    "STZ": "Constellation Brands, Inc.",
    "CEG": "Constellation Energy Corporation",
    "COO": "CooperCompanies, Inc.",
    "CPRT": "Copart, Inc.",
    "GLW": "Corning Incorporated",
    "CTVA": "Corteva, Inc.",
    "CSGP": "CoStar Group, Inc.",
    "COST": "Costco Wholesale Corporation",
    "CTRA": "Coterra Energy Inc.",
    "CCI": "Crown Castle Inc.",
    "CSX": "CSX Corporation",
    "CMI": "Cummins Inc.",
    "CVS": "CVS Health Corporation",
    "DHI": "D.R. Horton, Inc.",
    "DHR": "Danaher Corporation",
    "DRI": "Darden Restaurants, Inc.",
    "DVA": "DaVita Inc.",
    "DE": "Deere & Company",
    "DAL": "Delta Air Lines, Inc.",
    "XRAY": "Dentsply Sirona Inc.",
    "DVN": "Devon Energy Corporation",
    "DXCM": "Dexcom, Inc.",
    "FANG": "Diamondback Energy, Inc.",
    "DLR": "Digital Realty Trust, Inc.",
    "DFS": "Discover Financial Services",
    "DIS": "Walt Disney Company",
    "DG": "Dollar General Corporation",
    "DLTR": "Dollar Tree, Inc.",
    "D": "Dominion Energy, Inc.",
    "DPZ": "Domino's Pizza, Inc.",
    "DOV": "Dover Corporation",
    "DOW": "Dow Inc.",
    "DTE": "DTE Energy Company",
    "DUK": "Duke Energy Corporation",
    "DD": "DuPont de Nemours, Inc.",
    "DXC": "DXC Technology Company",
    "EMN": "Eastman Chemical Company",
    "ETN": "Eaton Corporation plc",
    "EBAY": "eBay Inc.",
    "ECL": "Ecolab Inc.",
    "EIX": "Edison International",
    "EW": "Edwards Lifesciences Corporation",
    "EA": "Electronic Arts Inc.",
    "ELV": "Elevance Health, Inc.",
    "LLY": "Eli Lilly and Company",
    "EMR": "Emerson Electric Co.",
    "ENPH": "Enphase Energy, Inc.",
    "ETR": "Entergy Corporation",
    "EOG": "EOG Resources, Inc.",
    "EPAM": "EPAM Systems, Inc.",
    "EQT": "EQT Corporation",
    "EFX": "Equifax Inc.",
    "EQIX": "Equinix, Inc.",
    "EQR": "Equity Residential",
    "ESS": "Essex Property Trust, Inc.",
    "EL": "Estée Lauder Companies Inc.",
    "ETSY": "Etsy, Inc.",
    "EG": "Everest Group, Ltd.",
    "EVRG": "Evergy, Inc.",
    "ES": "Eversource Energy",
    "EXC": "Exelon Corporation",
    "EXPE": "Expedia Group, Inc.",
    "EXPD": "Expeditors International of Washington, Inc.",
    "EXR": "Extra Space Storage Inc.",
    "XOM": "Exxon Mobil Corporation",
    "FFIV": "F5, Inc.",
    "FDS": "FactSet Research Systems Inc.",
    "FICO": "Fair Isaac Corporation",
    "FAST": "Fastenal Company",
    "FRT": "Federal Realty Investment Trust",
    "FDX": "FedEx Corporation",
    "FITB": "Fifth Third Bancorp",
    "FSLR": "First Solar, Inc.",
    "FE": "FirstEnergy Corp.",
    "FIS": "Fidelity National Information Services, Inc.",
    "FISV": "Fiserv, Inc.",
    "FLT": "Fleetcor Technologies, Inc.",
    "FMC": "FMC Corporation",
    "F": "Ford Motor Company",
    "FTNT": "Fortinet, Inc.",
    "FTV": "Fortive Corporation",
    "FOXA": "Fox Corporation",
    "FOX": "Fox Corporation",
    "BEN": "Franklin Resources, Inc.",
    "FCX": "Freeport-McMoRan Inc.",
    "FRO": "Frontline plc",
    "GRMN": "Garmin Ltd.",
    "IT": "Gartner, Inc.",
    "GEHC": "GE HealthCare Technologies Inc.",
    "GEN": "Gen Digital Inc.",
    "GNRC": "Generac Holdings Inc.",
    "GD": "General Dynamics Corporation",
    "GE": "General Electric Company",
    "GIS": "General Mills, Inc.",
    "GM": "General Motors Company",
    "GPC": "Genuine Parts Company",
    "GILD": "Gilead Sciences, Inc.",
    "GL": "Globe Life Inc.",
    "GPN": "Global Payments Inc.",
    "GS": "Goldman Sachs Group, Inc.",
    "GWW": "W. W. Grainger, Inc.",
    "HAL": "Halliburton Company",
    "HBI": "Hanesbrands Inc.",
    "HAS": "Hasbro, Inc.",
    "HCA": "HCA Healthcare, Inc.",
    "PEAK": "Healthpeak Properties, Inc.",
    "HSIC": "Henry Schein, Inc.",
    "HES": "Hess Corporation",
    "HPE": "Hewlett Packard Enterprise Company",
    "HLT": "Hilton Worldwide Holdings Inc.",
    "HOLX": "Hologic, Inc.",
    "HD": "Home Depot, Inc.",
    "HON": "Honeywell International Inc.",
    "HRL": "Hormel Foods Corporation",
    "HST": "Host Hotels & Resorts, Inc.",
    "HWM": "Howmet Aerospace Inc.",
    "HPQ": "HP Inc.",
    "HUBB": "Hubbell Incorporated",
    "HUM": "Humana Inc.",
    "HBAN": "Huntington Bancshares Incorporated",
    "HII": "Huntington Ingalls Industries, Inc.",
    "IBM": "International Business Machines Corporation",
    "IEX": "IDEX Corporation",
    "IDXX": "IDEXX Laboratories, Inc.",
    "ITW": "Illinois Tool Works Inc.",
    "ILMN": "Illumina, Inc.",
    "INCY": "Incyte Corporation",
    "IR": "Ingersoll Rand Inc.",
    "INTC": "Intel Corporation",
    "ICE": "Intercontinental Exchange, Inc.",
    "IFF": "International Flavors & Fragrances Inc.",
    "IP": "International Paper Company",
    "IPG": "Interpublic Group of Companies, Inc.",
    "INTU": "Intuit Inc.",
    "ISRG": "Intuitive Surgical, Inc.",
    "IVZ": "Invesco Ltd.",
    "INVH": "Invitation Homes Inc.",
    "IQV": "IQVIA Holdings Inc.",
    "IRM": "Iron Mountain Incorporated",
    "JBHT": "J.B. Hunt Transport Services, Inc.",
    "JKHY": "Jack Henry & Associates, Inc.",
    "J": "Jacobs Solutions Inc.",
    "JNJ": "Johnson & Johnson",
    "JCI": "Johnson Controls International plc",
    "JPM": "JPMorgan Chase & Co.",
    "JNPR": "Juniper Networks, Inc.",
    "K": "Kellogg Company",
    "KDP": "Keurig Dr Pepper Inc.",
    "KEY": "KeyCorp",
    "KEYS": "Keysight Technologies, Inc.",
    "KMB": "Kimberly-Clark Corporation",
    "KIM": "Kimco Realty Corporation",
    "KMI": "Kinder Morgan, Inc.",
    "KLAC": "KLA Corporation",
    "KHC": "Kraft Heinz Company",
    "KR": "Kroger Co.",
    "LHX": "L3Harris Technologies, Inc.",
    "LH": "Laboratory Corporation of America Holdings",
    "LRCX": "Lam Research Corporation",
    "LW": "Lamb Weston Holdings, Inc.",
    "LVS": "Las Vegas Sands Corp.",
    "LDOS": "Leidos Holdings, Inc.",
    "LEN": "Lennar Corporation",
    "LIN": "Linde plc",
    "LYV": "Live Nation Entertainment, Inc.",
    "LKQ": "LKQ Corporation",
    "LMT": "Lockheed Martin Corporation",
    "L": "Loews Corporation",
    "LOW": "Lowe's Companies, Inc.",
    "LYB": "LyondellBasell Industries N.V.",
    "MTB": "M&T Bank Corporation",
    "MRO": "Marathon Oil Corporation",
    "MPC": "Marathon Petroleum Corporation",
    "MKTX": "MarketAxess Holdings Inc.",
    "MAR": "Marriott International, Inc.",
    "MMC": "Marsh & McLennan Companies, Inc.",
    "MLM": "Martin Marietta Materials, Inc.",
    "MAS": "Masco Corporation",
    "MTCH": "Match Group, Inc.",
    "MKC": "McCormick & Company, Incorporated",
    "MCD": "McDonald's Corporation",
    "MCK": "McKesson Corporation",
    "MDT": "Medtronic plc",
    "MRK": "Merck & Co., Inc.",
    "META": "Meta Platforms, Inc.",
    "MET": "MetLife, Inc.",
    "MTD": "Mettler-Toledo International Inc.",
    "MGM": "MGM Resorts International",
    "MCHP": "Microchip Technology Incorporated",
    "MU": "Micron Technology, Inc.",
    "MSFT": "Microsoft Corporation",
    "MAA": "Mid-America Apartment Communities, Inc.",
    "MRNA": "Moderna, Inc.",
    "MHK": "Mohawk Industries, Inc.",
    "MOH": "Molina Healthcare, Inc.",
    "TAP": "Molson Coors Beverage Company",
    "MDLZ": "Mondelez International, Inc.",
    "MPWR": "Monolithic Power Systems, Inc.",
    "MNST": "Monster Beverage Corporation",
    "MCO": "Moody's Corporation",
    "MS": "Morgan Stanley",
    "MSI": "Motorola Solutions, Inc.",
    "MSCI": "MSCI Inc.",
    "NDAQ": "Nasdaq, Inc.",
    "NTAP": "NetApp, Inc.",
    "NFLX": "Netflix, Inc.",
    "NWL": "Newell Brands Inc.",
    "NEM": "Newmont Corporation",
    "NWSA": "News Corporation",
    "NWS": "News Corporation",
    "NEE": "NextEra Energy, Inc.",
    "NKE": "NIKE, Inc.",
    "NI": "NiSource Inc.",
    "NDSN": "Nordson Corporation",
    "NSC": "Norfolk Southern Corporation",
    "NTRS": "Northern Trust Corporation",
    "NOC": "Northrop Grumman Corporation",
    "NCLH": "Norwegian Cruise Line Holdings Ltd.",
    "NRG": "NRG Energy, Inc.",
    "NUE": "Nucor Corporation",
    "NVDA": "NVIDIA Corporation",
    "NVR": "NVR, Inc.",
    "NXPI": "NXP Semiconductors N.V.",
    "ORLY": "O'Reilly Automotive, Inc.",
    "OXY": "Occidental Petroleum Corporation",
    "ODFL": "Old Dominion Freight Line, Inc.",
    "OMC": "Omnicom Group Inc.",
    "ON": "ON Semiconductor Corporation",
    "OKE": "ONEOK, Inc.",
    "ORCL": "Oracle Corporation",
    "OTIS": "Otis Worldwide Corporation",
    "PCAR": "PACCAR Inc",
    "PKG": "Packaging Corporation of America",
    "PARA": "Paramount Global",
    "PH": "Parker-Hannifin Corporation",
    "PAYX": "Paychex, Inc.",
    "PAYC": "Paycom Software, Inc.",
    "PYPL": "PayPal Holdings, Inc.",
    "PNR": "Pentair plc",
    "PEP": "PepsiCo, Inc.",
    "PFE": "Pfizer Inc.",
    "PCG": "PG&E Corporation",
    "PM": "Philip Morris International Inc.",
    "PSX": "Phillips 66",
    "PNW": "Pinnacle West Capital Corporation",
    "PXD": "Pioneer Natural Resources Company",
    "PNC": "PNC Financial Services Group, Inc.",
    "POOL": "Pool Corporation",
    "PPG": "PPG Industries, Inc.",
    "PPL": "PPL Corporation",
    "PFG": "Principal Financial Group, Inc.",
    "PG": "Procter & Gamble Company",
    "PGR": "Progressive Corporation",
    "PLD": "Prologis, Inc.",
    "PRU": "Prudential Financial, Inc.",
    "PEG": "Public Service Enterprise Group Incorporated",
    "PTC": "PTC Inc.",
    "PSA": "Public Storage",
    "PHM": "PulteGroup, Inc.",
    "QRVO": "Qorvo, Inc.",
    "PWR": "Quanta Services, Inc.",
    "DGX": "Quest Diagnostics Incorporated",
    "RL": "Ralph Lauren Corporation",
    "RJF": "Raymond James Financial, Inc.",
    "RTX": "Raytheon Technologies Corporation",
    "O": "Realty Income Corporation",
    "REG": "Regency Centers Corporation",
    "REGN": "Regeneron Pharmaceuticals, Inc.",
    "RF": "Regions Financial Corporation",
    "RSG": "Republic Services, Inc.",
    "RMD": "ResMed Inc.",
    "RHI": "Robert Half Inc.",
    "ROK": "Rockwell Automation, Inc.",
    "ROL": "Rollins, Inc.",
    "ROP": "Roper Technologies, Inc.",
    "ROST": "Ross Stores, Inc.",
    "RCL": "Royal Caribbean Cruises Ltd.",
    "SPGI": "S&P Global Inc.",
    "CRM": "Salesforce, Inc.",
    "SBAC": "SBA Communications Corporation",
    "SLB": "Schlumberger Limited",
    "STX": "Seagate Technology Holdings plc",
    "SEE": "Sealed Air Corporation",
    "SRE": "Sempra",
    "NOW": "ServiceNow, Inc.",
    "SHW": "Sherwin-Williams Company",
    "SPG": "Simon Property Group, Inc.",
    "SWKS": "Skyworks Solutions, Inc.",
    "SJM": "J.M. Smucker Company",
    "SNA": "Snap-on Incorporated",
    "SEDG": "SolarEdge Technologies, Inc.",
    "SO": "Southern Company",
    "LUV": "Southwest Airlines Co.",
    "SWK": "Stanley Black & Decker, Inc.",
    "SBUX": "Starbucks Corporation",
    "STT": "State Street Corporation",
    "STE": "STERIS plc",
    "SYK": "Stryker Corporation",
    "SIVB": "SVB Financial Group",
    "SYF": "Synchrony Financial",
    "SNPS": "Synopsys, Inc.",
    "SYY": "Sysco Corporation",
    "TMUS": "T-Mobile US, Inc.",
    "TROW": "T. Rowe Price Group, Inc.",
    "TTWO": "Take-Two Interactive Software, Inc.",
    "TPR": "Tapestry, Inc.",
    "TRGP": "Targa Resources Corp.",
    "TGT": "Target Corporation",
    "TEL": "TE Connectivity Ltd.",
    "TDY": "Teledyne Technologies Incorporated",
    "TFX": "Teleflex Incorporated",
    "TER": "Teradyne, Inc.",
    "TSLA": "Tesla, Inc.",
    "TXN": "Texas Instruments Incorporated",
    "TXT": "Textron Inc.",
    "TMO": "Thermo Fisher Scientific Inc.",
    "TJX": "TJX Companies, Inc.",
    "TSCO": "Tractor Supply Company",
    "TT": "Trane Technologies plc",
    "TDG": "TransDigm Group Incorporated",
    "TRV": "Travelers Companies, Inc.",
    "TRMB": "Trimble Inc.",
    "TFC": "Truist Financial Corporation",
    "TYL": "Tyler Technologies, Inc.",
    "TSN": "Tyson Foods, Inc.",
    "USB": "U.S. Bancorp",
    "UDR": "UDR, Inc.",
    "ULTA": "Ulta Beauty, Inc.",
    "UNP": "Union Pacific Corporation",
    "UAL": "United Airlines Holdings, Inc.",
    "UNH": "UnitedHealth Group Incorporated",
    "UPS": "United Parcel Service, Inc.",
    "URI": "United Rentals, Inc.",
    "UHS": "Universal Health Services, Inc.",
    "UNM": "Unum Group",
    "VLO": "Valero Energy Corporation",
    "VTR": "Ventas, Inc.",
    "VRSN": "VeriSign, Inc.",
    "VRSK": "Verisk Analytics, Inc.",
    "VZ": "Verizon Communications Inc.",
    "VRTX": "Vertex Pharmaceuticals Incorporated",
    "VFC": "VF Corporation",
    "VICI": "VICI Properties Inc.",
    "V": "Visa Inc.",
    "VNO": "Vornado Realty Trust",
    "VMC": "Vulcan Materials Company",
    "WRB": "W. R. Berkley Corporation",
    "WAB": "Westinghouse Air Brake Technologies Corporation",
    "WBA": "Walgreens Boots Alliance, Inc.",
    "WMT": "Walmart Inc.",
    "WBD": "Warner Bros. Discovery, Inc.",
    "WM": "Waste Management, Inc.",
    "WAT": "Waters Corporation",
    "WEC": "WEC Energy Group, Inc.",
    "WFC": "Wells Fargo & Company",
    "WELL": "Welltower Inc.",
    "WST": "West Pharmaceutical Services, Inc.",
    "WDC": "Western Digital Corporation",
    "WU": "Western Union Company",
    "WRK": "WestRock Company",
    "WY": "Weyerhaeuser Company",
    "WHR": "Whirlpool Corporation",
    "WMB": "Williams Companies, Inc.",
    "WLTW": "Willis Towers Watson Public Limited Company",
    "WYNN": "Wynn Resorts, Limited",
    "XEL": "Xcel Energy Inc.",
    "XYL": "Xylem Inc.",
    "YUM": "Yum! Brands, Inc.",
    "ZBRA": "Zebra Technologies Corporation",
    "ZBH": "Zimmer Biomet Holdings, Inc.",
    "ZION": "Zions Bancorporation, N.A.",
    "ZTS": "Zoetis Inc."
}

# Crear los diccionarios inversos
ticker_to_name = sp500_ticker_to_name
name_to_ticker = {v: k for k, v in ticker_to_name.items()}

def clean_number(x):
    if isinstance(x, str):
        x = x.replace('\xa0', '').replace(' ', '')
        x = x.replace('.', '')
        x = x.replace(',', '.')
        x = re.sub(r'[^\d.-]', '', x)
    return x

def load_data(uploaded_file):
    content = uploaded_file.getvalue().decode('utf-8')
    df = pd.read_csv(io.StringIO(content), sep=';', skipinitialspace=True)
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

                return info[price_field] # Podriamos añadir una depuracion aqui para ver que campo se esta usando

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

def calculate_tir(cashflows, dates):
    if not cashflows:
        return 0
    try:
        # Asegurar que las fechas sean objetos date
        dates = [d.date() if isinstance(d, pd.Timestamp) else d for d in dates]
        tir = xirr(dates, cashflows) # El orden de los argumentos es distinto a npf.irr
        return tir
    except Exception as e:
        st.write(f"Error al calcular la TIR: {e}")
        return 0

def analyze_investments(df):
    results = []
    exchange_rate = get_exchange_rate()

    # Lista para almacenar todos los flujos de efectivo individuales
    all_cashflows = []

    for ticker in df['TICKER'].unique():
        
        ticker_df = df[df['TICKER'] == ticker]
        
        buy_df = ticker_df[ticker_df['TIPO_OP'] == 'BUY']
        dividend_df = ticker_df[ticker_df['TIPO_OP'] == 'Dividendo']

        total_invested = buy_df['PRECIO_OPERACION_EUR'].sum()
        total_shares = buy_df['VOLUMEN'].sum()
        avg_price = total_invested / total_shares if total_shares != 0 else 0
        total_dividends = dividend_df['PRECIO_OPERACION_EUR'].sum() if not dividend_df.empty else 0

        current_price = get_current_price(ticker)
        stock = yf.Ticker(ticker)
        full_name = stock.info.get('longName', ticker) 

        previous_close = None
        try:
            hist = get_historical_data(ticker, start_date=datetime.now() - timedelta(days=5), end_date=datetime.now())
            if not hist.empty and 'Close' in hist.columns:
                previous_close = hist['Close'].iloc[-2]
        except Exception as e:
            st.warning(f"Error obteniendo el cierre anterior para {ticker}: {e}")

        if ticker != '0P0000IKFS.F' and current_price is not None:
            current_price *= exchange_rate
        if previous_close is not None:
            previous_close *= exchange_rate

        current_value = current_price * total_shares if current_price is not None else None

        # --- Aquí se añade el flujo positivo del valor actual ---
        df_ticker_cashflows = pd.DataFrame()
        if not buy_df.empty:
            df_ticker_cashflows = pd.concat([
                df_ticker_cashflows,
                pd.DataFrame({
                    'FECHA': buy_df['FECHA'],
                    'cashflow': -buy_df['PRECIO_OPERACION_EUR']
                })
            ])

        if current_value is not None:
            df_ticker_cashflows = pd.concat([
                df_ticker_cashflows,
                pd.DataFrame({
                    'FECHA': [datetime.now()],
                    'cashflow': [current_value + total_dividends]
                })
            ])


        # Calcular TIR por ticker
        ticker_cashflows = df_ticker_cashflows['cashflow'].tolist()
        ticker_dates = df_ticker_cashflows['FECHA'].tolist()
        tir_ticker = calculate_tir(ticker_cashflows, ticker_dates)

        # --- Cálculo de variación diaria ---
        daily_change = current_price - previous_close if previous_close is not None else None
        daily_change_percentage = (daily_change / previous_close) * 100 if previous_close else None

        # Agregar flujos de efectivo del ticker a la lista global
        all_cashflows.extend(df_ticker_cashflows.to_dict('records'))

        results.append({
            'Nombre': full_name,
            'Ticker': ticker,
            'Total Invertido (EUR)': total_invested,
            'Acciones': total_shares,
            'Precio Promedio (EUR)': avg_price,
            'Precio Actual (EUR)': current_price,
            'Valor Actual (EUR)': current_value,
            'Dividendos Recibidos (EUR)': total_dividends,
            'Ganancia/Pérdida (EUR)': current_value - total_invested + total_dividends if current_value is not None else None,
            'Ganancia/Pérdida %': ((current_value - total_invested + total_dividends) / total_invested) * 100 if current_value is not None and total_invested != 0 else None,
            'Variación Diaria (EUR)': daily_change,
            'Variación Diaria %': daily_change_percentage,
            'TIR': tir_ticker,
        })

    # Calcular TIR global
    if all_cashflows:
        df_global_cashflows = pd.DataFrame(all_cashflows)
        df_global_cashflows = df_global_cashflows.groupby('FECHA')['cashflow'].sum().reset_index()
        global_cashflows = df_global_cashflows['cashflow'].tolist()
        global_dates = df_global_cashflows['FECHA'].tolist()
        tir_global = calculate_tir(global_cashflows, global_dates)

    else:
        tir_global = 0

    results_df = pd.DataFrame(results)
    results_df['TIR Global'] = tir_global

    return results_df

def plot_portfolio_distribution_bars(results):
    distribution_data = results.groupby('Ticker')['Valor Actual (EUR)'].sum().reset_index()
    distribution_data['Porcentaje'] = (distribution_data['Valor Actual (EUR)'] / distribution_data['Valor Actual (EUR)'].sum()) * 100
    distribution_data = distribution_data.sort_values('Porcentaje', ascending=True)

    # Añadir nombres completos de las empresas
    distribution_data['Nombre'] = distribution_data['Ticker'].map(ticker_to_name)
    distribution_data['Etiqueta'] = distribution_data['Ticker'] + ' - ' + distribution_data['Nombre']

    # Formatear los valores antes de usarlos en el f-string
    distribution_data['Texto'] = distribution_data.apply(
        lambda x: f"{x['Porcentaje']:.2f}% ({x['Valor Actual (EUR)']:,.2f} EUR)", axis=1
    )

    fig = go.Figure(go.Bar(
        x=distribution_data['Porcentaje'],
        y=distribution_data['Etiqueta'],
        orientation='h',
        text=distribution_data['Texto'],
        textposition='inside',  # Colocar el texto dentro de las barras
        marker=dict(
            color=distribution_data['Porcentaje'],
            colorscale='Sunset',  # Cambio aquí
            cmin=0.80,
            cmax=distribution_data['Porcentaje'].max(),
            showscale=False
        ),
        insidetextanchor="start"  # Centrar el texto dentro de la barra
    ))

    fig.update_layout(
        margin=dict(l=40, r=0, t=20, b=0),  # Ajustar el valor de t (margen superior)
        xaxis_title="Porcentaje (%)",
        yaxis_title="",
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor='rgba(200,200,200,0.5)'),
        title=dict(text=None),
        uniformtext_minsize=10, uniformtext_mode='hide'  # Ajustar tamaño mínimo de texto
    )

    fig.update_traces(
        hovertemplate="<br>".join([
            "Ticker: %{y}",
            "Porcentaje: %{x:.2f}%",
            "Valor Actual: %{text}"
        ]),
        textfont_size=12,  # Aumentar tamaño de fuente del texto en la barra
        textfont_color="white"  # Hacer el color del texto blanco para mejor contraste
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
        margin=dict(t=40, b=40, l=0, r=0),
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

# Modificar la función get_bg_color para aceptar el parámetro 'inverse'

def get_bg_color(value, thresholds, inverse=False):
    if value is None or value == "N/A":
        return "background-color: gray;"
    elif not thresholds:
        return ""  # No hay umbrales definidos, no aplicar color

    if inverse:
        if 'blue' in thresholds and value < thresholds['blue']:
            return "background-color: #cce0f5; color: #084594;"  # Azul
        elif 'green' in thresholds and value < thresholds['green']:
            return "background-color: #dcfce7; color: #166534;"  # Verde
        elif 'yellow' in thresholds and value < thresholds['yellow']:
            return "background-color: #fef9c3; color: #854d0e;"  # Amarillo
        else:
            return "background-color: #fee2e2; color: #991b1b;"  # Rojo
    else:
        if 'green' in thresholds and value > thresholds['green']:
            return "background-color: #dcfce7; color: #166534;"  # Verde
        elif 'yellow' in thresholds and value > thresholds['yellow']:
            return "background-color: #fef9c3; color: #854d0e;"  # Amarillo
        else:
            return "background-color: #fee2e2; color: #991b1b;"  # Rojo

@st.cache_data

def get_data_for_multiple_companies(tickers):
    data = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            cashflow = stock.quarterly_cashflow
            listing_years = calculate_listing_years(ticker)
            return_10y = calculate_annualized_return_10y(ticker)

            # Obtener la variación del CF respecto al trimestre anterior
            operating_cf_change = "N/A"
            investing_cf_change = "N/A"
            financing_cf_change = "N/A"
            if not cashflow.empty and cashflow.shape[1] >= 2:
                operating_cf_change = (
                    (
                        cashflow.loc["Operating Cash Flow", cashflow.columns[0]]
                        - cashflow.loc["Operating Cash Flow", cashflow.columns[1]]
                    )
                    / abs(cashflow.loc["Operating Cash Flow", cashflow.columns[1]])
                ) * 100 if cashflow.loc["Operating Cash Flow", cashflow.columns[1]] != 0 else 0
                investing_cf_change = (
                    (
                        cashflow.loc["Investing Cash Flow", cashflow.columns[0]]
                        - cashflow.loc["Investing Cash Flow", cashflow.columns[1]]
                    )
                    / abs(cashflow.loc["Investing Cash Flow", cashflow.columns[1]])
                ) * 100 if cashflow.loc["Investing Cash Flow", cashflow.columns[1]] != 0 else 0
                financing_cf_change = (
                    (
                        cashflow.loc["Financing Cash Flow", cashflow.columns[0]]
                        - cashflow.loc["Financing Cash Flow", cashflow.columns[1]]
                    )
                    / abs(cashflow.loc["Financing Cash Flow", cashflow.columns[1]])
                ) * 100 if cashflow.loc["Financing Cash Flow", cashflow.columns[1]] != 0 else 0

            # Función auxiliar para obtener valores numéricos o NaN
            def get_numeric_value(info_dict, key, operation=lambda x: x):
                value = info_dict.get(key)
                if value is not None:
                    try:
                        return operation(value)
                    except (TypeError, ValueError):
                        return float("nan")  # Usar NaN en lugar de "N/A"
                else:
                    return float("nan")  # Usar NaN en lugar de "N/A"

            data.append(
                {
                    "Ticker": ticker,
                    "MarketCap(B)": get_numeric_value(info, "marketCap", lambda x: round(x / 1e9, 2)),
                    "ROE(%)": get_numeric_value(info, "returnOnEquity", lambda x: round(x * 100, 2)),
                    "D/E": get_numeric_value(info, "debtToEquity", lambda x: round(x / 100, 2)),
                    "CR": get_numeric_value(info, "currentRatio"),
                    "PE": get_numeric_value(info, "trailingPE"),
                    "PBV": get_numeric_value(info, "priceToBook"),
                    "Ret10y": return_10y if return_10y is not None else float("nan"),
                    "Years": listing_years if listing_years is not None else float("nan"),
                    "Op.CF(Var%)": operating_cf_change if operating_cf_change is not None and operating_cf_change != "N/A" else float("nan"),
                    "Inv.CF(Var%)": investing_cf_change if investing_cf_change is not None and investing_cf_change != "N/A" else float("nan"),
                    "F.CF(Var%)": financing_cf_change if financing_cf_change is not None and financing_cf_change != "N/A" else float("nan"),
                }
            )
        except Exception as e:
            st.error(f"Error al obtener datos para {ticker}: {str(e)}")
            data.append(
                {
                    "Ticker": ticker,
                    "MarketCap(B)": float("nan"),
                    "ROE(%)": float("nan"),
                    "D/E": float("nan"),
                    "CR": float("nan"),
                    "PE": float("nan"),
                    "PBV": float("nan"),
                    "Ret10y": float("nan"),
                    "Years": float("nan"),
                    "Op.CF(Var%)": float("nan"),
                    "Inv.CF(Var%)": float("nan"),
                    "F.CF(Var%)": float("nan"),
                }
            )

    return pd.DataFrame(data)

def analyze_multiple_companies(tickers):
    df = get_data_for_multiple_companies(tickers)

    # Definir umbrales
    thresholds = {
        "MarketCap(B)": {"green": 10, "yellow": 5},
        "ROE(%)": {"green": 8, "yellow": 7},
        "D/E": {"blue": 0.6, "green": 1, "yellow": 2, "inverse": True},
        "CR": {"green": 1.5, "yellow": 1},
        "PE": {"green": 15, "yellow": 30, "inverse": True},
        "PBV": {"green": 1.5, "yellow": 4.5, "inverse": True},
        "Ret10y": {"green": 15, "yellow": 10},
        "Years": {"green": 10, "yellow": 8},
        "Op.CF(Var%)": {"green": 0, "yellow": -5},
        "Inv.CF(Var%)": {"green": 0, "yellow": 5, "inverse": True},
        "F.CF(Var%)": {"green": 0, "yellow": 5, "inverse": True},
    }

    def style_df(df, thresholds):
        styled_df = df.style.apply(
            lambda col: [
                get_bg_color(
                    val,
                    thresholds.get(col.name, {}),
                    inverse=thresholds.get(col.name, {}).get("inverse", False),
                )
                for val in col
            ],
            axis=0,
        )
        return styled_df

    # Aplicar estilos
    styled_df = style_df(df, thresholds)

    # Redondear a 2 decimales después de aplicar los estilos
    styled_df = styled_df.format(precision=2)

    return styled_df

def calculate_listing_years(ticker):
    """
    Calcula los años que lleva cotizando una empresa.

    Args:
        ticker (str): El ticker de la empresa.

    Returns:
        int: El número de años que lleva cotizando la empresa, o None si hay un error.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        first_trade_date = info.get("firstTradeDateEpochUtc")

        if first_trade_date is None:
            # Algunas empresas no tienen firstTradeDateEpochUtc, pero tienen ipoDate
            ipo_date = info.get("ipoDate")
            if ipo_date:
                first_trade_date = int(datetime.strptime(ipo_date, "%Y-%m-%d").timestamp())

        if first_trade_date:
            first_trade_datetime = datetime.utcfromtimestamp(first_trade_date)
            years_listed = datetime.now().year - first_trade_datetime.year
            return years_listed
        else:
            return None
    except Exception as e:
        print(f"Error al obtener datos para {ticker}: {e}")
        return None

def calculate_annualized_return_10y(ticker):
    """
    Calcula la revalorización anualizada de una empresa en los últimos 10 años.

    Args:
        ticker (str): El ticker de la empresa.

    Returns:
        float: La revalorización anualizada en porcentaje, o None si hay un error.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="10y")  # Obtener datos de los últimos 10 años

        if hist.empty:
            print(f"No se encontraron datos históricos para {ticker} en los últimos 10 años.")
            return None

        start_price = hist["Close"].iloc[0]
        end_price = hist["Close"].iloc[-1]

        total_return = (end_price / start_price) - 1
        annualized_return = (1 + total_return) ** (1 / 10) - 1  # Usar 10 para 10 años

        return round(annualized_return * 100, 2) # Convertir a porcentaje y solo 2 decimales

    except Exception as e:
        print(f"Error al obtener datos para {ticker}: {e}")
        return None

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
st.set_page_config(page_title="Análisis de Inversiones", page_icon="🐸", layout="centered")
add_logo("https://imgur.com/a/p9kb3Mh")

st.markdown(
    """
    <style>
    :root {
      --primary-color: #457b9d;
      --secondary-color: #1d3557;
      --background-color: #ffffff;
      --text-color: #1d3557;
      --accent-color: #e94957;
      --light-accent-color: #a8dadc;
      --success-color: #2a9d8f;
      --error-color: #e63946;
      --gray-color: #64748b;
    }

    /* Estilos generales */
    .stApp {
        background-color: var(--background-color) !important;
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
        color: var(--gray-color);
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
        color: var(--text-color);
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
    /* Estilo para el subheader */
    .subheader-container {
        font-family: 'Roboto', sans-serif;
        font-size: 18px;
        font-weight: 600;
        color: var(--secondary-color);
        margin: 10px 0;
        padding: 5px 0;
        position: relative;
    }
    .subheader-container::after {
        content: "";
        display: block;
        width: 100%;
        height: 2px;
        background-color: var(--primary-color);
        margin-top: 5px;
        border-radius: 1px;
    }
     /* Estilo para las tarjetas de resumen */
     [data-testid="stHorizontalBlock"] > div {
        margin: 0 !important; /* Elimina el margen para que las tarjetas se toquen */
    }
    /* Estilo para la info box */
    .info-box {
        background-color: var(--background-color);
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12);
        border: 1px solid #a8dadc;
    }
    .card-container {
        height: 150px; /* Ajusta la altura según sea necesario */
        margin-bottom: 10px; /* Espacio entre tarjetas */
        display: flex;
        flex-direction: column;
    }
    }
    .card-content {
        flex-grow: 1;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Inicializar el estado de la sesión ---
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False
    st.session_state.uploaded_file = None
    st.session_state.df = None
    st.session_state.rerun_triggered = False  # Nueva variable de estado

if "show_modal" not in st.session_state:
    st.session_state.show_modal = False

# Elementos del menú lateral
menu1 = "📊 Resumen"
menu2 = "👾 Work in Progress"
menu3 = "📋 Datos Cargados"
menu4 = "🏢 Análisis Stock"
menu5 = "📉 Análisis SP500"
menu6 = "📈 Análisis General"

# --- Barra lateral ---
with st.sidebar:

    st.title("🐸 Stonks")

    if st.session_state.file_uploaded:
        opciones_menu = [menu1, menu2, menu3, menu4, menu5, menu6]
    else:
        opciones_menu = [menu4, menu5, menu6]

    menu = st.radio("", opciones_menu, label_visibility="collapsed")

        # Botón para abrir el modal de carga
    if not st.session_state.file_uploaded:
        if st.button("Cargar CSV"):
            st.session_state.show_modal = True

# --- Página principal ---

# Modal para la carga de archivos
if st.session_state.show_modal:
    uploaded_file = st.file_uploader(" ", type="csv", label_visibility="collapsed")

    if uploaded_file is not None:
        # Procesar el archivo cargado
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.uploaded_file = uploaded_file
            st.session_state.df = df
            st.session_state.file_uploaded = True
            st.session_state.rerun_triggered = True  # Indicar que se debe actualizar
            st.success("✔️ Archivo cargado exitosamente.")
            # Cerrar el modal
            st.session_state.show_modal = False
        except Exception as e:
            st.error(f"❌ Error al cargar el archivo: {e}")

# Forzar la actualización de la página si es necesario
if st.session_state.rerun_triggered:
    st.session_state.rerun_triggered = False
    st.rerun()

if st.session_state.file_uploaded:
    df = load_data(st.session_state.uploaded_file)
    df['FECHA'] = pd.to_datetime(df['FECHA'])
    results = analyze_investments(df)

# Condiciones para las pestañas

if menu == menu1 and st.session_state.file_uploaded:

    styled_subheader('Resumen Total de la Cartera')

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
    # Obtener la TIR global del DataFrame 'results'
    tir_global = results['TIR Global'].iloc[0] if not results.empty else 0

    # --- Fila Superior ---
    col1, col2, col3 = st.columns(3)

    # Tarjeta para el Capital Invertido
    with col1:
        st.markdown(f"""
            <div style="background-color:#f0f8ff; padding:10px; border-radius:10px; text-align:center; box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1); height: 120px;">
                <h4 style="margin:0; color:#1d3557; font-size:16px;">💰 Invertido</h4>
                <p style="font-size:22px; font-weight:bold; margin:5px 0; color:#457b9d;">{total_invested:,.2f} €</p>
            </div>
        """, unsafe_allow_html=True)

    # Tarjeta para el Valor Actual
    with col2:
        st.markdown(f"""
            <div style="background-color:#f0f8ff; padding:10px; border-radius:10px; text-align:center; box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1); height: 120px;">
                <h4 style="margin:0; color:#1d3557; font-size:16px;">📈 Valor Actual</h4>
                <p style="font-size:22px; font-weight:bold; margin:5px 0; color:#457b9d;">{total_current_value:,.2f} €</p>
            </div>
        """, unsafe_allow_html=True)

    # Tarjeta para los Dividendos
    with col3:
        st.markdown(f"""
            <div style="background-color:#f0f8ff; padding:10px; border-radius:10px; text-align:center; box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1); height: 120px;">
                <h4 style="margin:0; color:#1d3557; font-size:16px;">💸 Dividendos</h4>
                <p style="font-size:22px; font-weight:bold; margin:5px 0; color:#457b9d;">{total_dividends:,.2f} €</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: -10px;'><br></div>", unsafe_allow_html=True)

    # --- Fila Inferior ---
    col4, col5 = st.columns(2)

    # Tarjeta para el Rendimiento (Valor Actual y Porcentaje)
    rendimiento_color = "#2a9d8f" if total_profit_loss >= 0 else "#e63946"
    with col4:
        st.markdown(f"""
            <div style="background-color:#f0f8ff; padding:10px; border-radius:10px; text-align:center; box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1); height: 120px; display: flex; flex-direction: column; justify-content: center;">
                <p style="font-size:22px; font-weight:bold; margin:0; color:{rendimiento_color};">
                    {total_profit_loss:,.2f} €
                </p>
                <p style="font-size:18px; font-weight:normal; margin:0; color:{rendimiento_color};">
                    {total_profit_loss_percentage:+.2f}%
                </p>
            </div>
        """, unsafe_allow_html=True)

    # Tarjeta para la TIR Global
    with col5:
        st.markdown(f"""
            <div style="background-color:#f0f8ff; padding:10px; border-radius:10px; text-align:center; box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1); height: 120px;">
                <h4 style="margin:0; color:#1d3557; font-size:16px;">🌐 TIR Global</h4>
                <p style="font-size:22px; font-weight:bold; margin:5px 0; color:#457b9d;">{tir_global:.2%} </p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <style>
        .compact-card {
            background-color: #f0f8ff; /* Fondo suave */
            padding: 5px 10px; /* Espacio interno reducido */
            margin: 2px;
            border-radius: 5px; /* Bordes redondeados */
            text-align: center; /* Centrar texto */
            box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1); /* Sombra sutil */
            font-family: 'Arial', sans-serif; /* Tipografía */
        }
        .metric-label {
            color: #1d3557; /* Color para la etiqueta */
            font-size: 12px; /* Tamaño de fuente reducido */
            font-weight: bold;
            margin-bottom: 2px; /* Espacio inferior */
        }
        .metric-value {
            font-size: 16px; /* Tamaño de fuente para el valor */
            font-weight: normal;
            margin: 0;
        }
        </style>
        """, unsafe_allow_html=True)

    # --- Resto del código ---
    exchange_rate = get_exchange_rate()
    st.info(f"Tipo de cambio: 1 USD = {exchange_rate:.4f} EUR", icon="💱")

    # styled_subheader('Detalle de Inversiones por Ticker')
    # st.markdown("<br>", unsafe_allow_html=True)  # Añadir un espacio

    # Crear el DataFrame con detalles de inversión por ticker
    data = []
    for ticker in results['Ticker'].unique():
        ticker_results = results[results['Ticker'] == ticker]
        ticker_invested = ticker_results['Total Invertido (EUR)'].values[0]
        ticker_current_value = ticker_results['Valor Actual (EUR)'].values[0]
        ticker_profit_loss = ticker_current_value - ticker_invested
        ticker_profit_loss_percentage = (ticker_profit_loss / ticker_invested) * 100 if ticker_invested != 0 else 0
        ticker_tir = ticker_results['TIR'].values[0] if 'TIR' in ticker_results.columns else 0

        # Obtener valores existentes
        ticker_daily_change = ticker_results['Variación Diaria (EUR)'].values[0] if 'Variación Diaria (EUR)' in ticker_results.columns else 0
        ticker_daily_change_percentage = ticker_results['Variación Diaria %'].values[0] if 'Variación Diaria %' in ticker_results.columns else 0
        ticker_shares = ticker_results['Acciones'].values[0] if 'Acciones' in ticker_results.columns else 0

        # Calcular Ganancia/Pérdida Diaria en dinero
        ticker_daily_profit_loss = ticker_daily_change * ticker_shares
        
        # Acceder al nombre completo de la empresa desde ticker_results
        full_name = ticker_results['Nombre'].iloc[0]  # Añadir esta línea

        data.append({
            'Nombre': ticker,
            'TIR': ticker_results['TIR'].iloc[0],
            'Invertido (€)': round(ticker_invested, 2),
            'Valor Actual (€)': round(ticker_current_value, 2),
            'G/P (€)': round(ticker_profit_loss, 2),
            'G/P (%)': round(ticker_profit_loss_percentage, 2),
            'Var. Diaria (€)': round(ticker_daily_profit_loss, 2),
            'Var. Diaria (%)': round(ticker_daily_change_percentage, 2),
        })

    ticker_details_df = pd.DataFrame(data)
    ticker_details_df = ticker_details_df.reset_index(drop=True)

    def apply_styles(df):
        styled = df.style.format(
            {
                'Invertido (€)': '{:.2f} €',
                'Valor Actual (€)': '{:.2f} €',
                'G/P (€)': '{:.2f} €',
                'G/P (%)': '{:.2f}%',
                'Var. Diaria (€)': '{:.2f} €',
                'Var. Diaria (%)': '{:.2f}%',
                'TIR': '{:.2%}',
            }
        )
        
        def color_tir(val):
            if val < 0.10:
                color = 'red'
            elif val >= 0.10 and val <= 0.15:
                color = 'green'
            elif val > 0.15:
                color = 'blue'
            else:
                color = 'black'  # Valor por defecto
            return f'color: {color}'

        styled = styled.applymap(color_tir, subset=['TIR'])

        styled = styled.applymap(
            lambda x: 'color: blue; font-weight: bold;',
            subset=['Nombre']
        )

        styled = styled.applymap(
            lambda x: 'color: green;' if isinstance(x, (int, float)) and x > 0 else 'color: red;' if isinstance(x, (int, float)) and x < 0 else '',
            subset=['G/P (€)', 'G/P (%)', 'Var. Diaria (€)', 'Var. Diaria (%)']
        )

        return styled

    # Aplicar estilos y mostrar la tabla
    styled_df = apply_styles(ticker_details_df)
    st.dataframe(styled_df, use_container_width=True, hide_index=True, height=500)


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

if menu == menu2 and st.session_state.file_uploaded:
    
        styled_subheader('Work in progress')

if menu == menu3 and st.session_state.file_uploaded:

    # --- Datos Cargados ---
    styled_subheader('Datos Cargados')

    # Seleccionar solo las columnas relevantes y reemplazar NaN con cadena vacía
    df_to_display = df[['FECHA', 'TIPO_OP', 'TICKER', 'VOLUMEN', 'PRECIO_ACCION', 'PRECIO_OPERACION_EUR', 'COMENTARIO']].copy()
    df_to_display.fillna("", inplace=True)  # Reemplaza NaN con cadena vacía
    df_to_display.reset_index(drop=True, inplace=True)

    # Formatear columnas numéricas y de fechas
    df_to_display['FECHA'] = df_to_display['FECHA'].apply(lambda x: x.strftime('%Y-%m-%d') if isinstance(x, pd.Timestamp) else x)
    df_to_display['VOLUMEN'] = df_to_display['VOLUMEN'].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
    df_to_display['PRECIO_ACCION'] = df_to_display['PRECIO_ACCION'].apply(lambda x: f"{x:,.2f} €" if isinstance(x, (int, float)) else x)
    df_to_display['PRECIO_OPERACION_EUR'] = df_to_display['PRECIO_OPERACION_EUR'].apply(lambda x: f"{x:,.2f} €" if isinstance(x, (int, float)) else x)

    # Configurar st-ag-grid para la tabla de Datos Cargados
    gb = GridOptionsBuilder.from_dataframe(df_to_display)
    gb.configure_pagination()
    gb.configure_side_bar()
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=False)

    # Aplicar estilos a las columnas 'Ticker' y 'TIPO_OP'
    gb.configure_column("TICKER", header_name="TICKER", cellStyle={'color': 'blue', 'font-weight': 'bold'})
    gb.configure_column("TIPO_OP", header_name="TIPO_OP", cellStyle={
        'styleConditions': [
            {
                'condition': "String(value) == 'BUY'",
                'style': {'color': 'green', 'font-weight': 'bold'}
            }
        ]
    })

    gridOptions = gb.build()

    AgGrid(
        df_to_display,
        gridOptions=gridOptions,
        height=500,
        width='100%',
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=True,
        enable_quicksearch=True,
        reload_data=True
    )

    # --- Información de Empresas ---
    styled_subheader('Información de Empresas')
    company_data = []
    for ticker in df['TICKER'].unique():
        # Excluir el ticker del fondo
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
            'Nº Splits': len(splits),
            'Detalles de Splits': ', '.join(split_details) if split_details else 'Ninguno',
        })

    company_info_df = pd.DataFrame(company_data)

    # Configurar st-ag-grid para la tabla de Información de Empresas
    gb = GridOptionsBuilder.from_dataframe(company_info_df)
    gb.configure_pagination()
    gb.configure_side_bar()
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=False)

    # Aplicar estilo a la columna 'Ticker'
    gb.configure_column("Ticker", header_name="Ticker", cellStyle={'color': 'blue', 'font-weight': 'bold'})

    gridOptions = gb.build()

    AgGrid(
        company_info_df,
        gridOptions=gridOptions,
        height=350,
        width='100%',
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=True,
        enable_quicksearch=True
    )

if menu == menu4:
    styled_subheader("Elección de empresa")

    # Crear una lista de opciones con el formato "ticker - nombre"
    options = [f"{ticker} - {name}" for ticker, name in ticker_to_name.items()]
    options.sort()  # Ordenar alfabéticamente

    # Encontrar el índice de la opción de NVIDIA
    default_index = options.index("NVDA - NVIDIA Corporation") if "NVDA - NVIDIA Corporation" in options else 0

    # Mostrar el selectbox con las opciones, con NVIDIA por defecto
    selected_option = st.selectbox("Selecciona una empresa:", options=options, index=default_index)

    # Extraer el ticker de la opción seleccionada
    selected_ticker = selected_option.split(" - ")[0]

    if selected_ticker:
        try:
            # Usar el ticker seleccionado para obtener datos y mostrar resultados
            # Fetch stock data
            stock = yf.Ticker(selected_ticker)
            info = stock.info
            # Recuperar EPS y EPS without NRI
            trailing_eps = info.get('trailingEps', None)
            listing_years = calculate_listing_years(selected_ticker)
            return_10y = calculate_annualized_return_10y(selected_ticker)

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
                    "thresholds": {"green": 8, "yellow": 6}                    
                },
                "Years": {
                    "value": listing_years,
                    "format": lambda x: f"{x}",
                    "thresholds": {"green": 10, "yellow": 8}
                },
                "Ret10y": {
                    "value": return_10y,
                    "format": lambda x: f"{x:.2f}%",
                    "thresholds": {"green": 15, "yellow": 10}
                },
                "D/E": {
                    "value": round(info.get('debtToEquity', 0) / 100, 2),
                    "format": lambda x: f"{x}",
                    "thresholds": {"blue": 0.6, "green": 1, "yellow": 2},
                    "inverse": True
                },
                "CR": {
                    "value": round(info.get('currentRatio', 0), 2),
                    "format": lambda x: f"{x}",
                    "thresholds": {"green": 1.5, "yellow": 1}
                },
                "PE": {
                    "value": round(info.get('trailingPE', 0), 2),
                    "format": lambda x: f"{x}",
                    "thresholds": {"green": 15, "yellow": 30},
                    "inverse": True
                },
                "PBV": {
                    "value": round(info.get('priceToBook', 0), 2),
                    "format": lambda x: f"{x}",
                    "thresholds": {"green": 1.5, "yellow": 4.5},
                    "inverse": True
                }
            }
            
            # Usar las variables de color en las tarjetas
            # Primera fila
            cols1 = st.columns(4)
            for idx, (metric_name, metric_data) in enumerate(list(metrics.items())[:4]):
                with cols1[idx]:
                    value = metric_data["value"]
                    formatted_value = metric_data["format"](value)
                    thresholds = metric_data.get("thresholds")
                    inverse = metric_data.get("inverse", False)

                    if thresholds:
                        bg_color = get_bg_color(value, thresholds, inverse)
                    else:
                        bg_color = ""

                    st.markdown(f"""
                        <div class="metric-card" style="{bg_color}">
                            <div class="metric-label">{metric_name}</div>
                            <div class="metric-value">{formatted_value}</div>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Segunda fila
            cols2 = st.columns(4)
            for idx, (metric_name, metric_data) in enumerate(list(metrics.items())[4:]):
                with cols2[idx]:
                    value = metric_data["value"]
                    formatted_value = metric_data["format"](value)
                    thresholds = metric_data.get("thresholds")
                    inverse = metric_data.get("inverse", False)

                    if thresholds:
                        bg_color = get_bg_color(value, thresholds, inverse)
                    else:
                        bg_color = ""

                    st.markdown(f"""
                        <div class="metric-card" style="{bg_color}">
                            <div class="metric-label">{metric_name}</div>
                            <div class="metric-value">{formatted_value}</div>
                        </div>
                    """, unsafe_allow_html=True)

            styled_subheader("Análisis de Flujo de Efectivo")
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
                            y=-0.3,
                            xanchor="right",
                            x=0.5
                        ),
                        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#f1f5f9'),
                        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#f1f5f9', title=dict(text="Millones USD", standoff=10))
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    styled_subheader("Decisión sobre Cashflows")
                    metric_cols = st.columns(4)  # Volvemos a 4 columnas

                    for idx, column in enumerate(main_cashflows):
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

                            # Usamos un div con una clase para forzar la altura
                            st.markdown(f"""
                                <div class="card-container">
                                    <div style="{card_color} padding: 10px; border-radius: 5px; text-align: center;" class="card-content">
                                        <h4 style="margin: 0; font-size: 14px;">{column.replace("Cash Flow", "CF")}</h4>
                                        <p style="margin: 0; font-size: 28px; font-weight: bold;">{change:+.1f}%</p>
                                        <p style="margin: 0; font-size: 18px;">${last_value:,.0f}M</p>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                else:
                    st.warning("No hay datos de flujo de efectivo disponibles para este stock.")
            except Exception as e:
                st.error(f"Error al procesar datos de flujo de efectivo: {str(e)}")

        except Exception as e:
            st.error(f"Error al obtener datos para {selected_ticker}: {str(e)}")

        start_date = '2021-01-01'
        end_date = datetime.now()
        
        # Plotear el rendimiento del ticker seleccionado
        ticker_performance_fig = plot_ticker_performance(selected_ticker, start_date, end_date)
        st.plotly_chart(ticker_performance_fig, use_container_width=True)

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

        # Inicializar variables en session_state si no existen
        if "iv_result" not in st.session_state:
            st.session_state.iv_result = None
            st.session_state.current_price = None
            st.session_state.mos = None
            st.session_state.calculate = False  # Control de cálculo

        # Importante: Resetear el estado de cálculo al cambiar de empresa
        if "previous_ticker" not in st.session_state or st.session_state.previous_ticker != selected_ticker:
            st.session_state.calculate = False
            st.session_state.iv_result = None # Importante resetear
            st.session_state.current_price = None # Importante resetear
            st.session_state.mos = None # Importante resetear
            st.session_state.previous_ticker = selected_ticker

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

if menu == menu5:

# Título de la sección
    styled_subheader('Análisis del S&P 500')
    
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
       # margin=dict(l=40, r=0, t=20, b=0),  # Ajustar el valor de t (margen superior)
        xaxis_title='Fecha',
        yaxis_title='Valor de Cierre ($)',
        yaxis_tickformat='$,.0f',
        showlegend=False

    )
    
    st.plotly_chart(fig)


if menu == menu6:
    styled_subheader("Análisis Multi-Empresa")
    sp500_tickers = ['AAPL', 'NVDA', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'META', 'TSLA', 'AVGO', 'WMT', 'LLY', 'JPM', 'V', 'UNH', 'ORCL', 'MA', 'XOM', 'COST', 'HD', 'PG', 'NFLX', 'JNJ', 'BAC', 'CRM', 'ABBV', 'CVX', 'KO', 'TMUS', 'MRK', 'WFC', 'ADBE', 'CSCO', 'BX', 'NOW', 'ACN', 'PEP', 'AXP', 'IBM', 'MCD', 'LIN', 'MS', 'DIS', 'TMO', 'AMD', 'ABT', 'PM', 'ISRG', 'CAT', 'GS', 'GE', 'INTU', 'VZ', 'QCOM', 'TXN', 'BKNG', 'DHR', 'T', 'PLTR', 'BLK', 'RTX', 'SPGI', 'NEE', 'CMCSA', 'LOW', 'PGR', 'HON', 'AMGN', 'PFE', 'KKR', 'SCHW', 'UNP', 'SYK', 'ETN', 'TJX', 'AMAT', 'ANET', 'C', 'COP', 'BSX', 'PANW', 'UBER', 'BA', 'DE', 'ADP', 'VRTX', 'LMT', 'MU', 'FI', 'NKE', 'GILD', 'BMY', 'CB', 'SBUX', 'UPS', 'ADI', 'MDT', 'MMC', 'PLD', 'LRCX', 'GEV', 'EQIX', 'AMT', 'MO', 'SHW', 'PYPL', 'SO', 'ELV', 'ICE', 'TT', 'CRWD', 'MCO', 'APH', 'KLAC', 'CMG', 'INTC', 'PH', 'WM', 'CTAS', 'CME', 'DUK', 'REGN', 'MDLZ', 'CDNS', 'ABNB', 'CI', 'DELL', 'HCA', 'MAR', 'WELL', 'ZTS', 'ITW', 'PNC', 'USB', 'MSI', 'AON', 'SNPS', 'CL', 'FTNT', 'CEG', 'EMR', 'ORLY', 'MCK', 'GD', 'EOG', 'AJG', 'COF', 'TDG', 'ECL', 'MMM', 'NOC', 'APD', 'FDX', 'SPG', 'RCL', 'WMB', 'CARR', 'RSG', 'ADSK', 'BDX', 'CVS', 'CSX', 'DLR', 'HLT', 'TGT', 'FCX', 'PCAR', 'TFC', 'OKE', 'KMI', 'CPRT', 'ROP', 'AFL', 'SLB', 'GM', 'MET', 'BK', 'AZO', 'SRE', 'TRV', 'PSA', 'NSC', 'GWW', 'NXPI', 'JCI', 'CHTR', 'AMP', 'FICO', 'ALL', 'URI', 'MNST', 'PSX', 'ROST', 'PAYX', 'CMI', 'AEP', 'AXON', 'PWR', 'VST', 'MSCI', 'MPC']
    
    # Aplicar estilos y mostrar el DataFrame
    styled_df = analyze_multiple_companies(sp500_tickers)
    st.dataframe(styled_df, use_container_width=True, hide_index=True, height=2500)
