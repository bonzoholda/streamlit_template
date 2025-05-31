import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import time
import os
import json
from datetime import datetime, timedelta
import asyncio
import threading
from dotenv import load_dotenv
import pytz
from PIL import Image  # Make sure to import this if not already

# Load environment variables
load_dotenv()

# Import custom modules
from components.exchange_connector import ExchangeConnector
from components.trading_engine import TradingEngine
from components.strategy import Strategy
from components.grid_strategy import GridStrategy
from components.trend_following_strategy import TrendFollowingStrategy
from components.risk_manager import RiskManager
from components.order_manager import OrderManager
from components.profit_compounder import ProfitCompounder
from components.market_data_manager import MarketDataManager
from components.config import Config
from components.database_service import DatabaseService
from components.performance_calculator import PerformanceCalculator

# Set page config
st.set_page_config(
    page_title="ShitBOX v2 - OKX Trading Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Display the logo at the top (adjust width if needed)
logo = Image.open("static/box-logo.png")
st.image(logo, width=150)

# CSS styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-title {
        font-size: 1.5rem;
        font-weight: 500;
        color: #1E88E5;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        height: 100%; /* Ensure full height in Streamlit columns */
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #6c757d;
    }
    .positive-value {
        color: #28a745;
    }
    .negative-value {
        color: #dc3545;
    }
    .status-running {
        color: #28a745;
        font-weight: 700;
    }
    .status-stopped {
        color: #dc3545;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'trading_engine' not in st.session_state:
    st.session_state.trading_engine = None
if 'trading_status' not in st.session_state:
    st.session_state.trading_status = "Stopped"
if 'portfolio_value' not in st.session_state:
    st.session_state.portfolio_value = 0
if 'trades' not in st.session_state:
    st.session_state.trades = []
if 'performance_metrics' not in st.session_state:
    st.session_state.performance_metrics = {
        'win_rate': 0,
        'profit': 0,
        'drawdown': 0,
        'roi': 0,
        'daily_growth': []
    }
if 'portfolio_history' not in st.session_state:
    st.session_state.portfolio_history = pd.DataFrame({
        'timestamp': [datetime.now() - timedelta(days=i) for i in range(7, 0, -1)],
        'value': [1000 * (1 + 0.01 * i) for i in range(7)]
    })
if 'trading_thread' not in st.session_state:
    st.session_state.trading_thread = None
if 'trading_stop_event' not in st.session_state:
    st.session_state.trading_stop_event = threading.Event()



# Bot logs
log_buffer = []

def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_buffer.append(f"[{timestamp}] {msg}")
    # Optional: Keep only the last 100 lines
    if len(log_buffer) > 100:
        log_buffer.pop(0)

# Helper functions
def load_config():
    """Load configuration from file or create default"""
    config_path = "config.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    else:
        # Default configuration
        default_config = {
            "exchange": {
                "api_key": "",
                "api_secret": "",
                "passphrase": "",
                "test_mode": True
            },
            "trading": {
                "symbol": "BTC-USDT",
                "strategy": "TrendFollowing",
                "interval": "15m",
                "max_open_positions": 3
            },
            "risk": {
                "max_position_size_percent": 10,
                "stop_loss_percent": 2,
                "take_profit_percent": 4, 
                "max_daily_risk_percent": 5,
                "max_drawdown_percent": 15
            },
            "compound": {
                "enabled": True,
                "target_compound_rate": 100,  # 100% (reinvest all profits)
                "compound_frequency": "Daily"
            }
        }
        with open(config_path, "w") as f:
            json.dump(default_config, f, indent=4)
        return default_config

def save_config(config):
    """Save configuration to file"""
    with open("config.json", "w") as f:
        json.dump(config, f, indent=4)

def initialize_trading_engine(config):
    """Initialize the trading engine with the given configuration"""
    try:
        exchange_connector = ExchangeConnector(
            api_key=config["exchange"]["api_key"],
            api_secret=config["exchange"]["api_secret"],
            passphrase=config["exchange"]["passphrase"],
            test_mode=config["exchange"]["test_mode"]
        )
        
        # Initialize all components
        market_data_manager = MarketDataManager(exchange_connector)
        
        # Select strategy based on config
        if config["trading"]["strategy"] == "GridTrading":
            strategy = GridStrategy(
                upper_limit=config.get("grid", {}).get("upper_limit", 0),
                lower_limit=config.get("grid", {}).get("lower_limit", 0),
                grid_levels=config.get("grid", {}).get("grid_levels", 5)
            )
        else:
            # Default to trend following
            strategy = TrendFollowingStrategy(
                short_ema=config.get("trend", {}).get("short_ema", 9),
                long_ema=config.get("trend", {}).get("long_ema", 21),
                stop_loss_percentage=config["risk"]["stop_loss_percent"],
                take_profit_percentage=config["risk"]["take_profit_percent"]
            )
        
        risk_manager = RiskManager(
            max_drawdown=config["risk"]["max_drawdown_percent"],
            max_position_size=config["risk"]["max_position_size_percent"],
            daily_risk_limit=config["risk"]["max_daily_risk_percent"]
        )
        
        order_manager = OrderManager(exchange_connector)
        
        profit_compounder = ProfitCompounder(
            target_compound_rate=config["compound"]["target_compound_rate"],
            compound_frequency=config["compound"]["compound_frequency"],
            order_manager=order_manager
        )
        
        database_service = DatabaseService()
        
        config_obj = Config()
        config_obj.update_config(config)
        
        # Create and initialize trading engine
        trading_engine = TradingEngine(
            strategies=[strategy],
            risk_manager=risk_manager,
            order_manager=order_manager,
            profit_compounder=profit_compounder,
            market_data_manager=market_data_manager,
            database_service=database_service,
            config=config_obj
        )
        
        return trading_engine
    
    except Exception as e:
        st.error(f"Error initializing trading engine: {str(e)}")
        return None

def start_trading():
    """Start the trading bot"""
    if st.session_state.trading_engine is None:
        config = load_config()
        st.session_state.trading_engine = initialize_trading_engine(config)

    if st.session_state.trading_engine is not None:
        try:
            def trading_loop(engine, stop_event):
                engine.start()
                while not stop_event.is_set():
                    time.sleep(5)
                engine.stop()

            # Set state and clear stop event
            st.session_state.trading_status = "Running"
            st.session_state.trading_stop_event.clear()

            # Start thread with arguments
            st.session_state.trading_thread = threading.Thread(
                target=trading_loop,
                args=(st.session_state.trading_engine, st.session_state.trading_stop_event),
                daemon=True
            )
            st.session_state.trading_thread.start()

            # Optional: preload data
            if st.session_state.trading_engine.get_status() != "Error":
                create_mock_data()

        except Exception as e:
            st.error(f"Error starting trading bot: {str(e)}")
            st.session_state.trading_status = "Stopped"


def stop_trading():
    """Stop the trading bot"""
    if 'trading_stop_event' in st.session_state:
        st.session_state.trading_stop_event.set()  # Signal the thread to stop

    st.session_state.trading_status = "Stopped"

    # Wait for the thread to finish
    if st.session_state.trading_thread is not None:
        st.session_state.trading_thread.join(timeout=5)
        st.session_state.trading_thread = None  # Reset the thread reference

    st.session_state.trading_engine = None


def create_mock_data():
    """Create mock data for demonstration purposes"""
    # Mock portfolio history
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    portfolio_values = [1000]
    for i in range(1, 30):
        # Generate daily growth between 0.5% and 1.5%
        daily_growth = np.random.uniform(0.005, 0.015)
        portfolio_values.append(portfolio_values[-1] * (1 + daily_growth))
    
    st.session_state.portfolio_history = pd.DataFrame({
        'timestamp': dates,
        'value': portfolio_values
    })
    
    # Mock trades
    trades = []
    symbols = ['BTC-USDT', 'ETH-USDT', 'SOL-USDT']
    
    for i in range(20):
        # 70% win rate
        is_win = np.random.random() < 0.7
        
        entry_time = datetime.now() - timedelta(days=np.random.randint(1, 30), 
                                            hours=np.random.randint(0, 24),
                                            minutes=np.random.randint(0, 60))
        exit_time = entry_time + timedelta(minutes=np.random.randint(15, 240))
        
        symbol = np.random.choice(symbols)
        trade_type = np.random.choice(['Long', 'Short'])
        
        # Generate price data
        entry_price = 100 + np.random.random() * 10
        
        if is_win:
            pnl_pct = np.random.uniform(0.5, 3.0)
        else:
            pnl_pct = np.random.uniform(-2.0, -0.2)
            
        exit_price = entry_price * (1 + pnl_pct/100) if trade_type == 'Long' else entry_price * (1 - pnl_pct/100)
        position_size = np.random.uniform(0.1, 1.0)
        pnl = position_size * entry_price * pnl_pct / 100
        
        trades.append({
            'id': i+1,
            'symbol': symbol,
            'type': trade_type,
            'entry_time': entry_time,
            'entry_price': entry_price,
            'exit_time': exit_time,
            'exit_price': exit_price,
            'position_size': position_size,
            'pnl': pnl,
            'pnl_percent': pnl_pct,
            'status': 'Closed'
        })
    
    # Sort by exit time (most recent first)
    trades.sort(key=lambda x: x['exit_time'], reverse=True)
    st.session_state.trades = trades
    
    # Calculate performance metrics
    wins = sum(1 for trade in trades if trade['pnl'] > 0)
    win_rate = wins / len(trades) * 100 if trades else 0
    
    total_profit = sum(trade['pnl'] for trade in trades)
    
    # Calculate maximum drawdown
    cumulative_returns = []
    running_max = 0
    current_drawdown = 0
    max_drawdown = 0
    
    for trade in sorted(trades, key=lambda x: x['exit_time']):
        cumulative_returns.append(trade['pnl'])
        current_sum = sum(cumulative_returns)
        running_max = max(running_max, current_sum)
        current_drawdown = running_max - current_sum
        max_drawdown = max(max_drawdown, current_drawdown)
    
    max_drawdown_pct = (max_drawdown / running_max) * 100 if running_max > 0 else 0
    
    # Calculate ROI
    initial_portfolio = 1000
    roi = (total_profit / initial_portfolio) * 100
    
    # Daily growth
    daily_growth = [(portfolio_values[i] / portfolio_values[i-1] - 1) * 100 for i in range(1, len(portfolio_values))]
    
    st.session_state.performance_metrics = {
        'win_rate': win_rate,
        'profit': total_profit,
        'drawdown': max_drawdown_pct,
        'roi': roi,
        'daily_growth': daily_growth
    }

def format_number(value, precision=2, with_plus=False):
    """Format numbers with commas as thousands separator"""
    if isinstance(value, (int, float)):
        if with_plus and value > 0:
            return f"+{value:,.{precision}f}"
        return f"{value:,.{precision}f}"
    return value

# Main layout
def main():
    st.markdown('<h1 class="main-title">OKX Trading Bot</h1>', unsafe_allow_html=True)

    # Top status row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_class = "status-running" if st.session_state.trading_status == "Running" else "status-stopped"
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">Bot Status</p>
            <p class="metric-value {status_class}">{st.session_state.trading_status}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        win_rate = format_number(st.session_state.performance_metrics['win_rate'])
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">Win Rate</p>
            <p class="metric-value">{win_rate}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        profit_class = "positive-value" if st.session_state.performance_metrics['profit'] >= 0 else "negative-value"
        profit = format_number(st.session_state.performance_metrics['profit'], with_plus=True)
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">Total Profit</p>
            <p class="metric-value {profit_class}">{profit} USDT</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        roi_class = "positive-value" if st.session_state.performance_metrics['roi'] >= 0 else "negative-value"
        roi = format_number(st.session_state.performance_metrics['roi'], with_plus=True)
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">ROI</p>
            <p class="metric-value {roi_class}">{roi}%</p>
        </div>
        """, unsafe_allow_html=True)


    # Control buttons
    col1, col2 = st.columns([1, 5])
    
    with col1:
        if st.session_state.trading_status == "Stopped":
            if st.button("▶️ Start Trading", use_container_width=True):
                start_trading()
        else:
            if st.button("⏹ Stop Trading", use_container_width=True):
                stop_trading()

    # Portfolio Value Chart
    st.markdown('<h2 class="sub-title">Portfolio Performance</h2>', unsafe_allow_html=True)
    
    fig = px.line(
        st.session_state.portfolio_history, 
        x='timestamp', 
        y='value',
        labels={'timestamp': 'Date', 'value': 'Portfolio Value (USDT)'},
        template='plotly_white'
    )
    
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Daily Returns Chart and Performance Metrics
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<h2 class="sub-title">Daily Returns</h2>', unsafe_allow_html=True)
        
        # Get daily returns from performance metrics
        daily_returns = st.session_state.performance_metrics['daily_growth']
        
        if daily_returns:
            daily_df = pd.DataFrame({
                'date': pd.date_range(end=datetime.now(), periods=len(daily_returns), freq='D'),
                'return': daily_returns
            })
            
            # Create bar chart
            fig = px.bar(
                daily_df,
                x='date',
                y='return',
                labels={'date': 'Date', 'return': 'Daily Return (%)'},
                color='return',
                color_continuous_scale=['#dc3545', '#dc3545', '#28a745', '#28a745'],
                color_continuous_midpoint=0,
                template='plotly_white'
            )
            
            fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=10, b=0),
                coloraxis_showscale=False,
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No daily return data available yet.")
    
    with col2:
        st.markdown('<h2 class="sub-title">Performance</h2>', unsafe_allow_html=True)
        
        metrics = [
            {"label": "Win Rate", "value": f"{format_number(st.session_state.performance_metrics['win_rate'])}%"},
            {"label": "Max Drawdown", "value": f"{format_number(st.session_state.performance_metrics['drawdown'])}%"},
            {"label": "Avg. Daily Return", "value": f"{format_number(np.mean(st.session_state.performance_metrics['daily_growth']) if st.session_state.performance_metrics['daily_growth'] else 0)}%"},
        ]
        
        for metric in metrics:
            st.markdown(f"""
            <div style="margin-bottom:10px;">
                <span style="font-size:14px;">{metric['label']}</span><br>
                <span style="font-size:20px;font-weight:bold;">{metric['value']}</span>
            </div>
            """, unsafe_allow_html=True)

    # Trade History
    st.markdown('<h2 class="sub-title">Trade History</h2>', unsafe_allow_html=True)
    
    if st.session_state.trades:
        # Convert to DataFrame for easier manipulation
        trades_df = pd.DataFrame(st.session_state.trades)
        
        # Format columns
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time']).dt.strftime('%Y-%m-%d %H:%M')
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time']).dt.strftime('%Y-%m-%d %H:%M')
        trades_df['entry_price'] = trades_df['entry_price'].map(lambda x: format_number(x, 4))
        trades_df['exit_price'] = trades_df['exit_price'].map(lambda x: format_number(x, 4))
        trades_df['position_size'] = trades_df['position_size'].map(lambda x: format_number(x, 4))
        
        # Format PnL with colors
        def format_pnl(row):
            pnl = row['pnl']
            pnl_pct = row['pnl_percent']
            
            if pnl > 0:
                return f'<span style="color:#28a745">+{format_number(pnl)} USDT (+{format_number(pnl_pct)}%)</span>'
            else:
                return f'<span style="color:#dc3545">{format_number(pnl)} USDT ({format_number(pnl_pct)}%)</span>'
        
        trades_df['pnl_formatted'] = trades_df.apply(format_pnl, axis=1)
        
        # Select columns to display
        display_df = trades_df[['id', 'symbol', 'type', 'entry_time', 'entry_price', 'exit_time', 'exit_price', 'position_size', 'pnl_formatted']]
        display_df.columns = ['ID', 'Symbol', 'Type', 'Entry Time', 'Entry Price', 'Exit Time', 'Exit Price', 'Size', 'PnL']
        
        # Display table with scrollbar
        st.markdown(
            """
            <style>
            .trade-table {
                height: 300px;
                overflow-y: auto;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown('<div class="trade-table">', unsafe_allow_html=True)
        st.write(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No trade history available yet.")

    # Tabs for configuration
    st.markdown('<h2 class="sub-title">Configuration</h2>', unsafe_allow_html=True)
    
    config = load_config()
    
    tabs = st.tabs(["API Settings", "Trading Parameters", "Risk Management", "Profit Compounding"])
    
    # API Settings Tab
    with tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            config["exchange"]["api_key"] = st.text_input("API Key", value=config["exchange"]["api_key"], type="password")
        
        with col2:
            config["exchange"]["api_secret"] = st.text_input("API Secret", value=config["exchange"]["api_secret"], type="password")
        
        col1, col2 = st.columns(2)
        
        with col1:
            config["exchange"]["passphrase"] = st.text_input("API Passphrase", value=config["exchange"]["passphrase"], type="password")
        
        with col2:
            config["exchange"]["test_mode"] = st.checkbox("Test Mode (Paper Trading)", value=config["exchange"]["test_mode"])
    
    # Trading Parameters Tab
    with tabs[1]:
        col1, col2 = st.columns(2)
        
        with col1:
            config["trading"]["symbol"] = st.text_input("Trading Symbol", value=config["trading"]["symbol"])
        
        with col2:
            config["trading"]["strategy"] = st.selectbox("Trading Strategy", options=["TrendFollowing", "GridTrading"], index=0 if config["trading"]["strategy"] == "TrendFollowing" else 1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            config["trading"]["interval"] = st.selectbox("Timeframe", options=["1m", "5m", "15m", "30m", "1h", "4h", "1d"], index=["1m", "5m", "15m", "30m", "1h", "4h", "1d"].index(config["trading"]["interval"]))
        
        with col2:
            config["trading"]["max_open_positions"] = st.number_input("Max Open Positions", min_value=1, max_value=10, value=config["trading"]["max_open_positions"])
        
        # Strategy-specific parameters
        if config["trading"]["strategy"] == "TrendFollowing":
            st.subheader("Trend Following Parameters")
            
            if "trend" not in config:
                config["trend"] = {"short_ema": 9, "long_ema": 21}
            
            col1, col2 = st.columns(2)
            
            with col1:
                config["trend"]["short_ema"] = st.number_input("Short EMA Period", min_value=3, max_value=50, value=config["trend"].get("short_ema", 9))
            
            with col2:
                config["trend"]["long_ema"] = st.number_input("Long EMA Period", min_value=5, max_value=200, value=config["trend"].get("long_ema", 21))
        
        elif config["trading"]["strategy"] == "GridTrading":
            st.subheader("Grid Trading Parameters")
            
            if "grid" not in config:
                config["grid"] = {"upper_limit": 0, "lower_limit": 0, "grid_levels": 5}
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                config["grid"]["upper_limit"] = st.number_input("Upper Price Limit", min_value=0.0, value=float(config["grid"].get("upper_limit", 0)))
            
            with col2:
                config["grid"]["lower_limit"] = st.number_input("Lower Price Limit", min_value=0.0, value=float(config["grid"].get("lower_limit", 0)))
            
            with col3:
                config["grid"]["grid_levels"] = st.number_input("Grid Levels", min_value=3, max_value=100, value=config["grid"].get("grid_levels", 5))
    
    # Risk Management Tab
    with tabs[2]:
        col1, col2 = st.columns(2)
    
        with col1:
            value = config["risk"].get("max_position_size_percent", 10)
            if isinstance(value, list):
                value = value[0] if value else 10
            config["risk"]["max_position_size_percent"] = st.slider(
                "Max Position Size (%)", min_value=1, max_value=100, value=int(value)
            )
    
        with col2:
            value = config["risk"].get("max_daily_risk_percent", 5)
            if isinstance(value, list):
                value = value[0] if value else 5
            config["risk"]["max_daily_risk_percent"] = st.slider(
                "Max Daily Risk (%)", min_value=1, max_value=50, value=int(value)
            )
    
        col1, col2 = st.columns(2)
    
        with col1:
            value = config["risk"].get("stop_loss_percent", 1.0)
            if isinstance(value, list):
                value = value[0] if value else 1.0
            config["risk"]["stop_loss_percent"] = st.slider(
                "Stop Loss (%)", min_value=0.1, max_value=10.0, value=float(value), step=0.1
            )
    
        with col2:
            value = config["risk"].get("take_profit_percent", 2.0)
            if isinstance(value, list):
                value = value[0] if value else 2.0
            config["risk"]["take_profit_percent"] = st.slider(
                "Take Profit (%)", min_value=0.1, max_value=20.0, value=float(value), step=0.1
            )
    
        value = config["risk"].get("max_drawdown_percent", 20)
        if isinstance(value, list):
            value = value[0] if value else 20
        config["risk"]["max_drawdown_percent"] = st.slider(
            "Max Drawdown (%)", min_value=5, max_value=50, value=int(value)
        )

    
    # Profit Compounding Tab
    with tabs[3]:
        config["compound"]["enabled"] = st.checkbox("Enable Profit Compounding", value=config["compound"]["enabled"])
        
        if config["compound"]["enabled"]:
            col1, col2 = st.columns(2)
            
            with col1:
                config["compound"]["target_compound_rate"] = st.slider("Compounding Rate (%)", min_value=10, max_value=100, value=config["compound"]["target_compound_rate"])
            
            with col2:
                config["compound"]["compound_frequency"] = st.selectbox("Compounding Frequency", options=["Daily", "Weekly", "Monthly"], index=["Daily", "Weekly", "Monthly"].index(config["compound"]["compound_frequency"]))
    
    # Save configuration button
    if st.button("Save Configuration"):
        save_config(config)
        st.success("Configuration saved successfully!")
    
        if st.session_state.trading_status == "Running":
            st.warning("Please restart the trading bot to apply the new configuration.")
    
        # Always initialize if not present
        if "trading_engine" not in st.session_state or st.session_state.trading_engine is None:
            st.session_state.trading_engine = initialize_trading_engine(config)
        else:
            # Reinitialize if already present
            st.session_state.trading_engine = initialize_trading_engine(config)


    # After your tabs or main layout
    with st.expander("Bot Logs"):
        st.text("\n".join(log_buffer))



if __name__ == "__main__":
    main()
