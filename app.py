import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
from streamlit_gsheets import GSheetsConnection

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(
    page_title="松熙TMT模拟仓 | Songxi Capital",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. CSS 样式
# ==========================================
st.markdown("""
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 3rem; }
    .header-wrapper {
        display: flex; flex-direction: row; align-items: center; justify-content: flex-start;
        flex-wrap: wrap; gap: 30px; width: 100%; margin-bottom: 10px;
        border-bottom: 1px solid #f0f0f0; padding-bottom: 15px; padding-right: 60px;
    }
    .main-title { font-size: 2.4rem; font-weight: 800; color: #2c3e50; margin: 0; line-height: 1.1; white-space: nowrap; }
    .sub-info { font-size: 0.95rem; color: #7f8c8d; margin-top: 5px; }
    .kpi-box {
        border: 1px solid #e1e4e8; border-radius: 8px; padding: 0 15px; min-width: 100px; height: 75px;
        display: flex; flex-direction: column; justify-content: center; align-items: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
    }
    .kpi-label { font-size: 0.85rem; font-weight: 600; color: #6c757d; }
    .kpi-value { font-size: 1.35rem; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 3. 数据层函数
# ==========================================
conn = st.connection("gsheets", type=GSheetsConnection)

def load_data():
    try:
        df = conn.read(ttl=600) 
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=['Date', 'Ticker', 'Action', 'Shares', 'Price', 'Reason'])
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df['Shares'] = pd.to_numeric(df['Shares'], errors='coerce').fillna(0)
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0)
        df['Ticker'] = df['Ticker'].astype(str).str.upper().str.strip()
        return df
    except Exception as e:
        st.error(f"数据读取错误: {e}")
        return pd.DataFrame(columns=['Date', 'Ticker', 'Action', 'Shares', 'Price', 'Reason'])

def save_transaction(new_row_dict):
    try:
        current_df = conn.read(ttl=0)
        new_row_df = pd.DataFrame([new_row_dict])
        updated_df = pd.concat([current_df, new_row_df], ignore_index=True)
        conn.update(data=updated_df)
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"写入失败: {e}")
        return False

# ==========================================
# 4. 计算引擎
# ==========================================
def get_price_history(tickers, start_date):
    if not tickers: return pd.DataFrame()
    all_tickers = list(set(tickers) | {'SPY'})
    if 'CASH' in all_tickers: all_tickers.remove('CASH')
    try:
        start_ts = pd.to_datetime(start_date)
        buffer_date = start_ts - pd.Timedelta(days=400)
        data = yf.download(all_tickers, start=buffer_date, progress=False)['Close']
        if isinstance(data, pd.Series):
            data = data.to_frame()
            data.columns = [all_tickers[0]]
        if data.index.tz is not None: data.index = data.index.tz_localize(None)
        return data.ffill()
    except: return pd.DataFrame()

def calculate_full_history(df_trans, price_data, sys_start_date):
    if df_trans.empty: return pd.DataFrame(), {}, 0
    sys_start_ts = pd.to_datetime(sys_start_date)
    df_trans = df_trans.sort_values('Date')
    end_date = datetime.now()
    full_dates = pd.date_range(start=sys_start_ts, end=end_date, freq='D')
    
    curr_trans = df_trans.copy()
    curr_trans['Date_Norm'] = curr_trans['Date'].dt.normalize()
    trans_grouped = curr_trans.groupby('Date_Norm')
    
    cash, holdings, nav_history, daily_snapshots = 0, {}, [], {}
    
    for d in full_dates:
        d_norm = d.normalize()
        if d_norm in trans_grouped.groups:
            for _, row in trans_grouped.get_group(d_norm).iterrows():
                t, s, p, a = row['Ticker'], row['Shares'], row['Price'], row['Action']
                if t == 'CASH' or a == 'DEPOSIT': cash += s
                else: cash -= (s * p); holdings[t] = holdings.get(t, 0) + s
        
        daily_snapshots[d_norm] = (holdings.copy(), cash)
        mkt_val = 0
        if not price_data.empty and d_norm in price_data.index:
            for t, s in holdings.items():
                if abs(s) > 0.001 and t in price_data.columns:
                    mkt_val += s * price_data.loc[d_norm, t]
            spy_val = price_data.loc[d_norm, 'SPY'] if 'SPY' in price_data.columns else 0
            nav_history.append({'Date': d_norm, 'Total Assets': cash + mkt_val, 'Cash': cash, 'Market Value': mkt_val, 'SPY': spy_val})
            
    df_nav = pd.DataFrame(nav_history)
    if not df_nav.empty: df_nav = df_nav.set_index('Date')
    return df_nav, daily_snapshots, cash

def calculate_period_attribution(df_trans, price_data, daily_snapshots, start_date, end_date):
    start_ts, end_ts = pd.to_datetime(start_date), pd.to_datetime(end_date)
    valid_dates = sorted(daily_snapshots.keys())
    if not valid_dates: return pd.DataFrame(), 0
    actual_start = min(valid_dates, key=lambda x: abs(x - start_ts))
    actual_end = min(valid_dates, key=lambda x: abs(x - end_ts))
    
    holdings_start, _ = daily_snapshots[actual_start]
    holdings_end, cash_end = daily_snapshots[actual_end]
    if price_data.empty: return pd.DataFrame(), cash_end
    
    prices_start = price_data.loc[price_data.index <= actual_start].iloc[-1] if not price_data.loc[price_data.index <= actual_start].empty else pd.Series()
    prices_end = price_data.loc[price_data.index <= actual_end].iloc[-1] if not price_data.loc[price_data.index <= actual_end].empty else pd.Series()
    
    mask = (df_trans['Date'] > actual_start) & (df_trans['Date'] <= actual_end)
    period_trans = df_trans.loc[mask]
    all_tickers = (set(holdings_start.keys()) | set(holdings_end.keys()) | set(period_trans['Ticker'].unique())) - {'CASH'}
    
    perf_stats = []
    for t in all_tickers:
        p_s, p_e = prices_start.get(t, 0), prices_end.get(t, 0)
        qty_s, qty_e = holdings_start.get(t, 0), holdings_end.get(t, 0)
        val_s, val_e = qty_s * p_s, qty_e * p_e
        t_tx = period_trans[period_trans['Ticker'] == t]
        net_invest = (t_tx[t_tx['Action'] == 'BUY']['Shares'] * t_tx[t_tx['Action'] == 'BUY']['Price']).sum() + \
                     (t_tx[t_tx['Action'] == 'SELL']['Shares'] * t_tx[t_tx['Action'] == 'SELL']['Price']).sum()
        pnl = (val_e - val_s) - net_invest
        perf_stats.append({'代码': t, '总盈亏': pnl, '当前持仓': qty_e, '当前市值': val_e, '类型': '多头' if qty_e > 0 else ('空头' if qty_e < 0 else '已平仓')})
    
    df_perf = pd.DataFrame(perf_stats)
    return df_perf.sort_values('总盈亏', ascending=False) if not df_perf.empty else df_perf, cash_end

# ==========================================
# 5. 初始化与全局计算 (仅执行一次)
# ==========================================
df_trans = load_data()
if not df_trans.empty:
    sys_start = df_trans['Date'].min().date()
    st.session_state['sys_start_date'] = sys_start
    tickers = df_trans[df_trans['Ticker']!='CASH']['Ticker'].unique().tolist()
    price_data = get_price_history(tickers, sys_start)
    df_nav_full, daily_snapshots, current_cash = calculate_full_history(df_trans, price_data, sys_start)
else:
    st.session_state['sys_start_date'] = date.today()
    df_nav_full, daily_snapshots, current_cash = pd.DataFrame(), {}, 0

# ==========================================
# 6. 侧边栏
# ==========================================
with st.sidebar:
    st.title("🌲 松熙基金工作台")
    if st.button("🔄 刷新数据", use_container_width=True):
        st.cache_data.clear(); st.rerun()
    st.divider()

    if not df_nav_full.empty:
        current_nav = df_nav_full.iloc[-1]['Total Assets']
        latest_date = sorted(daily_snapshots.keys())[-1]
        current_holdings, _ = daily_snapshots[latest_date]
    else:
        current_nav, current_holdings = 0, {}

    st.header("📝 交易录入")
    input_mode = st.radio("计算方式", ["按股数", "按净资产比例 %"], horizontal=True)
    
    with st.container(border=True):
        col1, col2 = st.columns(2)
        tx_date = col1.date_input("日期", date.today())
        tx_ticker = col2.text_input("代码", "").upper().strip()
        tx_action = st.selectbox("动作", ["BUY", "SELL", "DEPOSIT"])
        
        current_price = 0.0
        if tx_ticker and tx_ticker != 'CASH':
            ticker_obj = yf.Ticker(tx_ticker)
            current_price = ticker_obj.fast_info.get('last_price', 0.0)
            if current_price == 0:
                hist = ticker_obj.history(period="1d")
                current_price = hist['Close'].iloc[-1] if not hist.empty else 0.0
        
        tx_price = st.number_input("成交价格", min_value=0.0, value=float(current_price))
        
        if input_mode == "按股数":
            tx_shares_input = st.number_input("交易数量", min_value=0.0, value=100.0)
            final_shares = tx_shares_input if tx_action != "SELL" else -tx_shares_input
        else:
            tx_pct = st.number_input("占 NAV %", min_value=0.0, value=5.0)
            final_shares = (current_nav * (tx_pct / 100)) / tx_price if tx_price > 0 else 0
            final_shares = final_shares if tx_action != "SELL" else -final_shares
            st.caption(f"预估股数: {abs(final_shares):.2f}")

        tx_reason = st.text_area("投资逻辑", height=60)

    if st.button("🔍 预览交易", use_container_width=True, type="primary"):
        if tx_ticker or tx_action == 'DEPOSIT':
            st.session_state['show_preview'] = True
            st.session_state['temp_trade'] = {
                'Date': tx_date.strftime('%Y-%m-%d'), 'Ticker': tx_ticker if tx_ticker else 'CASH',
                'Action': tx_action, 'Shares': final_shares, 'Price': tx_price, 'Reason': tx_reason
            }

    if st.session_state.get('show_preview'):
        t = st.session_state['temp_trade']
        with st.expander("📊 预览确认", expanded=True):
            st.write(f"标的: {t['Ticker']} | 动作: {t['Action']}")
            st.write(f"股数: {t['Shares']:,.2f} | 价格: ${t['Price']}")
            if st.button("✅ 确认提交", use_container_width=True):
                if save_transaction(t):
                    st.session_state['show_preview'] = False
                    st.rerun()

# ==========================================
# 7. 主界面渲染
# ==========================================
if df_nav_full.empty:
    st.info("👋 欢迎！请在侧边栏录入初始资金 (DEPOSIT)。")
    st.stop()

# KPI Header
latest = df_nav_full.iloc[-1]
st.markdown(f"### 松熙 TMT 模拟仓 | 净值: ${latest['Total Assets']:,.0f} | 现金: ${latest['Cash']:,.0f}")

# 筛选器
time_range = st.radio("⏱️ 观察周期", ["近 1 月", "近 3 月", "近 1 年", "成立至今"], horizontal=True)
today = date.today()
start_filter = st.session_state['sys_start_date']
if time_range == "近 1 月": start_filter = today - timedelta(days=30)
elif time_range == "近 3 月": start_filter = today - timedelta(days=90)
elif time_range == "近 1 年": start_filter = today - timedelta(days=365)

filter_start_ts = pd.Timestamp(start_filter)
df_nav_filtered = df_nav_full[df_nav_full.index >= filter_start_ts]

tab1, tab2, tab3 = st.tabs(["📊 走势与持仓", "🏆 业绩归因", "📝 调仓记录"])

with tab1:
    col_chart, col_pos = st.columns([2, 1])
    # 净值图
    fig_nav = px.line(df_nav_filtered, y='Total Assets', title="组合净值走势")
    col_chart.plotly_chart(fig_nav, use_container_width=True)
    # 持仓分布
    df_perf_period, _ = calculate_period_attribution(df_trans, price_data, daily_snapshots, start_filter, date.today())
    if not df_perf_period.empty:
        fig_pos = px.bar(df_perf_period[df_perf_period['当前持仓']!=0], x='代码', y='当前市值', color='类型', title="当前持仓分布")
        col_pos.plotly_chart(fig_pos, use_container_width=True)

with tab2:
    if not df_perf_period.empty:
        st.subheader("区间盈亏贡献")
        fig_pnl = px.bar(df_perf_period, x='总盈亏', y='代码', orientation='h', color='总盈亏', color_continuous_scale='RdYlGn')
        st.plotly_chart(fig_pnl, use_container_width=True)

with tab3:
    st.subheader("调仓历史")
    st.dataframe(df_trans.sort_values('Date', ascending=False), use_container_width=True)
