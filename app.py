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
# 2. CSS 样式 (V15.5 兼容版：保留侧边栏开关)
# ==========================================
st.markdown("""
    <style>
    .block-container { 
        padding-top: 1.5rem; 
        padding-bottom: 3rem; 
    }
    .header-wrapper {
        display: flex; flex-direction: row; align-items: center; justify-content: flex-start;
        flex-wrap: wrap; gap: 30px; width: 100%; margin-bottom: 10px;
        border-bottom: 1px solid #f0f0f0; padding-bottom: 15px; padding-right: 60px;
    }
    .main-title {
        font-size: 2.4rem; font-weight: 800; color: #2c3e50; margin: 0; line-height: 1.1;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; white-space: nowrap;
    }
    .sub-info { font-size: 0.95rem; color: #7f8c8d; margin-top: 5px; font-weight: 400; }
    .kpi-box {
        border: 1px solid #e1e4e8; border-radius: 8px; padding: 0 15px; min-width: 100px; height: 75px;
        display: flex; flex-direction: column; justify-content: center; align-items: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
    }
    .kpi-label { font-size: 0.85rem; margin-bottom: 3px; font-weight: 600; }
    .kpi-value { font-size: 1.35rem; font-weight: 700; line-height: 1.1; }
    div.stRadio > div { display: flex; gap: 15px; align-items: center; }
    .plotly-notifier { display: none; }
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
        df['Ticker'] = df['Ticker'].astype(str).upper().strip()
        return df
    except Exception as e:
        st.error(f"读取错误: {e}")
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
# 4. 金融计算引擎
# ==========================================
def get_price_history(tickers, start_date):
    if not tickers: return pd.DataFrame()
    all_tickers = list(set(tickers) | {'SPY'}) 
    if 'CASH' in all_tickers: all_tickers.remove('CASH')
    try:
        data = yf.download(all_tickers, start=pd.to_datetime(start_date)-pd.Timedelta(days=5), progress=False)['Close']
        if isinstance(data, pd.Series):
            data = data.to_frame(); data.columns = [all_tickers[0]]
        if data.index.tz is not None: data.index = data.index.tz_localize(None)
        return data.ffill()
    except: return pd.DataFrame()

def calculate_full_history(df_trans, price_data, sys_start_date):
    if df_trans.empty: return pd.DataFrame(), {}, 0
    sys_start_ts = pd.to_datetime(sys_start_date)
    df_trans = df_trans.sort_values('Date')
    full_dates = pd.date_range(start=sys_start_ts, end=datetime.now(), freq='D')
    
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
            spy_val = price_data.loc[d_norm, 'SPY'] if 'SPY' in price_data.columns else 1
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
        net_invest = (t_tx[t_tx['Action'].str.contains('BUY')]['Shares'] * t_tx[t_tx['Action'].str.contains('BUY')]['Price']).sum() + \
                     (t_tx[t_tx['Action'].str.contains('SELL')]['Shares'] * t_tx[t_tx['Action'].str.contains('SELL')]['Price']).sum()
        pnl = (val_e - val_s) - net_invest
        perf_stats.append({'代码': t, '总盈亏': pnl, '当前持仓': qty_e, '当前市值': val_e, '类型': '多头' if qty_e > 0 else ('空头' if qty_e < 0 else '已平仓')})
    
    df_perf = pd.DataFrame(perf_stats)
    return df_perf.sort_values('总盈亏', ascending=False) if not df_perf.empty else df_perf, cash_end

# ==========================================
# 5. 初始化
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
# 6. 侧边栏 (比例下单功能)
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
        tx_action = st.selectbox("动作", ["BUY (做多/平空)", "SELL (卖出/做空)", "DEPOSIT"])
        
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
            final_shares = tx_shares_input if "SELL" not in tx_action else -tx_shares_input
        else:
            tx_pct = st.number_input("占 NAV %", min_value=0.0, value=5.0)
            calc_shares = (current_nav * (tx_pct / 100)) / tx_price if tx_price > 0 else 0
            final_shares = calc_shares if "SELL" not in tx_action else -calc_shares
            st.caption(f"预估股数: {abs(final_shares):.2f}")

        tx_reason = st.text_area("投资逻辑", height=60)

    if st.button("🔍 预览与提交", use_container_width=True, type="primary"):
        if tx_ticker or tx_action == 'DEPOSIT':
            trade = {
                'Date': tx_date.strftime('%Y-%m-%d'), 'Ticker': tx_ticker if tx_ticker else 'CASH',
                'Action': tx_action.split(" ")[0], 'Shares': final_shares, 'Price': tx_price, 'Reason': tx_reason
            }
            if save_transaction(trade): st.success("已提交！"); st.rerun()

# ==========================================
# 7. 主界面渲染 (恢复所有原功能)
# ==========================================
if df_nav_full.empty:
    st.info("👋 欢迎！请先在侧边栏录入初始资金。")
    st.stop()

# --- Header ---
latest = df_nav_full.iloc[-1]
st.markdown(f'<div class="header-wrapper"><h1 class="main-title">松熙 TMT 模拟仓</h1><div class="sub-info">📅 {latest.name.strftime("%Y-%m-%d")} | 💵 净值: ${latest["Total Assets"]:,.0f}</div></div>', unsafe_allow_html=True)

# 观察周期
time_range = st.radio("⏱️ 观察周期", ["近 1 月", "近 3 月", "近 1 年", "成立至今"], horizontal=True)
today = pd.Timestamp(date.today())
start_filter = pd.Timestamp(st.session_state['sys_start_date'])
if time_range == "近 1 月": start_filter = today - timedelta(days=30)
elif time_range == "近 3 月": start_filter = today - timedelta(days=90)
elif time_range == "近 1 年": start_filter = today - timedelta(days=365)

df_nav_filtered = df_nav_full[df_nav_full.index >= start_filter]
df_perf_period, cash_end = calculate_period_attribution(df_trans, price_data, daily_snapshots, start_filter, today)
mask_trans = (df_trans['Date'] >= start_filter) & (df_trans['Date'] <= today)
df_trans_filtered = df_trans.loc[mask_trans]

tab1, tab2, tab3 = st.tabs(["📊 走势与持仓", "🏆 业绩归因", "📝 交易流水"])

with tab1:
    col_chart, col_pos = st.columns([2, 1])
    with col_chart:
        st.subheader("净值走势 (归一化)")
        base = df_nav_filtered['Total Assets'].iloc[0] if not df_nav_filtered.empty else 1
        plot_df = df_nav_filtered.copy()
        plot_df['松熙组合'] = plot_df['Total Assets'] / base * 100
        if 'SPY' in plot_df:
            spy_base = plot_df['SPY'].iloc[0] if plot_df['SPY'].iloc[0] != 0 else 1
            plot_df['Ref'] = plot_df['SPY'] / spy_base * 100

        fig_nav = go.Figure()
        fig_nav.add_trace(go.Scatter(x=plot_df.index, y=plot_df['松熙组合'], name='松熙组合', line=dict(color='#2c3e50', width=2.5)))
        if 'Ref' in plot_df:
            fig_nav.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Ref'], name='Index Ref', line=dict(color='#BDC3C7', dash='dot')))
        
        # V15.0 高低点标注
        max_idx = plot_df['松熙组合'].idxmax(); max_val = plot_df.loc[max_idx, '松熙组合']
        fig_nav.add_annotation(x=max_idx, y=max_val, text=f"High: {max_val:.1f}", showarrow=True, arrowcolor="#9B59B6", ax=0, ay=-40, font=dict(color="#9B59B6", size=12))
        min_idx = plot_df['松熙组合'].idxmin(); min_val = plot_df.loc[min_idx, '松熙组合']
        fig_nav.add_annotation(x=min_idx, y=min_val, text=f"Low: {min_val:.1f}", showarrow=True, arrowcolor="#E67E22", ax=0, ay=40, font=dict(color="#E67E22", size=12))

        # V14.2 聚合交易点
        visible_trades = df_trans_filtered[df_trans_filtered['Ticker'] != 'CASH'].copy()
        if not visible_trades.empty:
            visible_trades['Date_Norm'] = visible_trades['Date'].dt.normalize()
            for d, group in visible_trades.groupby('Date_Norm'):
                if d in plot_df.index:
                    y_val = plot_df.loc[d, '松熙组合']
                    has_buy = any('BUY' in a for a in group['Action'])
                    has_sell = any('SELL' in a for a in group['Action'])
                    color = '#FFD700' if has_buy and has_sell else ('#E74C3C' if has_buy else '#2ECC71')
                    card_text = "<br>".join([f"<span style='color:{'#D32F2F' if 'BUY' in r['Action'] else '#2E7D32'}'><b>{r['Action'][:3]} {r['Ticker']}</b></span>" for _, r in group.iterrows()][:3])
                    fig_nav.add_trace(go.Scatter(x=[d], y=[y_val], mode='markers', marker=dict(symbol='square', size=10, color=color), showlegend=False))
                    fig_nav.add_annotation(x=d, y=y_val, text=card_text, bgcolor="white", bordercolor=color, font=dict(size=11), showarrow=True, ay=-35)

        fig_nav.update_layout(height=480, margin=dict(l=20, r=20, t=30, b=20), hovermode="x unified")
        st.plotly_chart(fig_nav, use_container_width=True)

    with col_pos:
        st.subheader("持仓占比 (%)")
        if not df_perf_period.empty:
            df_bar = df_perf_period[df_perf_period['当前市值'] != 0].copy()
            fig_bar = px.bar(df_bar, x='代码', y='当前市值', color='类型', color_discrete_map={'多头':'#E74C3C','空头':'#2ECC71'})
            fig_bar.update_layout(height=480, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    st.subheader("盈亏贡献")
    if not df_perf_period.empty:
        fig_pnl = px.bar(df_perf_period, x='总盈亏', y='代码', orientation='h', color='总盈亏', color_continuous_scale='RdYlGn')
        st.plotly_chart(fig_pnl, use_container_width=True)

with tab3:
    st.subheader("历史调仓")
    c1, c2 = st.columns([1, 4])
    show_all = c1.checkbox("显示全部历史", value=False)
    sort_mode = c2.radio("排序", ["日期(新)", "代码(AZ)"], horizontal=True)
    target_df = df_trans if show_all else df_trans_filtered
    if not target_df.empty:
        df_disp = target_df.sort_values('Date', ascending=False) if "日期" in sort_mode else target_df.sort_values('Ticker')
        st.dataframe(df_disp, use_container_width=True)
