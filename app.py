import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta

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
# 2. CSS 样式深度定制 (V12.0 紧凑布局版)
# ==========================================
st.markdown("""
    <style>
    /* 1. 全局容器: 顶部更紧凑 */
    .block-container { 
        padding-top: 1rem; 
        padding-bottom: 3rem;
    }
    
    /* 2. Header 容器: 核心修改 -> 左对齐 + 固定间距 + 垂直居中 */
    .header-wrapper {
        display: flex;
        flex-direction: row;
        align-items: center; /* 关键：垂直居中对齐 */
        justify-content: flex-start; /* 关键：左对齐，不两端撑开 */
        flex-wrap: wrap; 
        gap: 40px; /* 标题与卡片之间的固定间距 */
        width: 100%;
        margin-bottom: 10px;
        border-bottom: 1px solid #f0f0f0; 
        padding-bottom: 15px;
    }
    
    /* 3. 左侧标题 */
    .header-left { 
        flex-shrink: 0; 
        /* 移除 min-width 限制，让它自然收缩 */
    }
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #2c3e50;
        margin: 0;
        line-height: 1.0; /* 更紧凑的行高 */
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .sub-info { 
        font-size: 0.95rem; 
        color: #7f8c8d; 
        margin-top: 5px; 
        font-weight: 400; 
    }
    
    /* 4. 右侧指标区 */
    .header-right { 
        display: flex; 
        gap: 12px; 
        flex-wrap: wrap; 
        align-items: center; 
    }
    
    /* 5. 指标卡片 (KPI Box) */
    .kpi-box {
        border: 1px solid #e1e4e8;
        border-radius: 8px;
        padding: 0 15px;
        min-width: 100px;
        height: 75px; /* 高度微调，与标题视觉平衡 */
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
        transition: all 0.3s ease;
        position: relative; 
        overflow: hidden; 
    }
    
    .kpi-label { font-size: 0.85rem; margin-bottom: 3px; font-weight: 600; z-index: 2; }
    .kpi-value { font-size: 1.35rem; font-weight: 700; line-height: 1.1; white-space: nowrap; z-index: 2; }
    
    /* 6. Radio Button 优化 */
    div.stRadio > div { display: flex; gap: 0px; align-items: center; }
    div.stRadio > div label { margin-right: 15px; cursor: pointer; }
    
    .plotly-notifier { display: none; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 3. 模拟数据初始化
# ==========================================

DEFAULT_DEMO_DATA = [
    {'Date': '2023-01-01', 'Ticker': 'CASH', 'Action': 'DEPOSIT', 'Shares': 1000000.0, 'Price': 1.0, 'Reason': '初始募集'},
    {'Date': '2023-03-15', 'Ticker': 'NVDA', 'Action': 'BUY', 'Shares': 400.0, 'Price': 240.0, 'Reason': 'AI Infra 爆发，重仓算力龙头'},
    {'Date': '2023-04-10', 'Ticker': 'MSFT', 'Action': 'BUY', 'Shares': 600.0, 'Price': 280.0, 'Reason': 'Copilot 商业化落地预期'},
    {'Date': '2023-06-20', 'Ticker': 'SNOW', 'Action': 'SELL', 'Shares': -300.0, 'Price': 180.0, 'Reason': '消费疲软，SaaS 估值回调，建立空单'},
    {'Date': '2023-09-01', 'Ticker': 'AAPL', 'Action': 'BUY', 'Shares': 500.0, 'Price': 175.0, 'Reason': '现金流配置'},
]

if 'transactions' not in st.session_state:
    st.session_state['transactions'] = DEFAULT_DEMO_DATA
else:
    for t in st.session_state['transactions']:
        if 'Reason' not in t: t['Reason'] = '无记录'

if 'sys_start_date' not in st.session_state:
    st.session_state['sys_start_date'] = date(2023, 1, 1)

if 'sys_initial_capital' not in st.session_state:
    st.session_state['sys_initial_capital'] = 1000000.0

# ==========================================
# 4. 核心计算引擎
# ==========================================

def load_data():
    df = pd.DataFrame(st.session_state['transactions'])
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def get_price_history(tickers, start_date):
    if not tickers: return pd.DataFrame()
    all_tickers = list(set(tickers) | {'SPY'}) 
    if 'CASH' in all_tickers: all_tickers.remove('CASH')
    if not all_tickers: return pd.DataFrame()

    with st.spinner('🔄 同步 TMT 市场数据...'):
        try:
            buffer_date = start_date - pd.Timedelta(days=400) 
            data = yf.download(all_tickers, start=buffer_date, progress=False)['Close']
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            data = data.fillna(method='ffill')
            return data
        except Exception:
            return pd.DataFrame()

def calculate_full_history(df_trans, price_data):
    df_trans = df_trans.sort_values('Date')
    sys_start_date = pd.to_datetime(st.session_state['sys_start_date'])
    end_date = datetime.now()
    
    full_dates = pd.date_range(start=sys_start_date, end=end_date, freq='D')
    
    past_trans = df_trans[df_trans['Date'] < sys_start_date]
    curr_trans = df_trans[df_trans['Date'] >= sys_start_date].copy()
    curr_trans['Date_Norm'] = curr_trans['Date'].dt.normalize()
    trans_grouped = curr_trans.groupby('Date_Norm')
    
    cash = 0
    holdings = {}
    
    def process_tx(c, h, row):
        t, s, p, a = row['Ticker'], row['Shares'], row['Price'], row['Action']
        if t == 'CASH': c += s
        elif a == 'BUY':
            c -= (s * p)
            h[t] = h.get(t, 0) + s
        elif a == 'SELL':
            c += (abs(s) * p)
            h[t] = h.get(t, 0) + s
        return c, h

    for _, row in past_trans.iterrows():
        cash, holdings = process_tx(cash, holdings, row)

    nav_history = []
    daily_snapshots = {} 
    
    for d in full_dates:
        d_norm = d.normalize()
        if d_norm in trans_grouped.groups:
            for _, row in trans_grouped.get_group(d_norm).iterrows():
                cash, holdings = process_tx(cash, holdings, row)
        
        daily_snapshots[d_norm] = (holdings.copy(), cash)

        mkt_val = 0
        has_price = not price_data.empty and d_norm in price_data.index
        
        if has_price:
            for t, s in holdings.items():
                if abs(s) > 0.001 and t in price_data.columns:
                    mkt_val += s * price_data.loc[d_norm, t]
            
            total_assets = cash + mkt_val
            nav_history.append({
                'Date': d_norm, 'Total Assets': total_assets, 'Cash': cash, 
                'Market Value': mkt_val,
                'SPY': price_data.loc[d_norm, 'SPY'] if 'SPY' in price_data.columns else 0
            })
        elif price_data.empty:
             nav_history.append({
                'Date': d_norm, 'Total Assets': cash, 'Cash': cash, 'Market Value': 0, 'SPY': 100 
            })

    df_nav = pd.DataFrame(nav_history)
    if not df_nav.empty: df_nav = df_nav.set_index('Date')
    
    return df_nav, daily_snapshots, cash

def calculate_period_attribution(df_trans, price_data, daily_snapshots, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    valid_dates = sorted(daily_snapshots.keys())
    if not valid_dates: return pd.DataFrame(), 0
    
    def get_closest_date(target, dates):
        return min(dates, key=lambda x: abs(x - target))
    
    actual_start = get_closest_date(start_date, valid_dates)
    actual_end = get_closest_date(end_date, valid_dates)
    if actual_start > actual_end: actual_start = actual_end

    holdings_start, _ = daily_snapshots[actual_start]
    holdings_end, cash_end = daily_snapshots[actual_end]
    
    if price_data.empty: return pd.DataFrame(), cash_end
    
    price_idx = price_data.index
    p_start_idx = price_idx[price_idx <= actual_start]
    p_end_idx = price_idx[price_idx <= actual_end]
    
    if p_start_idx.empty or p_end_idx.empty: return pd.DataFrame(), cash_end
    
    prices_start = price_data.loc[p_start_idx[-1]]
    prices_end = price_data.loc[p_end_idx[-1]]
    
    mask = (df_trans['Date'] > actual_start) & (df_trans['Date'] <= actual_end)
    period_trans = df_trans.loc[mask]
    
    all_tickers = set(holdings_start.keys()) | set(holdings_end.keys()) | set(period_trans['Ticker'].unique())
    if 'CASH' in all_tickers: all_tickers.remove('CASH')
    
    perf_stats = []
    
    for t in all_tickers:
        qty_s = holdings_start.get(t, 0)
        val_s = qty_s * prices_start.get(t, 0) if t in prices_start else 0
        
        qty_e = holdings_end.get(t, 0)
        val_e = qty_e * prices_end.get(t, 0) if t in prices_end else 0
        
        t_tx = period_trans[period_trans['Ticker'] == t]
        buys = t_tx[t_tx['Action'] == 'BUY']
        sells = t_tx[t_tx['Action'] == 'SELL']
        
        cost_buy = (buys['Shares'] * buys['Price']).sum()
        proceeds_sell = (abs(sells['Shares']) * sells['Price']).sum()
        net_invest = cost_buy - proceeds_sell
        
        pnl = (val_e - val_s) - net_invest
        capital = abs(val_s) + cost_buy
        if capital == 0 and proceeds_sell > 0: capital = proceeds_sell
            
        roi = (pnl / capital * 100) if capital > 0 else 0
        
        if qty_e > 0: status = '多头 (Long)'
        elif qty_e < 0: status = '空头 (Short)'
        else: status = '已平仓 (Closed)'
        
        perf_stats.append({
            '代码': t, '总盈亏': pnl, '收益率': roi,
            '当前持仓': qty_e, '当前市值': val_e, '类型': status
        })
        
    df_perf = pd.DataFrame(perf_stats)
    if not df_perf.empty:
        df_perf = df_perf.sort_values('总盈亏', ascending=False)
        
    return df_perf, cash_end

# ==========================================
# 5. 侧边栏
# ==========================================
with st.sidebar:
    st.title("🌲 松熙基金工作台")
    
    with st.expander("⚙️ 系统设置", expanded=False):
        new_start_date = st.date_input("成立日期", st.session_state['sys_start_date'])
        if new_start_date != st.session_state['sys_start_date']:
            st.session_state['sys_start_date'] = new_start_date
            st.rerun()
        set_capital = st.number_input("初始资金", value=st.session_state['sys_initial_capital'])
        if st.button("🔴 重置系统", type="primary"):
            st.session_state['transactions'] = [{
                'Date': new_start_date.strftime('%Y-%m-%d'), 'Ticker': 'CASH', 
                'Action': 'DEPOSIT', 'Shares': set_capital, 'Price': 1.0, 'Reason': '初始化'
            }]
            st.rerun()

    st.divider()

    st.header("📝 交易录入")
    with st.form("trade_form"):
        col1, col2 = st.columns(2)
        with col1: tx_date = st.date_input("日期", max(date.today(), st.session_state['sys_start_date']))
        with col2: tx_ticker = st.text_input("代码", "").upper()
        col3, col4 = st.columns(2)
        with col3: tx_action = st.selectbox("动作", ["BUY (做多/平空)", "SELL (卖出/做空)", "DEPOSIT"])
        with col4: tx_shares = st.number_input("数量", min_value=1.0, value=100.0)
        tx_price = st.number_input("价格", min_value=0.0)
        tx_reason = st.text_area("投资逻辑", height=68, placeholder="TMT 行业逻辑...")
        if st.form_submit_button("提交", type="secondary", use_container_width=True):
            if not tx_ticker and 'DEPOSIT' not in tx_action: st.error("代码为空")
            else:
                real_action = 'DEPOSIT' if 'DEPOSIT' in tx_action else ('BUY' if 'BUY' in tx_action else 'SELL')
                shares_final = tx_shares if real_action != 'SELL' else -tx_shares
                st.session_state['transactions'].append({
                    'Date': tx_date.strftime('%Y-%m-%d'), 'Ticker': tx_ticker if tx_ticker else 'CASH',
                    'Action': real_action, 'Shares': shares_final, 'Price': tx_price, 'Reason': tx_reason
                })
                st.success("已记录")
                st.rerun()

# ==========================================
# 6. 主界面渲染
# ==========================================

df_trans = load_data()
tickers = df_trans[df_trans['Ticker']!='CASH']['Ticker'].unique().tolist()
price_data = get_price_history(tickers, st.session_state['sys_start_date'])

# 全量计算
df_nav_full, daily_snapshots, _ = calculate_full_history(df_trans, price_data)

# --- Header 数据 ---
if not df_nav_full.empty:
    latest = df_nav_full.iloc[-1]
    net_assets_str = f"${latest['Total Assets']:,.0f}"
    date_str = latest.name.strftime('%Y-%m-%d')
    nav = latest['Total Assets']
    net_exp_val = (latest['Market Value']) / nav * 100 if nav != 0 else 0
    
    def get_ret(days):
        target = latest.name - timedelta(days=days)
        past = df_nav_full[df_nav_full.index <= target]
        if past.empty: return None
        p_nav = past.iloc[-1]['Total Assets']
        return (nav - p_nav)/p_nav if p_nav!=0 else 0
    rets = {'1W': get_ret(7), '1M': get_ret(30), '1Y': get_ret(365)}
else:
    net_assets_str = "-"; date_str = "-"; net_exp_val = 0; rets = {'1W':None, '1M':None, '1Y':None}

# --- 样式生成函数 (Heatmap + Vertical Progress) ---
def get_card_style(val):
    if val is None: return 'background-color: #fff;', '#95a5a6', '#95a5a6', 'N/A'
    pct = val * 100
    abs_pct = abs(pct)
    opacity = min(max(abs_pct / 20, 0.1), 1.0) 
    
    if pct > 0:
        bg = f"rgba(217, 48, 37, {opacity})"
        txt = "#ffffff" if opacity > 0.5 else "#8B0000"
        lbl = "#ffffff" if opacity > 0.5 else "#95a5a6"
        sign = "+"
    elif pct < 0:
        bg = f"rgba(24, 128, 56, {opacity})"
        txt = "#ffffff" if opacity > 0.5 else "#006400"
        lbl = "#ffffff" if opacity > 0.5 else "#95a5a6"
        sign = ""
    else:
        bg = "#ffffff"; txt = "#95a5a6"; lbl = "#95a5a6"; sign = ""
    return f'background-color: {bg};', txt, lbl, f"{sign}{pct:.1f}%"

s_1w, c_1w, l_1w, t_1w = get_card_style(rets['1W'])
s_1m, c_1m, l_1m, t_1m = get_card_style(rets['1M'])
s_1y, c_1y, l_1y, t_1y = get_card_style(rets['1Y'])

exp_pct = min(max(net_exp_val, 0), 100)
style_exp = f"background: linear-gradient(to top, #e0e0e0 {exp_pct}%, #ffffff {exp_pct}%);"
color_exp = "#2c3e50" 

# --- Header HTML (V12.0 紧凑布局 + 垂直居中) ---
html_parts = []
html_parts.append('<div class="header-wrapper">')

html_parts.append('<div class="header-left">')
html_parts.append('<h1 class="main-title">松熙 TMT 模拟仓</h1>')
html_parts.append(f'<div class="sub-info">📅 {date_str} | 💵 净值: {net_assets_str}</div>')
html_parts.append('</div>')

html_parts.append('<div class="header-right">')
html_parts.append(f'<div class="kpi-box" style="{style_exp}"><div class="kpi-label" style="color:#6c757d">净多头仓位</div><div class="kpi-value" style="color:{color_exp}">{net_exp_val:.1f}%</div></div>')
html_parts.append(f'<div class="kpi-box" style="{s_1w}"><div class="kpi-label" style="color:{l_1w}">近一周</div><div class="kpi-value" style="color:{c_1w}">{t_1w}</div></div>')
html_parts.append(f'<div class="kpi-box" style="{s_1m}"><div class="kpi-label" style="color:{l_1m}">近一月</div><div class="kpi-value" style="color:{c_1m}">{t_1m}</div></div>')
html_parts.append(f'<div class="kpi-box" style="{s_1y}"><div class="kpi-label" style="color:{l_1y}">近一年</div><div class="kpi-value" style="color:{c_1y}">{t_1y}</div></div>')
html_parts.append('</div></div>')
st.markdown("".join(html_parts), unsafe_allow_html=True)

if df_nav_full.empty:
    st.info("请录入交易开始...")
    st.stop()

# --- 主界面时间筛选 (V12.0 新增: 自定义日期) ---
st.write("") 
c_filter_type, c_filter_date = st.columns([3, 4])

with c_filter_type:
    time_range = st.radio(
        "⏱️ 观察周期",
        ["近 1 月", "近 3 月", "近 1 年", "本年至今 (YTD)", "成立至今 (ALL)", "自定义"],
        horizontal=True,
        label_visibility="collapsed"
    )

# --- 筛选逻辑 ---
sys_start = st.session_state['sys_start_date']
today = date.today()
start_filter = sys_start
end_filter = today

if time_range == "自定义":
    with c_filter_date:
        c_start, c_end = st.columns(2)
        start_filter = c_start.date_input("开始日期", sys_start, label_visibility="collapsed")
        end_filter = c_end.date_input("结束日期", today, label_visibility="collapsed")
else:
    if time_range == "本年至今 (YTD)": start_filter = max(date(today.year, 1, 1), sys_start)
    elif time_range == "近 1 年": start_filter = max(today - timedelta(days=365), sys_start)
    elif time_range == "近 3 月": start_filter = max(today - timedelta(days=90), sys_start)
    elif time_range == "近 1 月": start_filter = max(today - timedelta(days=30), sys_start)

# 转换为 Timestamp
filter_start_ts = pd.Timestamp(start_filter)
filter_end_ts = pd.Timestamp(end_filter)

# 1. 净值截取
if not df_nav_full.empty:
    df_nav_filtered = df_nav_full[(df_nav_full.index >= filter_start_ts) & (df_nav_full.index <= filter_end_ts)].copy()
else: 
    df_nav_filtered = pd.DataFrame()

# 2. 归因截取
df_perf_period, cash_period_end = calculate_period_attribution(
    df_trans, price_data, daily_snapshots, filter_start_ts, filter_end_ts
)

# 3. 流水截取
mask_trans = (df_trans['Date'] >= filter_start_ts) & (df_trans['Date'] <= filter_end_ts)
df_trans_filtered = df_trans.loc[mask_trans]

# --- Tabs ---
st.caption(f"📅 数据展示区间: **{start_filter}** 至 **{end_filter}**")
tab1, tab2, tab3 = st.tabs(["📊 走势与持仓", "🏆 业绩归因", "📝 交易流水"])

with tab1:
    col_chart, col_pos = st.columns([2, 1])
    
    with col_chart:
        st.subheader("净值走势 (归一化)")
        if not df_nav_filtered.empty:
            start_val = df_nav_filtered['Total Assets'].iloc[0]
            base = start_val if start_val > 0 else 1
            plot_df = df_nav_filtered.copy()
            plot_df['松熙组合'] = plot_df['Total Assets'] / base * 100
            if 'SPY' in plot_df: 
                spy_base = plot_df['SPY'].iloc[0]
                plot_df['纳斯达克100'] = plot_df['SPY'] / (spy_base if spy_base>0 else 1) * 100
            
            fig_nav = go.Figure()
            fig_nav.add_trace(go.Scatter(x=plot_df.index, y=plot_df['松熙组合'], name='松熙组合', line=dict(color='#2c3e50', width=2.5)))
            if 'SPY' in plot_df:
                fig_nav.add_trace(go.Scatter(x=plot_df.index, y=plot_df['纳斯达克100'], name='Ref Index', line=dict(color='#BDC3C7', dash='dot')))
            
            # Buy/Sell 标记点 (V11.3 回归)
            visible_trades = df_trans_filtered[df_trans_filtered['Ticker'] != 'CASH'].copy()
            visible_trades['Date_Norm'] = visible_trades['Date'].dt.normalize()
            nav_lookup = plot_df['松熙组合']
            
            for action, color, symbol in [('BUY', '#E74C3C', 'triangle-up'), ('SELL', '#2ECC71', 'triangle-down')]:
                subset = visible_trades[visible_trades['Action'] == action]
                if not subset.empty:
                    y_vals = []
                    hover_texts = []
                    valid_dates = []
                    for _, row in subset.iterrows():
                        d = row['Date_Norm']
                        if d in nav_lookup.index:
                            y_vals.append(nav_lookup.loc[d])
                            valid_dates.append(d)
                            label = "Buy/Cover" if action=='BUY' else "Sell/Short"
                            hover_texts.append(f"<b>{row['Ticker']}</b> ({label})<br>${row['Price']}<br><i>{row.get('Reason','')}</i>")
                    if valid_dates:
                        fig_nav.add_trace(go.Scatter(x=valid_dates, y=y_vals, mode='markers', name=label, 
                                               marker=dict(symbol=symbol, size=10, color=color, line=dict(width=1, color='white')), 
                                               text=hover_texts, hoverinfo='text'))
            
            fig_nav.update_layout(height=480, margin=dict(l=20, r=20, t=30, b=20), legend=dict(orientation="h", y=1.02, x=0), hovermode="x unified")
            st.plotly_chart(fig_nav, use_container_width=True)
        else: st.warning("该区间内无净值数据")

    with col_pos:
        st.subheader("期末持仓结构")
        if not df_perf_period.empty:
            pos_data = []
            for _, row in df_perf_period.iterrows():
                if abs(row['当前市值']) > 1 and row['类型'] != '已平仓 (Closed)':
                    pos_data.append({'Label': row['代码'], 'Size': abs(row['当前市值']), 'SignedValue': row['当前市值'], 'Type': row['类型']})
            if cash_period_end > 1:
                pos_data.append({'Label': '现金', 'Size': cash_period_end, 'SignedValue': 0, 'Type': 'Cash'})
            
            if pos_data:
                df_tree = pd.DataFrame(pos_data)
                max_abs = max(abs(df_tree['SignedValue'].min()), abs(df_tree['SignedValue'].max())) if not df_tree.empty else 1
                if max_abs == 0: max_abs = 1
                
                fig_tree = px.treemap(
                    df_tree, path=[px.Constant("组合"), 'Label'], values='Size', color='SignedValue',
                    color_continuous_scale=[(0.0, '#228B22'), (0.5, '#F5F5F5'), (1.0, '#B22222')],
                    range_color=[-max_abs, max_abs],
                )
                fig_tree.update_traces(hovertemplate='<b>%{label}</b><br>市值: %{value:,.0f}', marker=dict(line=dict(width=0)), root_color="rgba(0,0,0,0)")
                fig_tree.update_layout(height=480, margin=dict(t=30, b=20, l=0, r=0), coloraxis_showscale=False)
                st.plotly_chart(fig_tree, use_container_width=True)
            else: st.info("期末为空仓")
        else: st.info("无数据")

with tab2:
    st.subheader("区间盈亏贡献")
    if df_perf_period.empty: st.info("该区间无持仓变动")
    else:
        df_pnl_plot = df_perf_period.sort_values('总盈亏', ascending=True)
        colors = ['#E74C3C' if x >= 0 else '#2ECC71' for x in df_pnl_plot['总盈亏']]
        fig_pnl = go.Figure(go.Bar(
            y=df_pnl_plot['代码'], x=df_pnl_plot['总盈亏'], orientation='h',
            marker_color=colors, marker_line_color='black', marker_line_width=1, opacity=1.0,
            text=[f"${v:,.0f} ({r:.1f}%)" for v, r in zip(df_pnl_plot['总盈亏'], df_pnl_plot['收益率'])],
            textposition='outside', 
            textfont=dict(family="Arial", size=14, color="black", weight="bold")
        ))
        fig_pnl.add_vline(x=0, line_width=1.5, line_color="black")
        mx = df_pnl_plot['总盈亏'].max(); mn = df_pnl_plot['总盈亏'].min()
        if pd.isna(mx): mx=0; mn=0
        range_buffer = max(abs(mx), abs(mn)) * 1.3 
        fig_pnl.update_layout(
            xaxis_range=[-range_buffer, range_buffer], 
            height=600, showlegend=False, plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='#f0f0f0'), 
            yaxis=dict(showgrid=False, tickfont=dict(size=15, color='black', family='Arial Black'))
        )
        st.plotly_chart(fig_pnl, use_container_width=True)

with tab3:
    st.subheader("区间交易流水")
    if not df_trans_filtered.empty:
        display_df = df_trans_filtered.sort_values('Date', ascending=False).copy()
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
        st.dataframe(display_df[['Date', 'Ticker', 'Action', 'Shares', 'Price', 'Reason']], use_container_width=True, hide_index=True)
    else: st.info("无交易记录")