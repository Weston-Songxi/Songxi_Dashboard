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
# 2. CSS 样式 (标准版：不隐藏任何系统菜单)
# ==========================================
st.markdown("""
    <style>
    /* 仅调整边距，不隐藏顶部 Header */
    .block-container { 
        padding-top: 1rem; 
        padding-bottom: 3rem; 
    }

    /* 下面是您喜欢的 UI 样式 (卡片、标题等)，保留不动 */
    .header-wrapper {
        display: flex; flex-direction: row; align-items: center; justify-content: flex-start;
        flex-wrap: wrap; gap: 30px; width: 100%; margin-bottom: 10px;
        border-bottom: 1px solid #f0f0f0; padding-bottom: 15px; padding-right: 60px;
    }
    .header-left { flex-shrink: 0; max-width: 100%; }
    .main-title {
        font-size: 2.4rem; font-weight: 800; color: #2c3e50; margin: 0; line-height: 1.1;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; white-space: nowrap;
    }
    @media (max-width: 800px) { .main-title { white-space: normal; font-size: 2rem; } }
    .sub-info { font-size: 0.95rem; color: #7f8c8d; margin-top: 5px; font-weight: 400; }

    .header-right { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }

    .kpi-box {
        border: 1px solid #e1e4e8; border-radius: 8px; padding: 0 15px; min-width: 100px; height: 75px;
        display: flex; flex-direction: column; justify-content: center; align-items: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03); transition: all 0.3s ease; position: relative; overflow: hidden;
    }
    .kpi-label { font-size: 0.85rem; margin-bottom: 3px; font-weight: 600; z-index: 2; }
    .kpi-value { font-size: 1.35rem; font-weight: 700; line-height: 1.1; white-space: nowrap; z-index: 2; }

    div.stRadio > div { display: flex; gap: 0px; align-items: center; }
    div.stRadio > div label { margin-right: 15px; cursor: pointer; }
    .plotly-notifier { display: none; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 3. Google Sheets 连接
# ==========================================
conn = st.connection("gsheets", type=GSheetsConnection)


def load_data():
    """读取数据 (缓存10分钟)"""
    cols = ["Date", "Ticker", "Action", "Shares", "Price", "Reason"]
    try:
        df = conn.read(ttl=600)
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=cols)

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df["Shares"] = pd.to_numeric(df["Shares"], errors="coerce").fillna(0)
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0)
        df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
        return df
    except Exception as e:
        if "429" in str(e):
            st.warning("⚠️ 触发API频率限制，显示缓存数据。")
        else:
            st.error(f"数据读取错误: {str(e)}")
        return pd.DataFrame(columns=cols)


def save_transaction(new_row_dict):
    """写入数据"""
    cols = ["Date", "Ticker", "Action", "Shares", "Price", "Reason"]
    try:
        current_df = conn.read(ttl=0)
        if current_df is None or not isinstance(current_df, pd.DataFrame):
            current_df = pd.DataFrame(columns=cols)

        new_row_df = pd.DataFrame([new_row_dict])
        updated_df = pd.concat([current_df, new_row_df], ignore_index=True)
        conn.update(data=updated_df)
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"写入失败: {e}")
        return False


def clear_all_data():
    """清空数据"""
    try:
        empty_df = pd.DataFrame(columns=["Date", "Ticker", "Action", "Shares", "Price", "Reason"])
        conn.update(data=empty_df)
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"清空失败: {e}")
        return False


# ==========================================
# 4. 金融计算引擎
# ==========================================
@st.cache_data(ttl=600, show_spinner=False)
def _download_price_history(all_tickers_tuple, start_date_str):
    start_ts = pd.to_datetime(start_date_str)
    buffer_date = start_ts - pd.Timedelta(days=400)

    raw = yf.download(list(all_tickers_tuple), start=buffer_date, progress=False, auto_adjust=False)
    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            return pd.DataFrame()
        data = raw["Close"].copy()
    else:
        if "Close" not in raw.columns:
            return pd.DataFrame()
        data = raw[["Close"]].copy()
        if len(all_tickers_tuple) == 1:
            data.columns = [all_tickers_tuple[0]]

    if isinstance(data, pd.Series):
        data = data.to_frame()
        if len(all_tickers_tuple) == 1:
            data.columns = [all_tickers_tuple[0]]

    data.index = pd.to_datetime(data.index)
    if getattr(data.index, "tz", None) is not None:
        data.index = data.index.tz_localize(None)

    data = data.sort_index()
    return data


def get_price_history(tickers, start_date):
    if not tickers:
        return pd.DataFrame()

    all_tickers = sorted(set(tickers) | {"SPY"})
    if "CASH" in all_tickers:
        all_tickers.remove("CASH")
    if not all_tickers:
        return pd.DataFrame()

    with st.spinner("🔄 同步 TMT 市场数据..."):
        try:
            data = _download_price_history(
                tuple(all_tickers),
                str(pd.to_datetime(start_date).date())
            )
            if data.empty:
                return st.session_state.get("_last_price_data", pd.DataFrame())

            # 统一为日频并补齐自然日，前向填充，保证净值曲线连续
            data.index = pd.to_datetime(data.index).normalize()
            data = data[~data.index.duplicated(keep="last")].sort_index()

            daily_idx = pd.date_range(
                start=data.index.min(),
                end=pd.Timestamp.today().normalize(),
                freq="D"
            )
            data = data.reindex(daily_idx).ffill()

            st.session_state["_last_price_data"] = data
            return data
        except Exception:
            return st.session_state.get("_last_price_data", pd.DataFrame())


@st.cache_data(ttl=20, show_spinner=False)
def get_realtime_price(ticker):
    if not ticker or ticker == "CASH":
        return 0.0
    try:
        tk = yf.Ticker(ticker)
        px = tk.fast_info.get("last_price", 0.0)
        if px and px > 0:
            return float(px)
        hist = tk.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return 0.0


def calculate_full_history(df_trans, price_data, sys_start_date):
    if df_trans.empty:
        return pd.DataFrame(), {}, 0.0

    sys_start_ts = pd.to_datetime(sys_start_date).normalize()
    df_trans = df_trans.sort_values("Date").copy()
    df_trans["Date_Norm"] = pd.to_datetime(df_trans["Date"]).dt.normalize()

    end_date = pd.Timestamp.today().normalize()
    full_dates = pd.date_range(start=sys_start_ts, end=end_date, freq="D")

    past_trans = df_trans[df_trans["Date_Norm"] < sys_start_ts]
    curr_trans = df_trans[df_trans["Date_Norm"] >= sys_start_ts]
    trans_grouped = curr_trans.groupby("Date_Norm")

    cash = 0.0
    holdings = {}
    last_px = {}  # 截至当日可见价格，避免未来数据穿越

    def process_tx(c, h, row):
        t = row["Ticker"]
        s = float(row["Shares"])
        p = float(row["Price"])
        a = row["Action"]

        if t == "CASH":
            c += s
            return c, h

        if p > 0:
            last_px[t] = p

        if a == "BUY":
            c -= (s * p)
            h[t] = h.get(t, 0.0) + s
        elif a == "SELL":
            c += (abs(s) * p)
            h[t] = h.get(t, 0.0) + s
        return c, h

    for _, row in past_trans.iterrows():
        cash, holdings = process_tx(cash, holdings, row)

    nav_history = []
    daily_snapshots = {}

    for d in full_dates:
        d_norm = d.normalize()

        # 先处理当日交易
        if d_norm in trans_grouped.groups:
            for _, row in trans_grouped.get_group(d_norm).iterrows():
                cash, holdings = process_tx(cash, holdings, row)

        # 再更新当日收盘价
        if (not price_data.empty) and (d_norm in price_data.index):
            px_row = price_data.loc[d_norm]
            if isinstance(px_row, pd.Series):
                for t, p in px_row.items():
                    if pd.notna(p) and float(p) > 0:
                        last_px[t] = float(p)

        # 用截至当日价格估值
        mkt_val = 0.0
        for t, s in holdings.items():
            if abs(s) <= 0.001:
                continue
            p = last_px.get(t)
            if p is not None:
                mkt_val += s * p

        spy_val = last_px.get("SPY", np.nan)
        total_assets = cash + mkt_val

        daily_snapshots[d_norm] = (holdings.copy(), cash)
        nav_history.append(
            {
                "Date": d_norm,
                "Total Assets": total_assets,
                "Cash": cash,
                "Market Value": mkt_val,
                "SPY": spy_val
            }
        )

    df_nav = pd.DataFrame(nav_history)
    if not df_nav.empty:
        df_nav = df_nav.set_index("Date")
    return df_nav, daily_snapshots, cash


def calculate_period_attribution(df_trans, price_data, daily_snapshots, start_date, end_date):
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    valid_dates = sorted(daily_snapshots.keys())
    if not valid_dates:
        return pd.DataFrame(), 0

    def get_closest_date(target, dates):
        return min(dates, key=lambda x: abs(x - target))

    actual_start = get_closest_date(start_ts, valid_dates)
    actual_end = get_closest_date(end_ts, valid_dates)
    if actual_start > actual_end:
        actual_start = actual_end

    holdings_start, _ = daily_snapshots[actual_start]
    holdings_end, cash_end = daily_snapshots[actual_end]

    if price_data.empty:
        return pd.DataFrame(), cash_end

    price_idx = price_data.index
    p_start_idx = price_idx[price_idx <= actual_start]
    p_end_idx = price_idx[price_idx <= actual_end]
    if p_start_idx.empty or p_end_idx.empty:
        return pd.DataFrame(), cash_end

    prices_start = price_data.loc[p_start_idx[-1]]
    prices_end = price_data.loc[p_end_idx[-1]]

    mask = (df_trans["Date"] > actual_start) & (df_trans["Date"] <= actual_end)
    period_trans = df_trans.loc[mask]

    all_tickers = set(holdings_start.keys()) | set(holdings_end.keys()) | set(period_trans["Ticker"].unique())
    if "CASH" in all_tickers:
        all_tickers.remove("CASH")

    perf_stats = []
    for t in all_tickers:
        p_s = prices_start.get(t, 0) if isinstance(prices_start, pd.Series) else 0
        p_e = prices_end.get(t, 0) if isinstance(prices_end, pd.Series) else 0
        qty_s = holdings_start.get(t, 0)
        val_s = qty_s * p_s
        qty_e = holdings_end.get(t, 0)
        val_e = qty_e * p_e

        t_tx = period_trans[period_trans["Ticker"] == t]
        buys = t_tx[t_tx["Action"] == "BUY"]
        sells = t_tx[t_tx["Action"] == "SELL"]
        cost_buy = (buys["Shares"] * buys["Price"]).sum()
        proceeds_sell = (abs(sells["Shares"]) * sells["Price"]).sum()

        net_invest = cost_buy - proceeds_sell
        pnl = (val_e - val_s) - net_invest

        capital = abs(val_s) + cost_buy
        if capital == 0 and proceeds_sell > 0:
            capital = proceeds_sell
        roi = (pnl / capital * 100) if capital > 0 else 0

        if qty_e > 0:
            status = "多头"
        elif qty_e < 0:
            status = "空头"
        else:
            status = "已平仓"

        perf_stats.append(
            {
                "代码": t,
                "总盈亏": pnl,
                "收益率": roi,
                "当前持仓": qty_e,
                "当前市值": val_e,
                "类型": status
            }
        )

    df_perf = pd.DataFrame(perf_stats)
    if not df_perf.empty:
        df_perf = df_perf.sort_values("总盈亏", ascending=False)
    return df_perf, cash_end


# ==========================================
# 5. 初始化
# ==========================================
df_trans = load_data()

if not df_trans.empty:
    min_db_date = df_trans["Date"].min().date()
    if "sys_start_date" not in st.session_state:
        st.session_state["sys_start_date"] = min_db_date
    elif st.session_state["sys_start_date"] > min_db_date:
        st.session_state["sys_start_date"] = min_db_date
else:
    if "sys_start_date" not in st.session_state:
        st.session_state["sys_start_date"] = date.today()

if not df_trans.empty:
    tickers = df_trans[df_trans["Ticker"] != "CASH"]["Ticker"].unique().tolist()
    price_data = get_price_history(tickers, st.session_state["sys_start_date"])
    df_nav_full, daily_snapshots, current_cash = calculate_full_history(
        df_trans, price_data, st.session_state["sys_start_date"]
    )
else:
    price_data = pd.DataFrame()
    df_nav_full = pd.DataFrame()
    daily_snapshots = {}
    current_cash = 0.0

# ==========================================
# 6. 侧边栏：支持比例下单与预览
# ==========================================
with st.sidebar:
    st.title("🌲 松熙基金工作台")
    if st.button("🔄 刷新数据", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.divider()

    st.header("📝 交易录入")

    if not df_nav_full.empty:
        current_nav = float(df_nav_full.iloc[-1]["Total Assets"])
        current_cash_balance = float(df_nav_full.iloc[-1]["Cash"])
        latest_date = sorted(daily_snapshots.keys())[-1]
        current_holdings, _ = daily_snapshots[latest_date]
    else:
        current_nav = 0.0
        current_cash_balance = 0.0
        current_holdings = {}

    input_mode = st.radio("计算方式", ["按股数", "按净资产比例 %"], horizontal=True)

    with st.container(border=True):
        tx_action = st.selectbox("动作", ["BUY (做多/平空)", "SELL (卖出/做空)", "DEPOSIT"])

        col1, col2 = st.columns(2)
        with col1:
            tx_date = st.date_input("日期", date.today())
        with col2:
            raw_ticker = st.text_input("代码", "", disabled=("DEPOSIT" in tx_action)).upper().strip()

        tx_ticker = "CASH" if "DEPOSIT" in tx_action else raw_ticker

        current_price = 0.0
        if tx_ticker and tx_ticker != "CASH":
            with st.spinner(f"正在获取 {tx_ticker} 现价..."):
                current_price = get_realtime_price(tx_ticker)

        default_price = 1.0 if "DEPOSIT" in tx_action else float(current_price)
        tx_price = st.number_input(
            "成交价格",
            min_value=0.0,
            value=default_price,
            disabled=("DEPOSIT" in tx_action),
            help="默认拉取最新价，可手动修改"
        )

        if input_mode == "按股数":
            tx_shares_input = st.number_input("交易数量", min_value=0.0, value=100.0)
            if "SELL" in tx_action:
                final_shares = -abs(tx_shares_input)
            else:
                final_shares = abs(tx_shares_input)
        else:
            tx_pct = st.number_input("净资产比例 (%)", min_value=0.0, max_value=200.0, value=5.0, step=0.5)
            if tx_price > 0 and current_nav > 0:
                calculated_shares = (current_nav * (tx_pct / 100.0)) / tx_price
                if "SELL" in tx_action:
                    final_shares = -abs(calculated_shares)
                else:
                    final_shares = abs(calculated_shares)
                st.info(f"计算股数: {abs(final_shares):.2f} 股")
            else:
                final_shares = 0.0
                st.warning("无法计算：请检查价格或净资产是否大于0")

        tx_reason = st.text_area("投资逻辑", height=68, placeholder="输入买入/做空理由...")

    if st.button("🔍 预览交易", use_container_width=True, type="primary"):
        if not tx_ticker and "DEPOSIT" not in tx_action:
            st.error("请输入股票代码")
        elif final_shares == 0 and "DEPOSIT" not in tx_action:
            st.error("交易数量不能为0")
        else:
            st.session_state["show_preview"] = True
            st.session_state["temp_trade"] = {
                "Date": tx_date.strftime("%Y-%m-%d"),
                "Ticker": "CASH" if "DEPOSIT" in tx_action else tx_ticker,
                "Action": "DEPOSIT" if "DEPOSIT" in tx_action else ("BUY" if "BUY" in tx_action else "SELL"),
                "Shares": float(final_shares),
                "Price": float(tx_price),
                "Reason": tx_reason
            }

    if st.session_state.get("show_preview"):
        t = st.session_state["temp_trade"]
        with st.expander("📊 交易预检 (Preview)", expanded=True):
            st.write(f"**标的:** {t['Ticker']}")

            if t["Action"] == "DEPOSIT":
                op_label = "入金"
            elif t["Shares"] > 0:
                op_label = "做多/平空"
            else:
                op_label = "卖出/做空"
            st.write(f"**操作:** {op_label}")

            if t["Ticker"] == "CASH":
                old_cash = current_cash_balance
                new_cash = old_cash + t["Shares"]
                old_weight = (old_cash / current_nav * 100) if current_nav > 0 else 0
                new_weight = (new_cash / current_nav * 100) if current_nav > 0 else 0

                preview_df = pd.DataFrame({
                    "维度": ["现金余额", "现金权重 %"],
                    "交易前": [f"{old_cash:,.2f}", f"{old_weight:.2f}%"],
                    "交易后": [f"{new_cash:,.2f}", f"{new_weight:.2f}%"],
                    "变动": [f"{t['Shares']:+,.2f}", f"{(new_weight - old_weight):+.2f}%"]
                })
            else:
                old_shares = current_holdings.get(t["Ticker"], 0.0)
                new_shares = old_shares + t["Shares"]
                old_weight = (old_shares * t["Price"] / current_nav * 100) if current_nav > 0 else 0
                new_weight = (new_shares * t["Price"] / current_nav * 100) if current_nav > 0 else 0

                preview_df = pd.DataFrame({
                    "维度": ["持仓股数", "组合权重 %"],
                    "交易前": [f"{old_shares:,.2f}", f"{old_weight:.2f}%"],
                    "交易后": [f"{new_shares:,.2f}", f"{new_weight:.2f}%"],
                    "变动": [f"{t['Shares']:+,.2f}", f"{(new_weight - old_weight):+.2f}%"]
                })

            st.table(preview_df)

            c1, c2 = st.columns(2)
            if c1.button("✅ 确认提交", use_container_width=True):
                with st.spinner("☁️ 正在写入云端..."):
                    if save_transaction(t):
                        st.success("交易已记录！")
                        st.session_state["show_preview"] = False
                        st.rerun()

            if c2.button("❌ 取消", use_container_width=True):
                st.session_state["show_preview"] = False
                st.rerun()

    st.divider()
    # (保留原有的数据管理 Expander...)

# ==========================================
# 7. 主界面渲染
# ==========================================
if df_trans.empty:
    st.info("👋 欢迎！数据库为空。请先在左侧录入第一笔资金。")
    st.stop()

# 不重复拉行情/重算，复用初始化结果

if not df_nav_full.empty:
    latest = df_nav_full.iloc[-1]
    net_assets_str = f"${latest['Total Assets']:,.0f}"
    date_str = latest.name.strftime("%Y-%m-%d")
    nav = latest["Total Assets"]
    net_exp_val = (latest["Market Value"]) / nav * 100 if nav != 0 else 0

    def get_ret(days):
        target = latest.name - timedelta(days=days)
        past = df_nav_full[df_nav_full.index <= target]
        if past.empty:
            return None
        p_nav = past.iloc[-1]["Total Assets"]
        return (nav - p_nav) / p_nav if p_nav != 0 else 0

    rets = {"1W": get_ret(7), "1M": get_ret(30), "1Y": get_ret(365)}
else:
    net_assets_str = "-"
    date_str = "-"
    net_exp_val = 0
    rets = {"1W": None, "1M": None, "1Y": None}


def get_card_style(val):
    if val is None:
        return "background-color: #fff;", "#95a5a6", "#95a5a6", "N/A"

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
        bg = "#ffffff"
        txt = "#95a5a6"
        lbl = "#95a5a6"
        sign = ""

    return f"background-color: {bg};", txt, lbl, f"{sign}{pct:.1f}%"


s_1w, c_1w, l_1w, t_1w = get_card_style(rets["1W"])
s_1m, c_1m, l_1m, t_1m = get_card_style(rets["1M"])
s_1y, c_1y, l_1y, t_1y = get_card_style(rets["1Y"])

exp_pct = min(max(net_exp_val, 0), 100)
style_exp = f"background: linear-gradient(to top, #e0e0e0 {exp_pct}%, #ffffff {exp_pct}%);"
color_exp = "#2c3e50"

html_parts = []
html_parts.append('<div class="header-wrapper">')
html_parts.append('<div class="header-left">')
html_parts.append('<h1 class="main-title">松熙 TMT 模拟仓</h1>')
html_parts.append(f'<div class="sub-info">📅 {date_str} | 💵 净值: {net_assets_str}</div>')
html_parts.append("</div>")
html_parts.append('<div class="header-right">')
html_parts.append(f'<div class="kpi-box" style="{style_exp}"><div class="kpi-label" style="color:#6c757d">净多头仓位</div><div class="kpi-value" style="color:{color_exp}">{net_exp_val:.1f}%</div></div>')
html_parts.append(f'<div class="kpi-box" style="{s_1w}"><div class="kpi-label" style="color:{l_1w}">近一周</div><div class="kpi-value" style="color:{c_1w}">{t_1w}</div></div>')
html_parts.append(f'<div class="kpi-box" style="{s_1m}"><div class="kpi-label" style="color:{l_1m}">近一月</div><div class="kpi-value" style="color:{c_1m}">{t_1m}</div></div>')
html_parts.append(f'<div class="kpi-box" style="{s_1y}"><div class="kpi-label" style="color:{l_1y}">近一年</div><div class="kpi-value" style="color:{c_1y}">{t_1y}</div></div>')
html_parts.append("</div></div>")
st.markdown("".join(html_parts), unsafe_allow_html=True)

# 筛选
st.write("")
c_filter_type, c_filter_date = st.columns([3, 4])
with c_filter_type:
    time_range = st.radio("⏱️ 观察周期", ["近 1 月", "近 3 月", "近 1 年", "本年至今 (YTD)", "成立至今 (ALL)", "自定义"], horizontal=True, label_visibility="collapsed")

today = date.today()
start_filter = st.session_state["sys_start_date"]
end_filter = today

if time_range == "自定义":
    with c_filter_date:
        c_start, c_end = st.columns(2)
        start_filter = c_start.date_input("开始", start_filter, label_visibility="collapsed")
        end_filter = c_end.date_input("结束", today, label_visibility="collapsed")
else:
    if time_range == "本年至今 (YTD)":
        start_filter = max(date(today.year, 1, 1), start_filter)
    elif time_range == "近 1 年":
        start_filter = max(today - timedelta(days=365), start_filter)
    elif time_range == "近 3 月":
        start_filter = max(today - timedelta(days=90), start_filter)
    elif time_range == "近 1 月":
        start_filter = max(today - timedelta(days=30), start_filter)

filter_start_ts = pd.Timestamp(start_filter)
filter_end_ts = pd.Timestamp(end_filter)

if not df_nav_full.empty:
    df_nav_filtered = df_nav_full[(df_nav_full.index >= filter_start_ts) & (df_nav_full.index <= filter_end_ts)].copy()
else:
    df_nav_filtered = pd.DataFrame()

df_perf_period, cash_period_end = calculate_period_attribution(df_trans, price_data, daily_snapshots, filter_start_ts, filter_end_ts)
mask_trans = (df_trans["Date"] >= filter_start_ts) & (df_trans["Date"] <= filter_end_ts)
df_trans_filtered = df_trans.loc[mask_trans]

# --- Tabs ---
st.caption(f"📅 数据展示区间: **{start_filter}** 至 **{end_filter}**")
tab1, tab2, tab3 = st.tabs(["📊 走势与持仓", "🏆 业绩归因", "📝 交易流水"])

with tab1:
    col_chart, col_pos = st.columns([2, 1])

    with col_chart:
        st.subheader("净值走势")
        if not df_nav_filtered.empty:
            start_val = df_nav_filtered["Total Assets"].iloc[0]
            base = start_val if start_val > 0 else 1
            plot_df = df_nav_filtered.copy()
            plot_df["松熙组合"] = plot_df["Total Assets"] / base * 100

            if "SPY" in plot_df.columns:
                spy_base = plot_df["SPY"].iloc[0]
                plot_df["标普500(SPY)"] = plot_df["SPY"] / (spy_base if spy_base > 0 else 1) * 100

            fig_nav = go.Figure()
            fig_nav.add_trace(go.Scatter(x=plot_df.index, y=plot_df["松熙组合"], name="松熙组合", line=dict(color="#2c3e50", width=2.5)))

            if "SPY" in plot_df.columns:
                fig_nav.add_trace(go.Scatter(x=plot_df.index, y=plot_df["标普500(SPY)"], name="标普500(SPY)", line=dict(color="#BDC3C7", dash="dot")))

            # V15.0 高低点
            if not plot_df.empty:
                max_idx = plot_df["松熙组合"].idxmax()
                max_val = plot_df.loc[max_idx, "松熙组合"]
                fig_nav.add_annotation(x=max_idx, y=max_val, text=f"<b>High: {max_val:.1f}</b>", showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#9B59B6", ax=0, ay=-45, bgcolor="white", bordercolor="#9B59B6", borderwidth=1, borderpad=4, font=dict(size=12, color="#9B59B6", family="Arial Black"))

                min_idx = plot_df["松熙组合"].idxmin()
                min_val = plot_df.loc[min_idx, "松熙组合"]
                fig_nav.add_annotation(x=min_idx, y=min_val, text=f"<b>Low: {min_val:.1f}</b>", showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#E67E22", ax=0, ay=45, bgcolor="white", bordercolor="#E67E22", borderwidth=1, borderpad=4, font=dict(size=12, color="#E67E22", family="Arial Black"))

            # V14.2 买卖点
            visible_trades = df_trans_filtered[df_trans_filtered["Ticker"] != "CASH"].copy()
            if not visible_trades.empty:
                visible_trades["Date_Norm"] = visible_trades["Date"].dt.normalize()
                nav_lookup = plot_df["松熙组合"]

                for d, group in visible_trades.groupby("Date_Norm"):
                    if d in nav_lookup.index:
                        y_val = nav_lookup.loc[d]
                        has_buy = any("BUY" in a for a in group["Action"])
                        has_sell = any("SELL" in a for a in group["Action"])

                        if has_buy and has_sell:
                            color = "#FFD700"; symbol = "diamond"; size = 13
                        elif has_buy:
                            color = "#E74C3C"; symbol = "square"; size = 11
                        else:
                            color = "#2ECC71"; symbol = "square"; size = 11

                        card_lines = []
                        hover_lines = [f"<span style='font-size:16px'><b>📅 {d.strftime('%Y-%m-%d')}</b></span>"]

                        for _, row in group.iterrows():
                            txt_color = "#D32F2F" if "BUY" in row["Action"] else "#2E7D32"
                            line_str = f"<span style='color:{txt_color}'><b>{row['Action'][:3]} {row['Ticker']}</b></span>"
                            card_lines.append(line_str)
                            hover_lines.append(f"{line_str}<br>   💵 ${row['Price'] * abs(row['Shares']):,.0f} | 📝 {row['Reason']}")

                        if len(card_lines) > 3:
                            card_text = "<br>".join(card_lines[:3]) + f"<br><span style='color:black'>...(+{len(card_lines)-3})</span>"
                        else:
                            card_text = "<br>".join(card_lines)

                        hover_content = "<br>".join(hover_lines)
                        fig_nav.add_trace(go.Scatter(x=[d], y=[y_val], mode="markers", name="Trade", marker=dict(symbol=symbol, size=size, color=color, line=dict(width=1, color="white")), showlegend=False, hovertext=hover_content, hoverinfo="text"))
                        fig_nav.add_annotation(x=d, y=y_val, text=card_text, showarrow=True, arrowhead=0, arrowsize=1, arrowwidth=1, arrowcolor=color, ax=0, ay=-35, bgcolor="white", bordercolor=color, borderwidth=1, borderpad=6, font=dict(size=13, family="Arial Black", color="black"), opacity=0.9)

            fig_nav.update_layout(height=480, margin=dict(l=20, r=20, t=30, b=20), legend=dict(orientation="h", y=1.02, x=0), hovermode="x unified", hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial"))
            st.plotly_chart(fig_nav, use_container_width=True)
        else:
            st.warning("该区间内无净值数据")

    with col_pos:
        st.subheader("期末持仓结构")
        if not df_perf_period.empty:
            total_mv = df_perf_period["当前市值"].sum()
            nav_end = cash_period_end + total_mv
            pos_data = []

            for _, row in df_perf_period.iterrows():
                if abs(row["当前市值"]) > 1 and row["类型"] != "已平仓":
                    pos_data.append({
                        "Ticker": row["代码"],
                        "Value": row["当前市值"],
                        "Pct": (row["当前市值"] / nav_end) * 100 if nav_end != 0 else 0,
                        "Type": row["类型"]
                    })

            if cash_period_end > 1:
                pos_data.append({
                    "Ticker": "CASH",
                    "Value": cash_period_end,
                    "Pct": (cash_period_end / nav_end) * 100 if nav_end != 0 else 0,
                    "Type": "Cash"
                })

            if pos_data:
                df_bar = pd.DataFrame(pos_data)
                sort_order = st.selectbox("排序方式", ["占比从大到小", "占比从小到大", "代码 A-Z"], label_visibility="collapsed")

                if sort_order == "占比从大到小":
                    df_bar = df_bar.sort_values("Pct", ascending=False)
                elif sort_order == "占比从小到大":
                    df_bar = df_bar.sort_values("Pct", ascending=True)
                else:
                    df_bar = df_bar.sort_values("Ticker")

                colors = ["#E74C3C" if v > 0 else "#2ECC71" for v in df_bar["Value"]]
                fig_bar = go.Figure(go.Bar(
                    x=df_bar["Ticker"], y=df_bar["Pct"],
                    text=[f"{'+' if p > 0 else ''}{p:.1f}%" for p in df_bar["Pct"]], textposition="outside",
                    textfont=dict(family="Arial Black", size=12, color="black"),
                    marker_color=colors, marker_line_color="black", marker_line_width=1.5,
                    hovertemplate="<b>%{x}</b><br>市值: $%{customdata:,.0f}<br>占比: %{y:.2f}%<extra></extra>", customdata=df_bar["Value"]
                ))

                fig_bar.update_layout(height=480, margin=dict(t=40, b=40, l=20, r=20), xaxis=dict(title=None, tickfont=dict(size=12)), yaxis=dict(title="占净值比例 (%)", showgrid=True, gridcolor="#f0f0f0"), plot_bgcolor="rgba(0,0,0,0)", dragmode=False)
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("期末为空仓")
        else:
            st.info("无数据")

with tab2:
    st.subheader("区间盈亏贡献")
    if df_perf_period.empty:
        st.info("无数据")
    else:
        df_pnl_plot = df_perf_period.sort_values("总盈亏", ascending=True)
        colors = ["#E74C3C" if x >= 0 else "#2ECC71" for x in df_pnl_plot["总盈亏"]]

        fig_pnl = go.Figure(go.Bar(
            y=df_pnl_plot["代码"], x=df_pnl_plot["总盈亏"], orientation="h",
            marker_color=colors, marker_line_color="black", marker_line_width=1, opacity=1.0,
            text=[f"${v:,.0f} ({r:.1f}%)" for v, r in zip(df_pnl_plot["总盈亏"], df_pnl_plot["收益率"])],
            textposition="outside", textfont=dict(family="Arial", size=14, color="black")
        ))

        fig_pnl.add_vline(x=0, line_width=1.5, line_color="black")
        mx = df_pnl_plot["总盈亏"].max()
        mn = df_pnl_plot["总盈亏"].min()
        if pd.isna(mx):
            mx = 0
            mn = 0

        range_buffer = max(abs(mx), abs(mn)) * 1.3
        fig_pnl.update_layout(xaxis_range=[-range_buffer, range_buffer], height=600, showlegend=False, plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(showgrid=True, gridcolor="#f0f0f0"), yaxis=dict(showgrid=False, tickfont=dict(size=15, color="black", family="Arial Black")))
        st.plotly_chart(fig_pnl, use_container_width=True)

with tab3:
    st.subheader("区间调仓记录")

    c1, c2 = st.columns([1, 4])
    with c1:
        show_all = st.checkbox("显示全部历史 (忽略筛选)", value=False)
    with c2:
        sort_mode = st.radio("排序方式", ["日期 (最新)", "日期 (最早)", "代码 (A-Z)"], horizontal=True, label_visibility="collapsed")

    target_df = df_trans if show_all else df_trans_filtered

    if not target_df.empty:
        if sort_mode == "日期 (最新)":
            display_df = target_df.sort_values("Date", ascending=False)
        elif sort_mode == "日期 (最早)":
            display_df = target_df.sort_values("Date", ascending=True)
        else:
            display_df = target_df.sort_values(["Ticker", "Date"], ascending=[True, False])

        display_df = display_df.copy()
        display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(display_df[["Date", "Ticker", "Action", "Shares", "Price", "Reason"]], use_container_width=True, hide_index=True)
    else:
        st.info("无交易")
