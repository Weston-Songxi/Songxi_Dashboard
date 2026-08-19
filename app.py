import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
# 2. CSS 样式 (不隐藏系统菜单)
# ==========================================
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: "IBM Plex Sans", "Source Han Sans SC", "Noto Sans SC",
                     "PingFang SC", "Helvetica Neue", Arial, sans-serif;
    }
    .stApp { background: #f4f5f7; }
    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 2.4rem;
        max-width: min(1680px, 100%);
    }
    [data-testid="stSidebar"] {
        background: #fbfbfc;
        border-right: 1px solid #e6e8ec;
    }
    [data-testid="stSidebar"] h1 { font-size: 1.15rem !important; letter-spacing: 0.02em; }
    [data-testid="stSidebar"] h2 { font-size: 0.95rem !important; color: #1f3a4d; }
    .header-wrapper {
        display: flex; flex-direction: row; align-items: center; justify-content: space-between;
        flex-wrap: wrap; gap: 18px 28px; width: 100%; margin-bottom: 6px;
        border-bottom: 1px solid #e6e8ec; padding-bottom: 14px;
    }
    .header-left { flex-shrink: 0; max-width: 100%; }
    .main-title {
        font-size: 1.7rem; font-weight: 720; color: #1c2430; margin: 0; line-height: 1.15;
        letter-spacing: -0.02em; white-space: nowrap;
    }
    @media (max-width: 800px) { .main-title { white-space: normal; font-size: 1.35rem; } }
    .sub-info { font-size: 0.82rem; color: #6b7280; margin-top: 6px; font-weight: 400; line-height: 1.45; }
    .header-right { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    .kpi-box {
        border: 1px solid #e6e8ec; border-radius: 10px; padding: 0 14px; min-width: 88px; height: 62px;
        display: flex; flex-direction: column; justify-content: center; align-items: center;
        background: #fff; box-shadow: 0 1px 2px rgba(28,36,48,0.04); position: relative; overflow: hidden;
    }
    .kpi-label { font-size: 0.7rem; margin-bottom: 2px; font-weight: 600; letter-spacing: 0.04em; z-index: 2; text-transform: uppercase; }
    .kpi-value { font-size: 1.15rem; font-weight: 700; line-height: 1.1; white-space: nowrap; z-index: 2; }
    div.stRadio > div { display: flex; gap: 2px; align-items: center; flex-wrap: wrap; }
    div.stRadio > div label { margin-right: 10px; cursor: pointer; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; border-bottom: 1px solid #e6e8ec;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 14px; font-weight: 600; color: #6b7280;
    }
    .stTabs [aria-selected="true"] { color: #1f3a4d !important; }
    [data-testid="stMetric"] {
        background: #fff; border: 1px solid #e6e8ec; border-radius: 10px;
        padding: 10px 14px;
    }
    [data-testid="stMetricValue"] { font-size: 1.2rem; }
    .stCaption { color: #6b7280 !important; }
    footer { visibility: hidden; }
    .plotly-notifier, .modebar { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

CHART_FONT = dict(
    family="IBM Plex Sans, Source Han Sans SC, Noto Sans SC, Helvetica Neue, Arial, sans-serif",
    size=12,
    color="#1c2430",
)
C_LONG = "#C0392B"
C_SHORT = "#1E8449"
C_CASH = "#6B7280"
C_NAV = "#1F3A4D"
C_SPY = "#9AA3AD"
C_DD = "#C47A2C"


def apply_chart_style(fig, height=420, showlegend=True):
    fig.update_layout(
        height=height,
        font=CHART_FONT,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        margin=dict(l=12, r=18, t=28, b=12),
        showlegend=showlegend,
        legend=dict(
            orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)",
            font=dict(size=12, color="#4b5563"),
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#ffffff",
            font_size=12,
            font_family=CHART_FONT["family"],
            bordercolor="#e6e8ec",
        ),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#e6e8ec", tickfont=dict(size=11, color="#6b7280"))
    fig.update_yaxes(showgrid=True, gridcolor="#f0f2f5", zeroline=False, linecolor="#e6e8ec", tickfont=dict(size=11, color="#6b7280"))
    return fig

# ==========================================
# 3. Google Sheets 连接
# ==========================================
conn = st.connection("gsheets", type=GSheetsConnection)

# FIX 5: 给 load_data 加 @st.cache_data，后续可精确清除此缓存，
#         不影响价格历史缓存（_download_price_history）
@st.cache_data(ttl=600, show_spinner=False)
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
        # FIX 5: 只清除交易数据缓存，保留价格历史缓存，避免每次提交都重新拉行情
        load_data.clear()
        return True
    except Exception as e:
        st.error(f"写入失败: {e}")
        return False

def clear_all_data():
    """清空数据"""
    try:
        empty_df = pd.DataFrame(columns=["Date", "Ticker", "Action", "Shares", "Price", "Reason"])
        conn.update(data=empty_df)
        # FIX 5: 同上，精确清除
        load_data.clear()
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
    # FIX 1: 移除已弃用的 auto_adjust=False，改为默认（auto_adjust=True）
    #         使用复权价可正确反映股票拆分后的历史净值
    raw = yf.download(list(all_tickers_tuple), start=buffer_date, progress=False)
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

            # FIX 8: 只填充到「最后一个有真实收盘价的日期」，不强制延伸到今天。
            # 原因：若今日市场尚未收盘或 yfinance 数据未更新，
            #        ffill 会用昨日价格填充今日，导致净值曲线末端
            #        出现连续两天完全相同的值（视觉上像 bug）。
            # 用 dropna(how='all') 找到最后一个至少有一只标的有价格的日期，
            # 以此为终点，保证图表只展示有真实价格支撑的区间。
            last_real_date = data.dropna(how="all").index.max()
            daily_idx = pd.date_range(
                start=data.index.min(),
                end=last_real_date,
                freq="D"
            )
            data = data.reindex(daily_idx).ffill()
            st.session_state["_last_price_data"] = data
            return data
        except Exception:
            return st.session_state.get("_last_price_data", pd.DataFrame())

def _positive(x):
    try:
        v = float(x)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def get_anytime_price(ticker):
    """Anytime fill: extended-hours print if yfinance has one, else last regular close."""
    if not ticker or ticker == "CASH":
        return 1.0, "cash"
    tk = yf.Ticker(ticker)
    pre = post = last = prev = None
    try:
        fi = tk.fast_info
        last = _positive(getattr(fi, "last_price", None))
        prev = _positive(getattr(fi, "previous_close", None))
        pre = _positive(getattr(fi, "pre_market_price", None)) or _positive(getattr(fi, "preMarketPrice", None))
        post = _positive(getattr(fi, "post_market_price", None)) or _positive(getattr(fi, "postMarketPrice", None))
    except Exception:
        pass
    try:
        info = tk.info or {}
        pre = pre or _positive(info.get("preMarketPrice"))
        post = post or _positive(info.get("postMarketPrice"))
        last = last or _positive(info.get("regularMarketPrice"))
        prev = prev or _positive(info.get("regularMarketPreviousClose")) or _positive(info.get("previousClose"))
    except Exception:
        pass
    try:
        h = tk.history(period="1d", interval="1m", prepost=True, auto_adjust=True)
        if h is not None and not h.empty:
            px = _positive(h["Close"].iloc[-1])
            if px:
                ts = h.index[-1]
                try:
                    hour = ts.tz_convert("America/New_York").hour
                except Exception:
                    hour = int(getattr(ts, "hour", 12))
                if hour < 9 or hour >= 16:
                    return px, "盘前盘后"
                return px, "最新"
    except Exception:
        pass
    if pre:
        return pre, "盘前"
    if post:
        return post, "盘后"
    if last:
        return last, "最新"
    if prev:
        return prev, "昨收"
    try:
        h = tk.history(period="5d", auto_adjust=True, prepost=False)
        if h is not None and not h.empty:
            px = _positive(h["Close"].iloc[-1])
            if px:
                return px, "昨收"
    except Exception:
        pass
    return 0.0, "无行情"


@st.cache_data(ttl=20, show_spinner=False)
def get_realtime_price(ticker):
    px, _src = get_anytime_price(ticker)
    return px

def calculate_full_history(df_trans, price_data, sys_start_date):
    if df_trans.empty:
        return pd.DataFrame(), {}, 0.0
    sys_start_ts = pd.to_datetime(sys_start_date).normalize()
    df_trans = df_trans.sort_values("Date").copy()
    df_trans["Date_Norm"] = pd.to_datetime(df_trans["Date"]).dt.normalize()
    # FIX 8 (配套)：净值曲线的终点与 price_data 保持一致。
    # price_data 已截止到最后一个真实收盘日，这里用它的 max index 作为终点，
    # 确保不会出现「末尾两天净值完全相同」的情况。
    last_px_day = price_data.index.max() if not price_data.empty else pd.NaT
    last_tx_day = df_trans["Date_Norm"].max() if not df_trans.empty else pd.NaT
    today_ts = pd.Timestamp.today().normalize()
    candidates = [d for d in (last_px_day, last_tx_day, today_ts) if pd.notna(d)]
    end_date = max(candidates) if candidates else today_ts
    full_dates = pd.date_range(start=sys_start_ts, end=end_date, freq="D")
    past_trans = df_trans[df_trans["Date_Norm"] < sys_start_ts]
    curr_trans = df_trans[df_trans["Date_Norm"] >= sys_start_ts]
    trans_grouped = curr_trans.groupby("Date_Norm")
    cash = 0.0
    holdings = {}
    # FIX 6: last_px 通过闭包被 process_tx 修改（有意为之）
    #         作用：记录每只股票截至当日可见的最新成交价，避免未来数据穿越
    #         这是一个设计上的 side effect，已知且可接受
    last_px = {}

    def process_tx(c, h, row):
        t = row["Ticker"]
        s = float(row["Shares"])
        p = float(row["Price"])
        a = row["Action"]
        if t == "CASH":
            c += s
            return c, h
        # 记录最新成交价到外层 last_px（通过闭包修改）
        if p > 0:
            last_px[t] = p
        if a == "BUY":
            c -= (s * p)
            h[t] = h.get(t, 0.0) + s
        elif a == "SELL":
            # s 在录入时已设为负数，abs(s) 为卖出股数，h[t] += s 减少持仓
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
    last_fill = {}
    if not df_trans.empty:
        non_cash = df_trans[df_trans["Ticker"] != "CASH"]
        for tkr, g in non_cash.groupby("Ticker"):
            last_fill[tkr] = float(g.sort_values("Date").iloc[-1]["Price"])
    if price_data.empty:
        prices_start = pd.Series(dtype=float)
        prices_end = pd.Series(dtype=float)
    else:
        price_idx = price_data.index
        p_start_idx = price_idx[price_idx <= actual_start]
        p_end_idx = price_idx[price_idx <= actual_end]
        prices_start = price_data.loc[p_start_idx[-1]] if not p_start_idx.empty else pd.Series(dtype=float)
        prices_end = price_data.loc[p_end_idx[-1]] if not p_end_idx.empty else pd.Series(dtype=float)
    mask = (df_trans["Date"] > actual_start) & (df_trans["Date"] <= actual_end)
    period_trans = df_trans.loc[mask]
    all_tickers = set(holdings_start.keys()) | set(holdings_end.keys()) | set(period_trans["Ticker"].unique())
    if "CASH" in all_tickers:
        all_tickers.remove("CASH")
    perf_stats = []
    for t in all_tickers:
        def _px(series, ticker, fallback):
            if isinstance(series, pd.Series) and ticker in series.index:
                v = series.get(ticker)
                if pd.notna(v) and float(v) > 0:
                    return float(v), False
            fb = fallback.get(ticker)
            if fb and float(fb) > 0:
                return float(fb), True
            return 0.0, True
        p_s, _ = _px(prices_start, t, last_fill)
        p_e, px_stale = _px(prices_end, t, last_fill)
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
                "类型": status,
                "价格来源": "昨收/成交价" if px_stale else "行情",
            }
        )
    df_perf = pd.DataFrame(perf_stats)
    if not df_perf.empty:
        df_perf = df_perf.sort_values("总盈亏", ascending=False)
    return df_perf, cash_end

def calculate_vwap_cost_basis(df_trans):
    """
    计算每个当前持仓标的的加权平均建仓价 (VWAP)，同时支持多头与空头。

    返回值：dict，key=Ticker，value=(vwap_price, direction)
        direction: "多头" | "空头"

    核心规则
    ────────
    • net_shares > 0 → 多头；< 0 → 空头；= 0 → 无仓位
    • open_cost 始终表示当前持仓的「建仓总成本」（正值）
      - 多头：BUY 的价格 × 股数之和
      - 空头：SELL 的价格 × 股数之和（做空时收到的对价）
    • 加仓（同方向扩大）：追加成本
    • 减仓（同方向缩小）：按比例缩减成本，均价不变
    • 完全平仓：成本归零
    • 仓位穿越零点（多翻空 / 空翻多）：
        - 原仓位全部平掉，成本清零
        - 超额部分以当笔价格作为新仓位成本
    """
    result = {}
    tickers = df_trans[df_trans["Ticker"] != "CASH"]["Ticker"].unique()

    for ticker in tickers:
        ticker_trans = (
            df_trans[df_trans["Ticker"] == ticker]
            .sort_values("Date")
        )
        net_shares = 0.0   # 正 = 多头, 负 = 空头
        open_cost  = 0.0   # 当前仓位的建仓总成本（始终 >= 0）

        for _, row in ticker_trans.iterrows():
            s = float(row["Shares"])   # BUY 为正, SELL 为负
            p = float(row["Price"])
            prev_net = net_shares
            new_net  = prev_net + s

            # ── 情况 1：当前无仓位 → 直接开新仓 ──────────────────
            if abs(prev_net) < 1e-6:
                net_shares = new_net
                open_cost  = abs(s) * p

            # ── 情况 2：完全平仓（新净持仓趋近 0）────────────────
            elif abs(new_net) < 1e-6:
                net_shares = 0.0
                open_cost  = 0.0

            # ── 情况 3：同向操作（方向未变且未归零）──────────────
            elif (prev_net > 0) == (new_net > 0):
                if abs(new_net) > abs(prev_net):
                    # 加仓：追加建仓成本
                    open_cost += abs(s) * p
                else:
                    # 减仓：按剩余比例缩减成本（均价不变）
                    open_cost *= abs(new_net) / abs(prev_net)
                net_shares = new_net

            # ── 情况 4：仓位穿越零点（方向翻转）─────────────────
            else:
                # 超额部分 = 翻转后的新仓位大小
                residual_qty = abs(new_net)
                open_cost    = residual_qty * p   # 新仓位以当笔价格为成本
                net_shares   = new_net

        if abs(net_shares) > 1e-6:
            vwap      = open_cost / abs(net_shares)
            direction = "多头" if net_shares > 0 else "空头"
            result[ticker] = (vwap, direction)

    return result

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
    st.markdown("##### 松熙 · 工作台")
    st.caption("模拟仓录入与刷新")
    if st.button("🔄 刷新数据", use_container_width=True):
        load_data.clear()
        try:
            _download_price_history.clear()
        except Exception:
            pass
        st.session_state.pop("_last_price_data", None)
        st.rerun()
    st.divider()
    st.markdown("**交易录入**")
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
        px_src = ""
        if tx_ticker and tx_ticker != "CASH":
            with st.spinner(f"正在获取 {tx_ticker} 现价..."):
                current_price, px_src = get_anytime_price(tx_ticker)
        default_price = 1.0 if "DEPOSIT" in tx_action else float(current_price or 0.0)
        tx_price = st.number_input(
            "成交价格",
            min_value=0.0,
            value=default_price if default_price > 0 else 0.0,
            disabled=("DEPOSIT" in tx_action),
            help="盘前盘后有价用扩展行情，否则用昨收，可手动改"
        )
        if px_src:
            st.caption(f"默认价来源：{px_src}")
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

    # FIX 4: 补充数据管理 expander，接入已定义的 clear_all_data()
    with st.expander("⚙️ 数据管理"):
        st.warning("以下操作不可恢复，请谨慎使用！")
        if st.button("🗑️ 清空所有数据", type="secondary", use_container_width=True):
            if clear_all_data():
                st.success("所有数据已清空")
                st.session_state.clear()
                st.rerun()

# ==========================================
# 7. 主界面渲染
# ==========================================
if df_trans.empty:
    st.info("👋 欢迎！数据库为空。请先在左侧录入第一笔资金。")
    st.stop()

if not df_nav_full.empty:
    latest = df_nav_full.iloc[-1]
    nav = float(latest["Total Assets"])
    cash_now = float(latest["Cash"]) if "Cash" in latest.index else 0.0
    mkt_now = float(latest["Market Value"]) if "Market Value" in latest.index else 0.0
    net_assets_str = f"${nav:,.0f}"
    date_str = latest.name.strftime("%Y-%m-%d")
    init_nav = float(df_nav_full["Total Assets"].iloc[0])
    since_incept = nav - init_nav
    since_incept_pct = (since_incept / init_nav) if init_nav else 0.0
    net_exp_val = (mkt_now / nav * 100) if nav != 0 else 0.0
    cash_pct = (cash_now / nav * 100) if nav != 0 else 0.0

    long_mv = short_mv = 0.0
    if daily_snapshots:
        last_d = max(daily_snapshots)
        h_end, _c = daily_snapshots[last_d]
        px_row = price_data.iloc[-1] if not price_data.empty else pd.Series(dtype=float)
        for t, s in h_end.items():
            if abs(s) < 0.001:
                continue
            p = None
            if isinstance(px_row, pd.Series) and t in px_row.index:
                raw_p = px_row.get(t)
                if pd.notna(raw_p) and float(raw_p) > 0:
                    p = float(raw_p)
            if p is None:
                continue
            mv = s * p
            if mv >= 0:
                long_mv += mv
            else:
                short_mv += abs(mv)
    gross_exp_val = ((long_mv + short_mv) / nav * 100) if nav != 0 else 0.0

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
    cash_now = 0.0
    cash_pct = 0.0
    net_exp_val = 0
    gross_exp_val = 0.0
    since_incept = 0.0
    since_incept_pct = 0.0
    rets = {"1W": None, "1M": None, "1Y": None}

def get_card_style(val):
    if val is None:
        return "background-color: #fff;", "#95a5a6", "#95a5a6", "N/A"
    pct = val * 100
    abs_pct = abs(pct)
    opacity = min(max(abs_pct / 40, 0.06), 0.18)
    if pct > 0:
        bg = f"rgba(192, 57, 43, {opacity})"
        txt = "#C0392B"
        lbl = "#8a9199"
        sign = "+"
    elif pct < 0:
        bg = f"rgba(30, 132, 73, {opacity})"
        txt = "#1E8449"
        lbl = "#8a9199"
        sign = ""
    else:
        bg = "#ffffff"
        txt = "#95a5a6"
        lbl = "#95a5a6"
        sign = ""
    return f"background-color: {bg};", txt, lbl, f"{sign}{pct:.1f}%"

s_1w, c_1w, l_1w, t_1w = get_card_style(rets["1W"])
s_1m, c_1m, l_1m, t_1m = get_card_style(rets["1M"])
s_si, c_si, l_si, t_si = get_card_style(since_incept_pct)
exp_pct = min(max(abs(net_exp_val), 0), 100)
style_exp = f"background: linear-gradient(to top, #e0e0e0 {exp_pct}%, #ffffff {exp_pct}%);"
color_exp = "#2c3e50"
gross_fill = min(max(gross_exp_val, 0), 160)
style_gross = f"background: linear-gradient(to top, #dfe6e9 {min(gross_fill, 100)}%, #ffffff {min(gross_fill, 100)}%);"
cash_str = f"${cash_now:,.0f}"
incept_sign = "+" if since_incept >= 0 else ""
incept_str = f"{incept_sign}${since_incept:,.0f} ({since_incept_pct*100:+.1f}%)"
cash_note = "杠杆" if cash_now < -0.5 else "现金"

html_parts = []
html_parts.append('<div class="header-wrapper">')
html_parts.append('<div class="header-left">')
html_parts.append('<h1 class="main-title">松熙 TMT 模拟仓</h1>')
html_parts.append(
    f'<div class="sub-info">{date_str}&nbsp;&nbsp;·&nbsp;&nbsp;净值 {net_assets_str}&nbsp;&nbsp;·&nbsp;&nbsp;{cash_note} {cash_str} ({cash_pct:+.1f}%)</div>'
)
html_parts.append("</div>")
html_parts.append('<div class="header-right">')
html_parts.append(f'<div class="kpi-box" style="{style_exp}"><div class="kpi-label" style="color:#6c757d">净敞口</div><div class="kpi-value" style="color:{color_exp}">{net_exp_val:.1f}%</div></div>')
html_parts.append(f'<div class="kpi-box" style="{style_gross}"><div class="kpi-label" style="color:#6c757d">毛敞口</div><div class="kpi-value" style="color:{color_exp}">{gross_exp_val:.1f}%</div></div>')
html_parts.append(f'<div class="kpi-box" style="{s_1w}"><div class="kpi-label" style="color:{l_1w}">近一周</div><div class="kpi-value" style="color:{c_1w}">{t_1w}</div></div>')
html_parts.append(f'<div class="kpi-box" style="{s_1m}"><div class="kpi-label" style="color:{l_1m}">近一月</div><div class="kpi-value" style="color:{c_1m}">{t_1m}</div></div>')
html_parts.append(f'<div class="kpi-box" style="{s_si}"><div class="kpi-label" style="color:{l_si}">成立以来</div><div class="kpi-value" style="color:{c_si}">{t_si}</div></div>')
html_parts.append("</div></div>")
st.markdown("".join(html_parts), unsafe_allow_html=True)

# 筛选
st.write("")
_PERIODS = ["近 1 月", "近 3 月", "近 1 年", "本年至今 (YTD)", "成立至今 (ALL)", "自定义"]
c_filter_type, c_filter_date = st.columns([3, 2])
with c_filter_type:
    if hasattr(st, "pills"):
        time_range = st.pills("观察周期", _PERIODS, default="近 1 月", label_visibility="collapsed")
        if not time_range:
            time_range = "近 1 月"
    else:
        time_range = st.radio(
            "观察周期",
            _PERIODS,
            horizontal=True,
            label_visibility="collapsed",
        )
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
    df_nav_filtered = df_nav_full[
        (df_nav_full.index >= filter_start_ts) & (df_nav_full.index <= filter_end_ts)
    ].copy()
else:
    df_nav_filtered = pd.DataFrame()

df_perf_period, cash_period_end = calculate_period_attribution(
    df_trans, price_data, daily_snapshots, filter_start_ts, filter_end_ts
)
mask_trans = (df_trans["Date"] >= filter_start_ts) & (df_trans["Date"] <= filter_end_ts)
df_trans_filtered = df_trans.loc[mask_trans]

# --- Tabs ---
st.caption(f"展示区间  {start_filter}  →  {end_filter}")
tab1, tab2, tab3, tab4 = st.tabs(["净值", "持仓", "归因", "流水"])

# 期末持仓数据（持仓页 / 成本图共用）
pos_data = []
stale_names = []
nav_end = 0.0
if not df_perf_period.empty:
    stale_names = [
        r["代码"] for _, r in df_perf_period.iterrows()
        if r.get("价格来源") == "昨收/成交价" and r.get("类型") != "已平仓"
    ]
    total_mv = df_perf_period["当前市值"].sum()
    nav_end = cash_period_end + total_mv
    for _, row in df_perf_period.iterrows():
        open_pos = abs(row.get("当前持仓", 0)) > 0.001
        if row["类型"] != "已平仓" and (open_pos or abs(row["当前市值"]) > 1):
            pos_data.append({
                "Ticker": row["代码"],
                "Value": row["当前市值"],
                "Pct": (row["当前市值"] / nav_end) * 100 if nav_end != 0 else 0,
                "Type": row["类型"],
                "PxSrc": row.get("价格来源", "行情"),
                "Shares": row.get("当前持仓", 0),
            })
    if abs(cash_period_end) > 0.5:
        pos_data.append({
            "Ticker": "CASH",
            "Value": cash_period_end,
            "Pct": (cash_period_end / nav_end) * 100 if nav_end != 0 else 0,
            "Type": "Cash",
            "PxSrc": "",
            "Shares": cash_period_end,
        })

with tab1:
    if df_nav_filtered.empty:
        st.warning("该区间内无净值数据")
    else:
        c_scale, c_marks, _sp = st.columns([2, 1, 3])
        with c_scale:
            _scales = ["指数 (起点=100)", "美元净值"]
            if hasattr(st, "pills"):
                nav_scale = st.pills("净值刻度", _scales, default="美元净值", label_visibility="collapsed", key="nav_scale")
                if not nav_scale:
                    nav_scale = "美元净值"
            else:
                nav_scale = st.radio(
                    "净值刻度",
                    _scales,
                    index=1,
                    horizontal=True,
                    label_visibility="collapsed",
                    key="nav_scale",
                )
        with c_marks:
            show_trades = st.checkbox("买卖点", value=False, key="nav_trade_marks")

        plot_src = df_nav_filtered.copy()
        weekday_src = plot_src[plot_src.index.dayofweek < 5]
        if not weekday_src.empty:
            plot_src = weekday_src

        start_val = float(plot_src["Total Assets"].iloc[0])
        base = start_val if start_val > 0 else 1.0
        plot_df = plot_src.copy()
        use_index = nav_scale.startswith("指数")
        if use_index:
            plot_df["松熙组合"] = plot_df["Total Assets"] / base * 100
            y_title = "指数"
        else:
            plot_df["松熙组合"] = plot_df["Total Assets"]
            y_title = "美元"

        if "SPY" in plot_df.columns:
            spy_series = plot_df["SPY"].dropna()
            if not spy_series.empty:
                spy_base = float(spy_series.iloc[0])
                spy_den = spy_base if spy_base > 0 else 1.0
                if use_index:
                    plot_df["标普500(SPY)"] = plot_df["SPY"] / spy_den * 100
                else:
                    plot_df["标普500(SPY)"] = plot_df["SPY"] / spy_den * start_val

        peak = plot_df["松熙组合"].cummax()
        dd = (plot_df["松熙组合"] / peak.replace(0, np.nan) - 1.0) * 100

        fig_nav = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.78, 0.22],
            vertical_spacing=0.04,
        )
        fig_nav.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df["松熙组合"],
            name="松熙组合",
            line=dict(color=C_NAV, width=2.2),
        ), row=1, col=1)
        if "标普500(SPY)" in plot_df.columns:
            fig_nav.add_trace(go.Scatter(
                x=plot_df.index, y=plot_df["标普500(SPY)"],
                name="标普500 (SPY)",
                line=dict(color=C_SPY, width=1.4, dash="dot"),
            ), row=1, col=1)

        if not plot_df.empty:
            max_idx = plot_df["松熙组合"].idxmax()
            max_val = plot_df.loc[max_idx, "松熙组合"]
            min_idx = plot_df["松熙组合"].idxmin()
            min_val = plot_df.loc[min_idx, "松熙组合"]
            hi_txt = f"高 {max_val:.1f}" if use_index else f"高 ${max_val:,.0f}"
            lo_txt = f"低 {min_val:.1f}" if use_index else f"低 ${min_val:,.0f}"
            fig_nav.add_annotation(
                x=max_idx, y=max_val, text=hi_txt,
                showarrow=True, arrowhead=0, arrowwidth=1,
                arrowcolor="#9B59B6", ax=0, ay=-28,
                bgcolor="#fff", bordercolor="#e6e8ec", borderwidth=1, borderpad=3,
                font=dict(size=11, color="#7d3c98"),
                xref="x", yref="y",
            )
            fig_nav.add_annotation(
                x=min_idx, y=min_val, text=lo_txt,
                showarrow=True, arrowhead=0, arrowwidth=1,
                arrowcolor=C_DD, ax=0, ay=28,
                bgcolor="#fff", bordercolor="#e6e8ec", borderwidth=1, borderpad=3,
                font=dict(size=11, color=C_DD),
                xref="x", yref="y",
            )

        visible_trades = df_trans_filtered[df_trans_filtered["Ticker"] != "CASH"].copy()
        if show_trades and not visible_trades.empty:
            visible_trades["Date_Norm"] = visible_trades["Date"].dt.normalize()
            nav_lookup = plot_df["松熙组合"]
            n_groups = visible_trades["Date_Norm"].nunique()
            annotate = n_groups <= 6
            for d, group in visible_trades.groupby("Date_Norm"):
                if d not in nav_lookup.index:
                    continue
                y_val = nav_lookup.loc[d]
                has_buy = any("BUY" in a for a in group["Action"])
                has_sell = any("SELL" in a for a in group["Action"])
                if has_buy and has_sell:
                    color, symbol, size = "#D4A017", "diamond", 11
                elif has_buy:
                    color, symbol, size = C_LONG, "square", 9
                else:
                    color, symbol, size = C_SHORT, "square", 9
                hover_lines = [f"<b>{d.strftime('%Y-%m-%d')}</b>"]
                card_lines = []
                for _, row in group.iterrows():
                    txt_color = C_LONG if "BUY" in row["Action"] else C_SHORT
                    line_str = f"<span style='color:{txt_color}'><b>{row['Action'][:3]} {row['Ticker']}</b></span>"
                    card_lines.append(line_str)
                    hover_lines.append(
                        f"{line_str}<br>   ${row['Price'] * abs(row['Shares']):,.0f} · {row['Reason']}"
                    )
                card_text = "<br>".join(card_lines[:3]) + (f"<br>+{len(card_lines)-3}" if len(card_lines) > 3 else "")
                fig_nav.add_trace(go.Scatter(
                    x=[d], y=[y_val], mode="markers", name="Trade",
                    marker=dict(symbol=symbol, size=size, color=color, line=dict(width=0.6, color="white")),
                    showlegend=False, hovertext="<br>".join(hover_lines), hoverinfo="text"
                ), row=1, col=1)
                if annotate:
                    fig_nav.add_annotation(
                        x=d, y=y_val, text=card_text,
                        showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor=color,
                        ax=0, ay=36 if (has_sell and not has_buy) else -28,
                        bgcolor="#fff", bordercolor="#e6e8ec", borderwidth=1, borderpad=4,
                        font=dict(size=11, color="#1c2430"), opacity=0.95,
                        xref="x", yref="y",
                    )

        fig_nav.add_trace(go.Scatter(
            x=dd.index, y=dd, name="回撤",
            fill="tozeroy",
            line=dict(color=C_DD, width=1),
            fillcolor="rgba(196,122,44,0.16)",
            showlegend=False,
            hovertemplate="%{x|%Y-%m-%d}<br>回撤 %{y:.1f}%<extra></extra>",
        ), row=2, col=1)
        fig_nav.update_yaxes(title_text=y_title, row=1, col=1)
        fig_nav.update_yaxes(title_text="回撤%", row=2, col=1)
        apply_chart_style(fig_nav, height=520)
        st.plotly_chart(fig_nav, use_container_width=True, config={"displayModeBar": False})
        st.caption("日线用官方收盘，周末不画。盘后成交只进持仓，不改这条曲线。")

with tab2:
    if not pos_data:
        st.info("期末为空仓")
    else:
        if stale_names:
            st.caption("无实时行情，已用昨收或最近成交价： " + " · ".join(stale_names))
        df_bar = pd.DataFrame(pos_data)
        long_p = df_bar.loc[df_bar["Type"] == "多头", "Pct"].sum()
        short_p = df_bar.loc[df_bar["Type"] == "空头", "Pct"].sum()
        cash_p = df_bar.loc[df_bar["Ticker"] == "CASH", "Pct"].sum()
        long_v = df_bar.loc[df_bar["Type"] == "多头", "Value"].sum()
        short_v = df_bar.loc[df_bar["Type"] == "空头", "Value"].sum()
        cash_v = df_bar.loc[df_bar["Ticker"] == "CASH", "Value"].sum()
        e1, e2, e3 = st.columns(3)
        e1.metric("多头", f"{long_p:.1f}%", f"${long_v:,.0f}")
        e2.metric("空头", f"{short_p:.1f}%", f"${short_v:,.0f}")
        e3.metric("现金", f"{cash_p:.1f}%", f"${cash_v:,.0f}")

        col_w, col_c = st.columns(2)
        with col_w:
            df_w = df_bar.assign(_abs=df_bar["Pct"].abs()).sort_values("_abs", ascending=True)
            colors = [
                C_CASH if r["Ticker"] == "CASH" else (C_LONG if r["Value"] > 0 else C_SHORT)
                for _, r in df_w.iterrows()
            ]
            fig_bar = go.Figure(go.Bar(
                y=df_w["Ticker"], x=df_w["Pct"], orientation="h",
                text=[f"{p:+.1f}%" for p in df_w["Pct"]],
                textposition="outside",
                textfont=dict(size=11, color="#1c2430"),
                marker_color=colors, marker_line_width=0,
                customdata=np.stack([df_w["Value"], df_w["Type"]], axis=1),
                hovertemplate="<b>%{y}</b> · %{customdata[1]}<br>市值 $%{customdata[0]:,.0f}<br>权重 %{x:.2f}%<extra></extra>",
            ))
            max_abs = float(df_w["Pct"].abs().max()) if not df_w.empty else 10
            fig_bar.add_vline(x=0, line_width=1, line_color="#1c2430")
            fig_bar.update_xaxes(title="占净值 %", range=[-max(max_abs * 1.35, 8), max(max_abs * 1.35, 8)])
            apply_chart_style(fig_bar, height=max(300, len(df_w) * 38 + 60), showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

        with col_c:
            vwap_costs = calculate_vwap_cost_basis(df_trans)
            latest_prices = price_data.iloc[-1].dropna().to_dict() if not price_data.empty else {}
            cost_rows = []
            for pd_row in pos_data:
                ticker = pd_row["Ticker"]
                if ticker == "CASH":
                    continue
                cost_info = vwap_costs.get(ticker)
                cur_px = latest_prices.get(ticker)
                if not cost_info or not cur_px or cur_px <= 0:
                    continue
                vwap, direction = cost_info
                if direction == "多头":
                    pnl_pct = (cur_px - vwap) / vwap * 100
                    pnl_abs = (cur_px - vwap) * abs(pd_row["Value"] / cur_px)
                else:
                    pnl_pct = (vwap - cur_px) / vwap * 100
                    pnl_abs = (vwap - cur_px) * abs(pd_row["Value"] / cur_px)
                cost_rows.append({
                    "代码": ticker, "方向": direction,
                    "建仓均价": round(vwap, 2), "现价": round(cur_px, 2),
                    "浮盈亏%": pnl_pct, "浮盈亏$": pnl_abs,
                    "持仓量": round(abs(pd_row["Value"]) / cur_px, 2),
                    "市值": pd_row["Value"], "权重%": pd_row["Pct"],
                })
            if cost_rows:
                df_cost = pd.DataFrame(cost_rows).sort_values("浮盈亏%", ascending=True)
                df_cost["标签"] = df_cost.apply(
                    lambda r: f"{r['代码']}  [{'空' if r['方向'] == '空头' else '多'}]",
                    axis=1,
                )
                bar_colors = [C_LONG if v >= 0 else C_SHORT for v in df_cost["浮盈亏%"]]
                fig_cost = go.Figure(go.Bar(
                    y=df_cost["标签"], x=df_cost["浮盈亏%"], orientation="h",
                    marker_color=bar_colors, marker_line_width=0,
                    text=[f"{v:+.1f}%" for v in df_cost["浮盈亏%"]],
                    textposition="outside",
                    textfont=dict(size=11, color="#1c2430"),
                    customdata=df_cost[["建仓均价", "现价", "浮盈亏$", "方向", "权重%"]].values,
                    hovertemplate=(
                        "<b>%{y}</b> · %{customdata[3]}<br>"
                        "成本 $%{customdata[0]:,.2f} → 现价 $%{customdata[1]:,.2f}<br>"
                        "权重 %{customdata[4]:+.1f}% · 浮盈亏 $%{customdata[2]:,.0f}"
                        "<extra></extra>"
                    ),
                ))
                fig_cost.add_vline(x=0, line_width=1, line_color="#1c2430")
                max_abs = float(df_cost["浮盈亏%"].abs().max())
                fig_cost.update_xaxes(title="浮盈亏 %", range=[-max(max_abs * 1.35, 5), max(max_abs * 1.35, 5)])
                apply_chart_style(fig_cost, height=max(300, len(cost_rows) * 38 + 60), showlegend=False)
                st.plotly_chart(fig_cost, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("暂无可计算成本的持仓")
                df_cost = pd.DataFrame()

        show_tbl = df_bar[["Ticker", "Type", "Shares", "Value", "Pct"]].copy()
        show_tbl = show_tbl.rename(columns={
            "Ticker": "代码", "Type": "方向", "Shares": "数量",
            "Value": "市值", "Pct": "权重%",
        })
        if cost_rows:
            extra = df_cost[["代码", "建仓均价", "现价", "浮盈亏%", "浮盈亏$"]]
            show_tbl = show_tbl.merge(extra, on="代码", how="left")
        fmt = show_tbl.copy()
        fmt["数量"] = fmt["数量"].map(lambda v: f"{v:,.0f}")
        fmt["市值"] = fmt["市值"].map(lambda v: f"${v:,.0f}")
        fmt["权重%"] = fmt["权重%"].map(lambda v: f"{v:+.1f}%")
        if "建仓均价" in fmt.columns:
            fmt["建仓均价"] = fmt["建仓均价"].map(lambda v: f"${v:,.2f}" if pd.notna(v) else "—")
            fmt["现价"] = fmt["现价"].map(lambda v: f"${v:,.2f}" if pd.notna(v) else "—")
            fmt["浮盈亏%"] = fmt["浮盈亏%"].map(lambda v: f"{v:+.1f}%" if pd.notna(v) else "—")
            fmt["浮盈亏$"] = fmt["浮盈亏$"].map(lambda v: f"{v:+,.0f}" if pd.notna(v) else "—")
        st.dataframe(fmt, use_container_width=True, hide_index=True)

with tab3:
    if df_perf_period.empty:
        st.info("无数据")
    else:
        start_nav = float(df_nav_filtered["Total Assets"].iloc[0]) if not df_nav_filtered.empty else 0.0
        end_nav = float(df_nav_filtered["Total Assets"].iloc[-1]) if not df_nav_filtered.empty else 0.0
        period_pnl = end_nav - start_nav
        period_ret = (period_pnl / start_nav) if start_nav else 0.0
        df_pnl_plot = df_perf_period.copy()
        df_pnl_plot["贡献%"] = df_pnl_plot["总盈亏"] / start_nav * 100 if start_nav else 0.0
        long_pnl = df_pnl_plot.loc[df_pnl_plot["类型"] == "多头", "总盈亏"].sum()
        short_pnl = df_pnl_plot.loc[df_pnl_plot["类型"] == "空头", "总盈亏"].sum()
        closed_pnl = df_pnl_plot.loc[df_pnl_plot["类型"] == "已平仓", "总盈亏"].sum()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("区间净值", f"{period_ret*100:+.1f}%", f"${period_pnl:,.0f}")
        m2.metric("多头贡献", f"${long_pnl:,.0f}")
        m3.metric("空头贡献", f"${short_pnl:,.0f}")
        m4.metric("已平仓", f"${closed_pnl:,.0f}")

        eps = max(50.0, abs(start_nav) * 0.0005)
        quiet = df_pnl_plot[df_pnl_plot["总盈亏"].abs() < eps]
        df_pnl_plot = df_pnl_plot[df_pnl_plot["总盈亏"].abs() >= eps]
        if df_pnl_plot.empty:
            st.info("区间内没有超过阈值的盈亏贡献")
            if not quiet.empty:
                st.caption("接近零未画柱：" + "、".join(quiet["代码"].astype(str).tolist()))
        else:
            df_pnl_plot = df_pnl_plot.sort_values("总盈亏", ascending=True)
            df_pnl_plot["标签"] = df_pnl_plot.apply(
                lambda r: f"{r['代码']}  [{'空' if r['类型']=='空头' else ('平' if r['类型']=='已平仓' else '多')}]",
                axis=1,
            )
            colors = [C_LONG if x >= 0 else C_SHORT for x in df_pnl_plot["总盈亏"]]
            fig_pnl = go.Figure(go.Bar(
                y=df_pnl_plot["标签"], x=df_pnl_plot["总盈亏"], orientation="h",
                marker_color=colors, marker_line_width=0,
                text=[f"${v:,.0f}  {c:+.1f}%" for v, c in zip(df_pnl_plot["总盈亏"], df_pnl_plot["贡献%"])],
                textposition="outside",
                textfont=dict(size=12, color="#1c2430"),
                customdata=np.stack([df_pnl_plot["收益率"], df_pnl_plot["类型"]], axis=1),
                hovertemplate="<b>%{y}</b> · %{customdata[1]}<br>贡献 $%{x:,.0f}<br>资本收益率 %{customdata[0]:.1f}%<extra></extra>",
            ))
            fig_pnl.add_vline(x=0, line_width=1, line_color="#1c2430")
            mx = df_pnl_plot["总盈亏"].max()
            mn = df_pnl_plot["总盈亏"].min()
            if pd.isna(mx):
                mx = mn = 0
            rng = max(abs(float(mx)), abs(float(mn)), 1) * 1.35
            fig_pnl.update_xaxes(range=[-rng, rng], title="区间贡献 $（右侧百分比为占期初净值）")
            apply_chart_style(fig_pnl, height=max(320, len(df_pnl_plot) * 38 + 70), showlegend=False)
            st.plotly_chart(fig_pnl, use_container_width=True, config={"displayModeBar": False})
            bits = ["柱上百分比是对期初净值的贡献，不是单票资本收益率。"]
            if not quiet.empty:
                bits.append("接近零未画柱：" + "、".join(quiet["代码"].astype(str).tolist()))
            st.caption(" ".join(bits))

with tab4:
    c1, c2 = st.columns([1, 3])
    with c1:
        show_all = st.checkbox("显示全部历史", value=False)
    with c2:
        sort_mode = st.radio(
            "排序",
            ["日期 (最新)", "日期 (最早)", "代码 (A-Z)"],
            horizontal=True,
            label_visibility="collapsed",
        )
    target_df = df_trans if show_all else df_trans_filtered
    if target_df.empty:
        st.info("无交易")
    else:
        if sort_mode == "日期 (最新)":
            display_df = target_df.sort_values("Date", ascending=False)
        elif sort_mode == "日期 (最早)":
            display_df = target_df.sort_values("Date", ascending=True)
        else:
            display_df = target_df.sort_values(["Ticker", "Date"], ascending=[True, False])
        display_df = display_df.copy()
        act_map = {"BUY": "买入", "SELL": "卖出", "DEPOSIT": "入金"}
        show = pd.DataFrame({
            "日期": display_df["Date"].dt.strftime("%Y-%m-%d"),
            "代码": display_df["Ticker"],
            "动作": display_df["Action"].map(lambda a: act_map.get(str(a), str(a))),
            "数量": display_df["Shares"].astype(float),
            "价格": display_df["Price"].astype(float),
            "金额": display_df["Shares"].astype(float) * display_df["Price"].astype(float),
            "逻辑": display_df["Reason"].fillna("").astype(str),
        })

        def _style_action(col):
            colors = []
            for v in col:
                if v == "买入":
                    colors.append("color: #C0392B; font-weight: 600")
                elif v == "卖出":
                    colors.append("color: #1E8449; font-weight: 600")
                else:
                    colors.append("color: #6B7280")
            return colors

        styled = show.style.apply(_style_action, subset=["动作"]).format({
            "数量": "{:,.0f}",
            "价格": "${:,.2f}",
            "金额": "${:,.0f}",
        })
        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
            column_config={
                "逻辑": st.column_config.TextColumn("逻辑", width="large"),
                "代码": st.column_config.TextColumn("代码", width="small"),
                "动作": st.column_config.TextColumn("动作", width="small"),
            },
        )
