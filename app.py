import altair as alt
import pandas as pd
import pandas_market_calendars as mcal
import streamlit as st
import yfinance as yf

CONSTITUENTS_URL = (
    "https://raw.githubusercontent.com/datasets/"
    "s-and-p-500-companies/main/data/constituents.csv"
)
RSI_PERIOD = 14
DMA_PERIOD = 30
RSI_LOWER_REFERENCE = 36
MARKET_TIMEZONE = "America/New_York"


@st.cache_data(ttl=86_400)
def load_constituents():
    stocks = pd.read_csv(
        CONSTITUENTS_URL, usecols=["Symbol", "Security", "GICS Sector"]
    )
    stocks["Yahoo symbol"] = stocks["Symbol"].str.replace(".", "-", regex=False)
    return stocks.set_index("Yahoo symbol")


def scan_calendar(now, lookback_weeks):
    today = now.tz_convert(MARKET_TIMEZONE).tz_localize(None).normalize()
    calendar = mcal.get_calendar("NYSE")
    schedule = calendar.schedule(
        start_date=(today - pd.Timedelta(weeks=lookback_weeks + 104)).date(),
        end_date=today.date(),
    )
    completed = schedule.loc[schedule["market_close"] <= now]
    if completed.empty:
        raise ValueError("No completed trading sessions are available.")
    cutoff = completed.index[-1] - pd.Timedelta(weeks=lookback_weeks)
    scan_dates = completed.index[completed.index > cutoff]
    return completed.index, scan_dates


@st.cache_data(ttl=300, show_spinner=False)
def load_closes(symbols, start, end):
    batches = []
    for offset in range(0, len(symbols), 50):
        try:
            data = yf.download(
                list(symbols[offset : offset + 50]),
                start=start, end=end, interval="1d", auto_adjust=False,
                group_by="column", multi_level_index=True, ignore_tz=True,
                threads=5, progress=False, timeout=20,
            )
            if data is not None and not data.empty:
                batches.append(data["Close"])
        except Exception:
            # Missing batch data is surfaced by coverage reporting.
            continue
    if not batches:
        return pd.DataFrame(columns=symbols)
    closes = pd.concat(batches, axis=1).sort_index()
    closes.index = pd.DatetimeIndex(closes.index).tz_localize(None).normalize()
    return closes.reindex(columns=symbols).where(lambda frame: frame.gt(0))


def wilder_average(values, period):
    averages = pd.Series(float("nan"), index=values.index, dtype=float)
    valid = values.notna()
    runs = (~valid).cumsum()
    for _, run in values[valid].groupby(runs[valid]):
        if len(run) < period:
            continue
        seeded = run.iloc[period - 1 :].copy()
        seeded.iloc[0] = run.iloc[:period].mean()
        averages.loc[seeded.index] = seeded.ewm(
            alpha=1 / period, adjust=False
        ).mean()
    return averages


def wilder_rsi(closes):
    changes = closes.diff()
    gain = wilder_average(changes.clip(lower=0), RSI_PERIOD)
    loss = wilder_average(-changes.clip(upper=0), RSI_PERIOD)
    total = gain + loss
    return (100 * gain / total).mask(total.eq(0), 50.0)


def sell_signals(closes, threshold):
    rsi = closes.apply(wilder_rsi)
    dma = closes.rolling(DMA_PERIOD, min_periods=DMA_PERIOD).mean()
    previous_rsi = rsi.shift(1)
    eligible = rsi.notna() & previous_rsi.notna() & dma.notna() & closes.notna()
    signals = (
        eligible
        & previous_rsi.ge(threshold)
        & rsi.lt(threshold)
        & closes.lt(dma)
    )
    return rsi, dma, eligible, signals


def build_results(stocks, closes, rsi, dma, signals):
    rows = []
    for symbol in signals.columns[signals.any(axis=0)]:
        stock = stocks.loc[symbol]
        for date in signals.index[signals[symbol]]:
            price = closes.at[date, symbol]
            average = dma.at[date, symbol]
            rows.append({
                "Ticker": stock["Symbol"],
                "Company": stock["Security"],
                "Sector": stock["GICS Sector"],
                "Signal date": date,
                "Signal close (USD)": price,
                "30 DMA (USD)": average,
                "Below 30 DMA (%)": (1 - price / average) * 100,
                "Previous RSI": rsi[symbol].shift(1).at[date],
                "Signal RSI": rsi.at[date, symbol],
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["Signal date", "Ticker"], ascending=[False, True]
    )


def show_charts(symbol, ticker, closes, rsi, dma, results, threshold, scan_dates):
    first_date = min(closes.index[-min(90, len(closes))], scan_dates[0])
    history = pd.DataFrame({
        "Close": closes[symbol], "30 DMA": dma[symbol], "RSI": rsi[symbol]
    }).loc[first_date:].rename_axis("Date").reset_index()
    markers = results.loc[
        results["Ticker"].eq(ticker),
        ["Signal date", "Signal close (USD)", "Signal RSI"],
    ].rename(columns={"Signal date": "Date"})

    prices = history.melt(
        id_vars="Date", value_vars=["Close", "30 DMA"],
        var_name="Series", value_name="Price",
    )
    lines = alt.Chart(prices).mark_line().encode(
        x=alt.X("Date:T", title="Trading session"),
        y=alt.Y("Price:Q", title="Price (USD)", scale=alt.Scale(zero=False)),
        color=alt.Color("Series:N", title=None),
        tooltip=["Series:N", alt.Tooltip("Date:T", format="%b %d, %Y"),
                 alt.Tooltip("Price:Q", format="$,.2f")],
    )
    price_markers = alt.Chart(markers).mark_point(
        color="#d1495b", filled=True, shape="triangle-down", size=130
    ).encode(
        x="Date:T", y="Signal close (USD):Q",
        tooltip=[alt.Tooltip("Date:T", title="Sell signal", format="%b %d, %Y"),
                 alt.Tooltip("Signal close (USD):Q", format="$,.2f"),
                 alt.Tooltip("Signal RSI:Q", format=".2f")],
    )
    labels = alt.Chart(markers).mark_text(
        align="left", dx=8, dy=-12, color="#d1495b"
    ).encode(
        x="Date:T", y="Signal close (USD):Q",
        text=alt.Text("Signal close (USD):Q", format="$,.2f"),
    )
    st.subheader(f"{ticker} daily close and 30 DMA")
    st.altair_chart(lines + price_markers + labels, use_container_width=True)

    rsi_line = alt.Chart(history).mark_line().encode(
        x=alt.X("Date:T", title="Trading session"),
        y=alt.Y("RSI:Q", title="Wilder RSI (14)", scale=alt.Scale(domain=[0, 100])),
        tooltip=[alt.Tooltip("Date:T", format="%b %d, %Y"),
                 alt.Tooltip("RSI:Q", format=".2f")],
    )
    levels = pd.DataFrame({"Level": [threshold, RSI_LOWER_REFERENCE]})
    rules = alt.Chart(levels).mark_rule(color="#888888", strokeDash=[5, 5]).encode(
        y="Level:Q", tooltip=[alt.Tooltip("Level:Q", format=".1f")]
    )
    rsi_markers = alt.Chart(markers).mark_point(
        color="#d1495b", filled=True, size=90
    ).encode(x="Date:T", y="Signal RSI:Q")
    st.subheader(f"{ticker} daily RSI")
    st.altair_chart(rsi_line + rules + rsi_markers, use_container_width=True)
    st.caption(f"Dashed RSI reference levels: {threshold:g} sell trigger and 36.")


def main():
    st.set_page_config(page_title="S&P 500 Sell Signal Scanner", layout="wide")
    st.title("S&P 500 Sell Signal Scanner")
    lookback = int(st.sidebar.number_input(
        "Weeks to scan", min_value=1, value=1, step=1,
        help="Trailing calendar weeks ending on the latest completed trading session.",
    ))
    threshold = st.sidebar.number_input(
        "RSI sell trigger", min_value=0.0, max_value=100.0, value=50.0, step=0.5
    )
    if st.sidebar.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()
    st.caption(
        f"Sell signal when the prior daily RSI is at least {threshold:g}, today's RSI "
        f"closes below {threshold:g}, and the daily close is below its 30-day simple moving average."
    )
    try:
        stocks = load_constituents()
        completed, scan_dates = scan_calendar(pd.Timestamp.now(tz="UTC"), lookback)
    except Exception as exc:
        st.error(f"Could not load the stock list or market calendar. {exc}")
        st.stop()

    with st.spinner("Scanning S&P 500 daily closing prices..."):
        closes = load_closes(
            tuple(stocks.index), completed[0].date().isoformat(),
            (completed[-1] + pd.Timedelta(days=1)).date().isoformat(),
        ).reindex(index=completed)
        rsi, dma, eligible, signals = sell_signals(closes, threshold)
    coverage = eligible.reindex(scan_dates)
    scanned = int(coverage.any(axis=0).sum())
    st.caption(
        f"Scan dates {scan_dates[0]:%b %d, %Y} through {scan_dates[-1]:%b %d, %Y} · "
        f"{scanned} of {len(stocks)} tickers had at least one scorable session."
    )
    incomplete = stocks.loc[~coverage.all(axis=0), "Symbol"]
    if not incomplete.empty:
        with st.expander(f"{len(incomplete)} tickers with incomplete scan coverage"):
            st.write(", ".join(incomplete))
    if scanned == 0:
        st.error("No tickers had sufficient price history. Try refreshing shortly.")
        st.stop()
    results = build_results(stocks, closes, rsi, dma, signals.reindex(scan_dates))
    if results.empty:
        st.info("No sell signals matched both conditions during this scan window.")
        st.stop()

    st.subheader(f"Sell signals ({len(results)})")
    config = {
        column: st.column_config.NumberColumn(format="$%.2f")
        for column in ["Signal close (USD)", "30 DMA (USD)"]
    }
    config.update({
        column: st.column_config.NumberColumn(format="%.2f")
        for column in ["Previous RSI", "Signal RSI"]
    })
    config["Signal date"] = st.column_config.DateColumn(format="YYYY-MM-DD")
    config["Below 30 DMA (%)"] = st.column_config.NumberColumn(format="%.2f%%")
    st.dataframe(results, hide_index=True, column_config=config)
    st.download_button(
        "Download sell signals as CSV", results.to_csv(index=False, date_format="%Y-%m-%d"),
        file_name="daily_rsi_sell_signals.csv", mime="text/csv",
    )
    ticker = st.selectbox("View signal charts", results["Ticker"].drop_duplicates())
    symbol = stocks.reset_index().set_index("Symbol").at[ticker, "Yahoo symbol"]
    show_charts(symbol, ticker, closes, rsi, dma, results, threshold, scan_dates)
    st.caption(
        "Signals use completed regular-session daily candles. RSI and the 30 DMA both use "
        "Yahoo Finance Close prices on the same basis. The 30 DMA includes the signal day's close. "
        "The current S&P 500 constituent list is scanned. Data is cached for five minutes."
    )


if __name__ == "__main__":
    main()
