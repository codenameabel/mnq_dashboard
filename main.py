from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse


app = FastAPI()

TICKER = "MNQ=F"
SESSION_TZ = ZoneInfo("America/New_York")

RETRACE_PERCENT = 0.10
STOP_POINTS = 30.0


def flatten_columns(data: pd.DataFrame) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


def get_data() -> pd.DataFrame:
    data = yf.download(
        TICKER,
        period="7d",
        interval="5m",
        progress=False,
        auto_adjust=False,
    )

    if data.empty:
        return data

    data = flatten_columns(data)
    data = data.dropna()

    if data.index.tz is None:
        data.index = data.index.tz_localize("UTC")

    data.index = data.index.tz_convert(SESSION_TZ)

    return data


def get_overnight_window(now: datetime) -> tuple[datetime, datetime, str]:
    market_open = time(9, 30)
    overnight_start = time(18, 0)

    if now.time() >= overnight_start:
        start = datetime.combine(now.date(), overnight_start, tzinfo=SESSION_TZ)
        end = now
        status = "Forming"

    elif now.time() < market_open:
        start_date = now.date() - timedelta(days=1)
        start = datetime.combine(start_date, overnight_start, tzinfo=SESSION_TZ)
        end = now
        status = "Forming"

    else:
        start_date = now.date() - timedelta(days=1)
        start = datetime.combine(start_date, overnight_start, tzinfo=SESSION_TZ)
        end = datetime.combine(now.date(), market_open, tzinfo=SESSION_TZ)
        status = "Locked"

    return start, end, status


def money_format(value):
    if value is None:
        return "None"
    return f"{value:,.2f}"


def calculate_dashboard() -> dict:
    data = get_data()

    if data.empty:
        raise ValueError("No market data returned from yfinance.")

    now = datetime.now(SESSION_TZ)
    overnight_start, overnight_end, session_status = get_overnight_window(now)

    overnight_data = data[
        (data.index >= overnight_start)
        & (data.index <= overnight_end)
    ]

    if overnight_data.empty:
        raise ValueError("No overnight data found for the selected session.")

    current_price = float(data["Close"].iloc[-1])
    overnight_high = float(overnight_data["High"].max())
    overnight_low = float(overnight_data["Low"].min())
    overnight_range = overnight_high - overnight_low
    retrace_points = overnight_range * RETRACE_PERCENT
    midpoint = (overnight_high + overnight_low) / 2

    bias = "Bullish" if current_price > midpoint else "Bearish"

    signal = "No Trade"
    setup = "Waiting"
    entry = None
    stop = None
    notes = "Overnight levels are still forming."

    if session_status == "Locked":
        rth_data = data[data.index > overnight_end]

        touched_high = rth_data[rth_data["High"] >= overnight_high]
        touched_low = rth_data[rth_data["Low"] <= overnight_low]

        first_high_time = touched_high.index[0] if not touched_high.empty else None
        first_low_time = touched_low.index[0] if not touched_low.empty else None

        if first_high_time is None and first_low_time is None:
            setup = "No level tested yet"
            notes = "Price has not tested the overnight high or overnight low."

        elif first_high_time is not None and (
            first_low_time is None or first_high_time < first_low_time
        ):
            setup = "Short setup"
            trigger_price = overnight_high - retrace_points

            if current_price <= trigger_price:
                signal = "SHORT"
                entry = trigger_price
                stop = overnight_high + STOP_POINTS
                notes = "Price tested the overnight high first, then retraced 10% of the overnight range."
            else:
                notes = "Overnight high tested first. Waiting for 10% retrace."

        elif first_low_time is not None:
            setup = "Long setup"
            trigger_price = overnight_low + retrace_points

            if current_price >= trigger_price:
                signal = "LONG"
                entry = trigger_price
                stop = overnight_low - STOP_POINTS
                notes = "Price tested the overnight low first, then retraced 10% of the overnight range."
            else:
                notes = "Overnight low tested first. Waiting for 10% retrace."

    risk_points = abs(entry - stop) if entry is not None and stop is not None else None

    return {
        "ticker": TICKER,
        "current_price": current_price,
        "overnight_high": overnight_high,
        "overnight_low": overnight_low,
        "overnight_range": overnight_range,
        "retrace_points": retrace_points,
        "midpoint": midpoint,
        "bias": bias,
        "session_status": session_status,
        "overnight_start": overnight_start.strftime("%Y-%m-%d %I:%M %p %Z"),
        "overnight_end": overnight_end.strftime("%Y-%m-%d %I:%M %p %Z"),
        "setup": setup,
        "signal": signal,
        "entry": entry,
        "stop": stop,
        "risk_points": risk_points,
        "notes": notes,
        "last_updated": now.strftime("%Y-%m-%d %I:%M:%S %p %Z"),
    }


@app.get("/api")
def api():
    try:
        return JSONResponse(calculate_dashboard())
    except Exception as error:
        return JSONResponse(
            {"error": str(error)},
            status_code=500,
        )


@app.get("/", response_class=HTMLResponse)
def home():
    try:
        data = calculate_dashboard()
    except Exception as error:
        return HTMLResponse(
            f"""
            <html>
                <body style="font-family: Arial; padding: 40px;">
                    <h1>MNQ Overnight Range Dashboard</h1>
                    <p>Error: {error}</p>
                </body>
            </html>
            """,
            status_code=500,
        )

    signal_class = "neutral"

    if data["signal"] == "LONG":
        signal_class = "long"

    if data["signal"] == "SHORT":
        signal_class = "short"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>MNQ Overnight Range Dashboard</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: #f4f6f8;
                margin: 0;
                padding: 30px;
                color: #111827;
            }}

            .container {{
                max-width: 1000px;
                margin: auto;
            }}

            .header {{
                margin-bottom: 25px;
            }}

            .header h1 {{
                margin-bottom: 6px;
            }}

            .grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 16px;
            }}

            .card {{
                background: white;
                border-radius: 14px;
                padding: 20px;
                box-shadow: 0 4px 14px rgba(0,0,0,0.08);
            }}

            .label {{
                color: #6b7280;
                font-size: 14px;
                margin-bottom: 8px;
            }}

            .value {{
                font-size: 26px;
                font-weight: bold;
            }}

            .signal {{
                font-size: 34px;
                font-weight: bold;
            }}

            .long {{
                color: #047857;
            }}

            .short {{
                color: #b91c1c;
            }}

            .neutral {{
                color: #374151;
            }}

            .wide {{
                grid-column: 1 / -1;
            }}

            .small {{
                font-size: 14px;
                color: #6b7280;
                line-height: 1.5;
            }}

            .footer {{
                margin-top: 24px;
                font-size: 13px;
                color: #6b7280;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>MNQ Overnight Range Dashboard</h1>
                <p>Educational dashboard tracking MNQ overnight levels and a 10% retrace model.</p>
            </div>

            <div class="grid">
                <div class="card">
                    <div class="label">Current Price</div>
                    <div class="value">{money_format(data["current_price"])}</div>
                </div>

                <div class="card">
                    <div class="label">Bias</div>
                    <div class="value">{data["bias"]}</div>
                </div>

                <div class="card">
                    <div class="label">Session Status</div>
                    <div class="value">{data["session_status"]}</div>
                </div>

                <div class="card">
                    <div class="label">Signal</div>
                    <div class="signal {signal_class}">{data["signal"]}</div>
                </div>

                <div class="card">
                    <div class="label">Overnight High</div>
                    <div class="value">{money_format(data["overnight_high"])}</div>
                </div>

                <div class="card">
                    <div class="label">Overnight Low</div>
                    <div class="value">{money_format(data["overnight_low"])}</div>
                </div>

                <div class="card">
                    <div class="label">Overnight Range</div>
                    <div class="value">{money_format(data["overnight_range"])}</div>
                </div>

                <div class="card">
                    <div class="label">10% Retrace</div>
                    <div class="value">{money_format(data["retrace_points"])}</div>
                </div>

                <div class="card">
                    <div class="label">Setup</div>
                    <div class="value">{data["setup"]}</div>
                </div>

                <div class="card">
                    <div class="label">Entry</div>
                    <div class="value">{money_format(data["entry"])}</div>
                </div>

                <div class="card">
                    <div class="label">Stop</div>
                    <div class="value">{money_format(data["stop"])}</div>
                </div>

                <div class="card">
                    <div class="label">Risk Points</div>
                    <div class="value">{money_format(data["risk_points"])}</div>
                </div>

                <div class="card wide">
                    <div class="label">Notes</div>
                    <p>{data["notes"]}</p>
                    <p class="small">
                        Overnight window: {data["overnight_start"]} to {data["overnight_end"]}
                    </p>
                    <p class="small">
                        Last updated: {data["last_updated"]}
                    </p>
                </div>
            </div>

            <div class="footer">
                This dashboard is for education and research only. It does not place trades and is not financial advice.
            </div>
        </div>
    </body>
    </html>
    """

    return HTMLResponse(html)


@app.get("/health")
def health():
    return {"status": "ok"}