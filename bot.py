#!/usr/bin/env python3
"""
bot.py - Binance USDT-M Futures signal bot (SIGNAL ONLY, never places orders).

What it does
------------
* Analyses 10 fixed USDT-perpetual symbols on closed candles (default 5m).
* Indicators (only): EMA 9/20/100, session VWAP (UTC day), RSI 14, ATR 14, volume.
* SHORT / LONG "double crossover" setups with RSI + volume filters.
* ATR-based entry trigger, stop-loss and 2R take-profit.
* Backtests the *same* functions (detect_setup + Trade.step) on historical data.
* Sends results and live alerts to Telegram.

Setup definition (identical for backtest and live; W = CROSS_WINDOW_CANDLES)
---------------------------------------------------------------------------
SHORT at closed candle i:
  1. state(i-W) == ABOVE : EMA9 and EMA20 both strictly above BOTH EMA100 and VWAP
  2. state(i)   == BELOW : EMA9 and EMA20 both strictly below BOTH EMA100 and VWAP
     -> every (EMA, reference) pair therefore crossed down inside the last W candles
  3. close(i) < min(EMA100(i), VWAP(i))
  4. RSI(i) in [SHORT_RSI_MIN, SHORT_RSI_MAX]       (default 25-45)
  5. volume(i) >= mean(volume of previous VOLUME_LOOKBACK closed candles) (default 20)
  6. no raw setup of the same side in candles i-W+1 .. i-1 (one alert per crossover)
LONG is the exact mirror (RSI default 55-75).

Trade model (identical for backtest and live tracking)
------------------------------------------------------
LONG : trigger = high(i) + TRIG*ATR ; stop = trigger - STOP*ATR ; R = trigger-stop ; TP = trigger + RR*R
SHORT: trigger = low(i)  - TRIG*ATR ; stop = trigger + STOP*ATR ; R = stop-trigger ; TP = trigger - RR*R
* Trigger is looked for ONLY in the next TRIGGER_EXPIRY_CANDLES closed candles (else EXPIRED).
* The trigger candle itself is also checked for stop/target. If a candle touches both stop and
  target, the STOP is counted first (conservative). Same for stop touched in the trigger candle.
* No stop/target within MAX_HOLDING_CANDLES -> TIME_EXPIRED: closed at that candle's close.
  It is reported as UNRESOLVED (never a win); its (mark-to-market) R is still part of net return.
* Fills: entry at trigger price, stop/target at their prices, with adverse slippage on both sides and
  taker fee on entry AND exit (percentages are per side). No gap modelling.
* Returns: fixed fractional risk, NOT compounded: each trade risks RISK_PER_TRADE_PCT of the
  initial equity, so return% = sum(R multiples after costs) * RISK_PER_TRADE_PCT.
  Drawdown is the peak-to-trough fall of that cumulative curve (trades ordered by exit time).
* Win rate = TP trades / (TP trades + SL trades) * 100 (resolved trades only).

Backtest results are historical simulations. They do not guarantee future performance.
"""
from __future__ import annotations

import asyncio
import html
import json
import logging
import math
import os
import random
import re
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Set, Tuple

import aiohttp

log = logging.getLogger("bot")

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
SYMBOLS: Tuple[str, ...] = (
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "LINKUSDT", "AVAXUSDT", "LTCUSDT",
)
TF_MS: Dict[str, int] = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000,
    "8h": 28_800_000, "12h": 43_200_000, "1d": 86_400_000,
}
LONG, SHORT = "LONG", "SHORT"
EMA_FAST, EMA_MID, EMA_SLOW = 9, 20, 100
DAY_MS = 86_400_000
NAN = float("nan")
TERMINAL = frozenset({"STOP", "TARGET", "TIME_EXPIRED", "EXPIRED"})
TG_LIMIT = 3900
DISCLAIMER = ("⚠️ Historical backtest results are simulations and do not guarantee future "
              "performance. Signals are informational only, not financial advice.")

# Row = (open_time_ms, open, high, low, close, volume, close_time_ms)
Row = Tuple[int, float, float, float, float, float, int]


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
class ConfigError(Exception):
    pass


class _Env:
    def __init__(self) -> None:
        self.errors: List[str] = []

    def get_str(self, name: str, default: str = "") -> str:
        v = os.getenv(name)
        return default if v is None or v.strip() == "" else v.strip()

    def get_int(self, name: str, default: int) -> int:
        raw = self.get_str(name, str(default))
        try:
            return int(raw)
        except ValueError:
            self.errors.append(f"{name} must be an integer (got {raw!r})")
            return default

    def get_float(self, name: str, default: float) -> float:
        raw = self.get_str(name, str(default))
        try:
            val = float(raw)
            if not math.isfinite(val):
                raise ValueError
            return val
        except ValueError:
            self.errors.append(f"{name} must be a finite number (got {raw!r})")
            return default

    def get_bool(self, name: str, default: bool) -> bool:
        raw = self.get_str(name, "true" if default else "false").lower()
        if raw in ("1", "true", "yes", "on"):
            return True
        if raw in ("0", "false", "no", "off"):
            return False
        self.errors.append(f"{name} must be true/false (got {raw!r})")
        return default


@dataclass(frozen=True)
class Config:
    telegram_token: str
    telegram_chat_id: str
    binance_api_key: str
    binance_api_secret: str
    base_url: str
    timeframe: str
    lookback_days: int
    scan_interval: float
    cross_window: int
    trigger_expiry: int
    atr_period: int
    atr_trigger_mult: float
    atr_stop_mult: float
    rr: float
    max_holding: int
    rsi_period: int
    volume_lookback: int
    short_rsi: Tuple[float, float]
    long_rsi: Tuple[float, float]
    fee_pct: float
    slippage_pct: float
    risk_pct: float
    allow_overlap: bool
    scan_candles: int
    min_samples: int
    refresh_hours: float
    cache_dir: str
    rate_budget: int
    log_level: str

    @property
    def tf_ms(self) -> int:
        return TF_MS[self.timeframe]

    @property
    def tf_minutes(self) -> float:
        return self.tf_ms / 60_000.0


def load_dotenv_file(path: str = ".env") -> None:
    """Optional local convenience: read KEY=VALUE lines from ./.env.
    Real environment variables (e.g. on Render) always win; a missing file is fine."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key.startswith("export "):
            key = key[7:].strip()
        if key and key not in os.environ:
            os.environ[key] = val


def load_config() -> Config:
    load_dotenv_file()
    e = _Env()
    token = e.get_str("TELEGRAM_BOT_TOKEN")
    chat = e.get_str("TELEGRAM_CHAT_ID")
    if not token:
        e.errors.append("TELEGRAM_BOT_TOKEN is required")
    if not chat:
        e.errors.append("TELEGRAM_CHAT_ID is required")
    cfg = Config(
        telegram_token=token,
        telegram_chat_id=chat,
        binance_api_key=e.get_str("BINANCE_API_KEY"),
        binance_api_secret=e.get_str("BINANCE_API_SECRET"),
        base_url=e.get_str("BINANCE_FUTURES_URL", "https://fapi.binance.com").rstrip("/"),
        timeframe=e.get_str("TIMEFRAME", "5m"),
        lookback_days=e.get_int("LOOKBACK_DAYS", 180),
        scan_interval=e.get_float("SCAN_INTERVAL_SECONDS", 15.0),
        cross_window=e.get_int("CROSS_WINDOW_CANDLES", 3),
        trigger_expiry=e.get_int("TRIGGER_EXPIRY_CANDLES", 6),
        atr_period=e.get_int("ATR_PERIOD", 14),
        atr_trigger_mult=e.get_float("ATR_TRIGGER_MULTIPLIER", 0.10),
        atr_stop_mult=e.get_float("ATR_STOP_MULTIPLIER", 1.50),
        rr=e.get_float("RISK_REWARD_RATIO", 2.0),
        max_holding=e.get_int("MAX_HOLDING_CANDLES", 48),
        rsi_period=e.get_int("RSI_PERIOD", 14),
        volume_lookback=e.get_int("VOLUME_LOOKBACK", 20),
        short_rsi=(e.get_float("SHORT_RSI_MIN", 25.0), e.get_float("SHORT_RSI_MAX", 45.0)),
        long_rsi=(e.get_float("LONG_RSI_MIN", 55.0), e.get_float("LONG_RSI_MAX", 75.0)),
        fee_pct=e.get_float("FEE_PCT_PER_SIDE", 0.05),
        slippage_pct=e.get_float("SLIPPAGE_PCT_PER_SIDE", 0.02),
        risk_pct=e.get_float("RISK_PER_TRADE_PCT", 1.0),
        allow_overlap=e.get_bool("ALLOW_OVERLAP", False),
        scan_candles=e.get_int("SCAN_CANDLES", 600),
        min_samples=e.get_int("MIN_HISTORICAL_SAMPLES", 30),
        refresh_hours=e.get_float("BACKTEST_REFRESH_HOURS", 24.0),
        cache_dir=e.get_str("CACHE_DIR", "./cache"),
        rate_budget=e.get_int("RATE_LIMIT_BUDGET_PER_MIN", 1800),
        log_level=e.get_str("LOG_LEVEL", "INFO").upper(),
    )
    err = e.errors
    if cfg.timeframe not in TF_MS:
        err.append(f"TIMEFRAME must be one of {sorted(TF_MS)}")
    for name, val, lo in (
        ("LOOKBACK_DAYS", cfg.lookback_days, 1), ("CROSS_WINDOW_CANDLES", cfg.cross_window, 1),
        ("TRIGGER_EXPIRY_CANDLES", cfg.trigger_expiry, 1), ("MAX_HOLDING_CANDLES", cfg.max_holding, 1),
        ("ATR_PERIOD", cfg.atr_period, 2), ("RSI_PERIOD", cfg.rsi_period, 2),
        ("VOLUME_LOOKBACK", cfg.volume_lookback, 2), ("SCAN_CANDLES", cfg.scan_candles, 300),
    ):
        if val < lo:
            err.append(f"{name} must be >= {lo}")
    if cfg.scan_candles > 1500:
        err.append("SCAN_CANDLES must be <= 1500")
    if cfg.scan_interval < 1:
        err.append("SCAN_INTERVAL_SECONDS must be >= 1")
    if cfg.atr_stop_mult <= 0 or cfg.rr <= 0 or cfg.atr_trigger_mult < 0:
        err.append("ATR_STOP_MULTIPLIER and RISK_REWARD_RATIO must be > 0; ATR_TRIGGER_MULTIPLIER >= 0")
    for name, (lo, hi) in (("SHORT_RSI", cfg.short_rsi), ("LONG_RSI", cfg.long_rsi)):
        if not (0 <= lo < hi <= 100):
            err.append(f"{name}_MIN/MAX must satisfy 0 <= MIN < MAX <= 100")
    if cfg.fee_pct < 0 or cfg.slippage_pct < 0 or cfg.risk_pct <= 0:
        err.append("FEE/SLIPPAGE must be >= 0 and RISK_PER_TRADE_PCT > 0")
    if cfg.rate_budget < 200 or cfg.rate_budget > 2400:
        err.append("RATE_LIMIT_BUDGET_PER_MIN must be between 200 and 2400")
    if err:
        raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(err))
    return cfg


def setup_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)sZ | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fmt.converter = time.gmtime  # UTC timestamps
    handler.setFormatter(fmt)
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, level, logging.INFO))
    logging.getLogger("aiohttp").setLevel(logging.WARNING)


# --------------------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------------------
def esc(x: Any) -> str:
    return html.escape(str(x), quote=False)


def fmt_ts(ms: Optional[int]) -> str:
    if ms is None:
        return "n/a"
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def fmt_num(x: Optional[float], places: int) -> str:
    if x is None or not math.isfinite(x):
        return "n/a"
    q = Decimal(1).scaleb(-places)
    return f"{Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP):f}"


def fpct(x: Optional[float], signed: bool = False) -> str:
    if x is None:
        return "n/a"
    return f"{x:+.1f}%" if signed else f"{x:.1f}%"


def fr(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:+.2f}R"


def places_from_tick(tick: str) -> int:
    try:
        exp = Decimal(tick).normalize().as_tuple().exponent
        return max(0, -int(exp))
    except Exception:
        return 4


# --------------------------------------------------------------------------------------
# Indicators (EMA, session VWAP, Wilder RSI, Wilder ATR, rolling volume mean)
# --------------------------------------------------------------------------------------
def ema(values: Sequence[float], period: int) -> List[float]:
    """EMA seeded with the SMA of the first `period` values (NaN before that)."""
    n = len(values)
    out = [NAN] * n
    if n < period:
        return out
    k = 2.0 / (period + 1)
    prev = sum(values[:period]) / period
    out[period - 1] = prev
    for i in range(period, n):
        prev = values[i] * k + prev * (1.0 - k)
        out[i] = prev
    return out


def session_vwap(t: Sequence[int], h: Sequence[float], l: Sequence[float],
                 c: Sequence[float], v: Sequence[float]) -> List[float]:
    """Session VWAP reset at 00:00 UTC; typical price = (H+L+C)/3."""
    out: List[float] = []
    day: Optional[int] = None
    cum_pv = cum_v = 0.0
    for i in range(len(t)):
        d = t[i] // DAY_MS
        if d != day:
            day, cum_pv, cum_v = d, 0.0, 0.0
        tp = (h[i] + l[i] + c[i]) / 3.0
        cum_pv += tp * v[i]
        cum_v += v[i]
        out.append(cum_pv / cum_v if cum_v > 0 else tp)
    return out


def _rsi_value(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def rsi_wilder(c: Sequence[float], period: int) -> List[float]:
    n = len(c)
    out = [NAN] * n
    if n <= period:
        return out
    gains = losses = 0.0
    for i in range(1, period + 1):
        d = c[i] - c[i - 1]
        gains += max(d, 0.0)
        losses += max(-d, 0.0)
    ag, al = gains / period, losses / period
    out[period] = _rsi_value(ag, al)
    for i in range(period + 1, n):
        d = c[i] - c[i - 1]
        ag = (ag * (period - 1) + max(d, 0.0)) / period
        al = (al * (period - 1) + max(-d, 0.0)) / period
        out[i] = _rsi_value(ag, al)
    return out


def atr_wilder(h: Sequence[float], l: Sequence[float], c: Sequence[float], period: int) -> List[float]:
    n = len(c)
    out = [NAN] * n
    if n <= period:
        return out
    tr = [h[0] - l[0]] + [max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])) for i in range(1, n)]
    prev = sum(tr[1:period + 1]) / period
    out[period] = prev
    for i in range(period + 1, n):
        prev = (prev * (period - 1) + tr[i]) / period
        out[i] = prev
    return out


def rolling_prev_mean(values: Sequence[float], n: int) -> List[float]:
    """out[i] = mean(values[i-n .. i-1]) -> the previous n completed candles, excluding i."""
    out = [NAN] * len(values)
    s = 0.0
    for i, x in enumerate(values):
        if i >= n:
            out[i] = s / n
        s += x
        if i >= n:
            s -= values[i - n]
    return out


@dataclass
class Indicators:
    t: List[int]
    o: List[float]
    h: List[float]
    l: List[float]
    c: List[float]
    v: List[float]
    ct: List[int]
    ema9: List[float]
    ema20: List[float]
    ema100: List[float]
    vwap: List[float]
    rsi: List[float]
    atr: List[float]
    vol_avg: List[float]
    min_index: int

    @property
    def n(self) -> int:
        return len(self.t)


def compute_indicators(rows: Sequence[Row], cfg: Config) -> Indicators:
    t = [r[0] for r in rows]
    o = [r[1] for r in rows]
    h = [r[2] for r in rows]
    l = [r[3] for r in rows]
    c = [r[4] for r in rows]
    v = [r[5] for r in rows]
    ct = [r[6] for r in rows]
    return Indicators(
        t=t, o=o, h=h, l=l, c=c, v=v, ct=ct,
        ema9=ema(c, EMA_FAST), ema20=ema(c, EMA_MID), ema100=ema(c, EMA_SLOW),
        vwap=session_vwap(t, h, l, c, v),
        rsi=rsi_wilder(c, cfg.rsi_period), atr=atr_wilder(h, l, c, cfg.atr_period),
        vol_avg=rolling_prev_mean(v, cfg.volume_lookback),
        min_index=max(EMA_SLOW - 1 + cfg.cross_window, cfg.atr_period, cfg.rsi_period, cfg.volume_lookback),
    )


# --------------------------------------------------------------------------------------
# Setup detection (shared by backtest and live)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Setup:
    side: str
    index: int
    open_time: int
    close_time: int
    close: float
    high: float
    low: float
    ema9: float
    ema20: float
    ema100: float
    vwap: float
    rsi: float
    atr: float
    volume: float
    vol_avg: float
    entry: float
    stop: float
    take_profit: float
    risk: float


def _state(ind: Indicators, i: int) -> int:
    """+1: EMA9 & EMA20 both strictly above BOTH EMA100 and VWAP; -1: both strictly below; else 0."""
    if i < 0 or i >= ind.n:
        return 0
    vals = (ind.ema9[i], ind.ema20[i], ind.ema100[i], ind.vwap[i])
    if any(math.isnan(x) for x in vals):
        return 0
    e9, e20, s, w = vals
    if min(e9, e20) > max(s, w):
        return 1
    if max(e9, e20) < min(s, w):
        return -1
    return 0


def _raw_setup(ind: Indicators, i: int, side: str, cfg: Config) -> bool:
    if i < ind.min_index or i >= ind.n:
        return False
    pre = 1 if side == SHORT else -1
    if _state(ind, i - cfg.cross_window) != pre or _state(ind, i) != -pre:
        return False
    close, s, w = ind.c[i], ind.ema100[i], ind.vwap[i]
    if side == SHORT and not close < min(s, w):
        return False
    if side == LONG and not close > max(s, w):
        return False
    lo, hi = cfg.short_rsi if side == SHORT else cfg.long_rsi
    r = ind.rsi[i]
    if math.isnan(r) or not (lo <= r <= hi):
        return False
    va, atr = ind.vol_avg[i], ind.atr[i]
    if math.isnan(va) or ind.v[i] < va:
        return False
    return not math.isnan(atr) and atr > 0


def detect_setup(ind: Indicators, i: int, side: str, cfg: Config) -> Optional[Setup]:
    if not _raw_setup(ind, i, side, cfg):
        return None
    for k in range(i - cfg.cross_window + 1, i):  # one alert per crossover episode
        if _raw_setup(ind, k, side, cfg):
            return None
    atr = ind.atr[i]
    if side == LONG:
        entry = ind.h[i] + cfg.atr_trigger_mult * atr
        stop = entry - cfg.atr_stop_mult * atr
        risk = entry - stop
        tp = entry + cfg.rr * risk
    else:
        entry = ind.l[i] - cfg.atr_trigger_mult * atr
        stop = entry + cfg.atr_stop_mult * atr
        risk = stop - entry
        tp = entry - cfg.rr * risk
    if stop <= 0 or tp <= 0 or risk <= 0:
        return None
    return Setup(side, i, ind.t[i], ind.ct[i], ind.c[i], ind.h[i], ind.l[i], ind.ema9[i], ind.ema20[i],
                 ind.ema100[i], ind.vwap[i], ind.rsi[i], atr, ind.v[i], ind.vol_avg[i], entry, stop, tp, risk)


# --------------------------------------------------------------------------------------
# Trade lifecycle (shared by backtest and live tracking)
# --------------------------------------------------------------------------------------
@dataclass
class Trade:
    symbol: str
    setup: Setup
    state: str = "WAITING"
    waited: int = 0
    held: int = 0
    trigger_time: Optional[int] = None
    exit_time: Optional[int] = None
    entry_fill: float = NAN
    exit_fill: float = NAN
    r_multiple: Optional[float] = None
    silent: bool = False

    def step(self, row: Row, cfg: Config) -> List[str]:
        """Feed ONE closed candle that occurred after the signal candle. Returns events."""
        if self.state in TERMINAL:
            return []
        t, _o, h, l, c, _v, _ct = row
        s = self.setup
        d = 1 if s.side == LONG else -1
        events: List[str] = []
        if self.state == "WAITING":
            self.waited += 1
            hit = h >= s.entry if d == 1 else l <= s.entry
            if hit:
                self.state = "ACTIVE"
                self.trigger_time = t
                self.entry_fill = s.entry * (1 + d * cfg.slippage_pct / 100.0)
                events.append("TRIGGER_HIT")
            elif self.waited >= cfg.trigger_expiry:
                self.state = "EXPIRED"
                return ["EXPIRED"]
            else:
                return events
        self.held += 1  # ACTIVE (includes the trigger candle itself)
        stop_hit = l <= s.stop if d == 1 else h >= s.stop
        tp_hit = h >= s.take_profit if d == 1 else l <= s.take_profit
        if stop_hit:  # conservative: stop first when both touched
            self._close("STOP", s.stop, t, cfg)
        elif tp_hit:
            self._close("TARGET", s.take_profit, t, cfg)
        elif self.held >= cfg.max_holding:
            self._close("TIME_EXPIRED", c, t, cfg)
        if self.state in TERMINAL:
            events.append(self.state)
        return events

    def _close(self, state: str, price: float, t: int, cfg: Config) -> None:
        d = 1 if self.setup.side == LONG else -1
        self.state = state
        self.exit_time = t
        self.exit_fill = price * (1 - d * cfg.slippage_pct / 100.0)
        pnl = (self.exit_fill - self.entry_fill) * d
        fees = (cfg.fee_pct / 100.0) * (self.entry_fill + self.exit_fill)
        self.r_multiple = (pnl - fees) / self.setup.risk


# --------------------------------------------------------------------------------------
# Backtesting + statistics
# --------------------------------------------------------------------------------------
@dataclass
class Record:
    side: str
    status: str  # EXPIRED STOP TARGET TIME_EXPIRED OPEN PENDING
    setup_time: int
    trigger_time: Optional[int] = None
    exit_time: Optional[int] = None
    r: Optional[float] = None
    hold_min: Optional[float] = None


@dataclass
class Stats:
    setups: int = 0
    skipped: int = 0
    pending: int = 0
    not_triggered: int = 0
    triggered: int = 0
    wins: int = 0
    losses: int = 0
    time_expired: int = 0
    open_at_end: int = 0
    resolved: int = 0
    unresolved: int = 0
    win_rate: Optional[float] = None
    trigger_rate: Optional[float] = None
    stop_rate: Optional[float] = None
    target_rate: Optional[float] = None
    avg_r: Optional[float] = None
    net_r: float = 0.0
    net_return_pct: float = 0.0
    max_dd_r: float = 0.0
    max_dd_pct: float = 0.0
    avg_hold_min: Optional[float] = None


@dataclass
class Report:
    generated_at: int
    by_key: Dict[Tuple[str, str], Stats]
    by_symbol: Dict[str, Stats]
    by_side: Dict[str, Stats]
    overall: Stats
    meta: Dict[str, Dict[str, Any]]


def max_drawdown(rs: Sequence[float]) -> float:
    cum = peak = dd = 0.0
    for r in rs:
        cum += r
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    return dd


def compute_stats(records: Sequence[Record], skipped: int, cfg: Config) -> Stats:
    s = Stats(skipped=skipped, setups=len(records))
    cnt: Dict[str, int] = {}
    for r in records:
        cnt[r.status] = cnt.get(r.status, 0) + 1
    s.pending, s.not_triggered = cnt.get("PENDING", 0), cnt.get("EXPIRED", 0)
    s.wins, s.losses = cnt.get("TARGET", 0), cnt.get("STOP", 0)
    s.time_expired, s.open_at_end = cnt.get("TIME_EXPIRED", 0), cnt.get("OPEN", 0)
    s.triggered = s.wins + s.losses + s.time_expired + s.open_at_end
    s.resolved = s.wins + s.losses
    s.unresolved = s.time_expired + s.open_at_end
    evaluated = s.setups - s.pending
    s.win_rate = s.wins / s.resolved * 100 if s.resolved else None
    s.trigger_rate = s.triggered / evaluated * 100 if evaluated else None
    s.stop_rate = s.losses / s.triggered * 100 if s.triggered else None
    s.target_rate = s.wins / s.triggered * 100 if s.triggered else None
    closed = sorted((r for r in records if r.status in ("STOP", "TARGET", "TIME_EXPIRED")),
                    key=lambda r: r.exit_time or 0)
    rs = [r.r for r in closed if r.r is not None]
    if rs:
        s.avg_r = sum(rs) / len(rs)
        s.net_r = sum(rs)
        s.net_return_pct = s.net_r * cfg.risk_pct
        s.max_dd_r = max_drawdown(rs)
        s.max_dd_pct = s.max_dd_r * cfg.risk_pct
    holds = [r.hold_min for r in closed if r.hold_min is not None]
    s.avg_hold_min = sum(holds) / len(holds) if holds else None
    return s


def backtest_symbol(symbol: str, rows: Sequence[Row], cfg: Config) -> Tuple[List[Record], Dict[str, int]]:
    """Chronological simulation using the exact live functions (detect_setup / Trade.step)."""
    records: List[Record] = []
    skipped = {LONG: 0, SHORT: 0}
    if len(rows) < 200:
        return records, skipped
    ind = compute_indicators(rows, cfg)
    n = ind.n
    busy_until = -1  # index of the last candle consumed by the previous trade of this symbol
    for i in range(ind.min_index, n):
        setup: Optional[Setup] = None
        for side in (LONG, SHORT):
            setup = detect_setup(ind, i, side, cfg)
            if setup:
                break
        if setup is None:
            continue
        if not cfg.allow_overlap and i < busy_until:
            skipped[setup.side] += 1
            continue
        trade = Trade(symbol=symbol, setup=setup)
        j = i + 1
        while j < n and trade.state not in TERMINAL:
            trade.step(rows[j], cfg)
            j += 1
        busy_until = j - 1 if trade.state in TERMINAL else n
        if trade.state in TERMINAL:
            status = trade.state
        elif trade.state == "WAITING":
            status = "PENDING"  # trigger window not finished before data ended
        else:
            status = "OPEN"  # triggered, no outcome before data ended
        rec = Record(side=setup.side, status=status, setup_time=setup.open_time,
                     trigger_time=trade.trigger_time, exit_time=trade.exit_time, r=trade.r_multiple)
        if trade.exit_time is not None and trade.trigger_time is not None:
            rec.hold_min = (trade.exit_time - trade.trigger_time) / 60_000.0 + cfg.tf_minutes
        records.append(rec)
    return records, skipped


def run_backtest(rows_by_symbol: Dict[str, List[Row]], cfg: Config) -> Report:
    all_rec: List[Record] = []
    by_key: Dict[Tuple[str, str], Stats] = {}
    by_symbol: Dict[str, Stats] = {}
    meta: Dict[str, Dict[str, Any]] = {}
    side_rec: Dict[str, List[Record]] = {LONG: [], SHORT: []}
    side_skip = {LONG: 0, SHORT: 0}
    for sym, rows in rows_by_symbol.items():
        recs, skipped = backtest_symbol(sym, rows, cfg)
        for side in (LONG, SHORT):
            sr = [r for r in recs if r.side == side]
            by_key[(sym, side)] = compute_stats(sr, skipped[side], cfg)
            side_rec[side].extend(sr)
            side_skip[side] += skipped[side]
        by_symbol[sym] = compute_stats(recs, skipped[LONG] + skipped[SHORT], cfg)
        all_rec.extend(recs)
        span = ((rows[-1][0] - rows[0][0]) / DAY_MS) if len(rows) > 1 else 0.0
        meta[sym] = {"candles": len(rows), "first": rows[0][0] if rows else None,
                     "last": rows[-1][0] if rows else None, "span_days": span}
    by_side = {s: compute_stats(side_rec[s], side_skip[s], cfg) for s in (LONG, SHORT)}
    overall = compute_stats(all_rec, side_skip[LONG] + side_skip[SHORT], cfg)
    return Report(int(time.time() * 1000), by_key, by_symbol, by_side, overall, meta)


# --------------------------------------------------------------------------------------
# Telegram
# --------------------------------------------------------------------------------------
def chunk_text(text: str, limit: int = TG_LIMIT) -> List[str]:
    chunks: List[str] = []
    cur = ""
    for line in text.split("\n"):
        while len(line) > limit:  # pathological single long line
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(line[:limit])
            line = line[limit:]
        if cur and len(cur) + 1 + len(line) > limit:
            chunks.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        chunks.append(cur)
    return chunks or [""]


class Telegram:
    def __init__(self, session: aiohttp.ClientSession, token: str, chat_id: str) -> None:
        self.session = session
        self.token = token
        self.chat_id = chat_id
        self._lock = asyncio.Lock()
        self._err_last: Dict[str, float] = {}

    def _scrub(self, s: str) -> str:
        return s.replace(self.token, "***") if self.token else s

    async def send(self, text: str) -> bool:
        ok = True
        async with self._lock:
            for part in chunk_text(text):
                ok = await self._post(part) and ok
                await asyncio.sleep(0.4)  # stay well under Telegram's per-chat rate limit
        return ok

    async def send_blocks(self, blocks: Sequence[str]) -> bool:
        msgs: List[str] = []
        cur = ""
        for b in blocks:
            if cur and len(cur) + 2 + len(b) > TG_LIMIT:
                msgs.append(cur)
                cur = b
            else:
                cur = f"{cur}\n\n{b}" if cur else b
        if cur:
            msgs.append(cur)
        ok = True
        for m in msgs:
            ok = await self.send(m) and ok
        return ok

    async def notify_error(self, key: str, text: str, cooldown: float = 1800.0) -> None:
        now = time.monotonic()
        if now - self._err_last.get(key, -1e9) < cooldown:
            return
        self._err_last[key] = now
        await self.send(f"🚨 <b>Bot warning</b>\n{esc(text)}")

    async def _post(self, text: str) -> bool:
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload: Dict[str, Any] = {"chat_id": self.chat_id, "text": text, "parse_mode": "HTML",
                                   "disable_web_page_preview": True}
        for attempt in range(1, 5):
            try:
                async with self.session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=20)) as r:
                    data = await r.json(content_type=None)
                    if r.status == 200 and isinstance(data, dict) and data.get("ok"):
                        return True
                    desc = str(data.get("description", "")) if isinstance(data, dict) else ""
                    if r.status == 429:
                        wait = float((data.get("parameters") or {}).get("retry_after", 5))
                        log.warning("Telegram rate limit, sleeping %.0fs", wait)
                        await asyncio.sleep(wait + 1)
                        continue
                    if r.status == 400 and "parse" in desc.lower() and "parse_mode" in payload:
                        payload.pop("parse_mode")  # fall back to plain text
                        payload["text"] = html.unescape(re.sub(r"<[^>]+>", "", text))
                        continue
                    if r.status in (400, 401, 403, 404):
                        log.error("Telegram rejected message (%s): %s", r.status, self._scrub(desc))
                        return False
                    raise aiohttp.ClientError(f"HTTP {r.status}")
            except asyncio.CancelledError:
                raise
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
                log.warning("Telegram send failed (attempt %d/4): %s", attempt, self._scrub(repr(exc)))
                await asyncio.sleep(min(30.0, 2.0 ** attempt))
        return False


# --------------------------------------------------------------------------------------
# Binance public market-data client
# --------------------------------------------------------------------------------------
class BinanceError(Exception):
    pass


class TransientError(BinanceError):
    pass


class FatalError(BinanceError):
    pass


class BanError(FatalError):
    """Binance banned/rate-limited this server IP for a long time (HTTP 418/429, long Retry-After)."""

    def __init__(self, msg: str, wait: float) -> None:
        super().__init__(msg)
        self.wait = wait


def parse_kline(r: Any) -> Optional[Row]:
    try:
        t, ct = int(r[0]), int(r[6])
        o, h, l, c, v = (float(r[i]) for i in range(1, 6))
    except (TypeError, ValueError, IndexError):
        return None
    if not all(math.isfinite(x) for x in (o, h, l, c, v)) or min(o, h, l, c) <= 0 or v < 0 or h < l:
        return None
    return (t, o, h, l, c, v, ct)


def kline_weight(limit: int) -> int:
    return 1 if limit < 100 else 2 if limit < 500 else 5 if limit <= 1000 else 10


class BinanceClient:
    def __init__(self, session: aiohttp.ClientSession, base_url: str, budget_per_min: int) -> None:
        self.session = session
        self.base = base_url
        self.budget = budget_per_min
        self._window: List[Tuple[float, int]] = []
        self._lock = asyncio.Lock()
        self._cooldown_until = 0.0

    async def _throttle(self, weight: int) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                if now < self._cooldown_until:
                    await asyncio.sleep(self._cooldown_until - now)
                    continue
                self._window = [(ts, w) for ts, w in self._window if now - ts < 60.0]
                if sum(w for _, w in self._window) + weight <= self.budget:
                    self._window.append((now, weight))
                    return
                await asyncio.sleep(max(0.2, 60.0 - (now - self._window[0][0]) + 0.05))

    async def _get(self, path: str, params: Optional[Dict[str, Any]] = None, weight: int = 1) -> Any:
        url = self.base + path
        last: Optional[BaseException] = None
        for attempt in range(1, 7):
            await self._throttle(weight)
            try:
                async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=20)) as r:
                    if r.status == 200:
                        return await r.json(content_type=None)
                    body = (await r.text())[:200]
                    if r.status in (418, 429):
                        wait = float(r.headers.get("Retry-After", "30" if r.status == 429 else "120"))
                        self._cooldown_until = time.monotonic() + wait
                        if wait > 900:  # do not sleep for hours: fail fast with a clear message
                            until = fmt_ts(int((time.time() + wait) * 1000))
                            raise BanError(f"Binance has banned this server's IP (HTTP {r.status}) until about "
                                           f"{until}. Redeploy the worker in another region "
                                           f"(Frankfurt/Singapore) or wait for the ban to expire.", wait)
                        raise TransientError(f"HTTP {r.status} rate limited, retry after {wait:.0f}s")
                    if r.status == 451:
                        raise FatalError("HTTP 451: Binance blocks this server location. "
                                         "Deploy the worker in a non-US region (e.g. Frankfurt/Singapore).")
                    if r.status >= 500:
                        raise TransientError(f"HTTP {r.status}")
                    raise BinanceError(f"HTTP {r.status} on {path}: {body}")
            except (FatalError, asyncio.CancelledError):
                raise
            except BinanceError as exc:
                if not isinstance(exc, TransientError):
                    raise
                last = exc
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
                last = exc
            delay = min(60.0, 2.0 ** attempt) + random.random()
            log.warning("Binance %s failed (attempt %d/6): %s; retry in %.1fs", path, attempt, last, delay)
            await asyncio.sleep(delay)
        raise BinanceError(f"{path} failed after retries: {last}")

    async def server_time(self) -> int:
        data = await self._get("/fapi/v1/time")
        try:
            return int(data["serverTime"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BinanceError(f"Malformed /time response: {data!r}") from exc

    async def exchange_info(self) -> Dict[str, Any]:
        data = await self._get("/fapi/v1/exchangeInfo")
        if not isinstance(data, dict) or "symbols" not in data:
            raise BinanceError("Malformed exchangeInfo response")
        return data

    async def klines(self, symbol: str, interval: str, start: Optional[int] = None,
                     end: Optional[int] = None, limit: int = 1000) -> Tuple[List[Row], int]:
        params: Dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": limit}
        if start is not None:
            params["startTime"] = start
        if end is not None:
            params["endTime"] = end
        data = await self._get("/fapi/v1/klines", params, kline_weight(limit))
        if not isinstance(data, list):
            raise BinanceError(f"Malformed klines response for {symbol}: {str(data)[:120]}")
        rows = [x for x in (parse_kline(r) for r in data) if x is not None]
        if len(rows) != len(data):
            log.warning("%s: dropped %d malformed kline rows", symbol, len(data) - len(rows))
        return rows, len(data)

    async def download_range(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> List[Row]:
        """Paginated download of klines with open_time in [start_ms, end_ms]."""
        tf = TF_MS[interval]
        limit = 1000
        out: Dict[int, Row] = {}
        cursor = start_ms
        while cursor <= end_ms:
            rows, raw_n = await self.klines(symbol, interval, cursor, end_ms, limit)
            if raw_n == 0:
                break
            for r in rows:
                out[r[0]] = r
            if not rows or rows[-1][0] < cursor:
                break
            cursor = rows[-1][0] + tf
            if raw_n < limit:
                break
        return [out[k] for k in sorted(out)]


# --------------------------------------------------------------------------------------
# Historical data store (disk cache + incremental update)
# --------------------------------------------------------------------------------------
class HistoryStore:
    def __init__(self, cfg: Config, client: BinanceClient) -> None:
        self.cfg = cfg
        self.client = client
        self.data: Dict[str, List[Row]] = {}

    def _path(self, symbol: str) -> Path:
        return Path(self.cfg.cache_dir) / f"{symbol}_{self.cfg.timeframe}.json"

    def _load(self, symbol: str) -> List[Row]:
        try:
            raw = json.loads(self._path(symbol).read_text())
            rows = [(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), int(r[6]))
                    for r in raw]
            return sorted(rows)
        except (OSError, ValueError, TypeError, IndexError):
            return []

    def _save(self, symbol: str, rows: List[Row]) -> None:
        p = self._path(symbol)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, separators=(",", ":")))
        os.replace(tmp, p)

    async def refresh(self, symbol: str, now_ms: int) -> List[Row]:
        cfg, tf = self.cfg, self.cfg.tf_ms
        start = now_ms - cfg.lookback_days * DAY_MS
        start -= start % tf
        rows = self.data.get(symbol)
        if rows is None:
            rows = await asyncio.to_thread(self._load, symbol)
        if rows and rows[0][0] <= start + 2 * tf:
            fetch_from = rows[-1][0] + tf  # incremental update
        else:
            rows, fetch_from = [], start  # no/insufficient cache -> full download
        fetched: List[Row] = []
        if fetch_from < now_ms:
            fetched = await self.client.download_range(symbol, cfg.timeframe, fetch_from, now_ms)
        merged = {r[0]: r for r in rows}
        merged.update({r[0]: r for r in fetched})
        out = [merged[k] for k in sorted(merged) if k >= start and merged[k][6] < now_ms]  # closed only
        expected = cfg.lookback_days * DAY_MS / tf
        if len(out) < 0.95 * expected:
            log.warning("%s: only %d of ~%d expected candles available", symbol, len(out), int(expected))
        try:
            await asyncio.to_thread(self._save, symbol, out)
        except OSError as exc:
            log.warning("%s: cache write failed: %s", symbol, exc)
        self.data[symbol] = out
        return out


# --------------------------------------------------------------------------------------
# Message formatting
# --------------------------------------------------------------------------------------
def stats_block(label: str, s: Stats, cfg: Config, span_days: Optional[float] = None) -> str:
    freq = ""
    if span_days and span_days > 1:
        freq = f" (~{(s.setups + s.skipped) / span_days * 30:.1f}/30d)"
    return "\n".join([
        f"<b>{esc(label)}</b>",
        f"Setups {s.setups}{freq} (+{s.skipped} skipped: overlap) · pending {s.pending}",
        f"Triggered {s.triggered} (trigger rate {fpct(s.trigger_rate)}) · not triggered {s.not_triggered}",
        f"Resolved {s.resolved}: {s.wins} TP / {s.losses} SL · unresolved {s.unresolved} "
        f"({s.time_expired} time-exp, {s.open_at_end} open at end)",
        f"Win rate {fpct(s.win_rate)} of {s.resolved} resolved · TP rate {fpct(s.target_rate)} · "
        f"SL rate {fpct(s.stop_rate)}",
        f"Avg {fr(s.avg_r)} · net {fpct(s.net_return_pct, True)} ({s.net_r:+.1f}R) · "
        f"max DD {s.max_dd_r:.1f}R ({s.max_dd_pct:.1f}%) · avg hold "
        f"{'n/a' if s.avg_hold_min is None else f'{s.avg_hold_min:.0f} min'}",
    ])


def format_backtest_blocks(report: Report, cfg: Config, title: str) -> List[str]:
    firsts = [m["first"] for m in report.meta.values() if m["first"]]
    lasts = [m["last"] for m in report.meta.values() if m["last"]]
    blocks = [
        f"📊 <b>{esc(title)}</b>\n<i>HISTORICAL SIMULATION — not live performance</i>\n"
        f"Data: {fmt_ts(min(firsts) if firsts else None)} → {fmt_ts(max(lasts) if lasts else None)}\n"
        f"Timeframe {esc(cfg.timeframe)} · lookback {cfg.lookback_days}d · {len(report.meta)} symbols\n"
        f"Rules: cross window {cfg.cross_window}c · trigger expiry {cfg.trigger_expiry}c · "
        f"trigger {cfg.atr_trigger_mult}×ATR · stop {cfg.atr_stop_mult}×ATR · TP {cfg.rr}R · "
        f"max hold {cfg.max_holding}c\n"
        f"Costs: fee {cfg.fee_pct}% per side (entry AND exit) + slippage {cfg.slippage_pct}% per side\n"
        f"Returns: fixed risk {cfg.risk_pct}% of initial equity per trade, non-compounded. "
        f"Win rate = TP / (TP+SL); time-expired trades are unresolved (never wins) but their "
        f"mark-to-market R is included in net return.",
        stats_block("🌐 PORTFOLIO (all symbols, both directions)", report.overall, cfg),
        stats_block("🟢 LONG — all symbols", report.by_side[LONG], cfg),
        stats_block("🔴 SHORT — all symbols", report.by_side[SHORT], cfg),
    ]
    for sym in report.by_symbol:
        span = report.meta[sym]["span_days"]
        blocks.append(stats_block(f"{sym} LONG", report.by_key[(sym, LONG)], cfg, span))
        blocks.append(stats_block(f"{sym} SHORT", report.by_key[(sym, SHORT)], cfg, span))
    blocks.append(DISCLAIMER)
    return blocks


def hist_text(report: Optional[Report], symbol: str, side: str, cfg: Config) -> str:
    if report is None:
        return "Historical stats: not available yet."
    s = report.by_key.get((symbol, side))
    if s is None or s.setups == 0:
        return f"Historical ({cfg.lookback_days}d backtest): no comparable setups — insufficient data."
    lines = [
        f"Historical sample ({cfg.lookback_days}d backtest): {s.setups} setups, {s.triggered} triggered "
        f"(trigger rate {fpct(s.trigger_rate)}), {s.resolved} resolved ({s.wins} TP / {s.losses} SL), "
        f"{s.unresolved} unresolved",
        f"Historical win rate (resolved only): {fpct(s.win_rate)}",
    ]
    if s.resolved < cfg.min_samples:
        lines.append(f"⚠️ INSUFFICIENT SAMPLE: only {s.resolved} resolved trades "
                     f"(< {cfg.min_samples}). Treat these statistics as unreliable.")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Live scanner
# --------------------------------------------------------------------------------------
@dataclass
class SymbolState:
    symbol: str
    places: int = 4
    last_processed: int = 0
    trades: List[Trade] = field(default_factory=list)
    alerted: Set[Tuple[str, int]] = field(default_factory=set)
    failures: int = 0


class Scanner:
    def __init__(self, cfg: Config, client: BinanceClient, tg: Telegram,
                 states: Dict[str, SymbolState], get_report: Callable[[], Optional[Report]]) -> None:
        self.cfg, self.client, self.tg = cfg, client, tg
        self.states, self.get_report = states, get_report
        self._offset_ms = 0
        self._cycles = 0

    async def _now_ms(self) -> int:
        try:
            server = await self.client.server_time()
            self._offset_ms = server - int(time.time() * 1000)
            return server
        except BinanceError as exc:
            log.warning("Could not read server time (%s); using local clock + last offset", exc)
            return int(time.time() * 1000) + self._offset_ms

    async def scan_once(self) -> None:
        now_ms = await self._now_ms()
        last_closed_open = (now_ms // self.cfg.tf_ms - 1) * self.cfg.tf_ms
        sem = asyncio.Semaphore(4)

        async def guarded(st: SymbolState) -> None:
            async with sem:
                await self.scan_symbol(st, now_ms, last_closed_open)

        await asyncio.gather(*(guarded(st) for st in self.states.values()))
        self._cycles += 1
        if self._cycles % 40 == 0:
            log.info("Heartbeat: %d scan cycles, last closed candle %s", self._cycles, fmt_ts(last_closed_open))

    async def scan_symbol(self, st: SymbolState, now_ms: int, last_closed_open: int) -> None:
        if st.last_processed >= last_closed_open:
            return  # nothing new has closed yet
        try:
            rows, _ = await self.client.klines(st.symbol, self.cfg.timeframe, limit=self.cfg.scan_candles)
            rows = [r for r in rows if r[6] < now_ms]  # closed candles only
            await self.handle_rows(st, rows)
            st.failures = 0
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # one failing symbol must never stop the others
            st.failures += 1
            log.warning("%s scan failed (%d in a row): %s", st.symbol, st.failures, exc)
            if st.failures in (5, 50):
                await self.tg.notify_error(f"scan-{st.symbol}", f"{st.symbol}: {st.failures} consecutive scan failures: {exc}")

    async def handle_rows(self, st: SymbolState, rows: List[Row]) -> None:
        cfg = self.cfg
        new = [i for i, r in enumerate(rows) if r[0] > st.last_processed]
        if not new:
            return
        if st.last_processed and (new[0] == 0 or rows[new[0] - 1][0] != st.last_processed):
            log.warning("%s: gap between last processed candle and fetched window; "
                        "some candles may be unrecoverable", st.symbol)
        ind = compute_indicators(rows, cfg)
        last_i = new[-1]
        for i in new:
            replay = i < last_i  # catching up after an outage -> avoid stale alert spam
            for tr in list(st.trades):
                events = tr.step(rows[i], cfg)
                if not tr.silent:
                    for ev in events:
                        await self.notify_event(st, tr, ev)
            st.trades = [t for t in st.trades if t.state not in TERMINAL]
            for side in (LONG, SHORT):
                setup = detect_setup(ind, i, side, cfg)
                if setup is None:
                    continue
                key = (side, setup.open_time)
                if key in st.alerted:
                    continue
                if len(st.alerted) > 2000:
                    st.alerted.clear()
                st.alerted.add(key)
                if not cfg.allow_overlap and st.trades:
                    log.info("%s %s setup at %s skipped: previous setup still active",
                             st.symbol, side, fmt_ts(setup.close_time))
                    break
                tr = Trade(symbol=st.symbol, setup=setup, silent=replay)
                st.trades.append(tr)
                log.info("SIGNAL %s %s close=%s entry=%s stop=%s", st.symbol, side,
                         fmt_ts(setup.close_time), setup.entry, setup.stop)
                if not replay:
                    await self.notify_setup(st, tr, late=False)
                break
        st.last_processed = rows[last_i][0]
        for tr in st.trades:  # announce setups found while catching up that are still alive
            if tr.silent:
                tr.silent = False
                await self.notify_setup(st, tr, late=True)

    # ---- notifications -------------------------------------------------------------
    def _fp(self, st: SymbolState, x: float) -> str:
        return fmt_num(x, st.places)

    async def notify_setup(self, st: SymbolState, tr: Trade, late: bool) -> None:
        cfg, s = self.cfg, tr.setup
        fp = lambda x: fmt_num(x, st.places)  # noqa: E731
        fa = lambda x: fmt_num(x, st.places + 2)  # noqa: E731
        icon = "🟢" if s.side == LONG else "🔴"
        rlo, rhi = cfg.long_rsi if s.side == LONG else cfg.short_rsi
        if s.side == LONG:
            ref, sign = f"candle high {fp(s.high)} + {cfg.atr_trigger_mult} × ATR {fa(s.atr)}", "+"
            stop_calc = f"entry − {cfg.atr_stop_mult} × ATR"
        else:
            ref, sign = f"candle low {fp(s.low)} − {cfg.atr_trigger_mult} × ATR {fa(s.atr)}", "−"
            stop_calc = f"entry + {cfg.atr_stop_mult} × ATR"
        status = "WAITING FOR TRIGGER"
        extra = f"expires after {cfg.trigger_expiry} closed candles"
        if tr.state == "ACTIVE":
            status = "TRIGGER HIT (during catch-up)"
            extra = f"triggered in candle {fmt_ts(tr.trigger_time)}"
        late_note = "\n⏳ Detected late (bot was catching up after a gap)." if late else ""
        text = "\n".join([
            f"{icon} <b>{s.side} setup — {esc(st.symbol)} ({esc(cfg.timeframe)})</b>",
            f"Status: <b>{status}</b> ({extra}){late_note}",
            f"Signal candle close: {fmt_ts(s.close_time + 1)}",
            f"Market price (signal close): <code>{fp(s.close)}</code>",
            f"EMA9 <code>{fa(s.ema9)}</code> · EMA20 <code>{fa(s.ema20)}</code> · EMA100 <code>{fa(s.ema100)}</code>",
            f"VWAP (UTC session): <code>{fa(s.vwap)}</code>",
            f"RSI14: <code>{s.rsi:.1f}</code> (filter {rlo:g}–{rhi:g})",
            f"ATR14: <code>{fa(s.atr)}</code>",
            f"Volume: {s.volume:,.2f} vs avg of prev {cfg.volume_lookback} candles {s.vol_avg:,.2f} → ✅ ≥ average",
            f"Entry trigger: <code>{fp(s.entry)}</code> ({ref})",
            f"Initial stop-loss: <code>{fp(s.stop)}</code> ({stop_calc})",
            f"Take-profit ({cfg.rr:g}R): <code>{fp(s.take_profit)}</code>",
            f"Risk distance (R): <code>{fp(s.risk)}</code>",
            "",
            esc(hist_text(self.get_report(), st.symbol, s.side, cfg)),
            "",
            DISCLAIMER,
        ])
        await self.tg.send(text)

    async def notify_event(self, st: SymbolState, tr: Trade, ev: str) -> None:
        cfg, s = self.cfg, tr.setup
        fp = lambda x: fmt_num(x, st.places)  # noqa: E731
        head = f"<b>{s.side} {esc(st.symbol)} ({esc(cfg.timeframe)})</b>"
        sig = f"Signal candle close: {fmt_ts(s.close_time + 1)}"
        note = "Tracked signal outcome (simulation) — no real order was placed."
        if ev == "TRIGGER_HIT":
            text = (f"🟡 TRIGGER HIT — {head}\n{sig}\nTrigger <code>{fp(s.entry)}</code> was reached in candle "
                    f"opening {fmt_ts(tr.trigger_time)} (detected at that candle's close).\n"
                    f"Stop <code>{fp(s.stop)}</code> · TP <code>{fp(s.take_profit)}</code>\n{note}")
        elif ev == "EXPIRED":
            text = (f"⌛ EXPIRED — {head}\n{sig}\nTrigger <code>{fp(s.entry)}</code> was not reached within "
                    f"{cfg.trigger_expiry} closed candles. No entry occurred.")
        elif ev == "STOP":
            text = (f"🔻 STOP-LOSS HIT — {head}\n{sig}\nStop <code>{fp(s.stop)}</code> touched in candle opening "
                    f"{fmt_ts(tr.exit_time)}.\nResult: <b>{fr(tr.r_multiple)}</b> (after fees+slippage)\n{note}")
        elif ev == "TARGET":
            text = (f"✅ TAKE-PROFIT HIT — {head}\n{sig}\nTarget <code>{fp(s.take_profit)}</code> touched in "
                    f"candle opening {fmt_ts(tr.exit_time)}.\nResult: <b>{fr(tr.r_multiple)}</b> "
                    f"(after fees+slippage)\n{note}")
        elif ev == "TIME_EXPIRED":
            text = (f"⏱ TIME EXPIRED (unresolved) — {head}\n{sig}\nNeither stop nor target within "
                    f"{cfg.max_holding} candles; closed at candle close.\nMark-to-market: "
                    f"<b>{fr(tr.r_multiple)}</b> — counted as unresolved, not a win.\n{note}")
        else:
            return
        log.info("EVENT %s %s %s", st.symbol, s.side, ev)
        await self.tg.send(text)

    async def run(self, stop: asyncio.Event) -> None:
        log.info("Live scanner started (interval %.0fs)", self.cfg.scan_interval)
        while not stop.is_set():
            try:
                await self.scan_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.exception("Scan cycle error: %s", exc)
                await self.tg.notify_error("scan-cycle", f"Scan cycle error: {exc}")
            if await _sleep(stop, self.cfg.scan_interval):
                break


async def _sleep(stop: asyncio.Event, seconds: float) -> bool:
    """Sleep but wake immediately on shutdown. Returns True if shutdown was requested."""
    try:
        await asyncio.wait_for(stop.wait(), timeout=seconds)
        return True
    except asyncio.TimeoutError:
        return False


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
class Bot:
    def __init__(self, cfg: Config, session: aiohttp.ClientSession, stop: asyncio.Event) -> None:
        self.cfg, self.stop = cfg, stop
        self.client = BinanceClient(session, cfg.base_url, cfg.rate_budget)
        self.tg = Telegram(session, cfg.telegram_token, cfg.telegram_chat_id)
        self.history = HistoryStore(cfg, self.client)
        self.report: Optional[Report] = None
        self.states: Dict[str, SymbolState] = {}

    async def verify_symbols(self) -> Dict[str, int]:
        info = await self.client.exchange_info()
        found = {s.get("symbol"): s for s in info["symbols"]}
        valid: Dict[str, int] = {}
        bad: List[str] = []
        for sym in SYMBOLS:
            s = found.get(sym)
            if (s and s.get("contractType") == "PERPETUAL" and s.get("quoteAsset") == "USDT"
                    and s.get("status") == "TRADING"):
                tick = next((f.get("tickSize") for f in s.get("filters", [])
                             if f.get("filterType") == "PRICE_FILTER"), "0.0001")
                valid[sym] = places_from_tick(str(tick))
            else:
                bad.append(sym)
        if bad:
            msg = f"Symbols not valid TRADING USDT perpetuals, skipped: {', '.join(bad)}"
            log.error(msg)
            await self.tg.send(f"🚨 {esc(msg)}")
        if not valid:
            raise FatalError("None of the configured symbols is a valid USDT perpetual")
        return valid

    async def startup(self) -> None:
        cfg = self.cfg
        ok = await self.tg.send(
            f"🚀 <b>Signal bot starting</b>\nTimeframe {esc(cfg.timeframe)} · lookback {cfg.lookback_days}d · "
            f"{len(SYMBOLS)} symbols\nSignal-only: no orders are ever placed.\nDownloading history and running "
            f"initial backtest — this can take a few minutes…")
        if not ok:
            raise FatalError("Telegram delivery failed — check TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
        if cfg.binance_api_key:
            log.info("Binance API credentials provided but not needed (public endpoints only); they are not used.")
        now_ms = await self.client.server_time()
        skew = now_ms - int(time.time() * 1000)
        log.info("Binance server time %s (local clock skew %+d ms)", fmt_ts(now_ms), skew)
        places = await self.verify_symbols()
        for sym, pl in places.items():
            try:
                rows = await self.history.refresh(sym, now_ms)
                log.info("%s: %d candles loaded (%s → %s)", sym, len(rows),
                         fmt_ts(rows[0][0] if rows else None), fmt_ts(rows[-1][0] if rows else None))
                if len(rows) < 300:
                    raise BinanceError(f"too little history ({len(rows)} candles)")
                self.states[sym] = SymbolState(symbol=sym, places=pl, last_processed=rows[-1][0])
            except FatalError:
                raise
            except BinanceError as exc:
                self.history.data.pop(sym, None)
                log.error("%s: history unavailable, symbol skipped: %s", sym, exc)
                await self.tg.send(f"🚨 {esc(sym)}: history unavailable, symbol skipped ({esc(exc)})")
        if not self.states:
            raise FatalError("No symbol has usable history")
        t0 = time.monotonic()
        self.report = await asyncio.to_thread(run_backtest, dict(self.history.data), cfg)
        log.info("Initial backtest finished in %.1fs", time.monotonic() - t0)
        await self.tg.send_blocks(format_backtest_blocks(self.report, cfg, "Initial backtest completed"))

    async def refresh_loop(self) -> None:
        cfg = self.cfg
        while not self.stop.is_set():
            if await _sleep(self.stop, cfg.refresh_hours * 3600):
                return
            try:
                now_ms = await self.client.server_time()
                for sym in list(self.states):
                    try:
                        await self.history.refresh(sym, now_ms)
                    except BinanceError as exc:
                        log.warning("%s: periodic history refresh failed: %s", sym, exc)
                report = await asyncio.to_thread(run_backtest, dict(self.history.data), cfg)
                self.report = report  # atomic swap; live scanning keeps running meanwhile
                await self.tg.send_blocks(format_backtest_blocks(report, cfg, "Periodic backtest update"))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.exception("Periodic backtest failed: %s", exc)
                await self.tg.notify_error("refresh", f"Periodic backtest failed: {exc}")

    async def run(self) -> None:
        await self.startup()
        scanner = Scanner(self.cfg, self.client, self.tg, self.states, lambda: self.report)
        tasks = [asyncio.create_task(scanner.run(self.stop), name="scanner"),
                 asyncio.create_task(self.refresh_loop(), name="refresh")]
        await self.tg.send("✅ <b>Live scanning started</b> — alerts are sent on completed candles only.")
        stop_wait = asyncio.create_task(self.stop.wait())
        done, _ = await asyncio.wait([stop_wait, *tasks], return_when=asyncio.FIRST_COMPLETED)
        for t in tasks:
            if t in done and t.exception():
                log.error("Background task %s crashed: %s", t.get_name(), t.exception())
        self.stop.set()
        for t in [stop_wait, *tasks]:
            t.cancel()
        await asyncio.gather(stop_wait, *tasks, return_exceptions=True)


async def start_health_server(port: int) -> Any:
    """Optional tiny HTTP endpoint, started ONLY when the PORT env var exists (Render Web Services set it;
    Background Workers do not). Lets an external pinger keep a free Web Service awake."""
    from aiohttp import web

    async def ok(_req: Any) -> Any:
        return web.json_response({"status": "ok", "time": fmt_ts(int(time.time() * 1000))})

    app = web.Application()
    app.router.add_get("/", ok)
    app.router.add_get("/health", ok)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", port).start()
    log.info("Health endpoint listening on port %d (/ and /health)", port)
    return runner


async def amain() -> int:
    cfg = load_config()
    setup_logging(cfg.log_level)
    port = os.getenv("PORT", "").strip()
    health = await start_health_server(int(port)) if port.isdigit() else None  # start before slow backtest
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # e.g. Windows
            signal.signal(sig, lambda *_: loop.call_soon_threadsafe(stop.set))
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=20)) as session:
        bot = Bot(cfg, session, stop)
        code = 0
        try:
            await bot.run()
        except BanError as exc:
            log.error("FATAL: %s", exc)
            await bot.tg.send(f"🛑 <b>Binance IP ban</b>\n{esc(exc)}")
            idle = min(exc.wait, 3600.0)  # wait before exiting so Render does not restart-loop into the ban
            log.info("Idling %.0f s before exit to avoid hammering Binance", idle)
            await _sleep(stop, idle)
            code = 1
        except FatalError as exc:
            log.error("FATAL: %s", exc)
            await bot.tg.send(f"🛑 <b>Fatal error</b>\n{esc(exc)}")
            code = 1
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.exception("Unhandled error: %s", exc)
            await bot.tg.send(f"🛑 <b>Unhandled error</b>\n{esc(repr(exc)[:500])}")
            code = 1
        try:
            await asyncio.wait_for(bot.tg.send("🔌 Signal bot shut down."), timeout=10)
        except Exception:
            pass
        log.info("Bye.")
        if health is not None:
            await health.cleanup()
        return code


def main() -> None:
    try:
        sys.exit(asyncio.run(amain()))
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
