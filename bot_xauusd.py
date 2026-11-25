#!/usr/bin/env python3
"""
Bot Alert Analisa XAU/USD -> Telegram   (ALERT SAJA, tanpa order otomatis)

SETUP
  1) pip install requests pandas numpy
  2) Daftar gratis di twelvedata.com -> salin API key
  3) Di Telegram buka @BotFather -> /newbot -> salin TOKEN
     Kirim satu pesan ke bot Anda, lalu buka:
     https://api.telegram.org/bot<TOKEN>/getUpdates  -> cari "chat":{"id": ...}
  4) Set variabel lingkungan:
       TWELVEDATA_KEY, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
     opsional: EQUITY (default 100), RISK_PCT (default 1.0)
  5) Tes sekali : python bot_xauusd.py --once
     Jalan terus: python bot_xauusd.py        (cek tiap 5 menit)

Tanpa TELEGRAM_TOKEN, pesan hanya dicetak di terminal.
Mode GitHub Actions: set STATE_FILE=state.json dan jalankan dengan --once
(bot ingat sinyal terakhir lewat file itu, jadi tidak kirim pesan dobel).
Kuota gratis Twelve Data 800 kredit/hari; bot ini memakai sekitar 400/hari.
"""
import os
import sys
import json
import time
import math
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import requests

# ---------------- KONFIGURASI ----------------
SYMBOL = "XAU/USD"
EQUITY = float(os.getenv("EQUITY") or 100)
RISK_PCT = float(os.getenv("RISK_PCT") or 1.0)
RR = 1.5                 # target reward:risk minimum
ADX_MIN = 20             # di bawah ini dianggap tren lemah -> NO SETUP
BIAS_MIN = 0.25          # ambang skor bias (-1..+1)
LOOP_SECONDS = 300
OZ_PER_LOT = 100         # 1 lot XAU/USD = 100 oz (cek spesifikasi broker Anda)
WIB = timezone(timedelta(hours=7))
TD_URL = "https://api.twelvedata.com/time_series"
STATE_FILE = os.getenv("STATE_FILE")  # opsional: simpan sinyal terakhir (mode --once terjadwal)


# ---------------- DATA ----------------
def fetch(interval, size):
    r = requests.get(TD_URL, params=dict(
        symbol=SYMBOL, interval=interval, outputsize=size, order="ASC",
        timezone="UTC", apikey=os.environ["TWELVEDATA_KEY"]), timeout=30)
    j = r.json()
    if j.get("status") == "error" or "values" not in j:
        raise RuntimeError(j.get("message", "respon API tidak valid"))
    df = pd.DataFrame(j["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.set_index("datetime")[["open", "high", "low", "close"]].astype(float)


_cache = {}


def cached(interval, size, ttl):
    t, df = _cache.get(interval, (0, None))
    if df is None or time.time() - t > ttl:
        df = fetch(interval, size)
        _cache[interval] = (time.time(), df)
    return df


# ---------------- INDIKATOR ----------------
def rsi(c, n=14):
    d = c.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    rs = up.ewm(alpha=1 / n, adjust=False).mean() / dn.ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + rs)


def macd_hist(c):
    m = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    return m - m.ewm(span=9, adjust=False).mean()


def atr(df, n=14):
    pc = df.close.shift()
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def adx(df, n=14):
    up, dn = df.high.diff(), -df.low.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    a = atr(df, n)
    pdi = 100 * pdm.ewm(alpha=1 / n, adjust=False).mean() / a
    mdi = 100 * mdm.ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def score_series(df):
    """Skor bias -1..+1 dari EMA50/200, MACD, RSI."""
    c = df.close
    s = np.sign(c.ewm(span=50, adjust=False).mean() - c.ewm(span=200, adjust=False).mean()) * 0.4
    s = s + np.sign(macd_hist(c)) * 0.3
    s = s + ((rsi(c) - 50) / 50).clip(-1, 1) * 0.3
    return s


# ---------------- STRUKTUR & LIKUIDITAS ----------------
def swings(df, k=3):
    h, l = df.high.values, df.low.values
    sh, sl = [], []
    for i in range(k, len(df) - k):
        if h[i] == h[i - k:i + k + 1].max():
            sh.append((i, h[i]))
        if l[i] == l[i - k:i + k + 1].min():
            sl.append((i, l[i]))
    return sh, sl


def structure(sh, sl):
    if len(sh) < 2 or len(sl) < 2:
        return "RANGE"
    hh, hl = sh[-1][1] > sh[-2][1], sl[-1][1] > sl[-2][1]
    if hh and hl:
        return "BULL HH/HL"
    if not hh and not hl:
        return "BEAR LH/LL"
    return "RANGE"


def sweep(df, sh, sl):
    c = df.close.iloc[-1]
    tail = df.iloc[-6:]
    if sl and tail.low.min() < sl[-1][1] < c:
        return "SELL-SIDE"
    if sh and tail.high.max() > sh[-1][1] > c:
        return "BUY-SIDE"
    return "none"


def fvg(df, look=30):
    h, l = df.high.values, df.low.values
    for i in range(len(df) - 1, len(df) - look, -1):
        if l[i] > h[i - 2]:
            return "BULL"
        if h[i] < l[i - 2]:
            return "BEAR"
    return "none"


def profile(df, bins=60, va=0.70):
    """Volume profile proxy (TPO): XAU/USD spot tidak punya volume asli."""
    edges = np.linspace(df.low.min(), df.high.max(), bins + 1)
    mids = (edges[:-1] + edges[1:]) / 2
    w = np.zeros(bins)
    for hi, lo in zip(df.high.values, df.low.values):
        w[(mids >= lo) & (mids <= hi)] += 1
    p = int(w.argmax())
    a = b = p
    acc, tot = w[p], w.sum()
    while acc < va * tot and (a > 0 or b < bins - 1):
        up = w[b + 1] if b < bins - 1 else -1
        dn = w[a - 1] if a > 0 else -1
        if up >= dn:
            b += 1
            acc += w[b]
        else:
            a -= 1
            acc += w[a]
    return mids[p], mids[b], mids[a]  # POC, VAH, VAL


def analog(h1, horizon=24):
    """Frekuensi historis kondisi serupa (bias searah, RSI mirip) -> arah 24 jam kemudian."""
    sc, r = score_series(h1), rsi(h1.close)
    fwd = h1.close.shift(-horizon) / h1.close - 1
    m = (np.sign(sc) == np.sign(sc.iloc[-1])) & ((r - r.iloc[-1]).abs() < 5) & fwd.notna()
    f = fwd[m]
    if len(f) < 20:
        return None
    return len(f), (f > 0).mean() * 100, (f < 0).mean() * 100


# ---------------- ANALISA ----------------
def analyze():
    m5 = cached("5min", 600, 240)
    h1 = cached("1h", 5000, 900)
    age_min = (pd.Timestamp.now("UTC").tz_localize(None) - m5.index[-1]).total_seconds() / 60
    if age_min > 30:
        return None, None  # pasar tutup / data tidak update

    last = m5.close.iloc[-1]
    a5 = atr(m5).iloc[-1]
    sc = score_series(h1).iloc[-1]
    rsi_h, mh, adx_h = rsi(h1.close).iloc[-1], macd_hist(h1.close).iloc[-1], adx(h1).iloc[-1]
    sh, sl = swings(m5)
    st = structure(sh, sl)
    poc, vah, val = profile(m5.iloc[-288:])
    an = analog(h1)
    bias = "BULL" if sc >= BIAS_MIN else "BEAR" if sc <= -BIAS_MIN else "NETRAL"

    # --- tentukan arah & rencana ---
    direction = None
    if adx_h >= ADX_MIN and sh and sl:
        if sc <= -BIAS_MIN and st != "BULL HH/HL":
            direction = "SELL"
        elif sc >= BIAS_MIN and st != "BEAR LH/LL":
            direction = "BUY"

    plan, sig = None, ("WAIT",)
    if direction:
        buf = 0.1 * a5
        if direction == "SELL":
            entry = sl[-1][1] - buf
            risk = min(max(sh[-1][1] + buf - entry, 1.5 * a5), 3 * a5)
            stop, tp = entry + risk, entry - RR * risk
            missed = last < entry - a5
        else:
            entry = sh[-1][1] + buf
            risk = min(max(entry - (sl[-1][1] - buf), 1.5 * a5), 3 * a5)
            stop, tp = entry - risk, entry + RR * risk
            missed = last > entry + a5
        if not missed:
            risk_usd = EQUITY * RISK_PCT / 100
            lot = max(0.01, math.floor(risk_usd / (risk * OZ_PER_LOT) * 100) / 100)
            real = lot * risk * OZ_PER_LOT
            plan = dict(d=direction, entry=entry, sl=stop, tp=tp, lot=lot,
                        risk_usd=risk_usd, real=real)
            sig = (direction, round(entry / 3))

    # --- susun pesan ---
    status = f"SETUP KONDISIONAL · {direction}" if plan else "NO SETUP · WAIT"
    ts = datetime.now(WIB).strftime("%d-%m-%Y %H:%M WIB")
    L = [f"XAU/USD · M5/H1 · {ts}", f"STATUS  {status}", "=" * 34,
         "MARKET",
         f"Last {last:.2f}   ATR(M5) {a5:.2f}",
         f"Bias {bias} ({sc:+.2f})",
         f"RSI {rsi_h:.1f}  MACD {'POS' if mh > 0 else 'NEG'}  ADX {adx_h:.1f}{' (lemah)' if adx_h < ADX_MIN else ''}",
         f"Struct {st}",
         f"Sweep {sweep(m5, sh, sl)}   FVG {fvg(m5)}",
         "-" * 34,
         "VOLUME PROFILE (proxy TPO, 1 hari)",
         f"POC {poc:.2f}  VAH {vah:.2f}  VAL {val:.2f}",
         "-" * 34, "HISTORIS (kondisi serupa, H1)"]
    if an:
        L += [f"n={an[0]} (tumpang tindih)",
              f"Naik 24j {an[1]:.1f}%  Turun 24j {an[2]:.1f}%"]
    else:
        L += ["Sampel kurang"]
    L += ["-" * 34, "PLAN · KONDISIONAL"]
    if plan:
        L += [f"{plan['d']} Entry {plan['entry']:.2f}",
              f"SL {plan['sl']:.2f}  TP {plan['tp']:.2f}  RR 1:{RR}",
              "Trigger: M5 close tembus entry,",
              "lalu retest terkonfirmasi.",
              f"Lot {plan['lot']:.2f} · risiko nyata ${plan['real']:.2f}"
              f" (target ${plan['risk_usd']:.2f})"]
        if plan["real"] > plan["risk_usd"] * 1.5:
            L += ["! Lot minimum 0.01 melebihi risiko",
                  "  target. Kecilkan risiko / skip."]
    else:
        L += ["Belum ada setup valid."]
    L += ["-" * 34, f"Equity ${EQUITY:.2f} · Risk {RISK_PCT:.2f}%",
          "Mode: ALERT SAJA · tanpa order",
          "Frekuensi historis ≠ probabilitas menang."]
    return "\n".join(L), sig


# ---------------- KIRIM ----------------
def deliver(text):
    token, chat = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not (token and chat):
        print(text)
        return
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json={"chat_id": chat, "text": f"<pre>{text}</pre>", "parse_mode": "HTML"},
                      timeout=20)
    r.raise_for_status()


def load_state():
    try:
        with open(STATE_FILE) as f:
            return tuple(json.load(f))
    except Exception:
        return None


def save_state(sig):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(list(sig), f)
    except Exception as e:
        print("Gagal simpan state:", e)


def main():
    once = "--once" in sys.argv
    last_sig = load_state() if STATE_FILE else None
    while True:
        try:
            msg, sig = analyze()
            # kirim hanya saat kondisi berubah (tes manual --once tanpa STATE_FILE: selalu kirim)
            if msg and (sig != last_sig or (once and not STATE_FILE)):
                deliver(msg)
                last_sig = sig
                if STATE_FILE:
                    save_state(sig)
        except Exception as e:
            print("Error:", e)
        if once:
            break
        time.sleep(LOOP_SECONDS)


if __name__ == "__main__":
    main()
