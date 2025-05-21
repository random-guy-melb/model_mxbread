# slack_bot_with_status_updates.py
"""Slack‑Bolt bot that shows the *original* staged hourglass animation
while a message is being processed by a Flask service.

Key points
~~~~~~~~~~
* Consolidated DM / mention handling (no duplicated code).
* Cached bot user‑id (one auth call at start‑up).
* Timeout‑protected Flask request.
* Three‑stage status messages – exactly the pattern you used originally:
  1. Processing your request
  2. Analysing data
  3. Generating response

Environment variables
---------------------
SLACK_BOT_TOKEN, SLACK_APP_TOKEN, and (optionally) FLASK_APP_URL.
"""

from __future__ import annotations

import os
import re
import tempfile
import time
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import requests
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------
load_dotenv()
app = App(token=os.getenv("SLACK_BOT_TOKEN"))
BOT_USER_ID = app.client.auth_test()["user_id"]  # cache – one API call
FLASK_APP_URL = os.getenv("FLASK_APP_URL", "http://localhost:5000/process")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _post_status_update(
    client,
    channel: str,
    ts: str,
    text: str,
    *,
    thread_ts: Optional[str] = None,
) -> None:
    """Update the in‑progress message (ignore failures silently)."""
    try:
        client.chat_update(
            channel=channel,
            ts=ts,
            thread_ts=thread_ts,
            text=f":hourglass_flowing_sand: {text}…",
        )
    except Exception:  # noqa: BLE001 – best‑effort only
        pass


def _call_flask_api(text: str) -> str:
    try:
        response = requests.post(
            FLASK_APP_URL, json={"text": text}, timeout=60
        )
        response.raise_for_status()
        return response.json().get("response", "(No content)")
    except Exception as exc:  # noqa: BLE001 – collapse any error
        print(f"Flask‑API error: {exc}")
        return "Sorry, there was an error while processing your request."


# ---------------------------- Plotly utilities ----------------------------

def _generate_plot_image(_: str) -> str:  # text ignored for demo chart
    df = pd.DataFrame({
        "x": range(10),
        "y": np.random.randint(0, 10, 10),
    })
    fig = px.line(df, x="x", y="y", title="Sample Chart")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.write_image(tmp.name)
    return tmp.name


def _upload_chart(client, channel: str, text: str, *, thread_ts: Optional[str] = None) -> None:
    path = _generate_plot_image(text)
    try:
        client.files_upload_v2(
            channels=channel,
            file=path,
            title="Plotly chart (static)",
            initial_comment="Here’s your chart ",
            thread_ts=thread_ts,
        )
    finally:
        os.remove(path)


# ---------------------------------------------------------------------------
# Core request handler
# ---------------------------------------------------------------------------

def _handle_request(client, event: dict, text: str, *, thread_ts: Optional[str] = None) -> None:
    channel = event["channel"]

    # 1️⃣ Initial message – Processing your request
    spinner_msg = client.chat_postMessage(
        channel=channel,
        thread_ts=thread_ts,
        text=":hourglass_flowing_sand: Processing your request…",
    )
    spinner_ts = spinner_msg["ts"]

    # 2️⃣ Update – Analysing data
    time.sleep(1)
    _post_status_update(client, channel, spinner_ts, "Analysing data", thread_ts=thread_ts)

    # Call the heavy Flask backend
    response_text = _call_flask_api(text)

    # 3️⃣ Update – Generating response (only if still processing)
    _post_status_update(client, channel, spinner_ts, "Generating response", thread_ts=thread_ts)

    # Small delay so user sees the final stage for a moment
    time.sleep(0.5)

    # Clean‑up – delete status message
    try:
        client.chat_delete(channel=channel, ts=spinner_ts)
    except Exception:
        pass

    # Final reply & chart
    client.chat_postMessage(channel=channel, thread_ts=thread_ts, text=response_text)
    _upload_chart(client, channel, text, thread_ts=thread_ts)


# ---------------------------------------------------------------------------
# Slack event subscriptions
# ---------------------------------------------------------------------------

@app.event("message")
def on_message(message, client, event):  # noqa: D401
    """Respond in DMs or when mentioned in channels."""
    if message.get("bot_id"):
        return

    channel_type = message.get("channel_type")
    text = message.get("text", "")
    is_dm = channel_type == "im"
    is_mentioned = f"<@{BOT_USER_ID}>" in text

    if not (is_dm or is_mentioned):
        return

    # Strip the mention from text for cleaner processing
    if is_mentioned:
        text = re.sub(f"<@{BOT_USER_ID}>", "", text).strip()

    thread_ts = message.get("ts") if not is_dm else None
    _handle_request(client, event, text, thread_ts=thread_ts)


@app.error
def on_error(error):
    print(f"Slack‑Bolt error: {error}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    handler = SocketModeHandler(app, os.getenv("SLACK_APP_TOKEN"))
    print("⚡  Bot is running – press Ctrl‑C to quit")
    handler.start()
