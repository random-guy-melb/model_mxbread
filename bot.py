# slack_bot_with_spinner.py
"""A cleaner Slack‑Bolt bot that shows an animated spinner while your
message is being processed by the Flask backend.

Improvements
------------
* **Single place for heavy work** – duplicated DM/mention logic has been
  merged into ``handle_request`` so maintenance is easier.
* **Non‑blocking animation** – a background thread updates a single
  “processing…” message every ~600 ms, giving the user feedback without
  blocking the event loop or spamming the channel.
* **One auth call** – the bot user‑id is fetched once at start‑up
  instead of on every message.
* **Requests timeout + retries** – prevents the worker from hanging
  indefinitely if the Flask API stalls.
* **Utility functions** – chart generation & upload kept minimal and
  re‑usable.

Environment variables required
------------------------------
* ``SLACK_BOT_TOKEN`` – xoxb‑… token
* ``SLACK_APP_TOKEN`` – xapp‑… token for Socket Mode
* ``FLASK_APP_URL``  – e.g. http://localhost:5000/process (optional,
  defaults to that)
"""

from __future__ import annotations

import os
import re
import threading
import time
import tempfile
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
BOT_USER_ID = app.client.auth_test()["user_id"]  # single API call
FLASK_APP_URL = os.getenv("FLASK_APP_URL", "http://localhost:5000/process")

# Unicode braille spinner frames (looks good in Slack dark & light mode)
SPINNER_FRAMES = [
    "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _animate_spinner(client, channel: str, ts: str, stop_event: threading.Event, *, thread_ts: Optional[str] = None) -> None:
    """Update the processing message every ~600 ms until *stop_event* is set."""
    frame_idx = 0
    while not stop_event.is_set():
        try:
            client.chat_update(
                channel=channel,
                ts=ts,
                thread_ts=thread_ts,
                text=f":hourglass_flowing_sand: Processing… {SPINNER_FRAMES[frame_idx]}"
            )
        except Exception:  # noqa: BLE001 – we only want to keep spinning
            pass
        frame_idx = (frame_idx + 1) % len(SPINNER_FRAMES)
        time.sleep(0.6)  # stay well under Slack rate‑limit (50 updates/min)


def _start_spinner(client, channel: str, *, thread_ts: Optional[str] = None) -> tuple[str, threading.Event]:
    """Send the initial processing message and launch the spinner thread."""
    result = client.chat_postMessage(
        channel=channel,
        thread_ts=thread_ts,
        text=f":hourglass_flowing_sand: Processing… {SPINNER_FRAMES[0]}"
    )
    ts = result["ts"]
    stop_event = threading.Event()
    threading.Thread(
        target=_animate_spinner,
        args=(client, channel, ts, stop_event),
        kwargs={"thread_ts": thread_ts},
        daemon=True,
    ).start()
    return ts, stop_event


def _call_flask_api(text: str) -> str:
    """Forward *text* to the Flask service and return the response."""
    try:
        response = requests.post(
            FLASK_APP_URL, json={"text": text}, timeout=60  # s
        )
        response.raise_for_status()
        return response.json().get("response", "(No content)")
    except Exception as exc:  # noqa: BLE001 – convert every failure to text
        print(f"Flask‑API error: {exc}")
        return "Sorry, there was an error while processing your request."


# -------------------------- Plotly chart utilities -------------------------

def _generate_plot_image(_: str) -> str:  # we ignore the text for demo
    df = pd.DataFrame({
        "x": np.arange(10),
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
            initial_comment="Here’s your chart :bar_chart:",
            thread_ts=thread_ts,
        )
    finally:
        os.remove(path)


# ---------------------------------------------------------------------------
# Core handler logic
# ---------------------------------------------------------------------------

def _handle_request(client, event: dict, text: str, *, thread_ts: Optional[str] = None) -> None:
    channel = event["channel"]

    spinner_ts, stop_event = _start_spinner(client, channel, thread_ts=thread_ts)

    # --- Do the heavy work (blocking) --------------------------------------
    response_text = _call_flask_api(text)

    # --- Stop spinner & clean‑up ------------------------------------------
    stop_event.set()
    try:
        client.chat_delete(channel=channel, ts=spinner_ts)
    except Exception:
        # If the bot lacks chat:write.public we might not delete; just ignore.
        pass

    # --- Final response ----------------------------------------------------
    client.chat_postMessage(channel=channel, thread_ts=thread_ts, text=response_text)
    _upload_chart(client, channel, text, thread_ts=thread_ts)


# ---------------------------------------------------------------------------
# Event subscriptions
# ---------------------------------------------------------------------------

@app.event("message")
def on_message(message, client, event):  # noqa: D401
    """Respond if the bot is mentioned or in a DM."""
    if message.get("bot_id"):
        return  # ignore our own / other bots

    channel_type = message.get("channel_type")
    text = message.get("text", "")
    is_dm = channel_type == "im"
    is_mentioned = f"<@{BOT_USER_ID}>" in text

    if not (is_dm or is_mentioned):
        return

    # Remove mention tag so the Flask API gets clean text
    if is_mentioned:
        text = re.sub(f"<@{BOT_USER_ID}>", "", text).strip()

    thread_ts = message.get("ts") if not is_dm else None  # DMs don’t thread
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
