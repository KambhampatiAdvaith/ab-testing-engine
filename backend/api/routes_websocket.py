import asyncio
import json
from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.config import settings

router = APIRouter()

_CONTROL_BASE_RATE = 0.10
_VARIANT_BASE_RATE = 0.12


@router.websocket("/ws/v1/experiment-stream/{experiment_id}")
async def experiment_stream(websocket: WebSocket, experiment_id: str):
    """
    Stream synthetic A/B test traffic in real time.

    Sends a JSON message every ``websocket_tick_ms`` milliseconds with
    cumulative click / impression / conversion-rate data for two variants
    (control and variant).
    """
    await websocket.accept()

    state = {
        "control": {"clicks": 0, "impressions": 0},
        "variant": {"clicks": 0, "impressions": 0},
    }

    tick_seconds = settings.websocket_tick_ms / 1000.0

    try:
        while True:
            for variant_name, base_rate in [
                ("control", _CONTROL_BASE_RATE),
                ("variant", _VARIANT_BASE_RATE),
            ]:
                new_impressions = int(np.random.poisson(5))
                new_clicks = int(np.random.binomial(new_impressions, base_rate))

                state[variant_name]["impressions"] += new_impressions
                state[variant_name]["clicks"] += new_clicks

                impressions = state[variant_name]["impressions"]
                clicks = state[variant_name]["clicks"]
                conv_rate = clicks / impressions if impressions > 0 else 0.0

                event_types = ["page_view", "click", "conversion", "bounce"]
                event_weights = [0.5, 0.25, base_rate, 0.25 - base_rate]
                event_type: str = np.random.choice(event_types, p=event_weights)

                message = {
                    "experiment_id": experiment_id,
                    "variant_name": variant_name,
                    "event_type": event_type,
                    "cumulative_clicks": clicks,
                    "cumulative_impressions": impressions,
                    "conversion_rate": round(conv_rate, 4),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                await websocket.send_text(json.dumps(message))

            await asyncio.sleep(tick_seconds)

    except WebSocketDisconnect:
        pass
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass
