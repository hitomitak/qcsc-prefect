"""Run Prefect with its server app in-process (no sockets), for sandboxes that forbid bind().

Import this module BEFORE anything else that imports Prefect. Only the HTTP transport changes:
- PREFECT_API_URL is a sentinel URL. Every PrefectClient gets httpx.ASGITransport and every
  SyncPrefectClient gets Starlette's TestClient transport, both into ONE in-process FastAPI app
  (prefect.server.api.server.create_app(ephemeral=True)); its SQLite DB lives under PREFECT_HOME
  and is migrated once at import.
- Client-side events (websockets) go to NullEventsClient via EventsWorker.set_client_override.
Flow, task, block, variable and artifact semantics are Prefect's own; the code under test is not
modified.
"""

import asyncio
import os

SENTINEL = "http://inprocess-prefect/api"
os.environ["PREFECT_API_URL"] = SENTINEL

import httpx  # noqa: E402
import prefect.client.orchestration as _orch  # noqa: E402
from prefect.events.clients import NullEventsClient  # noqa: E402
from prefect.events.worker import EventsWorker  # noqa: E402
from prefect.server.api.server import create_app  # noqa: E402
from prefect.server.database import provide_database_interface  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

asyncio.run(provide_database_interface().create_db())
_APP = create_app(ephemeral=True)
_SYNC_TRANSPORT = TestClient(_APP, base_url=SENTINEL)._transport


def _wrap(init, make_transport):
    def __init__(self, api, *args, httpx_settings=None, **kwargs):
        if isinstance(api, str) and api.rstrip("/").startswith(SENTINEL):
            httpx_settings = dict(httpx_settings or {})
            httpx_settings.setdefault("transport", make_transport())
        return init(self, api, *args, httpx_settings=httpx_settings, **kwargs)

    return __init__


_orch.PrefectClient.__init__ = _wrap(
    _orch.PrefectClient.__init__,
    lambda: httpx.ASGITransport(app=_APP, raise_app_exceptions=False),
)
_orch.SyncPrefectClient.__init__ = _wrap(_orch.SyncPrefectClient.__init__, lambda: _SYNC_TRANSPORT)
EventsWorker.set_client_override(NullEventsClient)
