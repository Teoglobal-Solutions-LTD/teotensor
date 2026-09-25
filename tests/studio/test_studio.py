"""The studio page lists models and can fit PCA without a browser."""

from __future__ import annotations

import json
from contextlib import suppress
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from teotensor.studio.server import Session, StudioServer


def test_studio_lists_pca_and_serves_the_legend() -> None:
    server = StudioServer(("127.0.0.1", 0), Session())
    thread_target = server.serve_forever
    import threading

    thread = threading.Thread(target=thread_target, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    base = f"http://{host}:{port}"
    try:
        page = urlopen(base + "/").read().decode("utf-8")
        assert "#0d9488" in page
        assert "#2563eb" in page
        assert "#d97706" in page
        assert "#e11d48" in page
        models = json.loads(urlopen(base + "/api/models").read().decode("utf-8"))
        names = {item["name"] for item in models["models"]}
        assert "pca" in names
        assert "mlp_classifier" in names
        body = json.dumps(
            {
                "csv": "a,b\n0,1\n1,3\n2,5\n",
                "target": "",
            }
        ).encode("utf-8")
        request = Request(
            base + "/api/data",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        loaded = json.loads(urlopen(request).read().decode("utf-8"))
        assert loaded["rows"] == 3
        assert loaded["train"] == 3
        page = json.loads(urlopen(base + "/api/rows?offset=0&limit=2").read())
        assert len(page["records"]) == 2
        assert page["records"][0]["values"] == [0.0, 1.0]
        prepared = json.dumps(
            {"scale_divisor": 2, "standardize": False, "val_fraction": 0}
        ).encode("utf-8")
        request = Request(
            base + "/api/prepare",
            data=prepared,
            headers={"Content-Type": "application/json"},
        )
        scaled = json.loads(urlopen(request).read().decode("utf-8"))
        assert scaled["scale_divisor"] == 2
        page = json.loads(urlopen(base + "/api/rows?offset=0&limit=1").read())
        assert page["records"][0]["values"] == [0.0, 0.5]
    finally:
        server.shutdown()


def _post(base: str, path: str, payload: dict[str, object]) -> dict[str, object]:
    request = Request(
        base + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urlopen(request).read().decode("utf-8"))


def test_layout_draws_input_hidden_and_output() -> None:
    server = StudioServer(("127.0.0.1", 0), Session())
    import threading

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    base = f"http://{host}:{port}"
    try:
        page = urlopen(base + "/").read().decode("utf-8")
        assert 'class="spinner"' in page
        assert 'id="check-box" hidden' in page
        assert 'id="net-zoom"' in page
        assert "function notePhase" in page
        assert "function playForward" in page
        assert 'name="scale"' in page
        assert 'id="report-block"' in page
        _post(base, "/api/select", {"model": "mlp_classifier"})
        _post(
            base,
            "/api/data",
            {"csv": "x,y\n0,0\n1,1\n0,1\n1,0\n", "target": "y"},
        )
        _post(
            base,
            "/api/params",
            {
                "fields": [
                    {
                        "key": "layers",
                        "kind": "layers",
                        "value": [{"neurons": 4, "activation": "ReLU"}],
                    }
                ]
            },
        )
        layout = json.loads(urlopen(base + "/api/layout").read().decode("utf-8"))
        assert layout["sizes"] == [1, 4, 2]
        try:
            urlopen(base + "/api/report")
        except HTTPError as exc:
            assert exc.code == 400
        else:
            raise AssertionError("a report before fit must be refused")
    finally:
        server.shutdown()


def test_held_out_rows_are_scored() -> None:
    server = StudioServer(("127.0.0.1", 0), Session())
    import threading

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    base = f"http://{host}:{port}"
    try:
        _post(base, "/api/select", {"model": "mlp_classifier"})
        _post(
            base,
            "/api/data",
            {"csv": "x,y\n0,0\n1,1\n0,1\n1,0\n", "target": "y"},
        )
        _post(
            base,
            "/api/prepare",
            {"scale_divisor": None, "standardize": False, "val_fraction": 0.5},
        )
        _post(
            base,
            "/api/params",
            {
                "fields": [
                    {
                        "key": "layers",
                        "kind": "layers",
                        "value": [{"neurons": 4, "activation": "ReLU"}],
                    },
                    {"key": "max_iter", "kind": "int", "value": 1},
                    {"key": "validation_fraction", "kind": "float", "value": 0},
                ]
            },
        )
        _post(base, "/api/fit", {"mode": "full"})
        checked = json.loads(urlopen(base + "/api/check").read().decode("utf-8"))
        assert checked["split"] == "check"
        assert checked["scored"] == 2
        assert len(checked["records"]) == 2
        assert "match" in checked["records"][0]
    finally:
        server.shutdown()


def test_a_running_fit_can_pause_and_resume() -> None:
    import threading
    import time

    server = StudioServer(("127.0.0.1", 0), Session())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    base = f"http://{host}:{port}"
    lines = ["a,b,y"]
    for index in range(800):
        bit = index % 2
        lines.append(f"{bit},{1 - bit},{bit}")
    fit_done = threading.Event()
    fit_error: list[BaseException] = []
    worker: threading.Thread | None = None

    def _run() -> None:
        try:
            _post(base, "/api/fit", {"mode": "full"})
        except BaseException as exc:
            fit_error.append(exc)
        finally:
            fit_done.set()

    try:
        page = urlopen(base + "/").read().decode("utf-8")
        assert 'id="pause"' in page
        assert 'id="check-box" hidden' in page
        try:
            _post(base, "/api/control", {"paused": True})
        except HTTPError as exc:
            assert exc.code == 400
        else:
            raise AssertionError("pause before teaching must be refused")
        _post(base, "/api/select", {"model": "mlp_classifier"})
        _post(base, "/api/data", {"csv": "\n".join(lines) + "\n", "target": "y"})
        _post(
            base,
            "/api/prepare",
            {"scale_divisor": None, "standardize": False, "val_fraction": 0},
        )
        _post(
            base,
            "/api/params",
            {
                "fields": [
                    {
                        "key": "layers",
                        "kind": "layers",
                        "value": [{"neurons": 64, "activation": "ReLU"}],
                    },
                    {"key": "max_iter", "kind": "int", "value": 30},
                    {"key": "batch_size", "kind": "int", "value": 32},
                    {"key": "validation_fraction", "kind": "float", "value": 0},
                    {"key": "inspect_every", "kind": "int", "value": 0},
                ]
            },
        )
        worker = threading.Thread(target=_run)
        worker.start()
        asked = False
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            box = json.loads(urlopen(base + "/api/progress").read())
            if box["running"] and not asked:
                _post(base, "/api/control", {"paused": True})
                asked = True
            if asked and box["paused"]:
                break
            time.sleep(0.02)
        held = json.loads(urlopen(base + "/api/progress").read())
        assert held["paused"]
        time.sleep(0.35)
        later = json.loads(urlopen(base + "/api/progress").read())
        assert later["paused"]
        assert later["epoch"] == held["epoch"]
        _post(base, "/api/control", {"paused": False})
        assert fit_done.wait(60)
        assert fit_error == []
    finally:
        with suppress(Exception):
            _post(base, "/api/control", {"paused": False})
        server.shutdown()
        if worker is not None:
            worker.join(timeout=5)


def test_infer_session_is_separate_from_train() -> None:
    server = StudioServer(("127.0.0.1", 0))
    import threading

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    base = f"http://{host}:{port}"
    try:
        body = json.dumps({"name": "score", "purpose": "infer"}).encode("utf-8")
        request = Request(
            base + "/api/sessions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        opened = json.loads(urlopen(request).read().decode("utf-8"))
        assert opened["purpose"] == "infer"
        assert opened["name"] == "score"
        listing = json.loads(urlopen(base + "/api/sessions").read().decode("utf-8"))
        assert listing["active"] == opened["id"]
        assert "Divide pixel values by 255" not in urlopen(base + "/").read().decode(
            "utf-8"
        )
        refused = json.dumps({"url": "file:///C:/tmp/a.csv"}).encode("utf-8")
        request = Request(
            base + "/api/url",
            data=refused,
            headers={"Content-Type": "application/json"},
        )
        try:
            urlopen(request)
        except HTTPError as exc:
            assert exc.code == 400
        else:
            raise AssertionError("a file URL must be refused")
    finally:
        server.shutdown()
