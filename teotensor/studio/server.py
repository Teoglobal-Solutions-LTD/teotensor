"""Local studio server. Python computes; the browser only displays.

The server binds to localhost. It uses the standard library only.
"""

from __future__ import annotations

import io
import json
import pickle
import tempfile
import threading
import urllib.error
import urllib.request
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qs, urlparse

import numpy as np
from numpy.typing import NDArray

from teotensor.artifacts import export_html
from teotensor.artifacts.weights import load_model
from teotensor.data import Dataset, Standardize, read_csv, read_mnist, scale
from teotensor.nn.training.loop import bind_progress, release_progress
from teotensor.studio.page import PAGE_HTML
from teotensor.studio.registry import model_registry, spec_by_name
from teotensor.studio.schema import apply_fields, form_fields

_MAX_URL_BYTES = 80 * 1024 * 1024


class Session:
    """One train or infer job: a table, preparation, and a model."""

    def __init__(
        self,
        name: str = "Session",
        purpose: str = "train",
        *,
        session_id: str | None = None,
    ) -> None:
        if purpose not in {"train", "infer"}:
            msg = "purpose must be 'train' or 'infer'."
            raise ValueError(msg)
        self.id = session_id or uuid.uuid4().hex[:8]
        self.name = name.strip() or "Session"
        self.purpose = purpose
        self.source: Dataset | None = None
        self.dataset: Dataset | None = None
        self.scale_divisor: float | None = None
        self.standardize = False
        self.val_fraction = 0.0
        self.model: Any = None
        self.model_name: str | None = None
        self.progress: dict[str, Any] = {
            "running": False,
            "epoch": 0,
            "epochs": 0,
            "phase": "",
        }


class SessionBook:
    """Named sessions. The page works on one of them at a time."""

    def __init__(self) -> None:
        self.sessions: dict[str, Session] = {}
        self.active_id: str | None = None

    def add(self, session: Session) -> Session:
        """Store ``session`` and make it the active one."""
        self.sessions[session.id] = session
        self.active_id = session.id
        return session

    def current(self) -> Session:
        """Return the session the page is editing."""
        if self.active_id is None or self.active_id not in self.sessions:
            msg = "Open a session first."
            raise RuntimeError(msg)
        return self.sessions[self.active_id]


class StudioServer(ThreadingHTTPServer):
    """HTTP server that keeps a :class:`SessionBook`."""

    def __init__(
        self,
        server_address: tuple[str, int],
        session: Session | None = None,
    ) -> None:
        super().__init__(server_address, StudioHandler)
        self.book = SessionBook()
        if session is not None:
            self.book.add(session)

    @property
    def session(self) -> Session:
        """The active session. Older callers read this attribute."""
        return self.book.current()


class StudioHandler(BaseHTTPRequestHandler):
    """Route studio commands. Errors come back as JSON."""

    server: StudioServer

    def log_message(self, format: str, *args: Any) -> None:
        """Stay quiet in the terminal. The page is the interface."""
        return None

    def do_GET(self) -> None:
        """Serve the page, the model list, state, and downloads."""
        path = urlparse(self.path).path
        try:
            if path == "/":
                self._bytes(200, PAGE_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/api/models":
                self._json(
                    200, {"models": [_spec_dict(spec) for spec in model_registry()]}
                )
            elif path == "/api/sessions":
                self._json(200, _session_list(self.server.book))
            elif path == "/api/session":
                self._json(200, _session_view(self.server.session))
            elif path == "/api/state":
                self._json(200, _state(self.server.session))
            elif path == "/api/layout":
                self._json(200, _layout(self.server.session))
            elif path == "/api/progress":
                self._json(200, _progress(self.server.session))
            elif path == "/api/check":
                query = parse_qs(urlparse(self.path).query)
                self._json(200, _check(self.server.session, query))
            elif path == "/api/report":
                self._inline_report()
            elif path == "/api/rows":
                query = parse_qs(urlparse(self.path).query)
                self._json(200, _rows(self.server.session, query))
            elif path == "/api/predict":
                query = parse_qs(urlparse(self.path).query)
                self._json(200, _predict_page(self.server.session, query))
            elif path == "/api/download/report":
                self._download_report()
            elif path == "/api/download/model":
                self._download_model()
            elif path == "/api/download/weights":
                self._download_weights()
            else:
                self._json(404, {"error": "not found"})
        except Exception as exc:
            self._json(400, {"error": str(exc)})

    def do_POST(self) -> None:
        """Accept data, settings, a fit command, or a playback request."""
        path = urlparse(self.path).path
        try:
            if path == "/api/sessions":
                payload = self._read_json()
                self._json(200, _open_session(self.server.book, payload))
                return
            if path == "/api/sessions/open":
                payload = self._read_json()
                opened = _activate(self.server.book, str(payload.get("id", "")))
                self._json(200, opened)
                return
            if path == "/api/sessions/drop":
                payload = self._read_json()
                self._json(200, _drop(self.server.book, str(payload.get("id", ""))))
                return
            session = self.server.session
            if path == "/api/upload":
                self._json(200, _load_upload(session, self))
                return
            if path == "/api/weights":
                self._json(200, _load_weights(session, self))
                return
            payload = self._read_json()
            if path == "/api/url":
                self._json(200, _load_url(session, payload))
            elif path == "/api/prepare":
                self._json(200, _prepare(session, payload))
            elif path == "/api/data":
                self._json(200, _load_table(session, payload))
            elif path == "/api/select":
                spec = spec_by_name(str(payload["model"]))
                session.model = spec.estimator_cls()
                session.model_name = spec.name
                self._json(
                    200, {"fields": form_fields(session.model), "model": spec.name}
                )
            elif path == "/api/params":
                if session.model is None:
                    msg = "Choose a model first."
                    raise RuntimeError(msg)
                apply_fields(session.model, list(payload.get("fields", [])))
                self._json(200, {"fields": form_fields(session.model)})
            elif path == "/api/fit":
                self._json(200, _fit(session, str(payload.get("mode", "full"))))
            elif path == "/api/control":
                self._json(200, _control(session, payload))
            elif path == "/api/playback":
                self._json(200, _playback(session, payload))
            else:
                self._json(404, {"error": "not found"})
        except Exception as exc:
            self._json(400, {"error": str(exc)})

    def _read_json(self) -> dict[str, Any]:
        """Read a JSON object from the request body."""
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        loaded = json.loads(raw.decode("utf-8"))
        if not isinstance(loaded, dict):
            msg = "Expected a JSON object."
            raise TypeError(msg)
        return loaded

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        """Send a JSON response."""
        body = json.dumps(_jsonable(payload)).encode("utf-8")
        self._bytes(status, body, "application/json; charset=utf-8")

    def _bytes(self, status: int, body: bytes, content_type: str) -> None:
        """Send raw bytes."""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _download_report(self) -> None:
        """Send the printable HTML report."""
        body = _report_html(self.server.session)
        self._attachment(body, "report.html", "text/html; charset=utf-8")

    def _inline_report(self) -> None:
        """Send the same report for the block under the train screen."""
        self._bytes(
            200,
            _report_html(self.server.session, theme="dark"),
            "text/html; charset=utf-8",
        )

    def _download_model(self) -> None:
        """Send the whole model as a pickle for this framework."""
        model = _require_fitted(self.server.session)
        buffer = io.BytesIO()
        pickle.dump(model, buffer, protocol=pickle.HIGHEST_PROTOCOL)
        self._attachment(buffer.getvalue(), "model.pkl", "application/octet-stream")

    def _download_weights(self) -> None:
        """Send the ``.ttw`` number file."""
        model = _require_fitted(self.server.session)
        if not hasattr(model, "export_weights"):
            msg = "This model cannot export weights."
            raise RuntimeError(msg)
        with tempfile.TemporaryDirectory() as folder:
            path = model.export_weights(str(Path(folder) / "weights.ttw"))
            body = Path(path).read_bytes()
        self._attachment(body, "weights.ttw", "application/zip")

    def _attachment(self, body: bytes, filename: str, content_type: str) -> None:
        """Send a file download."""
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def open_studio(*, open_browser: bool = True, port: int = 8765) -> str:
    """Serve the studio on localhost until the process is interrupted.

    The call blocks. A daemon thread would die as soon as the ``python -c``
    line finished, and the browser would open onto a server that is already
    gone. Ctrl+C stops the server.

    Parameters
    ----------
    open_browser : bool, default=True
        Open the address in the default browser.
    port : int, default=8765
        TCP port. ``0`` asks the operating system for a free port.

    Returns
    -------
    str
        ``http://127.0.0.1:<port>/``. Returned after the server stops.
    """
    server = StudioServer(("127.0.0.1", port))
    host, bound = server.server_address[:2]
    host_text = host.decode() if isinstance(host, bytes) else str(host)
    url = f"http://{host_text}:{bound}/"
    print(url, flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _release_paused_fits(server)
        print("Studio stopped.", flush=True)
    finally:
        _release_paused_fits(server)
        server.server_close()
    return url


def _spec_dict(spec: Any) -> dict[str, Any]:
    """JSON view of one registry entry."""
    return {
        "name": spec.name,
        "title": spec.title,
        "requires_y": spec.requires_y,
        "iterative": spec.iterative,
        "description": spec.description,
    }


def _load_table(session: Session, payload: dict[str, Any]) -> dict[str, Any]:
    """Parse CSV text into the session dataset."""
    target = str(payload.get("target", "")).strip() or None
    session.source = read_csv(str(payload.get("csv", "")), target=target)
    _rebuild(session)
    return _summary(_worked(session))


def _load_upload(session: Session, handler: StudioHandler) -> dict[str, Any]:
    """Store a CSV file or a MNIST zip as the raw table."""
    length = int(handler.headers.get("Content-Length", "0"))
    payload = handler.rfile.read(length) if length else b""
    filename = str(handler.headers.get("X-Filename", "upload.bin"))
    target = str(handler.headers.get("X-Target", "")).strip() or None
    if filename.lower().endswith(".zip"):
        session.source = read_mnist(payload)
    else:
        session.source = read_csv(payload.decode("utf-8"), target=target)
    _rebuild(session)
    return _summary(_worked(session))


def _load_url(session: Session, payload: dict[str, Any]) -> dict[str, Any]:
    """Fetch a CSV or a zip from an http(s) address and store it."""
    address = str(payload.get("url", "")).strip()
    parsed = urlparse(address)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        msg = "URL must be an http or https address."
        raise ValueError(msg)
    target = str(payload.get("target", "")).strip() or None
    request = urllib.request.Request(address, headers={"User-Agent": "TeoTensor"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            body = response.read(_MAX_URL_BYTES + 1)
    except urllib.error.URLError as exc:
        msg = f"Could not fetch the URL: {exc.reason}."
        raise ValueError(msg) from exc
    if len(body) > _MAX_URL_BYTES:
        msg = "The URL is larger than 80 MB."
        raise ValueError(msg)
    name = Path(parsed.path).name.lower()
    if name.endswith(".zip") or body[:2] == b"PK":
        session.source = read_mnist(body)
    else:
        session.source = read_csv(body.decode("utf-8"), target=target)
    _rebuild(session)
    return _summary(_worked(session))


def _load_weights(session: Session, handler: StudioHandler) -> dict[str, Any]:
    """Load a ``.ttw`` file into an inference session."""
    length = int(handler.headers.get("Content-Length", "0"))
    payload = handler.rfile.read(length) if length else b""
    filename = str(handler.headers.get("X-Filename", "weights.ttw"))
    if not filename.lower().endswith(".ttw"):
        msg = "Inference loads a .ttw weights file."
        raise ValueError(msg)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "weights.ttw"
        path.write_bytes(payload)
        session.model = load_model(path)
    session.model_name = type(session.model).__name__
    return _session_view(session)


def _prepare(session: Session, payload: dict[str, Any]) -> dict[str, Any]:
    """Store preparation and rebuild the working table from the raw source."""
    if session.source is None:
        msg = "Load a table first."
        raise RuntimeError(msg)
    raw = payload.get("scale_divisor")
    if raw is None or str(raw).strip() == "":
        session.scale_divisor = None
    else:
        divisor = float(raw)
        if divisor == 0.0:
            msg = "Divisor must not be 0."
            raise ValueError(msg)
        session.scale_divisor = divisor
    session.standardize = bool(payload.get("standardize", False))
    session.val_fraction = float(payload.get("val_fraction") or 0.0)
    _rebuild(session)
    summary = _summary(_worked(session))
    summary["scale_divisor"] = session.scale_divisor
    summary["standardize"] = session.standardize
    summary["val_fraction"] = session.val_fraction
    return summary


def _rebuild(session: Session) -> None:
    """Copy the raw table, scale, hold out a tail, then standardize."""
    if session.source is None:
        session.dataset = None
        return
    table = _clone_table(session.source)
    if session.scale_divisor is not None:
        table = scale(table, session.scale_divisor)
    if session.val_fraction > 0.0:
        table.split_tail(session.val_fraction)
    if session.standardize:
        fitted = Standardize().fit(table, table.train_idx)
        table = fitted.transform(table)
    session.dataset = table


def _clone_table(dataset: Dataset) -> Dataset:
    """Copy index lists so a later split does not move the raw table."""
    return Dataset(
        features=dataset.features,
        targets=dataset.targets,
        sample_weight=dataset.sample_weight,
        feature_names=dataset.feature_names,
        target_name=dataset.target_name,
        image_shape=dataset.image_shape,
        train_idx=np.array(dataset.train_idx, copy=True),
        val_idx=np.array(dataset.val_idx, copy=True),
        test_idx=np.array(dataset.test_idx, copy=True),
    )


def _rows(session: Session, query: dict[str, list[str]]) -> dict[str, Any]:
    """One page of the dataset. Image rows carry pixels instead of 784 columns."""
    table = session.dataset
    if table is None:
        msg = "Load a table first."
        raise RuntimeError(msg)
    offset = max(0, int(query.get("offset", ["0"])[0]))
    limit = min(50, max(1, int(query.get("limit", ["8"])[0])))
    end = min(offset + limit, table.n_rows)
    records: list[dict[str, Any]] = []
    for index in range(offset, end):
        record: dict[str, Any] = {"index": index}
        if table.targets is not None:
            record["target"] = _jsonable(table.targets[index])
        if table.image_shape is not None:
            record["pixels"] = _pixel_bytes(table.features[index])
        else:
            show = min(8, table.n_features)
            values = np.asarray(table.features[index, :show], dtype=np.float64)
            record["values"] = [float(value) for value in values]
            record["names"] = list(table.feature_names[:show])
        records.append(record)
    page = _summary(table)
    page["offset"] = offset
    page["records"] = records
    page["image_shape"] = None if table.image_shape is None else list(table.image_shape)
    return page


def _pixel_bytes(row: NDArray[np.generic]) -> list[int]:
    """Map one image row onto 0..255 for the preview only."""
    if row.dtype == np.uint8:
        return [int(value) for value in row]
    values = np.asarray(row, dtype=np.float64)
    low = float(np.min(values))
    high = float(np.max(values))
    if high <= low:
        return [0] * int(values.size)
    scaled = np.rint((values - low) / (high - low) * 255.0)
    return [int(value) for value in scaled]


def _worked(session: Session) -> Dataset:
    """Return the prepared table or raise when none is loaded."""
    table = session.dataset
    if table is None:
        msg = "Load a table first."
        raise RuntimeError(msg)
    return table


def _summary(table: Dataset) -> dict[str, Any]:
    """Return the counts the model screen needs before it trains."""
    return {
        "rows": table.n_rows,
        "features": table.n_features,
        "train": int(table.train_idx.size),
        "val": int(table.val_idx.size),
        "test": int(table.test_idx.size),
        "target": table.target_name,
    }


def _open_session(book: SessionBook, payload: dict[str, Any]) -> dict[str, Any]:
    """Create a train or infer session and select it."""
    name = str(payload.get("name", "")).strip() or "Session"
    purpose = str(payload.get("purpose", "train"))
    session = book.add(Session(name, purpose))
    return _session_view(session)


def _activate(book: SessionBook, session_id: str) -> dict[str, Any]:
    """Make an existing session the active one."""
    if session_id not in book.sessions:
        msg = f"No session {session_id!r}."
        raise KeyError(msg)
    book.active_id = session_id
    return _session_view(book.sessions[session_id])


def _drop(book: SessionBook, session_id: str) -> dict[str, Any]:
    """Delete a session. The page returns to the list."""
    book.sessions.pop(session_id, None)
    if book.active_id == session_id:
        book.active_id = None
    return _session_list(book)


def _session_list(book: SessionBook) -> dict[str, Any]:
    """Return every session and which one is open."""
    return {
        "active": book.active_id,
        "sessions": [_session_view(item) for item in book.sessions.values()],
    }


def _session_view(session: Session) -> dict[str, Any]:
    """JSON card for one session."""
    table = session.dataset
    fitted = session.model is not None and hasattr(session.model, "n_features_in_")
    return {
        "id": session.id,
        "name": session.name,
        "purpose": session.purpose,
        "model": session.model_name,
        "has_data": table is not None,
        "rows": 0 if table is None else table.n_rows,
        "features": 0 if table is None else table.n_features,
        "scale_divisor": session.scale_divisor,
        "standardize": session.standardize,
        "val": 0 if table is None else int(table.val_idx.size),
        "test": 0 if table is None else int(table.test_idx.size),
        "val_fraction": session.val_fraction,
        "fitted": fitted,
    }


def _predict_page(session: Session, query: dict[str, list[str]]) -> dict[str, Any]:
    """Score one page. Inference reads every row, not only the training list."""
    model = _require_fitted(session)
    table = _worked(session)
    offset = max(0, int(query.get("offset", ["0"])[0]))
    limit = min(50, max(1, int(query.get("limit", ["8"])[0])))
    end = min(offset + limit, table.n_rows)
    indices = np.arange(offset, end, dtype=np.int64)
    features, targets, _weights = table.take(indices)
    if hasattr(model, "predict"):
        raw = model.predict(features)
    elif hasattr(model, "transform"):
        raw = model.transform(features)
    else:
        msg = "This model has no predict or transform."
        raise RuntimeError(msg)
    predicted = np.asarray(raw)
    records: list[dict[str, Any]] = []
    for position, index in enumerate(indices):
        record: dict[str, Any] = {
            "index": int(index),
            "prediction": _cell(predicted[position]),
        }
        if targets is not None:
            record["target"] = _jsonable(targets[position])
        if table.image_shape is not None:
            record["pixels"] = _pixel_bytes(table.features[int(index)])
        records.append(record)
    page = _summary(table)
    page["offset"] = offset
    page["records"] = records
    page["image_shape"] = None if table.image_shape is None else list(table.image_shape)
    return page


def _cell(value: Any) -> Any:
    """Turn one prediction into JSON: a scalar or a list of scalars."""
    array = np.asarray(value)
    if array.shape == ():
        return _jsonable(array.item())
    return [_cell(item) for item in array]


def _fit(session: Session, mode: str) -> dict[str, Any]:
    """Fit the current model on the loaded table."""
    if session.purpose != "train":
        msg = "An infer session scores a loaded model. Fit in a train session."
        raise RuntimeError(msg)
    if session.model is None or session.dataset is None:
        msg = "Choose a model and load a table first."
        raise RuntimeError(msg)
    spec = spec_by_name(session.model_name or "")
    features, target, _weights = session.dataset.take(session.dataset.train_idx)
    if spec.requires_y and target is None:
        msg = "This model needs a target column."
        raise RuntimeError(msg)
    planned = int(getattr(session.model, "max_iter", 1) or 1)
    gate = threading.Event()
    gate.set()
    session.progress.update(
        running=True,
        epoch=0,
        epochs=planned,
        phase="feedforward",
        tick=0,
        gate=gate,
        stop=False,
    )
    token = bind_progress(session.progress)
    try:
        if spec.iterative and mode == "step":
            session.model.partial_fit(features, target, epochs=1)
        elif target is None:
            session.model.fit(features)
        else:
            session.model.fit(features, target)
    finally:
        release_progress(token)
        session.progress["running"] = False
    return _state(session)


def _playback(session: Session, payload: dict[str, Any]) -> dict[str, Any]:
    """Ask the model for one film, if it knows how."""
    model = _require_fitted(session)
    if not hasattr(model, "playback"):
        msg = "This model has no neuron film."
        raise RuntimeError(msg)
    table = session.dataset
    if table is None:
        msg = "Load a table first."
        raise RuntimeError(msg)
    index = int(payload.get("sample_index", 0))
    if index < 0 or index >= table.n_rows:
        msg = f"Row {index} is outside the table of {table.n_rows} rows."
        raise ValueError(msg)
    rows, targets, _weights = table.take(np.asarray([index], dtype=np.int64))
    row = rows[0]
    target = None if targets is None else targets[0]
    film = model.playback(
        row,
        None if target is None else [target],
        sample_index=index,
        with_backward=bool(payload.get("with_backward", True)),
    )
    payload_out: dict[str, Any] = film.to_dict()
    return payload_out


def _layout(session: Session) -> dict[str, Any]:
    """Layer widths for the picture, before or after ``fit``.

    Input width is the feature count. Hidden widths are the configured
    rows. Output width is the class count, or the target width.
    """
    model = session.model
    if model is None or not hasattr(model, "hidden_layer_sizes"):
        return {"sizes": [], "note": "This model has no neuron film."}
    if hasattr(model, "layer_sizes_"):
        return {"sizes": [int(size) for size in model.layer_sizes_], "note": ""}
    table = session.dataset
    if table is None:
        return {"sizes": [], "note": "Load a table to draw the network."}
    hidden = [int(size) for size in model.hidden_layer_sizes]
    sizes = [int(table.n_features), *hidden, _output_count(session)]
    return {"sizes": sizes, "note": ""}


def _output_count(session: Session) -> int:
    """How many neurons the output layer will have once ``fit`` runs."""
    model = session.model
    if hasattr(model, "n_outputs_"):
        return int(model.n_outputs_)
    table = session.dataset
    if table is None or table.targets is None:
        return 1
    target = np.asarray(table.targets)[np.asarray(table.train_idx)]
    if session.model_name == "mlp_classifier":
        return max(int(np.unique(target).shape[0]), 2)
    if target.ndim == 1:
        return 1
    return int(target.shape[1])


def _report_html(
    session: Session, *, theme: Literal["light", "dark"] = "light"
) -> bytes:
    """Render the observation report for a fitted model."""
    model = _require_fitted(session)
    return export_html(model.observe(), theme=theme).encode("utf-8")


def _progress(session: Session) -> dict[str, Any]:
    """Epoch counter written by the fit that is running on another thread."""
    box = session.progress
    gate = box.get("gate")
    paused = bool(box.get("running")) and gate is not None and not gate.is_set()
    return {
        "running": bool(box.get("running")),
        "paused": paused,
        "epoch": int(box.get("epoch", 0)),
        "epochs": int(box.get("epochs", 0)),
        "phase": str(box.get("phase", "")),
        "tick": int(box.get("tick", 0)),
    }


def _control(session: Session, payload: dict[str, Any]) -> dict[str, Any]:
    """Pause or resume the fit that is blocked between batches."""
    gate = session.progress.get("gate")
    if gate is None or not session.progress.get("running"):
        msg = "Nothing is teaching."
        raise RuntimeError(msg)
    if bool(payload.get("paused")):
        gate.clear()
        session.progress["phase"] = "paused"
    else:
        gate.set()
    return _progress(session)


def _release_paused_fits(server: StudioServer) -> None:
    """Unblock every paused fit so the process can leave."""
    for session in server.book.sessions.values():
        session.progress["stop"] = True
        gate = session.progress.get("gate")
        if gate is not None:
            gate.set()


def _check(session: Session, query: dict[str, list[str]]) -> dict[str, Any]:
    """Score rows that were not used to teach.

    Official test rows win. Otherwise the hold-out tail is used. A table
    with neither split has nothing to check.
    """
    model = _require_fitted(session)
    table = _worked(session)
    if not hasattr(model, "predict"):
        msg = "This model has no predict."
        raise RuntimeError(msg)
    split, indices = _held_out(table)
    page = _summary(table)
    page["split"] = split
    page["image_shape"] = None if table.image_shape is None else list(table.image_shape)
    if indices.size == 0:
        page["offset"] = 0
        page["scored"] = 0
        page["correct"] = 0
        page["records"] = []
        page["note"] = (
            "Every row was used for teaching. Hold some back, "
            "or load a zip that includes test images."
        )
        return page
    features, targets, _weights = table.take(indices)
    predicted = np.asarray(model.predict(features))
    matches = [
        _same(predicted[position], None if targets is None else targets[position])
        for position in range(indices.size)
    ]
    offset = max(0, int(query.get("offset", ["0"])[0]))
    limit = min(50, max(1, int(query.get("limit", ["8"])[0])))
    if offset >= indices.size:
        offset = 0
    end = min(offset + limit, indices.size)
    records: list[dict[str, Any]] = []
    for position in range(offset, end):
        index = int(indices[position])
        record: dict[str, Any] = {
            "index": index,
            "prediction": _cell(predicted[position]),
            "match": matches[position],
        }
        if targets is not None:
            record["target"] = _jsonable(targets[position])
        if table.image_shape is not None:
            record["pixels"] = _pixel_bytes(table.features[index])
        else:
            show = min(8, table.n_features)
            values = np.asarray(table.features[index, :show], dtype=np.float64)
            record["values"] = [float(value) for value in values]
            record["names"] = list(table.feature_names[:show])
        records.append(record)
    page["offset"] = offset
    page["scored"] = int(indices.size)
    page["correct"] = int(sum(matches))
    page["records"] = records
    page["note"] = ""
    return page


def _held_out(table: Dataset) -> tuple[str, NDArray[np.int64]]:
    """Prefer the official test list, then the hold-out tail."""
    if int(table.test_idx.size):
        return "test", np.asarray(table.test_idx, dtype=np.int64)
    if int(table.val_idx.size):
        return "check", np.asarray(table.val_idx, dtype=np.int64)
    return "none", np.empty(0, dtype=np.int64)


def _same(predicted: Any, target: Any) -> bool:
    """Whether one prediction agrees with the stored answer."""
    if target is None:
        return False
    left = np.asarray(predicted)
    right = np.asarray(target)
    if left.dtype.kind == "f" or right.dtype.kind == "f":
        return bool(
            np.allclose(
                np.asarray(left, dtype=np.float64).reshape(-1),
                np.asarray(right, dtype=np.float64).reshape(-1),
                rtol=0.0,
                atol=1e-6,
            )
        )
    return bool(np.array_equal(left.reshape(-1), right.reshape(-1)))


def _state(session: Session) -> dict[str, Any]:
    """Return a short JSON snapshot of the fitted model."""
    if session.model is None or not hasattr(session.model, "report"):
        return {"fitted": False}
    try:
        observation = session.model.observe()
    except Exception:
        return {"fitted": False, "model": session.model_name}
    report = observation.report
    return {
        "fitted": True,
        "model": session.model_name,
        "summary": "" if report is None else report.summary,
        "metrics": {} if report is None else report.metrics,
        "diagnostics": [
            {"code": item.code, "severity": item.severity, "message": item.message}
            for item in observation.diagnostics
        ],
    }


def _require_fitted(session: Session) -> Any:
    """Return the model or raise if ``fit`` has not been called."""
    if session.model is None:
        msg = "Choose a model first."
        raise RuntimeError(msg)
    return session.model


def _jsonable(value: Any) -> Any:
    """Convert NumPy scalars so ``json.dumps`` accepts the payload."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
