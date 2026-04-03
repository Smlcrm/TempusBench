"""
Unified logging utility for both standard Python logging and TensorBoard logging.

This logger combines standard Python logging (console/file) with TensorBoard logging
capabilities, providing a single interface for all logging needs in the benchmarking pipeline.
"""

import hashlib
import io
import json
import logging
import os
import sys
import numpy as np
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import re


def _sanitize_tb_component(name: str, max_len: int = 96) -> str:
    """Make a string safe for TensorBoard tag paths (no slashes, trimmed)."""
    s = re.sub(r"[^\w\-.]+", "_", name.replace("/", "__").replace(" ", "_"))
    s = s.strip("_") or "unknown"
    return s[:max_len] if len(s) > max_len else s


def _forecast_window_segment_and_title(forecast_start_ts) -> Tuple[str, str]:
    """
    Path segment and chart title fragment for a validation window.

    Uses the timestamp of the **first step in the forecast horizon** (same as the
    first entry in ``timestamps_pred``) so TensorBoard groups by calendar origin,
    not by rolling-window order. Segment is ``o`` + zero-padded pandas-ns int
    so lexical sort matches time order.

    Raises:
        ValueError: If ``forecast_start_ts`` is missing or not a finite, parseable time.
    """
    if forecast_start_ts is None:
        raise ValueError(
            "forecast_start_timestamp is required for TensorBoard forecast tags"
        )
    import pandas as pd

    # pd.Timestamp() rejects numpy.str_ in some versions; to_datetime accepts str_,
    # datetime64, Python datetime, and plain str.
    try:
        t = pd.to_datetime(forecast_start_ts, errors="raise")
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"forecast_start_timestamp is not parseable as a time: {forecast_start_ts!r}"
        ) from e
    if not isinstance(t, pd.Timestamp):
        t = pd.Timestamp(t)
    if pd.isna(t):
        raise ValueError(
            f"forecast_start_timestamp is NaT or invalid: {forecast_start_ts!r}"
        )
    ns = int(t.value)
    seg = f"o{ns:020d}"
    disp = t.isoformat(sep=" ", timespec="seconds")
    return seg, disp


def _hyperparam_trial_segment(hparams: Mapping[str, Any]) -> Tuple[str, str]:
    """
    Disambiguate TensorBoard tags when several grid points share the same forecast
    origin (same ``o{ns}``): path segment ``h{12-hex}`` plus a short subtitle.
    """
    if not hparams:
        return "default", ""
    key = json.dumps(sorted(hparams.items()), default=str, sort_keys=True)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    human = ",".join(f"{k}={v}" for k, v in sorted(hparams.items()))
    human = _sanitize_tb_component(human)
    if len(human) > 72:
        human = human[:69] + "..."
    return f"h{digest}", human


def _forecast_custom_category_id(sm: str, hyperparam_segment: str) -> str:
    """Stable Custom Scalars category key: sanitized model + hyperparam trial segment."""
    return _sanitize_tb_component(f"{sm}__{hyperparam_segment}")


def _forecast_custom_category_display_title(
    model_name: str,
    hyperparam_segment: str,
    hyperparam_subtitle: str,
) -> str:
    """Custom Scalars sidebar title: model only for the default trial, else model · params."""
    if hyperparam_segment == "default" and not (hyperparam_subtitle or "").strip():
        return model_name
    if hyperparam_subtitle:
        return f"{model_name} · {hyperparam_subtitle}"
    return f"{model_name} · {hyperparam_segment}"


# Defer TensorFlow import to avoid macOS mutex crash (TF 2.20+ on ARM).
# When TEMPUSBENCH_DISABLE_TENSORBOARD=1, skip TF entirely and use no-op writer.
_DISABLE_TB = os.environ.get("TEMPUSBENCH_DISABLE_TENSORBOARD", "").strip() in ("1", "true", "yes")


def _get_tf():
    """Lazy import TensorFlow (avoids mutex crash when disabled)."""
    if _DISABLE_TB:
        return None
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    return tf


def _get_hp():
    """Lazy import TensorBoard hparams plugin."""
    if _DISABLE_TB:
        return None
    from tensorboard.plugins.hparams import api as hp
    return hp


class _NoOpTfWriter:
    """No-op TensorBoard writer when TEMPUSBENCH_DISABLE_TENSORBOARD=1."""

    def as_default(self):
        from contextlib import nullcontext
        return nullcontext()

    def flush(self):
        pass

    def close(self):
        pass


class LogManager:
    """
    Unified logging utility for orchestration, status messages, and TensorBoard logging.

    This logger provides:
    - Standard Python logging to console and/or file
    - TensorBoard logging for metrics, plots, and hyperparameters
    - Independent control over each logging type

    This class implements a singleton pattern where the first instance created is stored
    in `LogManager.log_manager`, and subsequent instantiations return the same instance.
    """

    log_manager: Optional["LogManager"] = None

    @staticmethod
    def get_logger() -> "LogManager":
        """
        Get the LogManager singleton instance.

        Returns:
            LogManager: The singleton LogManager instance.

        Raises:
            RuntimeError: If LogManager has not been initialized yet.
        """
        if LogManager.log_manager is None:
            raise RuntimeError(
                "LogManager has not been initialized. Please create a LogManager instance first."
            )
        return LogManager.log_manager

    @classmethod
    def reset_singleton(cls) -> None:
        """Close and clear the singleton so a new run can open different log files.

        Long-lived worker processes (e.g. multiple benchmark plan steps) must call this
        after each :class:`~tempus_bench.run_benchmark.BenchmarkRunner` exits; otherwise
        later runs would keep using the first run's handlers and paths.
        """
        inst = cls.log_manager
        if inst is None:
            return
        try:
            inst.close()
        except Exception:
            pass
        cls.log_manager = None

    def __new__(cls, *args, **kwargs):
        """
        Override __new__ to implement singleton pattern.

        Returns the first instance created, stored in cls.log_manager.
        If no instance exists yet, creates a new one and stores it.
        """
        if cls.log_manager is None:
            cls.log_manager = super().__new__(cls)
            cls.log_manager._initialized = False
        return cls.log_manager

    def __init__(
        self,
        logs_path: str,
        name: str = "Tempus Bench",
        enable_logging: bool = True,
        console_logging: bool = True,
        file_logging: bool = True,
        console_log_level: str = "INFO",
        file_log_level: str = "DEBUG",
        tf_logs_path: Optional[str] = None,
        tensorboard_logging: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize logger with configuration for both standard and TensorBoard logging.

        Note: Logger and SummaryWriter objects are always created regardless of flag values.
        The actual logging behavior is controlled by `enable_logging` and `tensorboard_logging`.

        Note: This method only initializes the logger on the first call. Subsequent
        instantiations will return the same instance without re-initialization.

        Args:
            logs_path: Directory to write standard log files
            name: Name for the logger instance
            enable_logging: Controls whether standard logging methods actually log (Logger is always created)
            console_logging: Whether to create console handler
            file_logging: Whether to create file handler
            console_log_level: Console logging level (DEBUG, INFO, WARNING, ERROR)
            file_log_level: File logging level (DEBUG, INFO, WARNING, ERROR)
            tf_logs_path: Directory to write TensorBoard log files (optional, defaults to logs_path/tensorboard)
            tensorboard_logging: Controls whether TensorBoard logging methods actually log (SummaryWriter is always created)
        """
        # Prevent re-initialization of singleton instance
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.name = name
        self.enable_logging = enable_logging
        self.tensorboard_logging = tensorboard_logging
        self.verbose = verbose
        # Always create log directory
        Path(logs_path).mkdir(parents=True, exist_ok=True)

        # Always setup standard logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels
        # Clear any existing handlers to avoid duplicates
        self.logger.handlers.clear()

        # Always create console handler (configurable level)
        if console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            # Convert string level to logging constant
            log_level = getattr(logging, console_log_level.upper(), logging.INFO)
            console_handler.setLevel(log_level)
            console_format = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(console_format)
            self.logger.addHandler(console_handler)

        # Always create file handler (configurable level)
        if file_logging:
            log_file = Path(logs_path) / f"{name}.log"
            file_handler = logging.FileHandler(log_file)
            # Convert string level to logging constant
            log_level = getattr(logging, file_log_level.upper(), logging.DEBUG)
            file_handler.setLevel(log_level)
            file_format = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)

        # Prevent propagation to root logger
        self.logger.propagate = False

        # Always setup TensorBoard logging (unless TEMPUSBENCH_DISABLE_TENSORBOARD=1)
        if tf_logs_path is None:
            # Default to tensorboard subdirectory of logs_path
            self.tf_logs_path = str(Path(logs_path).parent / "tensorboard")
        else:
            self.tf_logs_path = tf_logs_path

        # Create TensorBoard writer (or no-op when disabled)
        tf = _get_tf()
        if tf is not None:
            os.makedirs(self.tf_logs_path, exist_ok=True)
            self.tf_writer = tf.summary.create_file_writer(self.tf_logs_path)
        else:
            self.tf_writer = _NoOpTfWriter()

        # Custom Scalars: one category per (model, hyperparam trial); charts per task/window/variate
        self._forecast_custom_charts: Dict[str, Dict[str, Tuple[str, str, str]]] = {}
        self._forecast_custom_category_titles: Dict[str, str] = {}

        # HParams: experiment config written once on root writer; each trial uses its own sub-run
        self._hparams_experiment_config_written = False

        # Mark as initialized
        self._initialized = True

    def _should_log(self) -> bool:
        """
        Check if standard logging should occur.

        Returns:
            bool: True if standard logging is enabled, False otherwise.
        """
        return self.enable_logging

    def _should_log_tensorboard(self) -> bool:
        """
        Check if TensorBoard logging should occur.

        Returns:
            bool: True if TensorBoard logging is enabled, False otherwise.
        """
        return self.tensorboard_logging and not _DISABLE_TB

    # ==================== Context Manager Methods ====================

    def __enter__(self):
        """
        Enter the context manager.

        Returns:
            LoggerManager: Returns self for use in with statements.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager, ensuring all logs are flushed and resources closed.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.

        Returns:
            bool: False, indicating that exceptions should not be suppressed.
        """
        self.close()
        return False  # Don't suppress exceptions

    # ==================== Standard Logging Methods ====================

    def info(self, module: str, message: str, is_verbose: bool = False):
        """
        Log an informational message with module context.

        Args:
            module (str): Module name for context.
            message (str): Message to log.
        """
        if self._should_log() and (self.verbose or not is_verbose):
            self.logger.info(f"[{module}] {message}")

    def warning(self, module: str, message: str):
        """
        Log a warning message with module context.

        Args:
            module (str): Module name for context.
            message (str): Message to log.
        """
        if self._should_log():
            self.logger.warning(f"[{module}] {message}")

    def error(self, module: str, message: str):
        """
        Log an error message with module context.

        Args:
            module (str): Module name for context.
            message (str): Message to log.
        """
        if self._should_log():
            self.logger.error(f"[{module}] {message}")

    def success(self, module: str, message: str):
        """
        Log a success message with module context.

        Args:
            module (str): Module name for context.
            message (str): Message to log.
        """
        if self._should_log():
            self.logger.info(f"[{module}] SUCCESS: {message}")

    def debug(self, module: str, message: str):
        """
        Log a debug message with module context.

        Args:
            module (str): Module name for context.
            message (str): Message to log.
        """
        if self._should_log():
            self.logger.debug(f"[{module}] {message}")

    def progress(self, module: str, message: str):
        """
        Log a progress message with module context.

        Args:
            module (str): Module name for context.
            message (str): Message to log.
        """
        if self._should_log():
            self.logger.info(f"[{module}] PROGRESS: {message}")

    # ==================== TensorBoard Logging Methods ====================

    def log_metrics(self, metrics: dict, step: int, model_name: str = ""):
        """
        Log evaluation metrics to TensorBoard.

        This method logs metrics to TensorBoard, handling various metric value types
        including scalars, arrays, and nested dictionaries. NaN values are skipped.

        Args:
            metrics (dict): Dictionary of metrics to log. Values may be scalars,
                arrays, or nested dictionaries.
            step (int): The current step (e.g., epoch, batch, or experiment ID).
            model_name (str): Optional prefix for metric names to group them in
                TensorBoard. Defaults to empty string.

        Returns:
            None: Logs metrics to TensorBoard if TensorBoard logging is enabled.
        """
        if not self._should_log_tensorboard():
            return

        tf = _get_tf()
        if tf is None:
            return

        group_prefix = f"{model_name}/" if model_name else ""
        with self.tf_writer.as_default():
            for metric_name, value in metrics.items():
                if value is None or (
                    isinstance(value, (float, int)) and np.isnan(value)
                ):
                    continue

                if isinstance(value, dict):  # For nested results like quantile losses
                    for sub_metric, sub_value in value.items():
                        tag = f"{group_prefix}{metric_name}/{sub_metric}"
                        tf.summary.scalar(tag, sub_value, step=step)
                elif isinstance(value, (np.ndarray, list)):
                    # For arrays log each element and the mean
                    value_arr = np.asarray(value)
                    tf.summary.scalar(
                        f"{group_prefix}{metric_name}/mean",
                        np.nanmean(value_arr),
                        step=step,
                    )
                    for i, elem in enumerate(value_arr):
                        tf.summary.scalar(
                            f"{group_prefix}{metric_name}/series_{i}", elem, step=step
                        )
                elif isinstance(value, (float, np.floating, int)):
                    tf.summary.scalar(f"{group_prefix}{metric_name}", value, step=step)
                else:
                    # Skip unsupported metric types silently (TensorBoard-only logging)
                    pass
        self.tf_writer.flush()

    def _register_forecast_custom_chart(
        self,
        *,
        model_name: str,
        sm: str,
        task_name: str,
        st: str,
        window_segment: str,
        window_title: str,
        hyperparam_segment: str,
        hyperparam_subtitle: str,
        vi: int,
        vw: int,
        tag_actual: str,
        tag_pred: str,
    ) -> None:
        """Record one Custom Scalars chart under category **model + hyperparam trial**.

        Matches TensorBoard's official pattern: one ``Chart`` with
        ``MultilineChartContent(tag=[regex_a, regex_b])`` so **both** series draw on
        the **same** axes (see ``custom_scalar_demo.py`` in the tensorboard repo:
        e.g. cosine + sine in one ``title="wave trig functions"`` chart).
        """
        cat_id = _forecast_custom_category_id(sm, hyperparam_segment)
        if cat_id not in self._forecast_custom_category_titles:
            disp = _forecast_custom_category_display_title(
                model_name, hyperparam_segment, hyperparam_subtitle
            )
            self._forecast_custom_category_titles[cat_id] = disp[:256]
        chart_id = f"{st}/{window_segment}/v{vi:0{vw}d}"
        title = f"{task_name} · {window_title} · v{vi}"
        if cat_id not in self._forecast_custom_charts:
            self._forecast_custom_charts[cat_id] = {}
        self._forecast_custom_charts[cat_id][chart_id] = (title, tag_actual, tag_pred)

    def _flush_forecast_custom_scalars_layout(self) -> None:
        """Write Custom Scalars layout (``custom_scalars__config__`` summary).

        Layout mirrors TB demo: ``Category`` = **model** (or **model · hyperparams**
        when tuning), each ``Chart`` is multiline with two tag regexes (actual +
        predicted). View in the **Custom Scalars** tab; the default **Scalars** tab
        lists raw tags separately.
        """
        if not self._forecast_custom_charts or not self._should_log_tensorboard():
            return
        tf = _get_tf()
        if tf is None:
            return
        try:
            from tensorboard.plugins.custom_scalar import layout_pb2
            from tensorboard.plugins.custom_scalar.summary import pb as custom_scalar_pb
        except ImportError:
            return

        def _cat_key(cat_id: str) -> Tuple[str, str]:
            title = self._forecast_custom_category_titles.get(cat_id, cat_id)
            return (title.lower(), cat_id)

        categories = []
        for cat_id in sorted(self._forecast_custom_charts.keys(), key=_cat_key):
            charts_pb = []
            for chart_id in sorted(self._forecast_custom_charts[cat_id].keys()):
                chart_title, ta, tp = self._forecast_custom_charts[cat_id][chart_id]
                charts_pb.append(
                    layout_pb2.Chart(
                        title=chart_title[:200],
                        multiline=layout_pb2.MultilineChartContent(
                            tag=[re.escape(ta), re.escape(tp)],
                        ),
                    )
                )
            cat_title = self._forecast_custom_category_titles.get(cat_id, cat_id)[:256]
            categories.append(layout_pb2.Category(title=cat_title, chart=charts_pb))

        layout = layout_pb2.Layout(category=categories)
        try:
            summary_proto = custom_scalar_pb(layout)
            raw = summary_proto.SerializeToString()
        except Exception:
            return

        with self.tf_writer.as_default():
            try:
                tf.summary.experimental.write_raw_pb(raw, step=0)
            except Exception:
                pass
        self.tf_writer.flush()

    def log_forecast_window_scalars(
        self,
        *,
        task_name: str,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        forecast_start_timestamp,
        hyperparameters: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Log actual vs predicted as TensorBoard **Scalars** (no PNG / image summaries).

        Tags mirror ``(model, task, forecast_origin, hyperparam_trial, variate)``,
        with **model** first so the Scalars sidebar groups under each model:

        ``forecast/<model>/<task>/o<nanoseconds>/h<hash-or-default>/v<variate>/{actual|predicted}``

        The ``h…`` segment separates **hyperparameter grid points** that share the
        same forecast origin (otherwise scalar tags collide and TensorBoard draws
        one mangled series). Use ``hyperparameters={}`` for a single configuration
        (e.g. foundation models).

        The ``o…`` segment is the **first validation timestamp** (forecast start),
        zero-padded so tag order matches time order.

        **Step** is the **forecast horizon index** ``0 … H-1`` (within that window).

        The **Custom Scalars** tab gets a layout aligned with TensorBoard's
        ``custom_scalar_demo.py``: **category** = model, or **model · hyperparams**
        when the trial is not the default empty grid, one **multiline chart** per
        (task, forecast origin, variate) with **two tag regexes** (actual +
        predicted). Use **Custom Scalars**, not only **Scalars**, for that overlay.

        Args:
            task_name: Benchmark task (folder name).
            model_name: Model name.
            y_true: Shape ``(H,)`` or ``(H, V)``.
            y_pred: Same shape as ``y_true``.
            forecast_start_timestamp: First timestep in the forecast (e.g.
                ``timestamps_pred[0]``). Required and must parse to a valid time.
            hyperparameters: Grid point used to produce ``y_true``/``y_pred`` (may be
                ``{}`` for a single-run model).

        Raises:
            ValueError: If ``forecast_start_timestamp`` is missing or invalid.
        """
        if not self._should_log_tensorboard():
            return

        tf = _get_tf()
        if tf is None:
            return

        yt = np.asarray(y_true, dtype=np.float64)
        yp = np.asarray(y_pred, dtype=np.float64)
        if yt.ndim == 1:
            yt = yt.reshape(-1, 1)
        if yp.ndim == 1:
            yp = yp.reshape(-1, 1)
        if yt.shape != yp.shape or yt.size == 0:
            return

        hp: Dict[str, Any] = (
            {} if hyperparameters is None else dict(hyperparameters)
        )
        wseg, wtitle = _forecast_window_segment_and_title(forecast_start_timestamp)
        hseg, hsubtitle = _hyperparam_trial_segment(hp)

        h, v_dim = yt.shape
        vw = max(2, len(str(v_dim - 1)))
        sm = _sanitize_tb_component(model_name)
        st = _sanitize_tb_component(task_name)
        prefix = f"forecast/{sm}/{st}/{wseg}/{hseg}"

        with self.tf_writer.as_default():
            for vi in range(v_dim):
                tag_actual = f"{prefix}/v{vi:0{vw}d}/actual"
                tag_pred = f"{prefix}/v{vi:0{vw}d}/predicted"
                self._register_forecast_custom_chart(
                    model_name=model_name,
                    sm=sm,
                    task_name=task_name,
                    st=st,
                    window_segment=wseg,
                    window_title=wtitle,
                    hyperparam_segment=hseg,
                    hyperparam_subtitle=hsubtitle,
                    vi=vi,
                    vw=vw,
                    tag_actual=tag_actual,
                    tag_pred=tag_pred,
                )
                for t in range(h):
                    v_true = float(yt[t, vi])
                    v_pred = float(yp[t, vi])
                    if np.isfinite(v_true):
                        tf.summary.scalar(tag_actual, v_true, step=t)
                    if np.isfinite(v_pred):
                        tf.summary.scalar(tag_pred, v_pred, step=t)
        self.tf_writer.flush()
        self._flush_forecast_custom_scalars_layout()

    def log_figure(
        self,
        figure,
        tag: str,
        step: int,
        *,
        dpi: int = 100,
    ):
        """
        Log a Matplotlib figure to TensorBoard.

        Args:
            figure: Matplotlib figure object to log.
            tag (str): Tag for the figure in TensorBoard.
            step (int): Step number for this figure.
            dpi: Resolution for the PNG written to TensorBoard.

        Returns:
            None: Logs figure to TensorBoard if TensorBoard logging is enabled.
        """
        if not self._should_log_tensorboard():
            return

        tf = _get_tf()
        if tf is None:
            return

        buf = io.BytesIO()
        figure.savefig(buf, format="png", dpi=int(dpi), bbox_inches="tight")
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue())
        image = tf.expand_dims(image, 0)
        with self.tf_writer.as_default():
            tf.summary.image(tag, image, step=step)
        self.tf_writer.flush()

    def log_image_file(self, image_path: str, tag: str, step: int):
        """
        Log an image from disk to TensorBoard.

        Args:
            image_path (str): Path to the image file to log.
            tag (str): Tag for the image in TensorBoard.
            step (int): Step number for this image.

        Returns:
            None: Logs image to TensorBoard if TensorBoard logging is enabled.
                Silently skips logging if file cannot be read.
        """
        if not self._should_log_tensorboard():
            return

        tf = _get_tf()
        if tf is None:
            return

        try:
            with open(image_path, "rb") as f:
                data = f.read()
            image = tf.image.decode_image(data)
            image = tf.expand_dims(image, 0)
            with self.tf_writer.as_default():
                tf.summary.image(tag, image, step=step)
            self.tf_writer.flush()
        except Exception as e:
            # Silently skip failed image logging (TensorBoard-only logging)
            pass

    def log_training_progress(
        self,
        model_name: str,
        epoch: int,
        loss: float,
        val_loss: Optional[float] = None,
        step: Optional[int] = None,
    ):
        """
        Log training progress for real-time monitoring.

        This method logs training and validation losses to TensorBoard for
        real-time monitoring of model training progress.

        Args:
            model_name (str): Name of the model being trained.
            epoch (int): Current epoch number.
            loss (float): Training loss value.
            val_loss (Optional[float]): Validation loss value. If None, only
                training loss is logged.
            step (Optional[int]): Global step for TensorBoard. If None, uses
                epoch as the step.

        Returns:
            None: Logs training progress to TensorBoard if TensorBoard logging
                is enabled.
        """
        if not self._should_log_tensorboard():
            return

        tf = _get_tf()
        if tf is None:
            return

        if step is None:
            step = epoch

        # Log to TensorBoard
        with self.tf_writer.as_default():
            tf.summary.scalar(f"{model_name}/train_loss", loss, step=step)
            if val_loss is not None:
                tf.summary.scalar(f"{model_name}/val_loss", val_loss, step=step)
        self.tf_writer.flush()

    @staticmethod
    def _coerce_hparam_value(v):
        """Convert a value to a TensorBoard-hparams-compatible scalar, or None to skip."""
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, (float, np.floating)):
            return float(v)
        if isinstance(v, str):
            return v
        return None

    def _ensure_hparams_experiment_config(self, tf, hp) -> None:
        """
        Write ``hp.hparams_config`` once on the root logdir (TensorBoard best practice).

        Declares searchable hyperparameter columns and metric tags so the HParams
        tab can compare trials. See tensorboard.plugins.hparams API docs.
        """
        if self._hparams_experiment_config_written:
            return
        from tempus_bench.pipeline.metric_registry import MetricRegistry

        registry = MetricRegistry()
        metric_names = sorted(
            set(registry.deterministic_metrics) | set(registry.stochastic_metrics)
        )
        metric_infos = [
            hp.Metric(
                name,
                display_name=name.replace("_", " ").upper(),
                dataset_type=hp.Metric.VALIDATION,
            )
            for name in metric_names
        ]
        hparam_defs = [
            hp.HParam("model"),
            hp.HParam("task"),
            hp.HParam("window"),
            hp.HParam("sp"),
            hp.HParam("theta_method"),
            hp.HParam("use_reduced_rank"),
        ]
        with self.tf_writer.as_default():
            hp.hparams_config(hparams=hparam_defs, metrics=metric_infos)
        self.tf_writer.flush()
        self._hparams_experiment_config_written = True

    @staticmethod
    def _scalar_metrics_from_eval(metrics: dict) -> Dict[str, float]:
        """Pick finite float metrics suitable for tf.summary.scalar / HParams."""
        out: Dict[str, float] = {}
        for name, value in metrics.items():
            tag = str(name)
            if "/" in tag:
                continue
            if isinstance(value, dict):
                continue
            try:
                if isinstance(value, str):
                    fv = float(value.strip())
                else:
                    arr = np.asarray(value)
                    if arr.size != 1:
                        continue
                    fv = float(arr.reshape(-1)[0])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(fv):
                continue
            out[tag] = fv
        return out

    def log_hparams(
        self,
        hparams: dict,
        metrics: dict,
        *,
        model_name: str = "",
        task_name: str = "",
        window_idx: int = 0,
    ):
        """
        Log one **trial** for TensorBoard **HParams** (comparison table + parallel coords).

        Follows TensorBoard guidance: declare the experiment with
        :func:`hp.hparams_config` on the root logdir once, then for each
        hyperparameter evaluation write a **session** (``hp.hparams`` +
        validation metric scalars + ``session_end``) under
        ``<tensorboard>/hparams_sessions/<trial_id>/`` so each trial is its own
        **run** and metrics do not overwrite each other.

        Always includes ``model``, ``task``, and ``window`` in the recorded
        hyperparameters so you can slice by model and task in the HParams UI.

        Args:
            hparams: Model hyperparameter grid point (e.g. ``{"sp": 12}``).
            metrics: Evaluation outputs containing numeric metrics (e.g. ``mae``).
            model_name: Benchmark model id (folder name).
            task_name: Task folder name.
            window_idx: Rolling validation window index for this row.
        """
        if not self._should_log_tensorboard():
            return

        tf = _get_tf()
        hp = _get_hp()
        if tf is None or hp is None:
            return

        from tensorboard.plugins.hparams import api_pb2 as hparams_api_pb2
        from tensorboard.plugins.hparams import summary as hparams_event_summary

        self._ensure_hparams_experiment_config(tf, hp)

        sanitized_model: Dict[str, object] = {}
        for k, v in hparams.items():
            coerced = self._coerce_hparam_value(v)
            if coerced is not None:
                sanitized_model[str(k)] = coerced

        full_hparams: Dict[str, object] = {
            "model": model_name or "unknown",
            "task": task_name or "unknown",
            "window": int(window_idx),
            **sanitized_model,
        }

        payload = json.dumps(
            [model_name, task_name, int(window_idx), sorted(sanitized_model.items())],
            default=str,
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        trial_fs = (
            f"{_sanitize_tb_component(model_name)}__{_sanitize_tb_component(task_name)}"
            f"__w{int(window_idx):03d}__{digest}"
        )
        trial_fs = trial_fs[:180]

        trial_logdir = str(Path(self.tf_logs_path) / "hparams_sessions" / trial_fs)
        os.makedirs(trial_logdir, exist_ok=True)
        trial_writer = tf.summary.create_file_writer(trial_logdir)

        scalars = self._scalar_metrics_from_eval(metrics)
        with trial_writer.as_default():
            hp.hparams(full_hparams, trial_id=trial_fs)
            step = int(window_idx)
            for m_name, m_val in scalars.items():
                tf.summary.scalar(m_name, m_val, step=step)
            end_pb = hparams_event_summary.session_end_pb(
                hparams_api_pb2.STATUS_SUCCESS
            )
            tf.summary.experimental.write_raw_pb(
                end_pb.SerializeToString(), step=0
            )
        trial_writer.flush()
        trial_writer.close()

        self.tf_writer.flush()

    def log_text(self, tag: str, text: str, step: int):
        """
        Log text to TensorBoard.

        Args:
            tag (str): Tag for the text in TensorBoard.
            text (str): Text content to log.
            step (int): Step number for this text.

        Returns:
            None: Logs text to TensorBoard if TensorBoard logging is enabled.
        """
        if not self._should_log_tensorboard():
            return

        tf = _get_tf()
        if tf is None:
            return

        with self.tf_writer.as_default():
            tf.summary.text(tag, text, step=step)
        self.tf_writer.flush()

    def log_scalar(self, tag: str, value: float, step: int):
        """
        Log a scalar value to TensorBoard.

        Args:
            tag (str): Tag for the scalar in TensorBoard.
            value (float): Scalar value to log.
            step (int): Step number for this scalar.

        Returns:
            None: Logs scalar to TensorBoard if TensorBoard logging is enabled.
        """
        if not self._should_log_tensorboard():
            return

        tf = _get_tf()
        if tf is None:
            return

        with self.tf_writer.as_default():
            tf.summary.scalar(tag, value, step=step)
        self.tf_writer.flush()

    def close(self):
        """
        Flush and close all logger resources.

        This method flushes all log handlers and closes the TensorBoard writer.
        It can be called independently without using the context manager.

        Returns:
            None: Flushes and closes all logging resources.
        """
        # Flush all handlers in the standard logger
        for handler in self.logger.handlers:
            handler.flush()

        # Flush and close TensorBoard writer
        if self.tf_writer is not None:
            try:
                self.tf_writer.flush()
            except Exception:
                pass
            try:
                self.tf_writer.close()
            except Exception:
                pass
