"""Application configuration with sensible defaults for the live pipeline.

Supports loading from a YAML file with CLI overrides.  If no YAML file
exists the dataclass defaults are used.
"""

import argparse
import dataclasses
import logging
import os
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class AppConfig:
    """Runtime configuration for ingestion and consumer threads.

    Attributes:
        samplerate_hz: Audio sample rate in hertz.
        channels: Number of input channels (1 = mono).
        blocksize: Frames per audio callback buffer.
        dtype: Sample format expected by sounddevice and aubio.
        recording_dir: Directory where WAV recordings are written.
        device: PortAudio device index, or ``None`` for system default.
        midi_port_name: Name of the virtual MIDI output port.
        diagnostics_interval_s: Seconds between diagnostics log lines.
    """

    samplerate_hz: int = 44100
    channels: int = 1
    blocksize: int = 1024
    dtype: str = "float32"
    recording_dir: str = "recordings"
    device: int | None = None
    midi_port_name: str = "InstrumentSampler"
    diagnostics_interval_s: float = 5.0


# Maps CLI arg names (as they appear in argparse.Namespace) to AppConfig
# field names.  Only args that can override config fields are listed.
_CLI_TO_FIELD: dict[str, str] = {
    "device": "device",
    "midi_port": "midi_port_name",
    "blocksize": "blocksize",
    "samplerate": "samplerate_hz",
    "recording_dir": "recording_dir",
}


def _load_yaml(path: str) -> dict[str, Any]:
    """Read a YAML config file and return its contents as a dict.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed mapping, or an empty dict if the file does not exist.
    """
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else {}


def load_config(args: argparse.Namespace) -> AppConfig:
    """Build an :class:`AppConfig` from YAML defaults + CLI overrides.

    Resolution order (highest priority last):

    1. Dataclass defaults
    2. Values from the YAML config file (if it exists)
    3. Explicit CLI arguments (non-``None`` values)

    Args:
        args: Parsed CLI namespace.  Expected to contain a ``config``
            attribute with the path to the YAML file.

    Returns:
        Fully resolved configuration.
    """
    yaml_path: str = getattr(args, "config", "config.yaml")
    yaml_data = _load_yaml(yaml_path)

    field_names = {f.name for f in dataclasses.fields(AppConfig)}
    yaml_kwargs = {k: v for k, v in yaml_data.items() if k in field_names}

    if yaml_kwargs:
        logger.info("Loaded config from %s: %s", yaml_path, yaml_kwargs)

    config = AppConfig(**yaml_kwargs)

    for cli_key, field_name in _CLI_TO_FIELD.items():
        cli_value = getattr(args, cli_key, None)
        if cli_value is not None:
            setattr(config, field_name, cli_value)

    return config
