"""Canonical checkpoint identities for the CMC-KF association project."""

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class CheckpointSpec:
    checkpoint_id: str
    config_name: str
    checkpoint_relative_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ParameterSpec:
    variant: str
    checkpoint: CheckpointSpec
    version: str

    @property
    def checkpoint_id(self):
        return self.checkpoint.checkpoint_id

    @property
    def config_name(self):
        return self.checkpoint.config_name

    @property
    def checkpoint_relative_path(self):
        return self.checkpoint.checkpoint_relative_path

    @property
    def size_bytes(self):
        return self.checkpoint.size_bytes

    @property
    def sha256(self):
        return self.checkpoint.sha256


CHECKPOINTS = {
    "official_vitb256_ce_ep300": CheckpointSpec(
        checkpoint_id="official_vitb256_ce_ep300",
        config_name="vitb_256_mae_ce_32x4_ep300",
        checkpoint_relative_path=(
            "checkpoints/train/ostrack/"
            "official_vitb_256_mae_ce_32x4_ep300/OSTrack_ep0300.pth.tar"
        ),
        size_bytes=370179249,
        sha256="8e01d6251569ac84dbb53fdd062cd938551be2d889582f789aa3070196f42f41",
    ),
    "retrained_vitb256_ce_ep300": CheckpointSpec(
        checkpoint_id="retrained_vitb256_ce_ep300",
        config_name="vitb_256_mae_ce_32x4_ep300",
        checkpoint_relative_path=(
            "checkpoints/train/ostrack/"
            "retrained_vitb_256_mae_ce_32x4_ep300/OSTrack_ep0300.pth.tar"
        ),
        size_bytes=1111836284,
        sha256="91f53eec3f33fb08f06d5f9ee726cb33fbf572520b68b0dc360044cfed458b63",
    ),
}


def parse_parameter_name(
        parameter_name, expected_variant="e0", expected_version="v1",
        allowed_variants=None, allowed_versions=None):
    """Resolve ``<variant>__<checkpoint_id>__<version>`` without ambiguity."""

    parts = parameter_name.lower().split("__")
    if len(parts) != 3:
        raise ValueError(
            "parameter name must be '<variant>__<checkpoint_id>__<version>', "
            f"got {parameter_name!r}"
        )
    variant, checkpoint_id, version = parts
    if allowed_variants is not None:
        allowed_variants = {item.lower() for item in allowed_variants}
        if variant not in allowed_variants:
            known = ", ".join(sorted(allowed_variants))
            raise ValueError(
                f"tracker expects one of variants {known}, got {variant!r}"
            )
    elif expected_variant is not None and variant != expected_variant:
        raise ValueError(
            f"tracker expects variant {expected_variant!r}, got {variant!r}"
        )
    if allowed_versions is not None:
        allowed_versions = {item.lower() for item in allowed_versions}
        if version not in allowed_versions:
            known = ", ".join(sorted(allowed_versions))
            raise ValueError(
                f"tracker expects one of config versions {known}, got {version!r}"
            )
    elif expected_version is not None and version != expected_version:
        raise ValueError(
            f"tracker expects config version {expected_version!r}, got {version!r}"
        )
    try:
        spec = CHECKPOINTS[checkpoint_id]
    except KeyError as exc:
        known = ", ".join(sorted(CHECKPOINTS))
        raise ValueError(
            f"unknown checkpoint_id {checkpoint_id!r}; known ids: {known}"
        ) from exc
    return ParameterSpec(variant=variant, checkpoint=spec, version=version)


def resolve_paths(spec, project_dir, save_dir):
    config_path = Path(project_dir) / "experiments" / "ostrack" / f"{spec.config_name}.yaml"
    checkpoint_path = Path(save_dir) / Path(spec.checkpoint_relative_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"OSTrack config not found: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    return config_path, checkpoint_path


@lru_cache(maxsize=8)
def _sha256_for_file(path_string, size_bytes, mtime_ns):
    del size_bytes, mtime_ns
    digest = hashlib.sha256()
    with Path(path_string).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(spec, checkpoint_path):
    """Fail before model loading when a registered checkpoint identity drifts."""

    checkpoint_path = Path(checkpoint_path)
    stat = checkpoint_path.stat()
    if stat.st_size != spec.size_bytes:
        raise ValueError(
            f"checkpoint size mismatch for {spec.checkpoint_id}: "
            f"expected {spec.size_bytes}, got {stat.st_size} at {checkpoint_path}"
        )
    actual = _sha256_for_file(
        str(checkpoint_path.resolve()), stat.st_size, stat.st_mtime_ns)
    if actual != spec.sha256:
        raise ValueError(
            f"checkpoint SHA256 mismatch for {spec.checkpoint_id}: "
            f"expected {spec.sha256}, got {actual} at {checkpoint_path}"
        )
    return actual
