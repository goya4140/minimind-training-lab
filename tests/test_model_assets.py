import hashlib
import runpy
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/fetch_models.py"
FETCH_MODELS = runpy.run_path(str(SCRIPT))


def test_every_frozen_model_asset_has_a_full_sha256():
    for model in FETCH_MODELS["MODELS"].values():
        for _, checksum in model["files"].values():
            assert isinstance(checksum, str)
            assert len(checksum) == 64
            int(checksum, 16)


def test_model_asset_validation_rejects_same_size_tampering(tmp_path):
    asset = tmp_path / "config.json"
    asset.write_bytes(b"abc")
    digest = hashlib.sha256(b"abc").hexdigest()
    valid = FETCH_MODELS["valid"]
    assert valid(asset, 3, digest)
    asset.write_bytes(b"abd")
    assert not valid(asset, 3, digest)
