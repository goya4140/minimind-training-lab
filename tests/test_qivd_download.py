import hashlib

from minimind_lab.data.integrity import file_matches, verified_dataset_manifest


def test_qivd_file_integrity_checks_size_and_sha256(tmp_path):
    path = tmp_path / "video.mp4"
    path.write_bytes(b"complete-video")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert file_matches(path, len(b"complete-video"), digest)
    assert not file_matches(path, len(b"complete-video") + 1, digest)
    assert not file_matches(path, len(b"complete-video"), "0" * 64)


def test_qivd_manifest_records_upstream_verification(tmp_path):
    relative = "videos/00000000.mp4"
    path = tmp_path / relative
    path.parent.mkdir()
    path.write_bytes(b"video")
    digest = hashlib.sha256(b"video").hexdigest()
    manifest = verified_dataset_manifest(
        tmp_path,
        [relative],
        {relative: {"bytes": len(b"video"), "sha256": digest}},
        repository="example/QIVD",
        revision="abc123",
    )
    assert manifest["video_count"] == 1
    assert manifest["upstream_lfs_verified"] is True
    assert manifest["files"][0]["sha256"] == digest
