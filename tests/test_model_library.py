from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from marathon_app.model_library import (
    HuggingFaceGguf,
    download_huggingface_gguf,
    list_huggingface_ggufs,
)


class ModelLibraryTests(unittest.TestCase):
    def test_huggingface_listing_filters_sidecars_and_prefers_q4(self) -> None:
        siblings = [
            types.SimpleNamespace(
                rfilename="Qwen3.8-27B-Q8_0.gguf",
                size=29,
                lfs=types.SimpleNamespace(sha256="q8hash"),
            ),
            types.SimpleNamespace(rfilename="mmproj-F16.gguf", size=2),
            types.SimpleNamespace(rfilename="Qwen3.8-27B-Q4_K_M.gguf", size=16),
            types.SimpleNamespace(rfilename="Qwen3.8-MTP-Q8_0.gguf", size=1),
            types.SimpleNamespace(
                rfilename="Qwen3.8-27B-DFlash2-Q4_K_M.gguf", size=1
            ),
            types.SimpleNamespace(
                rfilename="Qwen3.8-27B-DSpark-Q4_K_M.gguf", size=1
            ),
            types.SimpleNamespace(
                rfilename="Qwen3.8-27B-Eagle3-Q4_K_M.gguf", size=1
            ),
            types.SimpleNamespace(
                rfilename="Qwen3.8-27B-Q4-00001-of-00002.gguf", size=8
            ),
        ]
        info = types.SimpleNamespace(sha="abc123", siblings=siblings)
        api = mock.Mock()
        api.model_info.return_value = info
        module = types.SimpleNamespace(HfApi=mock.Mock(return_value=api))

        with mock.patch.dict(sys.modules, {"huggingface_hub": module}):
            files = list_huggingface_ggufs("unsloth/Qwen3.8-27B-GGUF")

        self.assertEqual([item.quant for item in files], ["Q4_K_M", "Q8_0"])
        self.assertTrue(all(item.revision == "abc123" for item in files))
        self.assertTrue(
            all(item.mmproj_filename == "mmproj-F16.gguf" for item in files)
        )
        self.assertEqual(files[1].sha256, "q8hash")

    def test_download_records_exact_repository_revision(self) -> None:
        model = HuggingFaceGguf(
            repository="author/model",
            revision="deadbeef",
            filename="model-Q4_K_M.gguf",
            size_bytes=None,
            quant="Q4_K_M",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            downloaded = root / "author--model" / model.filename
            downloaded.parent.mkdir()
            downloaded.write_bytes(b"gguf")
            downloader = mock.Mock(return_value=str(downloaded))
            module = types.SimpleNamespace(hf_hub_download=downloader)

            with mock.patch.dict(sys.modules, {"huggingface_hub": module}):
                result = download_huggingface_gguf(model, root)

            provenance = json.loads(
                result.with_suffix(result.suffix + ".marathon.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(provenance["revision"], "deadbeef")
        downloader.assert_called_once_with(
            repo_id="author/model",
            filename="model-Q4_K_M.gguf",
            revision="deadbeef",
            local_dir=root / "author--model",
        )

    def test_download_fetches_matching_multimodal_projector(self) -> None:
        model = HuggingFaceGguf(
            repository="author/model",
            revision="deadbeef",
            filename="model-Q4_K_M.gguf",
            size_bytes=None,
            quant="Q4_K_M",
            mmproj_filename="mmproj-F16.gguf",
            mmproj_size_bytes=123,
            mmproj_sha256="projector-hash",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "author--model"
            destination.mkdir()

            def download(*, filename: str, **_kwargs: object) -> str:
                path = destination / filename
                path.write_bytes(b"gguf")
                return str(path)

            downloader = mock.Mock(side_effect=download)
            module = types.SimpleNamespace(hf_hub_download=downloader)

            with mock.patch.dict(sys.modules, {"huggingface_hub": module}):
                result = download_huggingface_gguf(model, root)

            provenance = json.loads(
                result.with_suffix(result.suffix + ".marathon.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(
            [call.kwargs["filename"] for call in downloader.call_args_list],
            ["model-Q4_K_M.gguf", "mmproj-F16.gguf"],
        )
        self.assertEqual(
            provenance["multimodal_projector"],
            {
                "filename": "mmproj-F16.gguf",
                "size_bytes": 123,
                "sha256": "projector-hash",
            },
        )


if __name__ == "__main__":
    unittest.main()
