"""Text extraction with Kreuzberg (native text layer, or OCR for scans/images)."""

from pathlib import Path

from kreuzberg import ExtractionConfig, KreuzbergError, OcrConfig, extract_file_sync

OCR_HELP = (
    "OCR needs Tesseract language data, which `uv sync` does not install. Either install Tesseract "
    "(apt install tesseract-ocr / brew install tesseract) or download eng.traineddata from "
    "https://github.com/tesseract-ocr/tessdata_fast and point TESSDATA_PREFIX at its folder."
)


class ExtractError(Exception):
    """Extraction failed or produced no text; the message is meant for the CLI user."""


def extract_text(path: Path, force_ocr: bool = False) -> str:
    config = ExtractionConfig(
        force_ocr=force_ocr,
        ocr=OcrConfig(backend="tesseract", language="eng"),
    )
    try:
        result = extract_file_sync(path, config=config)
    except KreuzbergError as exc:
        raise ExtractError(f"Kreuzberg could not extract {path}: {exc}\n{OCR_HELP}") from exc

    if not result.content.strip():
        # Without language data Kreuzberg can silently fall back to non-OCR extraction, which yields
        # nothing for an image or a scanned PDF.
        raise ExtractError(f"No text extracted from {path}. If it is a scan or an image: {OCR_HELP}")
    return result.content
