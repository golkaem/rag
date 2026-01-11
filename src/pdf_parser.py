import pdfplumber
from pdf2image import convert_from_path
from pathlib import Path
import json
import os
import pytesseract
from dotenv import load_dotenv


load_dotenv()
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_PATH", "tesseract")


def save_extracted_text(pdf_path, output_dir="data/parsed"):
    """
    Извлекает текст и сохраняет как JSON.
    """
    extracted = extract_text_pdf(pdf_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / (Path(pdf_path).stem + ".json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(extracted, f, ensure_ascii=False, indent=2)

    print(f"Saved parsed text to {out_path}")


def extract_text_pdf(pdf_path):
    pdf_path = Path(pdf_path)
    result = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            if page_number % 10 == 0:
                print(f"[pdfplumber] {page_number}")

            text: str = page.extract_text() or ""

            is_bad = (
                not text.strip() or is_bad_encoded_text(text) or is_glued_text(text)
            )

            if is_bad:
                print(f"[OCR] Page {page_number}")
                text = perform_page_ocr(page_number, pdf_path)

            result.append({"page": page_number, "text": text.strip()})

    return result


def is_bad_encoded_text(text, min_ratio=0.1):
    """
    Возвращает True, если есть битая PDF-кодировка.
    """
    if not text:
        return True

    cid_count = text.count("(cid:")

    if cid_count / max(1, len(text.split()) // len("(cid:  ")) >= min_ratio:
        return True

    return False


def is_glued_text(text, max_word_len=30, max_long_word_ratio=0.2):
    """
    Возвращает True, если текст выглядит слипшимся.
    """
    if not text:
        return True

    lines = text.splitlines()
    if not lines:
        return True

    words = text.split()
    if not words:
        return True

    long_words = [w for w in words if len(w) > max_word_len]
    if len(long_words) / len(words) > max_long_word_ratio:
        return True

    return False


def clean_ocr_text(text: str):
    """
    Убирает OCR-шум.
    """
    clean_lines = []

    for line in text.splitlines():
        line = line.strip()

        if not any(c.isalnum() for c in line):
            continue

        non_alnum_ratio = sum(not c.isalnum() for c in line) / len(line)
        if non_alnum_ratio > 0.4:
            continue

        clean_lines.append(line)

    return "\n".join(clean_lines)


def perform_page_ocr(page_number, pdf_path):
    """
    OCR для всей страницы.
    """
    try:
        images = convert_from_path(
            str(pdf_path),
            dpi=300,
            first_page=page_number,
            last_page=page_number,
        )

        if images:
            pil_img = images[0]

            text = pytesseract.image_to_string(
                pil_img, lang="eng", config="--psm 3 --oem 3"
            )
            return clean_ocr_text(text)
    except Exception as e:
        print(f"Error in page OCR for page {page_number}: {e}")

    return ""


def parse_all_pdfs(input_dir="data/pdfs", output_dir="data/parsed"):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    for pdf in input_dir.glob("*.pdf"):
        output_json = output_dir / (pdf.stem + ".json")

        if output_json.exists():
            print(f"[SKIP] Already parsed: {pdf.name}")
            continue

        print(f"[INFO] Processing {pdf.name}")
        save_extracted_text(pdf)
