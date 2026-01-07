import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from pathlib import Path
import json

pytesseract.pytesseract.tesseract_cmd = r"D:\aaaeurovision\tesseract\tesseract.exe"


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
                print(f"[pdfplumber] Page number: {page_number}")

            words = page.extract_words(
                use_text_flow=True, keep_blank_chars=False, x_tolerance=2, y_tolerance=2
            )

            text = words_to_text(words)

            flag = is_bad_encoded_text(text)

            # Проверяем битую кодировку
            if flag:
                print(f"[BAD ENCODING] Page {page_number} → OCR")
                text = perform_page_ocr(page_number, pdf_path)

            # Проверяем наличие изображений на странице
            if page.images and not flag:
                print(f"[INFO] Page {page_number} has {len(page.images)} images")

                # Извлекаем текст с изображений
                image_texts = extract_text_from_page_images(page, page_number)

                if image_texts:
                    text += "\n\n".join(image_texts)
            else:
                if not text.strip():
                    print(f"[OCR] Page {page_number} is empty → OCR")
                    text = perform_page_ocr(page_number, pdf_path)

            result.append({"page": page_number, "text": text.strip()})

    return result


def words_to_text(words, y_tol=3):
    """
    Собирает текст построчно из extract_words.
    """
    if not words:
        return ""

    lines = {}

    for w in words:
        y = round(w["top"] / y_tol) * y_tol
        lines.setdefault(y, []).append(w)

    result_lines = []

    for y in sorted(lines):
        line_words = sorted(lines[y], key=lambda x: x["x0"])
        line = " ".join(w["text"] for w in line_words)
        result_lines.append(line)

    return "\n".join(result_lines)


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


def extract_text_from_page_images(page, page_number):
    """
    Извлекает текст с изображений на странице.
    """
    image_texts = []

    for i, img in enumerate(page.images):
        try:
            print(f"  Processing image {i+1} on page {page_number}...")
            # Вырезаем область изображения
            bbox = (img["x0"], img["top"], img["x1"], img["bottom"])

            # Проверяем, что область изображения достаточно большая
            width = img["x1"] - img["x0"]
            height = img["bottom"] - img["top"]

            if width < 10 or height < 10:
                continue

            raw_bbox = (img["x0"], img["top"], img["x1"], img["bottom"])
            bbox = clip_bbox(raw_bbox, page)

            if not bbox:
                print("    Skipping invalid bbox")
                continue

            cropped_page = page.within_bbox(bbox)
            if cropped_page:
                img_obj = cropped_page.to_image(resolution=300)
                pil_img = img_obj.original

                ocr_text = pytesseract.image_to_string(
                    pil_img, lang="eng", config="--psm 6 --oem 3"
                )

                if ocr_text and ocr_text.strip():
                    # Очищаем текст от шума
                    clean_text = clean_ocr_text(ocr_text)
                    if clean_text:
                        image_texts.append(f"{clean_text}")
                        print(f"    Found text in image {i+1}: {clean_text[:50]}...")
        except Exception as e:
            print(f"    Error processing image {i+1}: {str(e)}")
            continue

    return image_texts


def clip_bbox(bbox, page):
    x0, y0, x1, y1 = bbox
    px0, py0, px1, py1 = page.bbox

    x0 = max(x0, px0)
    y0 = max(y0, py0)
    x1 = min(x1, px1)
    y1 = min(y1, py1)

    if x1 <= x0 or y1 <= y0:
        return None

    return (x0, y0, x1, y1)


def clean_ocr_text(text):
    """
    Очистка текста после OCR.
    """
    # Лишние пробелы и переносы
    lines = [line.strip() for line in text.split("\n")]
    # Пустые строки
    lines = [line for line in lines if line and not line.isspace()]
    # Строки, состоящие только из символов
    lines = [line for line in lines if any(c.isalnum() for c in line)]

    return "\n".join(lines)


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
