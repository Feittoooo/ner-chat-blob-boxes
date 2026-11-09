"""Generate the ner-chat sample document as a text PDF and as a PNG (for the OCR path).

Both files are committed; rerun only if ARTICLE changes:

    uv run python scripts/make_samples.py
"""

import re
from pathlib import Path

from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).resolve().parent.parent / "samples"

TITLE = "Lumora Energy to build green hydrogen plant near Valencia"

# Fictional people and companies, real places. Every repeated entity always uses the same surface form, so the
# expected counts below are unambiguous. Core PDF fonts are latin-1 only: straight quotes, no em dashes.
ARTICLE = [
    "Lumora Energy announced on Tuesday that it will build a green hydrogen plant on the coast south of "
    "Valencia. The project is expected to cost 1.2 billion euros and to create around 800 jobs during "
    "construction.",
    "Marta Ibarra, chief executive of Lumora Energy, said the site was chosen for its solar capacity and its "
    'access to the port. "We want to show that heavy industry can run on sunlight," Marta Ibarra told '
    "reporters at the company's headquarters.",
    "Financing will come partly from the European Investment Bank, which has agreed to lend 400 million euros. "
    "Daniel Okafor, a vice-president at the European Investment Bank, described the loan as one of the "
    "largest clean-energy deals in Spain this year.",
    "The first hydrogen shipments are planned to leave Valencia in 2029, bound for industrial customers of "
    "Lumora Energy in Rotterdam. Daniel Okafor said the bank would review progress every six months.",
    "Local business groups welcomed the news, although environmental organisations asked for more detail on "
    "water use. Lumora Energy said it would publish a full environmental assessment early next year, and "
    "Marta Ibarra promised regular public meetings in Valencia.",
]

# Literal occurrences in the whole document, title included (extract sees the title too). Mirrored in README.md.
EXPECTED_COUNTS = {
    "Lumora Energy": 5,
    "Marta Ibarra": 3,
    "Valencia": 4,
    "European Investment Bank": 2,
    "Daniel Okafor": 2,
    "Rotterdam": 1,
    "Spain": 1,
}


def check_counts() -> None:
    body = "\n".join([TITLE, *ARTICLE])
    for name, expected in EXPECTED_COUNTS.items():
        actual = len(re.findall(rf"\b{re.escape(name)}\b", body))
        assert actual == expected, f"{name!r}: expected {expected}, found {actual}"


def write_pdf(path: Path) -> None:
    pdf = FPDF(format="A4")
    pdf.set_margins(20, 20)
    pdf.add_page()
    pdf.set_font("Helvetica", style="B", size=16)
    pdf.multi_cell(0, 8, TITLE, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", size=11)
    for paragraph in ARTICLE:
        pdf.multi_cell(0, 6, paragraph, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)
    pdf.output(str(path))


def wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if draw.textlength(candidate, font=font) <= width:
            line = candidate
        else:
            lines.append(line)
            line = word
    return [*lines, line]


def write_png(path: Path) -> None:
    # Large black-on-white text at ~A4 width: easy input for OCR.
    width, margin = 1700, 90
    title_font = ImageFont.load_default(size=44)
    body_font = ImageFont.load_default(size=32)
    scratch = ImageDraw.Draw(Image.new("L", (1, 1)))
    blocks = [(wrap(scratch, TITLE, title_font, width - 2 * margin), title_font, 58)]
    blocks += [(wrap(scratch, p, body_font, width - 2 * margin), body_font, 44) for p in ARTICLE]
    height = 2 * margin + sum(len(lines) * step + 30 for lines, _, step in blocks)

    image = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(image)
    y = margin
    for lines, font, step in blocks:
        for line in lines:
            draw.text((margin, y), line, font=font, fill=0)
            y += step
        y += 30
    image.save(path)


def main() -> None:
    check_counts()
    OUT_DIR.mkdir(exist_ok=True)
    write_pdf(OUT_DIR / "news_article.pdf")
    write_png(OUT_DIR / "news_article.png")
    print(f"wrote {OUT_DIR / 'news_article.pdf'} and {OUT_DIR / 'news_article.png'}")


if __name__ == "__main__":
    main()
