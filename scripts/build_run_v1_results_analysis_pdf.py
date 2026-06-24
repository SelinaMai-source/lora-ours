from __future__ import annotations

import argparse
import html
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


_FONT_NAME = "WQYZenHei"
_FONT_PATHS = [
    Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttf"),
]


def _register_fonts() -> None:
    for path in _FONT_PATHS:
        if path.is_file():
            try:
                pdfmetrics.registerFont(TTFont(_FONT_NAME, str(path), subfontIndex=0))
                return
            except Exception:
                continue
    try:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    except Exception:
        pass


def _text_font() -> str:
    if _FONT_NAME in pdfmetrics.getRegisteredFontNames():
        return _FONT_NAME
    return "STSong-Light"


def _escape_text(text: str) -> str:
    text = html.escape(text, quote=False)
    text = text.replace("`", "")
    text = text.replace("**", "")
    text = text.replace("*", "")
    return text


def _parse_table(lines: Sequence[str]) -> tuple[List[str], List[List[str]]]:
    rows: List[List[str]] = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows.append(cells)
    if len(rows) >= 2 and all(set(cell) <= {"-", ":"} for cell in rows[1]):
        rows = [rows[0]] + rows[2:]
    headers = rows[0] if rows else []
    body = rows[1:] if len(rows) > 1 else []
    return headers, body


def _is_special_line(line: str) -> bool:
    stripped = line.strip()
    return (
        not stripped
        or stripped.startswith("#")
        or stripped.startswith("![")
        or stripped.startswith("|")
        or stripped.startswith("- ")
        or bool(re.match(r"^\d+\.\s+", stripped))
    )


def _parse_markdown(md_text: str) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    lines = md_text.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i].rstrip()
        line = raw.strip()
        if not line or line.startswith("<!--"):
            i += 1
            continue

        if re.match(r"^#{1,6}\s+", line):
            level = len(line) - len(line.lstrip("#"))
            text = line[level:].strip()
            blocks.append({"type": "heading", "level": level, "text": text})
            i += 1
            continue

        if line.startswith("![") and "](" in line and line.endswith(")"):
            match = re.match(r"!\[(.*?)\]\((.*?)\)", line)
            if match:
                blocks.append({"type": "image", "caption": match.group(1).strip(), "path": match.group(2).strip()})
                i += 1
                continue

        if line.startswith("|"):
            table_lines = [line]
            i += 1
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            headers, rows = _parse_table(table_lines)
            blocks.append({"type": "table", "headers": headers, "rows": rows})
            continue

        if line.startswith("- "):
            items = [line[2:].strip()]
            i += 1
            while i < len(lines) and lines[i].strip().startswith("- "):
                items.append(lines[i].strip()[2:].strip())
                i += 1
            blocks.append({"type": "bullets", "ordered": False, "items": items})
            continue

        if re.match(r"^\d+\.\s+", line):
            items = [re.sub(r"^\d+\.\s+", "", line)]
            i += 1
            while i < len(lines) and re.match(r"^\d+\.\s+", lines[i].strip()):
                items.append(re.sub(r"^\d+\.\s+", "", lines[i].strip()))
                i += 1
            blocks.append({"type": "bullets", "ordered": True, "items": items})
            continue

        para_lines = [line]
        i += 1
        while i < len(lines) and not _is_special_line(lines[i]):
            para_lines.append(lines[i].strip())
            i += 1
        blocks.append({"type": "paragraph", "text": " ".join(para_lines)})

    return blocks


def _build_styles() -> Dict[str, ParagraphStyle]:
    _register_fonts()
    base = getSampleStyleSheet()
    font = _text_font()

    styles: Dict[str, ParagraphStyle] = {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName=font,
            fontSize=20,
            leading=26,
            alignment=TA_CENTER,
            spaceAfter=18,
            wordWrap="CJK",
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName=font,
            fontSize=16,
            leading=22,
            alignment=TA_LEFT,
            spaceBefore=10,
            spaceAfter=8,
            wordWrap="CJK",
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName=font,
            fontSize=13.5,
            leading=19,
            alignment=TA_LEFT,
            spaceBefore=8,
            spaceAfter=6,
            wordWrap="CJK",
        ),
        "h3": ParagraphStyle(
            "h3",
            parent=base["Heading3"],
            fontName=font,
            fontSize=11.5,
            leading=16,
            alignment=TA_LEFT,
            spaceBefore=6,
            spaceAfter=5,
            wordWrap="CJK",
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName=font,
            fontSize=10.5,
            leading=16,
            spaceAfter=6,
            wordWrap="CJK",
        ),
        "caption": ParagraphStyle(
            "caption",
            parent=base["BodyText"],
            fontName=font,
            fontSize=9,
            leading=13,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#444444"),
            spaceAfter=10,
            wordWrap="CJK",
        ),
    }
    return styles


def _make_list(items: Sequence[str], styles: Dict[str, ParagraphStyle], ordered: bool) -> List[Any]:
    list_style = ParagraphStyle(
        "list_body",
        parent=styles["body"],
        leftIndent=16,
        firstLineIndent=0,
        spaceAfter=4,
    )
    flowables: List[Any] = []
    for idx, item in enumerate(items, start=1):
        prefix = f"{idx}. " if ordered else "- "
        flowables.append(Paragraph(_escape_text(prefix + item), list_style))
    flowables.append(Spacer(1, 0.08 * cm))
    return flowables


def _make_table(headers: Sequence[str], rows: Sequence[Sequence[str]], doc_width: float) -> Table:
    data = [[Paragraph(_escape_text(cell), ParagraphStyle("tbl_h", fontName=_text_font(), fontSize=9.2, leading=12, alignment=TA_CENTER, wordWrap="CJK")) for cell in headers]]
    body_style = ParagraphStyle("tbl_b", fontName=_text_font(), fontSize=8.8, leading=11, wordWrap="CJK")
    for row in rows:
        data.append([Paragraph(_escape_text(cell), body_style) for cell in row])

    col_count = max(1, len(headers))
    col_widths = [doc_width / col_count] * col_count
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8ECF3")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#AAB4C0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _make_image(image_path: Path, caption: str, max_width: float) -> List[Any]:
    flowables: List[Any] = []
    if image_path.is_file():
        img = Image(str(image_path))
        draw_width = img.imageWidth
        draw_height = img.imageHeight
        if draw_width > max_width:
            scale = max_width / float(draw_width)
            draw_width *= scale
            draw_height *= scale
        img.drawWidth = draw_width
        img.drawHeight = draw_height
        flowables.append(img)
        flowables.append(Spacer(1, 0.18 * cm))
    flowables.append(Paragraph(_escape_text(caption), _build_styles()["caption"]))
    return flowables


def _header_footer(canvas: Any, doc: SimpleDocTemplate, title: str) -> None:
    canvas.saveState()
    canvas.setFont(_text_font(), 9)
    canvas.drawString(doc.leftMargin, A4[1] - 1.2 * cm, title)
    canvas.drawRightString(A4[0] - doc.rightMargin, 0.8 * cm, f"第 {canvas.getPageNumber()} 页")
    canvas.restoreState()


def build_pdf(md_path: Path, out_path: Path, repo_root: Path) -> None:
    styles = _build_styles()
    md_text = md_path.read_text(encoding="utf-8")
    blocks = _parse_markdown(md_text)

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.9 * cm,
        bottomMargin=1.5 * cm,
    )

    story: List[Any] = []
    first_heading_used = False
    title_text = "Run V1 结果分析报告"

    for block in blocks:
        block_type = block["type"]
        if block_type == "heading":
            text = _escape_text(block["text"])
            level = int(block["level"])
            if not first_heading_used and level == 1:
                story.append(Paragraph(text, styles["title"]))
                story.append(Paragraph(_escape_text("按 RP(Lora)_v2 的 method blocks 复盘当前 ours 设计"), styles["caption"]))
                story.append(Spacer(1, 0.15 * cm))
                first_heading_used = True
            elif level == 1:
                story.append(PageBreak())
                story.append(Paragraph(text, styles["h1"]))
            elif level == 2:
                story.append(Paragraph(text, styles["h1"]))
            else:
                story.append(Paragraph(text, styles["h2"] if level == 3 else styles["h3"]))
        elif block_type == "paragraph":
            story.append(Paragraph(_escape_text(block["text"]), styles["body"]))
        elif block_type == "bullets":
            story.extend(_make_list(block["items"], styles, bool(block.get("ordered"))))
        elif block_type == "table":
            story.append(_make_table(block["headers"], block["rows"], doc.width))
            story.append(Spacer(1, 0.2 * cm))
        elif block_type == "image":
            image_path = (repo_root / block["path"]).resolve()
            story.extend(_make_image(image_path, block["caption"], doc.width))

    doc.build(
        story,
        onFirstPage=lambda canvas, doc: _header_footer(canvas, doc, title_text),
        onLaterPages=lambda canvas, doc: _header_footer(canvas, doc, title_text),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/run_v1_results_analysis.md"))
    parser.add_argument("--output", type=Path, default=Path("run_v1_results_analysis.pdf"))
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    md_path = (repo_root / args.input).resolve()
    out_path = (repo_root / args.output).resolve()
    build_pdf(md_path=md_path, out_path=out_path, repo_root=repo_root)
    print(f"[run_v1_analysis_pdf] wrote {out_path}")


if __name__ == "__main__":
    main()
