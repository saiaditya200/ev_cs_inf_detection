import re
from pathlib import Path
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_table_borders(table, color="D3D3D3"):
    tblPr = table._tbl.tblPr
    tblBorders = OxmlElement('w:tblBorders')
    for border_name in ['top', 'left', 'bottom', 'right', 'insideH']:
        border = OxmlElement(f'w:{border_name}')
        border.set(qn('w:val'), 'single')
        border.set(qn('w:sz'), '4')
        border.set(qn('w:space'), '0')
        border.set(qn('w:color'), color)
        tblBorders.append(border)
    insideV = OxmlElement('w:insideV')
    insideV.set(qn('w:val'), 'none')
    tblBorders.append(insideV)
    tblPr.append(tblBorders)

def parse_markdown_to_docx(md_path, docx_path):
    doc = Document()
    
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(4)

    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    in_table = False
    table_lines = []
    in_code = False
    code_lines = []

    for line in lines:
        raw = line.rstrip('\r\n')
        
        if raw.startswith('```'):
            if in_code:
                in_code = False
                p = doc.add_paragraph('\n'.join(code_lines))
                p.paragraph_format.left_indent = Inches(0.4)
                p.paragraph_format.space_before = Pt(4)
                p.paragraph_format.space_after = Pt(6)
                for run in p.runs:
                    run.font.name = 'Courier New'
                    run.font.size = Pt(9.5)
                    run.font.color.rgb = RGBColor(0x33, 0x41, 0x55)
                code_lines = []
            else:
                in_code = True
                code_lines = []
            continue
        
        if in_code:
            code_lines.append(raw)
            continue

        if raw.strip().startswith('|') and raw.strip().endswith('|'):
            if not in_table:
                in_table = True
                table_lines = []
            table_lines.append(raw)
            continue
        else:
            if in_table:
                in_table = False
                render_table(doc, table_lines)
                table_lines = []

        stripped = raw.strip()
        if not stripped:
            continue

        if stripped == '---':
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(6)
            continue

        if stripped.startswith('# '):
            text = stripped[2:].strip()
            p = doc.add_heading(text, level=1)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(16)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.keep_with_next = True
            for r in p.runs:
                r.font.name = 'Times New Roman'
                r.font.size = Pt(18)
                r.font.bold = True
                r.font.color.rgb = RGBColor(0x0f, 0x17, 0x2a)
        elif stripped.startswith('## '):
            text = stripped[3:].strip()
            p = doc.add_heading(text, level=2)
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.keep_with_next = True
            for r in p.runs:
                r.font.name = 'Times New Roman'
                r.font.size = Pt(14)
                r.font.bold = True
                r.font.color.rgb = RGBColor(0x1e, 0x3a, 0x8a)
        elif stripped.startswith('### '):
            text = stripped[4:].strip()
            p = doc.add_heading(text, level=3)
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.keep_with_next = True
            for r in p.runs:
                r.font.name = 'Times New Roman'
                r.font.size = Pt(12)
                r.font.bold = True
                r.font.color.rgb = RGBColor(0x33, 0x41, 0x55)
        elif stripped.startswith('- ') or stripped.startswith('* '):
            p = doc.add_paragraph(style='List Bullet')
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.left_indent = Inches(0.25)
            format_inline_text(p, stripped[2:].strip())
        elif re.match(r'^\d+\.\s+', stripped):
            match = re.match(r'^\d+\.\s+', stripped)
            text = stripped[match.end():].strip()
            p = doc.add_paragraph(style='List Number')
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.left_indent = Inches(0.25)
            format_inline_text(p, text)
        else:
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(5)
            p.paragraph_format.line_spacing = 1.15
            format_inline_text(p, stripped)

    if in_table and table_lines:
        render_table(doc, table_lines)

    doc.save(docx_path)
    print(f"[OK] Word document saved successfully: {docx_path}")

def format_inline_text(paragraph, text):
    tokens = re.split(r'(\*\*.*?\*\*|\*.*?\*|`.*?`)', text)
    for token in tokens:
        if not token:
            continue
        if token.startswith('**') and token.endswith('**'):
            run = paragraph.add_run(token[2:-2])
            run.font.name = 'Times New Roman'
            run.font.bold = True
        elif token.startswith('*') and token.endswith('*'):
            run = paragraph.add_run(token[1:-1])
            run.font.name = 'Times New Roman'
            run.font.italic = True
        elif token.startswith('`') and token.endswith('`'):
            run = paragraph.add_run(token[1:-1])
            run.font.name = 'Courier New'
            run.font.size = Pt(9.5)
            run.font.color.rgb = RGBColor(0x0f, 0x17, 0x2a)
        else:
            run = paragraph.add_run(token)
            run.font.name = 'Times New Roman'

def render_table(doc, table_lines):
    rows = []
    for line in table_lines:
        cleaned = line.strip().strip('|')
        cells = [c.strip() for c in cleaned.split('|')]
        if cells and all(re.match(r'^:?-+:?$', c) for c in cells):
            continue
        rows.append(cells)

    if not rows:
        return

    n_cols = max(len(r) for r in rows)
    table = doc.add_table(rows=len(rows), cols=n_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    set_table_borders(table, "D0D5DD")

    for row_idx, row_data in enumerate(rows):
        is_header = (row_idx == 0)
        table_row = table.rows[row_idx]
        for col_idx in range(n_cols):
            cell = table_row.cells[col_idx]
            val = row_data[col_idx] if col_idx < len(row_data) else ""
            
            p = cell.paragraphs[0]
            p.text = ""
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.05
            
            format_inline_text(p, val)
            
            if is_header:
                set_cell_background(cell, "F1F5F9")
                for r in p.runs:
                    r.font.bold = True
                    r.font.color.rgb = RGBColor(0x0f, 0x17, 0x2a)
            else:
                if row_idx % 2 == 1:
                    set_cell_background(cell, "FFFFFF")
                else:
                    set_cell_background(cell, "F8FAFC")
            
            set_cell_margins(cell, top=120, bottom=120, left=140, right=140)

    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_after = Pt(6)

if __name__ == "__main__":
    base_dir = Path(r"C:\Users\Dell\Downloads\EV_Charging")
    md_file = base_dir / "MAJOR_PROJECT_PHASE_1_REPORT.md"
    docx_file = base_dir / "MAJOR_PROJECT_PHASE_1_REPORT.docx"
    parse_markdown_to_docx(md_file, docx_file)

