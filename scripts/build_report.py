"""
scripts/build_report.py
Comprehensive Generator for GovDoc-RAG RV University Project Report.
Strictly conforms to 'RV university report format.docx'.
"""

import os
import sys
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def set_cell_shading(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_table_borders(table, color="CCCCCC", sz="4", val="single"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(f'''
        <w:tblBorders {nsdecls("w")}>
            <w:top w:val="{val}" w:sz="8" w:space="0" w:color="000000"/>
            <w:bottom w:val="{val}" w:sz="8" w:space="0" w:color="000000"/>
            <w:left w:val="none"/>
            <w:right w:val="none"/>
            <w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:insideV w:val="none"/>
        </w:tblBorders>
    ''')
    tblPr.append(borders)

def add_page_number_field(run):
    fldChar1 = parse_xml(r'<w:fldChar %s w:fldCharType="begin"/>' % nsdecls('w'))
    instrText = parse_xml(r'<w:instrText %s xml:space="preserve"> PAGE </w:instrText>' % nsdecls('w'))
    fldChar2 = parse_xml(r'<w:fldChar %s w:fldCharType="separate"/>' % nsdecls('w'))
    fldChar3 = parse_xml(r'<w:fldChar %s w:fldCharType="end"/>' % nsdecls('w'))
    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)
    run._r.append(fldChar3)

def setup_section_footer(section, fmt='decimal', start=1, is_first_diff=False):
    sectPr = section._sectPr
    for el in sectPr.findall('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}pgNumType'):
        sectPr.remove(el)
    pgNumType = parse_xml(f'<w:pgNumType {nsdecls("w")} w:fmt="{fmt}" w:start="{start}"/>')
    sectPr.append(pgNumType)

    section.different_first_page_header_footer = is_first_diff
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.text = ''
    run = p.add_run()
    run.font.name = 'Times New Roman'
    run.font.size = Pt(11)
    add_page_number_field(run)

def generate_full_report():
    doc = docx.Document()
    
    # Configure A4 dimensions & strict RV University margins
    sec_prelim = doc.sections[0]
    sec_prelim.page_width = Inches(8.27)
    sec_prelim.page_height = Inches(11.69)
    sec_prelim.left_margin = Inches(3.81 / 2.54)
    sec_prelim.right_margin = Inches(2.54 / 2.54)
    sec_prelim.top_margin = Inches(2.54 / 2.54)
    sec_prelim.bottom_margin = Inches(2.54 / 2.54)
    setup_section_footer(sec_prelim, fmt='lowerRoman', start=1, is_first_diff=True)
    
    # Configure Default Style
    style_normal = doc.styles['Normal']
    font_normal = style_normal.font
    font_normal.name = 'Times New Roman'
    font_normal.size = Pt(12)
    font_normal.color.rgb = RGBColor(0, 0, 0)
    style_normal.paragraph_format.line_spacing = 1.5
    style_normal.paragraph_format.space_after = Pt(6)
    
    # Helper formatting closures
    def add_p(text="", align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=0, space_after=6, line_spacing=1.5, bold=False, italic=False, size=12, color=(0,0,0)):
        p = doc.add_paragraph()
        p.alignment = align
        p.paragraph_format.space_before = Pt(space_before)
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = line_spacing
        if text:
            run = p.add_run(text)
            run.font.name = 'Times New Roman'
            run.font.size = Pt(size)
            run.bold = bold
            run.italic = italic
            run.font.color.rgb = RGBColor(*color)
        return p

    def add_chap_heading(chap_num, title):
        doc.add_page_break()
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(24)
        p.paragraph_format.space_after = Pt(12)
        r = p.add_run(f"CHAPTER {chap_num}\n{title.upper()}")
        r.font.name = 'Times New Roman'
        r.font.size = Pt(16)
        r.bold = True
        r.font.color.rgb = RGBColor(0, 0, 0)
        return p

    def add_sec_heading(sec_num, title):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(16)
        p.paragraph_format.space_after = Pt(4)
        r = p.add_run(f"{sec_num} {title.upper()}")
        r.font.name = 'Times New Roman'
        r.font.size = Pt(14)
        r.bold = True
        r.font.color.rgb = RGBColor(0, 0, 0)
        return p

    def add_subsec_heading(subsec_num, title):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(f"{subsec_num} {title.upper()}")
        r.font.name = 'Times New Roman'
        r.font.size = Pt(12)
        r.bold = True
        r.font.color.rgb = RGBColor(0, 0, 0)
        return p

    def add_subsubsec_heading(subsubsec_num, title):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(f"{subsubsec_num} {title}")
        r.font.name = 'Times New Roman'
        r.font.size = Pt(12)
        r.bold = True
        r.italic = True
        r.font.color.rgb = RGBColor(0, 0, 0)
        return p

    def add_fig_image(img_path, caption_num, caption_text, width=Inches(5.4)):
        if os.path.exists(img_path):
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.paragraph_format.space_before = Pt(12)
            p_img.paragraph_format.space_after = Pt(4)
            run = p_img.add_run()
            run.add_picture(img_path, width=width)
            
            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap.paragraph_format.space_before = Pt(2)
            p_cap.paragraph_format.space_after = Pt(14)
            r_bold = p_cap.add_run(f"Figure {caption_num}: ")
            r_bold.font.name = 'Times New Roman'
            r_bold.font.size = Pt(10.5)
            r_bold.bold = True
            r_txt = p_cap.add_run(caption_text)
            r_txt.font.name = 'Times New Roman'
            r_txt.font.size = Pt(10.5)
            r_txt.italic = True
        else:
            add_p(f"[FIGURE {caption_num}: {caption_text}]", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True)

    def style_academic_table(table, col_widths, headers, data, alignments=None, caption=None):
        if caption:
            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap.paragraph_format.space_before = Pt(12)
            p_cap.paragraph_format.space_after = Pt(4)
            r_cap = p_cap.add_run(caption)
            r_cap.font.name = 'Times New Roman'
            r_cap.font.size = Pt(10.5)
            r_cap.bold = True

        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        set_table_borders(table, color="D3D3D3", sz="4", val="single")

        # Format Header Row
        hdr_row = table.rows[0]
        tblPr = table._tbl.tblPr
        tblHeader = parse_xml(f'<w:tblHeader {nsdecls("w")}/>')
        hdr_row._tr.get_or_add_trPr().append(tblHeader)

        for col_idx, h_text in enumerate(headers):
            c = hdr_row.cells[col_idx]
            c.width = Inches(col_widths[col_idx])
            set_cell_margins(c, top=80, bottom=80, left=100, right=100)
            set_cell_shading(c, "F1F5F9")
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            r = p.add_run(h_text)
            r.font.name = 'Times New Roman'
            r.font.size = Pt(9.5)
            r.bold = True

        # Format Data Rows
        for row_idx, r_data in enumerate(data):
            row = table.rows[row_idx + 1]
            for col_idx, val in enumerate(r_data):
                c = row.cells[col_idx]
                c.width = Inches(col_widths[col_idx])
                set_cell_margins(c, top=60, bottom=60, left=100, right=100)
                p = c.paragraphs[0]
                align = alignments[col_idx] if alignments else WD_ALIGN_PARAGRAPH.LEFT
                p.alignment = align
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.05
                r = p.add_run(str(val))
                r.font.name = 'Times New Roman'
                r.font.size = Pt(9.0)

        p_after = doc.add_paragraph()
        p_after.paragraph_format.space_after = Pt(8)

    # ════════════════════════════════════════════════════════════════════════════
    # 1. COVER PAGE
    # ════════════════════════════════════════════════════════════════════════════
    add_p("School of Computer Science and Engineering", align=WD_ALIGN_PARAGRAPH.CENTER, space_before=10, space_after=4, bold=True, size=14)
    add_p("October, 2026", align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=28, size=12)

    add_p("HYBRID MULTILINGUAL RAG-BASED DOCUMENT INTELLIGENCE SYSTEM FOR SEMANTIC SEARCH AND SUMMARIZATION FOR INDIAN GOVERNMENT NOTIFICATIONS",
          align=WD_ALIGN_PARAGRAPH.CENTER, space_before=10, space_after=20, line_spacing=1.2, bold=True, size=18)

    add_p("Submitted in partial fulfillment for the award of the degree of", align=WD_ALIGN_PARAGRAPH.CENTER, space_before=10, space_after=8, italic=True, size=13)
    add_p("Bachelor of Technology (Honours)\nin\nComputer Science and Engineering", align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=24, line_spacing=1.2, bold=True, size=18)

    add_p("by", align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=8, italic=True, size=13)
    
    authors = [
        ("NAGARJUNA V V", "nagarjunavvbtech24@rvu.edu.in"),
        ("KURLAPALLI CHAKRADHAR REDDY", "kurlapaliichakradharreddybtech24@rvu.edu.in"),
        ("M VIJAYA SNEHAN", "mvijayasnehanbtech24@rvu.edu.in"),
        ("MANOJ R", "manojrbtech24@rvu.edu.in")
    ]
    for name, email in authors:
        add_p(name, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=1, bold=True, size=13)
        add_p(f"({email})", align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=4, size=10)

    add_p("Guided by", align=WD_ALIGN_PARAGRAPH.CENTER, space_before=14, space_after=4, italic=True, size=13)
    add_p("Dr. Rashmi S", align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=2, bold=True, size=14)
    add_p("Assistant Professor, School of Computer Science and Engineering", align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=16, size=11)

    logo_path = "extracted_template_media/image1.png"
    if os.path.exists(logo_path):
        p_logo = doc.add_paragraph()
        p_logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_logo.paragraph_format.space_before = Pt(8)
        p_logo.paragraph_format.space_after = Pt(12)
        r_logo = p_logo.add_run()
        r_logo.add_picture(logo_path, width=Inches(2.2))

    add_p("School of Computer Science and Engineering\nRV University", align=WD_ALIGN_PARAGRAPH.CENTER, space_before=4, space_after=2, bold=True, size=13)
    add_p("RV Vidyaniketan, 8th Mile, Mysore Road, Bengaluru, Karnataka, India - 560059", align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=0, size=10.5)

    # ════════════════════════════════════════════════════════════════════════════
    # 2. TITLE PAGE
    # ════════════════════════════════════════════════════════════════════════════
    doc.add_page_break()
    add_p("School of Computer Science and Engineering", align=WD_ALIGN_PARAGRAPH.CENTER, space_before=10, space_after=4, bold=True, size=14)
    add_p("October, 2026", align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=28, size=12)

    add_p("HYBRID MULTILINGUAL RAG-BASED DOCUMENT INTELLIGENCE SYSTEM FOR SEMANTIC SEARCH AND SUMMARIZATION FOR INDIAN GOVERNMENT NOTIFICATIONS",
          align=WD_ALIGN_PARAGRAPH.CENTER, space_before=10, space_after=20, line_spacing=1.2, bold=True, size=18)

    add_p("Submitted in partial fulfillment for the award of the degree of", align=WD_ALIGN_PARAGRAPH.CENTER, space_before=10, space_after=8, italic=True, size=13)
    add_p("Bachelor of Technology (Honours)\nin\nComputer Science and Engineering", align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=24, line_spacing=1.2, bold=True, size=18)

    add_p("by", align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=8, italic=True, size=13)
    for name, email in authors:
        add_p(name, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=1, bold=True, size=13)
        add_p(f"({email})", align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=4, size=10)

    add_p("Guided by", align=WD_ALIGN_PARAGRAPH.CENTER, space_before=14, space_after=4, italic=True, size=13)
    add_p("Dr. Rashmi S", align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=2, bold=True, size=14)
    add_p("Assistant Professor, School of Computer Science and Engineering", align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=16, size=11)

    if os.path.exists(logo_path):
        p_logo = doc.add_paragraph()
        p_logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_logo.paragraph_format.space_before = Pt(8)
        p_logo.paragraph_format.space_after = Pt(12)
        r_logo = p_logo.add_run()
        r_logo.add_picture(logo_path, width=Inches(2.2))

    add_p("School of Computer Science and Engineering\nRV University", align=WD_ALIGN_PARAGRAPH.CENTER, space_before=4, space_after=2, bold=True, size=13)
    add_p("RV Vidyaniketan, 8th Mile, Mysore Road, Bengaluru, Karnataka, India - 560059", align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=0, size=10.5)

    # ════════════════════════════════════════════════════════════════════════════
    # 3. CERTIFICATE
    # ════════════════════════════════════════════════════════════════════════════
    doc.add_page_break()
    add_p("CERTIFICATE", align=WD_ALIGN_PARAGRAPH.CENTER, space_before=24, space_after=20, bold=True, size=16)

    cert_text = (
        "This is to certify that the project work titled “HYBRID MULTILINGUAL RAG-BASED DOCUMENT "
        "INTELLIGENCE SYSTEM FOR SEMANTIC SEARCH AND SUMMARIZATION FOR INDIAN GOVERNMENT NOTIFICATIONS” "
        "is a bonafide work carried out by Nagarjuna V V (nagarjunavvbtech24@rvu.edu.in), Kurlapalli Chakradhar Reddy "
        "(kurlapaliichakradharreddybtech24@rvu.edu.in), M Vijaya Snehan (mvijayasnehanbtech24@rvu.edu.in), and Manoj R "
        "(manojrbtech24@rvu.edu.in), bonafide students of Bachelor of Technology (Honours) at the School of Computer Science "
        "and Engineering, RV University, Bengaluru, in partial fulfillment for the award of the degree of Bachelor of Technology "
        "(Honours) in Computer Science and Engineering, during the Academic Year 2026–2027."
    )
    add_p(cert_text, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=8, space_after=28, line_spacing=1.5, size=12)

    sig_table = doc.add_table(rows=3, cols=3)
    sig_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    sigs = [
        ("Dr. Rashmi S", "Dr. Rashmi S", "Dr. G Shobha"),
        ("Internal Guide\nAssistant Professor, SoCSE\nRV University", "Program Director\nB.Tech (Honours), SoCSE\nRV University", "Dean\nSoCSE\nRV University"),
        ("Date: ________________", "Date: ________________", "Date: ________________")
    ]
    for row_idx, row_data in enumerate(sigs):
        for col_idx, cell_text in enumerate(row_data):
            c = sig_table.rows[row_idx].cells[col_idx]
            c.width = Inches(2.2)
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.line_spacing = 1.15
            p.paragraph_format.space_after = Pt(2)
            r = p.add_run(cell_text)
            r.font.name = 'Times New Roman'
            r.font.size = Pt(10)
            if row_idx == 0:
                r.bold = True

    add_p("", space_before=16, space_after=12)
    add_p("Examiners Committee:", align=WD_ALIGN_PARAGRAPH.LEFT, space_before=12, space_after=8, bold=True, size=11)
    
    exam_tbl = doc.add_table(rows=3, cols=2)
    exam_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(["Name of the Examiner", "Signature with Date"]):
        c = exam_tbl.rows[0].cells[i]
        c.width = Inches(3.2)
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(h)
        r.bold = True
        r.font.name = 'Times New Roman'
        r.font.size = Pt(11)
    
    for row_idx, (e_name, e_sig) in enumerate([("1. _________________________________________", "1. _________________________________________"), ("2. _________________________________________", "2. _________________________________________")]):
        r_cells = exam_tbl.rows[row_idx + 1].cells
        r_cells[0].paragraphs[0].add_run(e_name).font.size = Pt(10.5)
        r_cells[1].paragraphs[0].add_run(e_sig).font.size = Pt(10.5)

    # ════════════════════════════════════════════════════════════════════════════
    # 4. DECLARATION
    # ════════════════════════════════════════════════════════════════════════════
    doc.add_page_break()
    add_p("DECLARATION", align=WD_ALIGN_PARAGRAPH.CENTER, space_before=24, space_after=20, bold=True, size=16)

    decl_p1 = (
        "We hereby declare that the project report entitled “HYBRID MULTILINGUAL RAG-BASED DOCUMENT "
        "INTELLIGENCE SYSTEM FOR SEMANTIC SEARCH AND SUMMARIZATION FOR INDIAN GOVERNMENT NOTIFICATIONS” "
        "submitted by us to RV University, Bengaluru, for the award of the degree of Bachelor of Technology (Honours) "
        "in Computer Science and Engineering is a bonafide record of original work carried out by us under the supervision "
        "and guidance of Dr. Rashmi S, Assistant Professor, School of Computer Science and Engineering, RV University."
    )
    add_p(decl_p1, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=8, space_after=12, line_spacing=1.5, size=12)

    decl_p2 = (
        "We further declare that the content embodied in this project report has not been submitted previously, in part or "
        "in full, to any other university, institute, or examining body for the award of any degree, diploma, fellowship, or "
        "other academic distinction."
    )
    add_p(decl_p2, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=0, space_after=24, line_spacing=1.5, size=12)

    add_p("Place: Bengaluru\nDate: 30 October 2026", align=WD_ALIGN_PARAGRAPH.LEFT, space_before=12, space_after=20, bold=True, size=11)

    cand_tbl = doc.add_table(rows=4, cols=2)
    cand_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, (name, email) in enumerate(authors):
        c0 = cand_tbl.rows[i].cells[0]
        c1 = cand_tbl.rows[i].cells[1]
        c0.width = Inches(3.5)
        c1.width = Inches(3.0)
        p0 = c0.paragraphs[0]
        p0.add_run(f"{name}\n({email})").font.name = 'Times New Roman'
        p0.runs[0].font.size = Pt(10.5)
        p0.runs[0].bold = True
        p1 = c1.paragraphs[0]
        p1.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p1.add_run("Signature: _______________________").font.name = 'Times New Roman'
        p1.runs[0].font.size = Pt(10.5)

    # ════════════════════════════════════════════════════════════════════════════
    # 5. ABSTRACT
    # ════════════════════════════════════════════════════════════════════════════
    doc.add_page_break()
    add_p("ABSTRACT", align=WD_ALIGN_PARAGRAPH.CENTER, space_before=24, space_after=18, bold=True, size=16)

    abs_1 = (
        "Indian federal and state government bodies administer statutory notifications, administrative circulars, "
        "and public policies primarily in formal, bureaucratic English and Hindi gazettes. This creates an acute linguistic "
        "and structural barrier for vernacular citizens and rural administrative officers operating in resource-constrained regional "
        "offices. In this work, we design, implement, and empirically validate GovDoc-RAG, a cross-lingual Retrieval-Augmented Generation "
        "(RAG) and document intelligence framework optimized for offline edge deployment on commodity GPUs constrained to 4.0 GB of VRAM."
    )
    add_p(abs_1, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=6, space_after=10, line_spacing=1.5, size=12)

    abs_2 = (
        "GovDoc-RAG introduces four foundational technical pillars: (1) A dual-view query bridge that unifies native scripts and "
        "phonological transliterations (via IndicXlit and BAAI/bge-m3 dense embeddings) to overcome zero-shot retrieval vocabulary gaps "
        "across English, Hindi, Kannada, Hinglish, and Kanglish; (2) A calibrated cosine trust gate operating with dual decision thresholds "
        "(tau_hi = 0.65, tau_lo = 0.30, and calibrated tau* = 0.40) combined with an entity-level hallucination verifier that achieves "
        "a 100% detection rate on corrupted numerical values and dates without raising false alarms on grounded evidence; (3) A parameter-efficient "
        "abstractive summarisation engine leveraging IndicBART with Low-Rank Adaptation (LoRA, r=16, alpha=32) paired with IndicTrans2, "
        "elevating cross-lingual EN->HI summarisation from 0.0071 to 0.2208 ROUGE-L (a 31.1-fold relative improvement, p < 0.001) and achieving "
        "0.5048 ROUGE-L on monolingual Hindi; and (4) An edge memory co-residency manager that enforces mutual exclusion between model backbones, "
        "guaranteeing a peak GPU footprint of 3,219.0 MB against a 4,096.0 MB hard physical ceiling (maintaining an 877.0 MB safety margin) with "
        "a median QA latency of 1,494.0 ms."
    )
    add_p(abs_2, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=0, space_after=10, line_spacing=1.5, size=12)

    abs_3 = (
        "Extensive benchmarking on 1,500 multilingual queries across English, Hindi, Kannada, and code-mixed romanised dialects demonstrates "
        "that GovDoc-RAG maintains 84.67% Recall@1 for English, 47.83% for Hindi, 47.50% for Kannada, and 55.07% overall macro Recall@1. By eliminating "
        "cloud API dependencies and offering verifiable citation tracking down to exact document page numbers, GovDoc-RAG provides a scalable, "
        "privacy-preserving document intelligence architecture for Indian public governance."
    )
    add_p(abs_3, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=0, space_after=12, line_spacing=1.5, size=12)

    p_kw = add_p("Keywords: ", align=WD_ALIGN_PARAGRAPH.LEFT, space_before=6, space_after=12, bold=True, size=11)
    p_kw.runs[0].bold = True
    r_kw = p_kw.add_run("Retrieval-Augmented Generation (RAG), Edge Computing, Low-Resource Indic NLP, LoRA Fine-Tuning, IndicBART, BGE-M3, Trust Gate Abstention, Small Language Models, Document Intelligence, e-Governance.")
    r_kw.font.name = 'Times New Roman'
    r_kw.font.size = Pt(11)
    r_kw.italic = True

    # ════════════════════════════════════════════════════════════════════════════
    # 6. ACKNOWLEDGEMENTS
    # ════════════════════════════════════════════════════════════════════════════
    doc.add_page_break()
    add_p("ACKNOWLEDGEMENT", align=WD_ALIGN_PARAGRAPH.CENTER, space_before=24, space_after=18, bold=True, size=16)

    ack_1 = (
        "It is our pleasure to express our deepest sense of gratitude to our internal guide, Dr. Rashmi S, Assistant Professor "
        "and Program Director, School of Computer Science and Engineering, RV University, for her constant guidance, continual "
        "encouragement, and intellectual mentorship throughout the progression of this project. Her constructive critiques, technical insights, "
        "and patience enabled us to navigate complex research hurdles in low-resource cross-lingual processing and edge hardware optimization. "
        "Working under her supervision has been an invaluable learning experience."
    )
    add_p(ack_1, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=6, space_after=10, line_spacing=1.5, size=12)

    ack_2 = (
        "We wish to express our sincere gratitude to Prof. Dr. Ram Kumar Kakani, Vice Chancellor, RV University, and Prof. Dr. G Shobha, "
        "Dean, School of Computer Science and Engineering, RV University, for providing an exemplary academic and technological environment, "
        "access to computational infrastructure, and constant encouragement towards research excellence."
    )
    add_p(ack_2, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=0, space_after=10, line_spacing=1.5, size=12)

    ack_3 = (
        "We also extend our earnest appreciation to all the faculty members and technical staff of the School of Computer Science and "
        "Engineering, whose pedagogical support and timely feedback enriched our academic journey. Their dedication provided us with the foundational "
        "rigor essential to bring this complex engineering project to fruition."
    )
    add_p(ack_3, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=0, space_after=10, line_spacing=1.5, size=12)

    ack_4 = (
        "Finally, we are eternally grateful to our parents, family members, and friends for their unwavering love, moral encouragement, "
        "and support during late hours of model training and system debugging. We also thank all individuals who directly or indirectly "
        "assisted us in completing this research endeavour."
    )
    add_p(ack_4, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=0, space_after=24, line_spacing=1.5, size=12)

    add_p("Place: Bengaluru\nDate: 30 October 2026", align=WD_ALIGN_PARAGRAPH.LEFT, space_before=10, space_after=10, bold=True, size=11)
    add_p("Nagarjuna V V\nKurlapalli Chakradhar Reddy\nM Vijaya Snehan\nManoj R", align=WD_ALIGN_PARAGRAPH.RIGHT, space_before=0, space_after=0, bold=True, size=11)

    # ════════════════════════════════════════════════════════════════════════════
    # 7. TABLE OF CONTENTS
    # ════════════════════════════════════════════════════════════════════════════
    doc.add_page_break()
    add_p("TABLE OF CONTENTS", align=WD_ALIGN_PARAGRAPH.CENTER, space_before=24, space_after=18, bold=True, size=16)

    toc_items = [
        ("CERTIFICATE", "iii"),
        ("DECLARATION", "iv"),
        ("ABSTRACT", "v"),
        ("ACKNOWLEDGEMENT", "vi"),
        ("LIST OF FIGURES", "ix"),
        ("LIST OF TABLES", "x"),
        ("LIST OF ACRONYMS AND ABBREVIATIONS", "xi"),
        ("CHAPTER 1: INTRODUCTION", "1"),
        ("    1.1 Context and Background", "1"),
        ("    1.2 Problem Statement", "3"),
        ("    1.3 Project Objectives", "4"),
        ("    1.4 Scope and Delimitations of the Project", "5"),
        ("CHAPTER 2: BACKGROUND AND LITERATURE SURVEY", "7"),
        ("    2.1 Chronological Literature Survey (2022–2025)", "7"),
        ("    2.2 Comprehensive Literature Comparison Matrix", "13"),
        ("    2.3 Identified Research Gaps", "15"),
        ("    2.4 Supporting Tools and Technologies", "16"),
        ("CHAPTER 3: SYSTEM METHODOLOGY AND ARCHITECTURE", "19"),
        ("    3.1 End-to-End System Architecture", "19"),
        ("    3.2 Layout-Aware Document Ingestion and Chunking", "21"),
        ("    3.3 Cross-Lingual Dual-View Query Bridging", "23"),
        ("    3.4 Calibrated Trust Gate and Abstention Shield", "25"),
        ("    3.5 Grounded Answer Generation and Entity Verification", "28"),
        ("    3.6 Edge Memory Co-Residency Guard (Sub-4GB VRAM)", "30"),
        ("    3.7 Abstractive Summarisation Pipeline with LoRA", "32"),
        ("CHAPTER 4: IMPLEMENTATION AND EXPERIMENTAL SETUP", "35"),
        ("    4.1 Datasets and Curated Benchmarks", "35"),
        ("    4.2 System Components, Parameter Scales, and Quantisation", "37"),
        ("    4.3 Engineering Milestones and Workflow (M1–M11)", "39"),
        ("    4.4 Hardware and Experimental Environment", "42"),
        ("    4.5 Training Dynamics of IndicBART LoRA Adapter", "43"),
        ("CHAPTER 5: EXPERIMENTAL RESULTS AND EVALUATION", "45"),
        ("    5.1 Abstractive Summarisation Performance", "45"),
        ("    5.2 Cross-Lingual Dense Retrieval Performance", "48"),
        ("    5.3 Calibrated Trust Boundary and Abstention Analysis", "51"),
        ("    5.4 Edge Hardware Footprint and Latency Profiling", "54"),
        ("    5.5 Qualitative Case Studies and System Traces", "57"),
        ("    5.6 Baseline Comparisons and Architectural Ablations", "60"),
        ("CHAPTER 6: INFERENCES AND DISCUSSION", "62"),
        ("    6.1 Key Empirical Findings", "62"),
        ("    6.2 Fulfillment of Project Objectives", "64"),
        ("CHAPTER 7: IMPRESSION REPORT", "66"),
        ("    7.1 Personal Learning Experience", "66"),
        ("    7.2 Technical and Analytical Skill Enhancement", "67"),
        ("    7.3 Academic and Industrial Relevance", "68"),
        ("CHAPTER 8: CONCLUSION AND FUTURE WORK", "70"),
        ("    8.1 Summary of Contributions", "70"),
        ("    8.2 Practical Limitations", "71"),
        ("    8.3 Directions for Future Research", "72"),
        ("REFERENCES (APA FORMAT)", "74"),
        ("APPENDIX A: SYSTEM PROMPTS AND VERIFIER RULES", "78"),
        ("APPENDIX B: SYSTEM HYPERPARAMETERS AND HARDWARE LOGS", "81"),
        ("APPENDIX C: MULTILINGUAL BENCHMARK QUERY TRACES", "84"),
    ]

    toc_tbl = doc.add_table(rows=len(toc_items), cols=2)
    toc_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    for idx, (title, page_str) in enumerate(toc_items):
        c0 = toc_tbl.rows[idx].cells[0]
        c1 = toc_tbl.rows[idx].cells[1]
        c0.width = Inches(5.8)
        c1.width = Inches(0.8)
        p0 = c0.paragraphs[0]
        p0.paragraph_format.line_spacing = 1.15
        p0.paragraph_format.space_after = Pt(2)
        r0 = p0.add_run(title)
        r0.font.name = 'Times New Roman'
        r0.font.size = Pt(11)
        if title.startswith("CHAPTER") or title.startswith("REFERENCES") or title.startswith("APPENDIX") or title in ["CERTIFICATE", "DECLARATION", "ABSTRACT", "ACKNOWLEDGEMENT", "LIST OF FIGURES", "LIST OF TABLES", "LIST OF ACRONYMS AND ABBREVIATIONS"]:
            r0.bold = True
        p1 = c1.paragraphs[0]
        p1.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p1.paragraph_format.line_spacing = 1.15
        p1.paragraph_format.space_after = Pt(2)
        r1 = p1.add_run(page_str)
        r1.font.name = 'Times New Roman'
        r1.font.size = Pt(11)
        if r0.bold:
            r1.bold = True

    # ════════════════════════════════════════════════════════════════════════════
    # 8. LIST OF FIGURES
    # ════════════════════════════════════════════════════════════════════════════
    doc.add_page_break()
    add_p("LIST OF FIGURES", align=WD_ALIGN_PARAGRAPH.CENTER, space_before=24, space_after=18, bold=True, size=16)

    fig_items = [
        ("Figure 1", "End-to-End GovDoc-RAG System Architecture Pipeline", "20"),
        ("Figure 2", "Cross-Lingual Query Bridge and Calibrated Trust Gate Flowchart", "24"),
        ("Figure 3", "IndicBART LoRA Adapter Training and Validation Loss Curves", "44"),
        ("Figure 4", "Multilingual Summarisation ROUGE-L Comparison across Directions", "47"),
        ("Figure 5", "Dense Retrieval Recall@K on Benchmark A across Query Languages", "50"),
        ("Figure 6", "Top-1 Cosine Similarity Distributions for Answerable, Near-Miss, and OOD", "52"),
        ("Figure 7", "Threshold Sweep: Answerable Recall vs. False Accept Rate with ROC Inset", "53"),
        ("Figure 8", "Peak GPU Memory per Residency Mode against 4.0 GB Hard Cap", "55"),
        ("Figure 9", "End-to-End Latency Breakdown by Stage and Query Language", "56"),
    ]

    fig_tbl = doc.add_table(rows=len(fig_items), cols=3)
    fig_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    for idx, (f_num, f_desc, f_page) in enumerate(fig_items):
        c0 = fig_tbl.rows[idx].cells[0]
        c1 = fig_tbl.rows[idx].cells[1]
        c2 = fig_tbl.rows[idx].cells[2]
        c0.width = Inches(1.2)
        c1.width = Inches(4.6)
        c2.width = Inches(0.8)
        p0 = c0.paragraphs[0]
        p0.paragraph_format.space_after = Pt(3)
        r0 = p0.add_run(f_num)
        r0.bold = True
        r0.font.name = 'Times New Roman'
        r0.font.size = Pt(11)
        p1 = c1.paragraphs[0]
        p1.paragraph_format.space_after = Pt(3)
        r1 = p1.add_run(f_desc)
        r1.font.name = 'Times New Roman'
        r1.font.size = Pt(11)
        p2 = c2.paragraphs[0]
        p2.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p2.paragraph_format.space_after = Pt(3)
        r2 = p2.add_run(f_page)
        r2.font.name = 'Times New Roman'
        r2.font.size = Pt(11)

    # ════════════════════════════════════════════════════════════════════════════
    # 9. LIST OF TABLES
    # ════════════════════════════════════════════════════════════════════════════
    doc.add_page_break()
    add_p("LIST OF TABLES", align=WD_ALIGN_PARAGRAPH.CENTER, space_before=24, space_after=18, bold=True, size=16)

    table_items = [
        ("Table 2.1", "Comprehensive Literature Survey and Comparative Matrix of Prior Art", "13"),
        ("Table 4.1", "Dataset Statistics across Languages, Domains, and Evaluation Splits", "36"),
        ("Table 4.2", "System Components, Model Parameter Scales, Precision, and Hyperparameters", "38"),
        ("Table 5.1", "Multilingual Summarisation Results on Test Split with 95% Bootstrap CIs", "46"),
        ("Table 5.2", "Dense Retrieval Performance on Benchmark A (N = 1,500 Queries)", "49"),
        ("Table 5.3", "Trust Boundary Abstention and Grounding Verifier Performance", "51"),
        ("Table 5.4", "System Resource Footprint, Peak VRAM, and End-to-End Latency Profile", "54"),
        ("Table 5.5", "Qualitative Question Answering and Summarisation Evaluation Traces", "58"),
        ("Table 6.1", "Project Objective Fulfillment and Quantitative Validation Matrix", "64"),
    ]

    tbl_list = doc.add_table(rows=len(table_items), cols=3)
    tbl_list.alignment = WD_TABLE_ALIGNMENT.CENTER
    for idx, (t_num, t_desc, t_page) in enumerate(table_items):
        c0 = tbl_list.rows[idx].cells[0]
        c1 = tbl_list.rows[idx].cells[1]
        c2 = tbl_list.rows[idx].cells[2]
        c0.width = Inches(1.2)
        c1.width = Inches(4.6)
        c2.width = Inches(0.8)
        p0 = c0.paragraphs[0]
        p0.paragraph_format.space_after = Pt(3)
        r0 = p0.add_run(t_num)
        r0.bold = True
        r0.font.name = 'Times New Roman'
        r0.font.size = Pt(11)
        p1 = c1.paragraphs[0]
        p1.paragraph_format.space_after = Pt(3)
        r1 = p1.add_run(t_desc)
        r1.font.name = 'Times New Roman'
        r1.font.size = Pt(11)
        p2 = c2.paragraphs[0]
        p2.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p2.paragraph_format.space_after = Pt(3)
        r2 = p2.add_run(t_page)
        r2.font.name = 'Times New Roman'
        r2.font.size = Pt(11)

    # ════════════════════════════════════════════════════════════════════════════
    # 10. LIST OF ACRONYMS
    # ════════════════════════════════════════════════════════════════════════════
    doc.add_page_break()
    add_p("LIST OF ACRONYMS AND ABBREVIATIONS", align=WD_ALIGN_PARAGRAPH.CENTER, space_before=24, space_after=18, bold=True, size=16)

    acronyms = [
        ("API", "Application Programming Interface"),
        ("AUROC", "Area Under the Receiver Operating Characteristic"),
        ("BERT", "Bidirectional Encoder Representations from Transformers"),
        ("BGE", "BAAI General Embedding"),
        ("CI", "Confidence Interval"),
        ("CPU", "Central Processing Unit"),
        ("CUDA", "Compute Unified Device Architecture"),
        ("Dense Retrieval", "Embedding-based k-Nearest Neighbor Semantic Search"),
        ("DocAI", "Document Artificial Intelligence"),
        ("FAR", "False Accept Rate / False Abstain Rate"),
        ("FP16", "16-Bit Half Precision Floating-Point Format"),
        ("FP32", "32-Bit Single Precision Floating-Point Format"),
        ("GGUF", "GPT-Generated Unified Format (llama.cpp)"),
        ("GoK", "Government of Karnataka"),
        ("GPU", "Graphics Processing Unit"),
        ("Hinglish", "Hindi written in the Roman (Latin) Script"),
        ("IndicBART", "Indic Bidirectional and Auto-Regressive Transformer"),
        ("IndicTrans2", "Multilingual Neural Machine Translation for 22 Indic Languages"),
        ("IndicXlit", "Multilingual Roman-to-Native Script Transliteration Engine"),
        ("Kanglish", "Kannada written in the Roman (Latin) Script"),
        ("LLM", "Large Language Model"),
        ("LoRA", "Low-Rank Adaptation"),
        ("MRR", "Mean Reciprocal Rank"),
        ("NER", "Named Entity Recognition"),
        ("NLP", "Natural Language Processing"),
        ("OCR", "Optical Character Recognition"),
        ("OOD", "Out-of-Domain"),
        ("PIB", "Press Information Bureau (Government of India)"),
        ("RAG", "Retrieval-Augmented Generation"),
        ("ROUGE", "Recall-Oriented Understudy for Gisting Evaluation"),
        ("SLM", "Small Language Model (< 7 Billion Parameters)"),
        ("TKDE", "Transactions on Knowledge and Data Engineering"),
        ("UI", "User Interface"),
        ("USN", "University Seat Number"),
        ("VRAM", "Video Random Access Memory (GPU Dedicated Memory)"),
    ]

    acro_tbl = doc.add_table(rows=len(acronyms), cols=2)
    acro_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    for idx, (acro, desc) in enumerate(acronyms):
        c0 = acro_tbl.rows[idx].cells[0]
        c1 = acro_tbl.rows[idx].cells[1]
        c0.width = Inches(2.0)
        c1.width = Inches(4.6)
        p0 = c0.paragraphs[0]
        p0.paragraph_format.space_after = Pt(2)
        r0 = p0.add_run(acro)
        r0.bold = True
        r0.font.name = 'Times New Roman'
        r0.font.size = Pt(10.5)
        p1 = c1.paragraphs[0]
        p1.paragraph_format.space_after = Pt(2)
        r1 = p1.add_run(desc)
        r1.font.name = 'Times New Roman'
        r1.font.size = Pt(10.5)

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 2: MAIN REPORT CHAPTERS
    # ════════════════════════════════════════════════════════════════════════════
    sec_main = doc.add_section()
    sec_main.page_width = Inches(8.27)
    sec_main.page_height = Inches(11.69)
    sec_main.left_margin = Inches(3.81 / 2.54)
    sec_main.right_margin = Inches(2.54 / 2.54)
    sec_main.top_margin = Inches(2.54 / 2.54)
    sec_main.bottom_margin = Inches(2.54 / 2.54)
    sec_main.footer.is_linked_to_previous = False
    setup_section_footer(sec_main, fmt='decimal', start=1, is_first_diff=False)

    # ════════════════════════════════════════════════════════════════════════════
    # CHAPTER 1: INTRODUCTION
    # ════════════════════════════════════════════════════════════════════════════
    add_chap_heading(1, "INTRODUCTION")

    add_sec_heading("1.1", "Context and Background")
    add_p(
        "Over the past decade, the Government of India and various state administrations have undergone an extensive digital transformation "
        "under the flagship 'Digital India' program. Millions of statutory notifications, gazette releases, welfare guidelines, cabinet resolutions, "
        "and administrative circulars are now archived and disseminated electronically via public repositories such as the Press Information Bureau (PIB), "
        "e-Gazette portals, and state government circular hubs. However, the operational reality of these digital archives reveals an acute societal and "
        "functional disparity: while documents are technically public and indexed, they remain practically inaccessible to the vast majority of ordinary citizens, "
        "grassroots legal advocates, and regional administrative personnel."
    )
    add_p(
        "This accessibility gap arises from two fundamental structural factors: linguistic divergence and bureaucratic syntax. Indian government documentation "
        "is overwhelmingly authored in formal, statutory English or Sanskritised standard Hindi. These texts feature dense legalese, nested clauses, multi-page "
        "schedules, and bureaucratic cross-references. In contrast, India is home to over 1.4 billion people speaking 22 constitutionally recognized languages "
        "written in distinct indigenous scripts (such as Devanagari, Kannada, Telugu, Tamil, and Bengali). A substantial proportion of citizens interact with digital "
        "services through regional languages or colloquial code-mixed romanised dialects (such as Hinglish and Kanglish). When a farmer in rural Karnataka seeks to "
        "understand irrigation subsidy eligibility from a 40-page bilingual circular, standard keyword-based search portals fail entirely to bridge the phonological "
        "and script divide."
    )
    add_p(
        "Simultaneously, the recent explosion of generative Artificial Intelligence and Large Language Models (LLMs) has demonstrated remarkable question-answering "
        "and summarisation capabilities. Yet, existing enterprise solutions rely almost exclusively on commercial cloud APIs (such as OpenAI GPT-4, Google Gemini, "
        "or Anthropic Claude). In public administrative and judicial contexts, direct cloud dependence introduces three fatal vulnerabilities: (1) Data Sovereignty "
        "and Privacy Risks: Public servants cannot upload confidential draft policies, departmental disciplinary inquiries, or citizen records to external commercial servers; "
        "(2) Recurring Operational Costs: Per-token cloud API pricing poses a severe budgetary barrier for district-level panchayats and municipal offices; and "
        "(3) Severe Hallucination Hazards: General-purpose LLMs frequently produce plausible-sounding but factually fabricated statutory citations, wrong penalty amounts, "
        "or obsolete effective dates. In a legal or welfare context, an ungrounded generative hallucination can deprive a citizen of entitled benefits or misguide judicial proceedings."
    )

    add_sec_heading("1.2", "Problem Statement")
    add_p(
        "To bridge the gap between complex official documentation and multi-lingual citizens without compromising data privacy or exceeding edge compute budgets, "
        "there is an urgent need for an offline, edge-deployable document intelligence system. Specifically, the technical problem tackled in this project is "
        "formally formulated as follows:"
    )
    add_p(
        "Design, build, and empirically validate an end-to-end, privacy-preserving Retrieval-Augmented Generation (RAG) and abstractive summarisation system capable of: "
        "(a) Ingesting complex, multi-page Indian government gazettes and notifications while preserving hierarchical document layouts; "
        "(b) Executing accurate cross-lingual dense semantic retrieval across English, Hindi, Kannada, Hinglish, and Kanglish queries against English and bilingual knowledge bases; "
        "(c) Establishing a mathematically calibrated trust boundary that strictly abstains from answering out-of-domain, irrelevant, or unanswerable queries; "
        "(d) Synthesizing strictly grounded answers with mandatory [Page X] page-level citation anchors, supported by deterministic entity-level hallucination verification; "
        "(e) Producing fluent abstractive executive summaries in multiple Indic languages via parameter-efficient fine-tuning; and "
        "(f) Operating strictly within the physical hardware constraint of a single commodity edge GPU equipped with at most 4.0 GB of Dedicated VRAM, without invoking external cloud APIs."
    )

    add_sec_heading("1.3", "Project Objectives")
    add_p("To systematically achieve the problem statement, the project is structured around the following concrete research and engineering objectives:")
    
    obj_list = [
        "1. Hierarchical Layout-Aware Ingestion: Develop a multi-stage PDF ingestion pipeline that parses native searchable government notifications, extracts font and spatial layout hierarchies, and partitions documents into semantic section chunks while maintaining page number metadata.",
        "2. Cross-Lingual Script Unification: Construct a dual-view query bridging mechanism that couples phonological roman-to-native transliteration (IndicXlit) with a massive multi-lingual dense retriever (BAAI/bge-m3) to bridge script discrepancies across English, Hindi, Kannada, and code-mixed romanised variants.",
        "3. Calibrated Trust Gate & Abstention Engine: Formulate and calibrate a dual-threshold cosine decision shield (tau_hi, tau_lo, and calibrated tau*) that separates answerable evidence from near-miss distractions and completely out-of-domain (OOD) queries, preventing generative hallucination at the retrieval gateway.",
        "4. Grounded Small Language Model Generation: Integrate an efficient, quantised open-weights Small Language Model (Qwen 3 4B Instruct in Q4_K_M GGUF precision) to synthesize natural-language responses strictly constrained to retrieved evidence passages, embedding bracketed page citations.",
        "5. Deterministic Hallucination & Fact Verification: Implement an automated, entity-level verification filter that extracts dates, numerical values, monetary amounts, and statutory entities from generated responses and cross-validates them against source evidence chunks, rejecting unfaithful answers.",
        "6. Parameter-Efficient Indic Abstractive Summarisation: Fine-tune a compact sequence-to-sequence model (ai4bharat/IndicBART) using Low-Rank Adaptation (LoRA, r=16, alpha=32) on official government press releases (PMIndiaSum), elevating cross-lingual (EN->HI) and monolingual (HI->HI) abstractive headline summarisation.",
        "7. Edge Memory Co-Residency Manager: Engineer a mutual-exclusion memory state machine that orchestrates runtime transitions between heavy embedding models, generation models, and summarisation adapters, guaranteeing that peak GPU VRAM utilization never exceeds the 4,096 MB physical budget cap.",
        "8. Empirical Benchmarking and Validation: Curate standardized multilingual retrieval and grounded QA benchmarks (Benchmark A with 1,500 queries, Benchmark B with real circulars) to empirically evaluate ROUGE-L, Recall@K, AUROC, latency profiles, and VRAM margins under rigorous academic standards."
    ]
    for obj in obj_list:
        add_p(obj, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=2, space_after=4, size=11.5)

    add_sec_heading("1.4", "Scope and Delimitations of the Project")
    add_p(
        "Scope: The GovDoc-RAG system covers official notifications published by the Central Government of India (PIB, Ministry Gazettes) and State Governments "
        "(specifically Karnataka Government circulars). It supports user interaction in English, Hindi, Kannada, and romanised transliterated variants (Hinglish and Kanglish). "
        "The system provides two core operational modes: (1) Interactive Cross-Lingual Question Answering with trust-gated citation grounding; and (2) Abstractive Document "
        "Summarisation with multi-lingual translation bridging. The physical execution target is constrained to a local NVIDIA GeForce RTX 2050/3050 Laptop GPU with 4 GB VRAM."
    )
    add_p(
        "Delimitations: The system currently assumes digitally created, searchable PDF documents. Processing legacy scanned historical records characterized by heavy photocopier "
        "artifacts, ink bleeds, or handwritten rubber stamps is outside the current scope and would require specialized OCR layout pipelines. Furthermore, the summarisation model "
        "is trained on headline-length abstractive summaries (20–40 words), which optimizes concise executive gisting but is not designed for multi-page narrative synthesis."
    )

    # ════════════════════════════════════════════════════════════════════════════
    # CHAPTER 2: BACKGROUND AND LITERATURE SURVEY
    # ════════════════════════════════════════════════════════════════════════════
    add_chap_heading(2, "BACKGROUND AND LITERATURE SURVEY")

    add_sec_heading("2.1", "Chronological Literature Survey (2022–2025)")
    add_p(
        "In accordance with academic guidelines, this section provides an exhaustive, year-wise analytical review of 20 foundational and contemporary research papers "
        "that define the state-of-the-art in multilingual representation learning, Document AI, cross-lingual retrieval-augmented generation, trust verification, and summarisation."
    )

    # 2.1.1: 2022
    add_subsec_heading("2.1.1", "Foundational Research in Indic NLP and Document AI (2022)")
    add_p(
        "Dabre et al. (2022) [1] published 'IndicBART: A Pre-trained Model for Indic Natural Language Generation' in Findings of ACL 2022. They introduced a compact "
        "sequence-to-sequence model based on the mBART architecture, incorporating ALBERT-style parameter sharing across layers to restrict total model size to 244 million "
        "parameters. Trained on the IndicCorp corpus spanning 11 major Indic languages and English, IndicBART established that compact, language-family-specific models can rival "
        "massive multilingual LLMs if script unification is applied. However, the authors noted that without task-specific fine-tuning, IndicBART suffers from severe cross-lingual "
        "vocabulary fragmentation and produces repetitive or generic outputs on technical bureaucratic documents."
    )
    add_p(
        "Huang et al. (2022) [9] presented 'LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking' at ACM Multimedia 2022. LayoutLMv3 unified visual, "
        "textual, and spatial coordinates into a single multi-modal transformer architecture, demonstrating that preserving 2D spatial layouts and reading orders is essential "
        "for extracting tabular data and hierarchical sections from PDF circulars. While highly effective, LayoutLMv3 requires substantial GPU memory for continuous image patch "
        "tokenization, rendering full multi-modal pre-training impractical on sub-4GB edge devices."
    )

    # 2.1.2: 2023
    add_subsec_heading("2.1.2", "Emergence of Cross-Lingual RAG and Multimodal Augmentation (2023)")
    add_p(
        "Nie, Broscheit, and Schütze (2023) [3] published 'Cross-lingual Retrieval Augmented Prompt for Low-Resource Languages' at EMNLP 2023. They investigated scenarios "
        "where user queries are formulated in low-resource vernacular languages while authoritative evidence documents exist predominantly in high-resource English. Their findings "
        "demonstrated that naive translation before retrieval compounds compounding errors, whereas performing dense retrieval using cross-lingual bi-encoders yields significantly "
        "higher recall if phonological script differences are properly addressed."
    )
    add_p(
        "Panda, Chatterjee, and Sharma (2023) [4] authored 'Cross-modal Augmentation for Low-Resource Language Understanding and Generation' in ACM Transactions on Asian and "
        "Low-Resource Language Information Processing. They investigated data augmentation strategies across syntactic and acoustic modalities, establishing that phonological "
        "similarity between Indic language families can be leveraged through phonetic transliteration to dramatically expand effective vocabulary coverage in low-resource search tasks."
    )

    # 2.1.3: 2024
    add_subsec_heading("2.1.3", "Advances in Multilingual Embeddings, RAG Architectures, and Security (2024)")
    add_p(
        "Chen et al. (2024) [2] introduced 'BGE M3-Embedding: Multi-linguality, Multi-functionality, Multi-granularity Text Embeddings' (arXiv:2402.03216). BGE-M3 represents a "
        "breakthrough in multi-lingual representation learning, supporting over 100 languages with an input context length of 8,192 tokens. It supports multi-granularity retrieval "
        "by unifying dense embeddings, lexical sparse representations (BM25-like weights), and multi-vector ColBERT-style interactions into a single 568M parameter backbone. In GovDoc-RAG, "
        "we utilize the dense embedding capability of BGE-M3 (1024-dimensional normalized vectors) as the primary cross-lingual semantic search engine."
    )
    add_p(
        "Urlana, Rao, Ekbal, and Bhattacharyya (2024) [14] published 'PMIndiaSum: Multilingual and Cross-Lingual Headline Summarization for Languages of India' at LREC-COLING 2024. "
        "They released a curated corpus of over 30,000 official Indian government press releases published by the Press Information Bureau across 14 languages, paired with authentic "
        "administrative headline summaries. Their experiments revealed that zero-shot LLMs perform poorly on cross-lingual Indian summarisation, establishing that supervised adapter "
        "fine-tuning is strictly required for domain-accurate gisting."
    )
    add_p(
        "Ranaldi, Pucci, and Fallucchi (2024) [12] contributed 'Multilingual Retrieval-Augmented Generation for Knowledge-Intensive Question Answering' in Findings of EACL 2024. "
        "They evaluated cross-lingual RAG pipelines across disparate language typologies, revealing that retrieval recall plummets when user queries contain code-mixing or informal "
        "romanised spellings, underscoring the critical necessity of an explicit query transliteration bridge."
    )
    add_p(
        "Chirkova, Hou, and Vashishth (2024) [13] published 'Retrieval-Augmented Generation in Multilingual Settings' in Transactions of the ACL (TACL). They demonstrated that "
        "retrieving evidence in the original high-resource document language and feeding it to a multilingual generator yields superior factual accuracy compared to translating entire "
        "corpora beforehand, provided the generator is strictly instructed to avoid cross-lingual hallucination."
    )
    add_p(
        "Li et al. (2024) [5] conducted an exhaustive survey titled 'Mitigating Hallucination in Large Language Models: An Application-Oriented Survey' in ACM Computing Surveys. "
        "They classified hallucination mitigation into pre-generation, during-generation, and post-generation methods. They proved that post-generation deterministic entity extraction "
        "and numerical fact-checking against retrieved source text provides the highest safety guarantee in legal, medical, and administrative deployments."
    )
    add_p(
        "Yoon, Lee, Lee, and Han (2024) [15] proposed 'COMPACT: Compressing Retrieved Documents Actively for Question Answering' (arXiv:2410.02747). They introduced active "
        "evidence compression to strip irrelevant conversational tokens from retrieved contexts, showing that concise, high-density evidence prompts improve generation latency "
        "and prevent LLM attention drift on edge devices."
    )
    add_p(
        "Le (2024) [8] contributed 'Cross-Lingual Summarization with Pseudo-Label Regularization' at ACL 2024. The study showed that parameter-efficient adapters fine-tuned on "
        "pseudo-labeled cross-lingual pairs achieve higher factual consistency than unconstrained multi-task prompting."
    )
    add_p(
        "Rastogi, Gupta, and Sharma (2024) [10] published 'An Empirical Analysis of Leveraging Knowledge for Low-Resource Task-Oriented Semantic Parsing' at EMNLP 2024. They "
        "highlighted the fragility of semantic parsing when dealing with complex statutory terminology across Indic languages."
    )
    add_p(
        "Kumar, Singh, and Bansal (2024) [11] presented 'A Multimodal Retrieval-Augmented Generation System for Intelligent Document Intelligence' at AAAI 2024. They integrated "
        "multi-modal document tokenization with generative QA, emphasizing the necessity of preserving page-level coordinate anchors for legal verification."
    )
    add_p(
        "Zhang, Wang, and Derr (2024) [7] authored 'A Survey of Large Language Models on Generative Graph Analytics: Query, Learning, and Applications' in IEEE TKDE 2024, "
        "analyzing how structured knowledge representations can constrain generative output spaces."
    )
    add_p(
        "ElSaify, Hassan, and Eltoweissy (2024) [19] investigated 'Adversarial and Multilingual Threats in Retrieval-Augmented Generation' at ACM CCS 2024. They proved that "
        "unfiltered RAG systems are highly susceptible to out-of-domain prompt injection and false-positive evidence acceptance, advocating for mathematical abstention gates "
        "at the retrieval interface."
    )
    add_p(
        "Ahmed, Khan, and Islam (2024) [20] presented 'Context-Aware Multilingual Travel Assistant Using Retrieval-Augmented Generation' at IEEE Big Data 2024, demonstrating "
        "that localized domain adaptation is essential for accurate cross-lingual information retrieval."
    )

    # 2.1.4: 2025
    add_subsec_heading("2.1.4", "Recent Innovations in Trust Boundaries and Edge Domain RAG (2025)")
    add_p(
        "Kalra, Chauhan, and Shah (2025) [6] published 'HyPA-RAG: A Hybrid Parameter Adaptive Retrieval-Augmented Generation System for AI Legal and Policy Applications' "
        "in IEEE Access. They tackled statutory policy comprehension by dynamically adapting retrieval parameters based on query complexity. They established that statutory "
        "compliance requires strict abstention when document similarity falls below calibrated confidence thresholds, directly inspiring our trust gate formulation."
    )
    add_p(
        "Lahiani, Benmeriem, and Guessoum (2025) [16] contributed 'An Intelligent Multilingual Legal Chatbot for Algerian Law Based on RAG and LLMs' at the International Conference "
        "on Artificial Intelligence and Law (ICAIL 2025). They addressed bilingual legal ambiguity in French and Arabic, emphasizing that citation tracking down to specific statutory "
        "articles is non-negotiable for public trust."
    )
    add_p(
        "Al Masud, Roy, and Ahmed (2025) [17] published 'Quality-Based Evidence Summarization for Retrieval-Augmented Generation (RAG)' in Information Processing & Management. "
        "They demonstrated that pre-filtering and abstractively compressing retrieved chunks before passing them to the generator reduces token latency by 45% while preserving critical factual entities."
    )
    add_p(
        "Medina, Gomez, and Morales (2025) [18] presented 'EduX-RAG: Retrieval Augmented Generation Framework for Cross-Lingual Educational Chatbots' in IEEE Transactions on "
        "Learning Technologies. They deployed lightweight RAG pipelines in resource-poor classrooms, demonstrating that small quantised language models running locally can "
        "match cloud APIs in specialized domains when paired with focused retrieval."
    )

    # 2.2: Literature Comparison Matrix Table
    add_sec_heading("2.2", "Comprehensive Literature Comparison Matrix")
    add_p(
        "Table 2.1 provides a structured, comparative analysis of the 20 surveyed papers, summarizing their primary focus, underlying methodologies, target language "
        "settings, reported metrics, and fundamental operational limitations that GovDoc-RAG directly overcomes."
    )

    lit_cols = [1.2, 0.6, 1.4, 1.4, 1.4, 0.8]
    lit_hdrs = ["Author & Year", "Year", "Core Focus & Domain", "Methodology", "Key Findings & Limits", "Edge Fit"]
    lit_data = [
        ["Dabre et al. [1]", "2022", "Indic NLG / Pre-training", "mBART + ALBERT parameter sharing", "244M params; vocabulary fragmented without fine-tuning", "High"],
        ["Huang et al. [9]", "2022", "Document AI / Layout", "Multimodal Text + Vision masking", "State-of-the-art layout; high GPU memory footprint", "Low"],
        ["Nie et al. [3]", "2023", "Cross-Lingual RAG", "Cross-lingual prompt bi-encoders", "Query-doc script gap hurts zero-shot recall", "Medium"],
        ["Panda et al. [4]", "2023", "Low-Resource Indic NLP", "Cross-modal acoustic/phonetic aug", "Transliteration expands vocabulary overlap", "High"],
        ["Chen et al. [2]", "2024", "Universal Embeddings", "BGE-M3 multi-granularity dense", "1024-dim dense vectors; 100+ languages supported", "High"],
        ["Urlana et al. [14]", "2024", "Indic Summarisation", "PMIndiaSum 30k PIB corpus", "Zero-shot LLMs fail; supervised adapters required", "High"],
        ["Ranaldi et al. [12]", "2024", "Multilingual QA RAG", "Dense multilingual bi-encoders", "Code-mixing and romanised queries cause recall collapse", "Medium"],
        ["Chirkova et al. [13]", "2024", "Multilingual Retrieval", "Original-language evidence retrieval", "Retrieving English evidence avoids translation drift", "High"],
        ["Li et al. [5]", "2024", "Hallucination Survey", "Post-generation fact verification", "Entity/numeric checking yields highest safety", "High"],
        ["Yoon et al. [15]", "2024", "Evidence Compression", "COMPACT active token pruning", "Pruning irrelevant tokens improves edge generation", "High"],
        ["Le [8]", "2024", "Cross-Lingual Summary", "Pseudo-label regularization", "Adapters outperform unconstrained prompting", "High"],
        ["Rastogi et al. [10]", "2024", "Indic Semantic Parsing", "Knowledge-guided task parsing", "Out-of-domain technical terms degrade parsing", "Medium"],
        ["Kumar et al. [11]", "2024", "Doc Intelligence RAG", "Multimodal coordinate tokenization", "Page coordinate anchors ensure statutory compliance", "Medium"],
        ["Zhang et al. [7]", "2024", "Generative Graphs", "Graph-augmented grounding", "Graph constraints reduce hallucinations", "Low"],
        ["ElSaify et al. [19]", "2024", "RAG Security & Threats", "Adversarial prompt injection defense", "Mathematical confidence gates prevent false acceptance", "High"],
        ["Ahmed et al. [20]", "2024", "Localized RAG", "Domain-specific multilingual RAG", "Domain adaptation critical for regional dialects", "High"],
        ["Kalra et al. [6]", "2025", "Legal Policy RAG", "HyPA-RAG adaptive retrieval", "Strict confidence abstention required for statutory QA", "High"],
        ["Lahiani et al. [16]", "2025", "Bilingual Legal RAG", "RAG for Algerian legal statutes", "Exact statutory citation tracking required", "Medium"],
        ["Al Masud et al. [17]", "2025", "Quality Evidence RAG", "Quality-based evidence filtering", "Pre-filtering chunks reduces latency by 45%", "High"],
        ["Medina et al. [18]", "2025", "Edge Educational RAG", "Lightweight quantized SLMs", "Quantized SLMs match cloud APIs on specialized tasks", "High"],
    ]
    lit_aligns = [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER]
    
    t_lit = doc.add_table(rows=len(lit_data) + 1, cols=6)
    style_academic_table(t_lit, lit_cols, lit_hdrs, lit_data, lit_aligns, caption="Table 2.1: Comprehensive Literature Survey and Comparative Matrix of Prior Art.")

    # 2.3: Research Gap
    add_sec_heading("2.3", "Identified Research Gaps")
    add_p("A rigorous synthesis of the literature reveals four critical unresolved research gaps in existing document intelligence systems:")
    gaps = [
        "1. The Edge Compute vs. Model Size Paradox: State-of-the-art Document AI frameworks (such as LayoutLMv3 and proprietary cloud LLMs) demand substantial GPU memory (>16 GB VRAM) and continuous internet connectivity. Conversely, local public administration offices operate on commodity laptops or desktops with modest edge GPUs (<= 4 GB VRAM). Existing literature lacks an integrated, mutually exclusive memory architecture capable of running multilingual embeddings, generation, and summarisation concurrently under a strict 4.0 GB budget ceiling.",
        "2. The Script and Transliteration Divide: While BGE-M3 provides strong multi-lingual representations for formal native scripts (Devanagari, Kannada), its recall collapses dramatically on romanised, colloquial, or code-mixed dialects (Hinglish, Kanglish). Most Indian citizens type search queries using Roman keyboards. Existing RAG frameworks lack an automated dual-view bridging layer that reconciles phonological transliterations with native semantic indices without inducing double-retrieval overhead.",
        "3. Absence of Calibrated Abstention Guards: Standard enterprise RAG systems blindly pass retrieved evidence to generative models regardless of relevance. In statutory domains, this leads to hallucinated legal guidance when queries are near-misses or entirely out-of-domain. Prior art emphasizes the need for abstention but fails to provide mathematically calibrated, empirical decision boundaries on dense cosine metrics.",
        "4. Unverifiable Hallucination Vulnerability: Small Language Models (< 7B parameters) running at low-bit quantisation are prone to subtle numerical and temporal hallucinations (e.g., misstating an application deadline from 15th August to 25th August). Current open-source edge RAG systems lack deterministic, entity-level factual verifiers that cross-reference extracted entities directly against retrieved chunks before surfacing answers to users."
    ]
    for g in gaps:
        add_p(g, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=2, space_after=4, size=11.5)

    # 2.4: Tools and Technologies
    add_sec_heading("2.4", "Supporting Tools and Technologies")
    add_p(
        "To address these research gaps, GovDoc-RAG integrates a cohesive ecosystem of open-source models, libraries, and hardware execution runtimes:"
    )
    tools = [
        ("ai4bharat/IndicBART & LoRA", "A 244M parameter sequence-to-sequence model pre-trained on 11 Indic languages and English. We augment IndicBART with Low-Rank Adaptation (LoRA, rank r=16, alpha=32) applied to the self-attention projection layers (q_proj, v_proj), fine-tuning only 1.2M trainable parameters while freezing the backbone."),
        ("BAAI/bge-m3", "A 568M parameter universal embedding model optimized for 100+ languages. Outputs 1024-dimensional normalized dense vectors. In GovDoc-RAG, it indexes section chunks into ChromaDB and maps multilingual queries into a shared semantic manifold."),
        ("IndicXlit & IndicTrans2", "Open-source state-of-the-art neural engines developed by AI4Bharat. IndicXlit provides deterministic Roman-to-Indic phonological transliteration, while IndicTrans2 (212M parameter distilled variant) translates abstractive digests into regional target languages."),
        ("Qwen 3 4B Instruct (GGUF Q4_K_M)", "An ultra-efficient 4.0-billion parameter Small Language Model developed by Alibaba Cloud. Quantised to 4-bit medium precision via llama.cpp/Ollama, executing in 2.2 GB VRAM with sub-1.5 second token generation latency."),
        ("ChromaDB & PyMuPDF (fitz)", "PyMuPDF parses native searchable PDFs, extracting layout-aware bounding boxes and font hierarchies. ChromaDB acts as the local, persistent vector database utilizing cosine similarity metrics."),
        ("Streamlit & PyTorch CUDA", "Streamlit provides an intuitive, responsive web user interface with dedicated tabs for Document Ingestion, Semantic Question Answering, and Executive Summarisation. The backend runs on PyTorch 2.6 with CUDA 12.4 acceleration.")
    ]
    for t_title, t_desc in tools:
        p_tool = add_p(f"• {t_title}: ", align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=2, space_after=4, bold=True, size=11.5)
        p_tool.runs[0].bold = True
        r_t = p_tool.add_run(t_desc)
        r_t.font.name = 'Times New Roman'
        r_t.font.size = Pt(11.5)

    print("Chapter 2 completed. Loading Chapters 3-8...")

    helpers = {
        "add_p": add_p,
        "add_chap_heading": add_chap_heading,
        "add_sec_heading": add_sec_heading,
        "add_subsec_heading": add_subsec_heading,
        "add_subsubsec_heading": add_subsubsec_heading,
        "add_fig_image": add_fig_image,
        "style_academic_table": style_academic_table
    }

    import report_chapters_3_4 as c34
    import report_chapters_5_8 as c58

    c34.build_chapter_3(doc, helpers)
    c34.build_chapter_4(doc, helpers)
    c58.build_chapters_5_to_8(doc, helpers)

    output_path = "GovDoc_RAG_RVU_Project_Report.docx"
    doc.save(output_path)
    print(f"Full comprehensive report successfully saved to {output_path}!")
    return doc

if __name__ == "__main__":
    generate_full_report()
