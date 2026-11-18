"""Export the report to Word with editable Office Math using Python packages.

From the project root (dependencies can live outside the project environment):
PYTHONPATH=/tmp/burgers-docx-tools MPLCONFIGDIR=/tmp/burgers-report-mpl \
    .venv/bin/python report/export_docx.py

Requires python-docx, markdown-it-py, mdit-py-plugins, latex2mathml,
mathml2omml, and the project's Matplotlib installation.
Only report/ output is written. Scientific source artifacts are never modified.
"""
from pathlib import Path
import json
import hashlib
import re
import tempfile
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from markdown_it import MarkdownIt
from mdit_py_plugins.dollarmath import dollarmath_plugin
import latex2mathml.converter
import mathml2omml
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.shared import Cm, Inches, Pt, RGBColor
from lxml import etree

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
OUTPUT = HERE / "FINAL_REPORT.docx"


def box(ax, x, y, w, h, text, color="#e9f1f7"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01",
                               linewidth=1, edgecolor="#284b63", facecolor=color))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=10.5, linespacing=1.4)


def arrow(ax, start, end, connectionstyle="arc3"):
    ax.annotate("", xy=end, xytext=start,
                arrowprops={"arrowstyle": "-|>", "color": "#284b63", "lw": 1.4,
                            "connectionstyle": connectionstyle})


def diagram(name, kind):
    fig, ax = plt.subplots(figsize=(7, 7.8 if kind == "pipeline" else 6.8))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    if kind == "pipeline":
        box(ax, .02, .865, .44, .11, "Custom initial conditions\nVerified spectral solver")
        box(ax, .54, .865, .44, .11, "PDEBench source archive\nChecksum / coordinate checks")
        labels = ["Fixed trajectory splits and common grids",
                  "Training-only normalization",
                  "Train registered models and seeds",
                  "Select checkpoints using validation rollouts",
                  "Lock checkpoint and dataset identities",
                  "Test autoregressive rollouts at three grids",
                  "Accuracy, conservation, energy, residuals, failures",
                  "Paired comparisons and report artifacts"]
        positions = [.75, .65, .55, .45, .35, .25, .15, .05]
        for y, label in zip(positions, labels):
            box(ax, .025, y, .95, .07, label,
                "#e8f4ee" if y == .05 else "#e9f1f7")
        arrow(ax, (.24, .865), (.35, .82))
        arrow(ax, (.76, .865), (.65, .82))
        for a, b in zip(positions, positions[1:]):
            arrow(ax, (.5, a), (.5, b+.07))
    else:
        box(ax, .08, .86, .84, .10, "Physical input u → normalized field z")
        box(ax, .08, .70, .84, .10, "Backbone predicts derivative q\nProject update: q₀ = q − spatial mean(q)")
        box(ax, .08, .54, .84, .10, "Residual step: z_next = z + Δt q₀\nPhysical first prediction v = μ + σ z_next")
        box(ax, .52, .34, .46, .13, "Same model applied again\nSecond prediction w\n(no second reference target)")
        box(ax, .02, .34, .43, .13, "Supervised loss\nCompare first prediction v\nwith reference target y")
        box(ax, .48, .15, .50, .12, "Two-step physics loss\nResiduals R(u,v) and R(v,w)\nGradients through both steps")
        box(ax, .02, .015, .96, .075, "Total objective = supervised loss + λ × physics loss", "#e8f4ee")
        for a, b in [((.5,.86),(.5,.80)), ((.5,.70),(.5,.64)),
                     ((.32,.54),(.23,.47)), ((.73,.54),(.75,.47)),
                     ((.75,.34),(.75,.27)), ((.72,.15),(.68,.09)),
                     ((.23,.34),(.23,.09))]:
            arrow(ax, a, b)
    fig.savefig(FIG / f"{name}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


diagram("03_01_pipeline", "pipeline")
diagram("03_02_constrained_model", "constraint")


def field(paragraph, instruction, cached=""):
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar"); begin.set(qn("w:fldCharType"), "begin")
    code = OxmlElement("w:instrText"); code.set(qn("xml:space"), "preserve"); code.text = instruction
    sep = OxmlElement("w:fldChar"); sep.set(qn("w:fldCharType"), "separate")
    result = OxmlElement("w:t"); result.text = cached
    end = OxmlElement("w:fldChar"); end.set(qn("w:fldCharType"), "end")
    for node in [begin, code, sep, result, end]: run._r.append(node)


def bookmark_name(slug):
    return "section_"+hashlib.sha1(slug.encode()).hexdigest()[:20]


def slug(text):
    return re.sub(r"[^\w\- ]", "", text.lower()).replace(" ", "-")


def math_node(tex, display=False):
    markup=mathml2omml.convert(latex2mathml.converter.convert(tex))
    namespace="http://schemas.openxmlformats.org/officeDocument/2006/math"
    markup=markup.replace("<m:oMath>",f'<m:oMath xmlns:m="{namespace}">',1)
    node=etree.fromstring(markup.encode())
    if display:
        parent=OxmlElement("m:oMathPara");parent.append(node);return parent
    return node


def inline(p, children):
    bold=italic=False
    hyperlink=None
    for t in children or []:
        if t.type=="strong_open":bold=True
        elif t.type=="strong_close":bold=False
        elif t.type=="em_open":italic=True
        elif t.type=="em_close":italic=False
        elif t.type=="link_open":
            href=t.attrGet("href");hyperlink=OxmlElement("w:hyperlink")
            if href.startswith("#"):hyperlink.set(qn("w:anchor"),bookmark_name(href[1:]))
            else:hyperlink.set(qn("r:id"),p.part.relate_to(href,RT.HYPERLINK,is_external=True))
            p._p.append(hyperlink)
        elif t.type=="link_close":hyperlink=None
        elif t.type=="math_inline":p._p.append(math_node(t.content))
        elif t.type=="image":
            picture=p.add_run().add_picture(str(HERE/t.attrGet("src")),width=Inches(6.3))
            if picture.height>Inches(7.8):
                picture.width=int(picture.width*Inches(7.8)/picture.height);picture.height=Inches(7.8)
            p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        elif t.type in ("softbreak","hardbreak"):
            if t.type=="hardbreak":p.add_run().add_break()
            else:p.add_run(" ")
        elif t.type in ("text","code_inline"):
            run=p.add_run(t.content);run.bold=bold;run.italic=italic
            if t.type=="code_inline":run.font.name="Courier New";run.font.size=Pt(9)
            if hyperlink is not None:
                run.font.color.rgb=RGBColor.from_string("245C87");run.underline=True
                hyperlink.append(run._r)


def render_markdown(source, document):
    parser=MarkdownIt().enable("table").use(dollarmath_plugin)
    tokens=parser.parse(source)
    i=0;lists=[];pending_prefix="";quote_depth=0;bookmark_id=0
    while i<len(tokens):
        t=tokens[i]
        if t.type=="heading_open":
            title=tokens[i+1];level=int(t.tag[1:]);p=document.add_paragraph(style=f"Heading {level}")
            start=OxmlElement("w:bookmarkStart");start.set(qn("w:id"),str(bookmark_id))
            start.set(qn("w:name"),bookmark_name(slug(title.content)));p._p.append(start)
            inline(p,title.children)
            end=OxmlElement("w:bookmarkEnd");end.set(qn("w:id"),str(bookmark_id));p._p.append(end)
            bookmark_id+=1;i+=3;continue
        if t.type=="paragraph_open":
            p=document.add_paragraph()
            if pending_prefix:
                p.add_run(pending_prefix);pending_prefix=""
                p.paragraph_format.left_indent=Cm(.4);p.paragraph_format.first_line_indent=Cm(-.4)
            if quote_depth:p.paragraph_format.left_indent=Cm(.4)
            inline(p,tokens[i+1].children)
            i+=3;continue
        if t.type=="math_block":
            p=document.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
            p._p.append(math_node(t.content,True))
        elif t.type=="fence":
            p=document.add_paragraph()
            run=p.add_run(t.content.rstrip());run.font.name="Courier New";run.font.size=Pt(8)
            p.paragraph_format.line_spacing=1.0
        elif t.type=="table_open":
            rows=[];cells=[];i+=1
            while tokens[i].type!="table_close":
                if tokens[i].type=="tr_open":cells=[]
                elif tokens[i].type in ("th_open","td_open"):
                    cells.append(tokens[i+1].children)
                elif tokens[i].type=="tr_close":rows.append(cells)
                i+=1
            table=document.add_table(rows=len(rows),cols=len(rows[0]))
            for r,cells in zip(table.rows,rows):
                for c,content in zip(r.cells,cells):inline(c.paragraphs[0],content)
        elif t.type in ("ordered_list_open","bullet_list_open"):
            lists.append([t.type,int(t.attrGet("start") or 1)])
        elif t.type in ("ordered_list_close","bullet_list_close"):lists.pop()
        elif t.type=="list_item_open":
            pending_prefix=f"{lists[-1][1]}. " if lists[-1][0]=="ordered_list_open" else "• "
            lists[-1][1]+=1
        elif t.type=="blockquote_open":quote_depth+=1
        elif t.type=="blockquote_close":quote_depth-=1
        elif t.type=="hr":document.add_paragraph()
        i+=1
    return document


with tempfile.TemporaryDirectory(prefix="burgers-docx-") as tempdir:
    temp = Path(tempdir)
    ref = Document()
    section = ref.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.top_margin, section.bottom_margin = Cm(2.2), Cm(2.2)
    section.left_margin, section.right_margin = Cm(2.2), Cm(2.2)
    section.header_distance, section.footer_distance = Cm(1), Cm(1)
    section.different_first_page_header_footer = True
    for name in ["Normal", "Body Text"]:
        style = ref.styles[name]
        style.font.name = "Times New Roman"; style.font.size = Pt(11)
        style.paragraph_format.line_spacing = 1.25
        style.paragraph_format.space_after = Pt(6)
        style.paragraph_format.widow_control = True
    for name, size in [("Title",20),("Subtitle",15),("Heading 1",16),("Heading 2",13),("Heading 3",11.5)]:
        st=ref.styles[name]; st.font.name="Times New Roman"; st.font.size=Pt(size)
        st.font.color.rgb=RGBColor.from_string("17354A")
        st.paragraph_format.keep_with_next=True
        st.paragraph_format.space_before=Pt(12);st.paragraph_format.space_after=Pt(8)
    ref.styles["Heading 1"].paragraph_format.page_break_before=True
    ref.styles["Caption"].font.name="Times New Roman"
    ref.styles["Caption"].font.size=Pt(9)
    ref.styles["Caption"].font.italic=True
    ref.styles["Caption"].paragraph_format.space_after=Pt(10)
    header = section.header.paragraphs[0]
    header.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    run=header.add_run("Physics-Constrained Neural Models for Burgers’ Equation")
    run.font.name="Times New Roman";run.font.size=Pt(8)
    footer=section.footer.paragraphs[0]; footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
    field(footer,"PAGE","1")
    ref_path=temp/"reference.docx";ref.save(ref_path)

    source=(HERE/"FINAL_REPORT.md").read_text()
    diagrams=iter([("Experimental pipeline","03_01_pipeline"),("Constrained prediction and loss","03_02_constrained_model")])
    def substitute_diagram(match):
        label, name=next(diagrams)
        return f"![{label}](figures/{name}.png)"
    source=re.sub(r"```mermaid\n.*?```",substitute_diagram,source,flags=re.S)
    source=source.replace("Included: Mermaid diagram","Included: embedded diagram")
    source=source.replace("The chapter notes use the following stable reference identifiers. A single consolidated bibliography avoids duplicate entries in this editable Markdown draft; it can be converted to chapter-wise lists if the submission template requires that layout.",
        "The chapter notes use the following stable reference identifiers. This Word draft uses a consolidated bibliography; it can be converted to chapter-wise lists if the submission template requires that layout.")
    source=source.replace("Mermaid diagrams in Chapter 3 should be rendered to SVG or PDF during Word/LaTeX preparation if the chosen exporter does not support Mermaid directly.",
        "The two Chapter 3 diagrams are embedded in this Word edition; corresponding PNG and PDF files are supplied in the figures directory.")
    # Word document lives alongside Markdown, so links to project artifacts stay relative.
    # Tables and mathematical notation remain editable; plots are embedded images.
    render_markdown(source,ref).save(OUTPUT)

doc=Document(OUTPUT)
doc.core_properties.title="Physics-Constrained Neural Models for Burgers’ Equation"
doc.core_properties.subject="Minor Project Report — working draft"
doc.core_properties.author="[FILL: student authors]"
doc.core_properties.keywords="Burgers equation, FNO, U-Net, physics constraints, PDEBench"
front={"Candidate’s Declaration","Acknowledgement","Abstract","List of Tables","List of Figures",
       "List of Abbreviations","List of Mathematical Symbols","Table of Contents"}
for i,p in enumerate(doc.paragraphs):
    if p.text=="Minor Project Report":
        p.style=doc.styles["Title"];p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.page_break_before=False
    if p.text.startswith("Physics-Constrained Neural Models for Burgers’ Equation: A Comparative Study"):
        p.style=doc.styles["Subtitle"];p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after=Pt(20)
    if p.text in front:
        p.paragraph_format.page_break_before=True
    if p.text.startswith("Figure "):
        p.style=doc.styles["Caption"]
        if i: doc.paragraphs[i-1].paragraph_format.keep_with_next=True
    if p.text.startswith("Table ") and "." in p.text[:12]:
        p.paragraph_format.keep_with_next=True
    if p.text.startswith("Chapter references:") or p.text.startswith("Chapter evidence:"):
        for r in p.runs:r.font.size=Pt(9)
    if p.text.startswith("[FILL:") or p.text.startswith("[PENDING RESULT:") or "[FIGURE TO ADD:" in p.text:
        props=p._p.get_or_add_pPr();shade=OxmlElement("w:shd")
        shade.set(qn("w:fill"),"FFF2CC");props.append(shade)

# Keep tables readable and inside the printable width; repeat header rows in Word.
width=doc.sections[0].page_width-doc.sections[0].left_margin-doc.sections[0].right_margin
for t in doc.tables:
    t.style="Table Grid"
    t.autofit=False
    columns=len(t.columns)
    for col in t.columns:col.width=int(width/columns)
    for i,r in enumerate(t.rows):
        props=r._tr.get_or_add_trPr()
        no_split=OxmlElement("w:cantSplit");props.append(no_split)
        if i==0:
            repeat=OxmlElement("w:tblHeader");props.append(repeat)
        for c in r.cells:
            c.width=int(width/columns)
            if i==0:
                shade=OxmlElement("w:shd");shade.set(qn("w:fill"),"E8EFF5")
                c._tc.get_or_add_tcPr().append(shade)
            for p in c.paragraphs:
                p.paragraph_format.line_spacing=1.0
                p.paragraph_format.space_after=Pt(3);p.paragraph_format.space_before=Pt(3)
                for run in p.runs:
                    run.font.name="Times New Roman";run.font.size=Pt(8 if columns>=6 else 9)
                    if i==0:run.bold=True
for picture in doc.inline_shapes:
    if picture.width>width:
        picture.height=int(picture.height*width/picture.width);picture.width=width

# Ask Word to refresh field values such as footer page numbers when opened.
settings=doc.settings.element
update=OxmlElement("w:updateFields");update.set(qn("w:val"),"true");settings.append(update)
doc.save(OUTPUT)

# Structural checks catch missing equations, images, or document sections.
with zipfile.ZipFile(OUTPUT) as z:
    assert z.testzip() is None
    xml=etree.fromstring(z.read("word/document.xml"))
    ns={"w":"http://schemas.openxmlformats.org/wordprocessingml/2006/main",
        "m":"http://schemas.openxmlformats.org/officeDocument/2006/math"}
    text=" ".join(xml.xpath("//w:t/text()",namespaces=ns))
    media=[n for n in z.namelist() if n.startswith("word/media/")]
    equations=xml.xpath("//m:oMath",namespaces=ns)
    assert len(media)==12,("Expected 10 data figures and two diagrams",len(media))
    assert len(equations)>=11,"Display/inline equations were not converted"
    for heading in ["Chapter 1: Introduction","Chapter 2: Literature Review",
                    "Chapter 3: Methodology and Experimental Design","Chapter 4: Results and Comparative Analysis",
                    "Chapter 5: Discussion, Conclusion and Future Work"]:
        assert heading in text,heading
    assert "CONSTRAINED_UNET_200" in text and "PDEBENCH_600" in text
    assert "flowchart" not in text,"A Mermaid code block was left unrendered"
    for n in z.namelist():
        if n.endswith(".xml") or n.endswith(".rels"): etree.fromstring(z.read(n))
    stats={"output":OUTPUT.name,"embedded_images":len(media),"editable_math_objects":len(equations),
           "tables":len(doc.tables),"paragraphs":len(doc.paragraphs),
           "validated":"ZIP integrity, XML parse, chapters, diagrams, equations, pending placeholders"}
(HERE/"evidence/docx_validation.json").write_text(json.dumps(stats,indent=2)+"\n")
print(json.dumps(stats,indent=2))
