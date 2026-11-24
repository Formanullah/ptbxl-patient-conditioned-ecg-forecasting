"""Minimal, dependency-free .pptx writer (Office Open XML): text boxes with bullets,
pictures, tables, simple shapes and speaker notes, 16:9. Used by build_presentation.py."""

from __future__ import annotations

import re
import struct
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

EMU = 914400
NS = ('xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
      'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
      'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"')  # fmt: skip
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
CT = "application/vnd.openxmlformats-officedocument.presentationml"
FONT = "Calibri"
GRP = ('<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm>'
       '<a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>')  # fmt: skip


def emu(v: float) -> int:
    return int(round(v * EMU))


def png_size(path: Path) -> tuple[int, int]:
    with open(path, "rb") as f:
        head = f.read(24)
    return struct.unpack(">II", head[16:24])


def runs(text: str, size: float, color: str, bold: bool = False, italic: bool = False) -> str:
    """Text with **bold** segments -> a:r elements."""
    out = []
    for i, part in enumerate(re.split(r"\*\*", text)):
        if not part:
            continue
        b = bold or i % 2 == 1
        out.append(f'<a:r><a:rPr lang="en-US" sz="{int(size * 100)}" b="{int(b)}" i="{int(italic)}" dirty="0">'
                   f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill><a:latin typeface="{FONT}"/>'
                   f'<a:cs typeface="{FONT}"/></a:rPr><a:t>{escape(part)}</a:t></a:r>')  # fmt: skip
    return "".join(out) or f'<a:endParaRPr lang="en-US" sz="{int(size * 100)}"/>'


def para(text: str, size: float, color: str, level: int = 0, bullet: bool = True, bold: bool = False,
         align: str = "l", space_before: float = 4, italic: bool = False) -> str:  # fmt: skip
    if bullet:
        mar, ind = emu(0.22 + 0.25 * level), -emu(0.2)
        ch = "•" if level == 0 else "–"
        bu = f'<a:buClr><a:srgbClr val="{color}"/></a:buClr><a:buFont typeface="Arial"/><a:buChar char="{ch}"/>'
    else:
        mar, ind, bu = 0, 0, "<a:buNone/>"
    return (f'<a:p><a:pPr marL="{mar}" indent="{ind}" algn="{align}"><a:spcBef><a:spcPts val="{int(space_before * 100)}"/>'
            f"</a:spcBef>{bu}</a:pPr>{runs(text, size, color, bold, italic)}</a:p>")  # fmt: skip


class Slide:
    def __init__(self, deck: Deck):
        self.deck, self.shapes, self.images, self.notes = deck, [], [], ""
        self._id = 1

    def _nid(self) -> int:
        self._id += 1
        return self._id

    def text(self, x, y, w, h, paras: list, size=16, color="222222", bullet=True, fill=None, line=None,
             anchor="t", bold=False, align="l", inset=0.08, autofit=True):  # fmt: skip
        """paras: strings, or (text, level) tuples; '' gives an empty line."""
        ps = []
        for p in paras:
            t, lvl = (p, 0) if isinstance(p, str) else p
            ps.append(para(t, size - 1.5 * lvl, color, lvl, bullet and bool(t), bold, align))
        fill_x = f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>' if fill else "<a:noFill/>"
        line_x = f'<a:ln w="12700"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>' if line else "<a:ln><a:noFill/></a:ln>"
        i = self._nid()
        ins = emu(inset)
        fit = "<a:normAutofit/>" if autofit else ""
        self.shapes.append(
            f'<p:sp><p:nvSpPr><p:cNvPr id="{i}" name="Text {i}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>'
            f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>{fill_x}{line_x}</p:spPr>'
            f'<p:txBody><a:bodyPr wrap="square" lIns="{ins}" tIns="{ins}" rIns="{ins}" bIns="{ins}" anchor="{anchor}">{fit}'
            f'</a:bodyPr><a:lstStyle/>{"".join(ps)}</p:txBody></p:sp>')  # fmt: skip

    def shape(self, x, y, w, h, prst="rect", fill="1F2A44", text="", size=12, color="FFFFFF", bold=True):
        i = self._nid()
        body = (f'<p:txBody><a:bodyPr wrap="square" lIns="45720" tIns="45720" rIns="45720" bIns="45720" anchor="ctr">'
                f'<a:normAutofit/></a:bodyPr><a:lstStyle/>'
                + "".join(para(t, size, color, bullet=False, bold=bold, align="ctr", space_before=0) for t in text.split("\n"))
                + "</p:txBody>")  # fmt: skip
        self.shapes.append(
            f'<p:sp><p:nvSpPr><p:cNvPr id="{i}" name="Shape {i}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>'
            f'<a:prstGeom prst="{prst}"><a:avLst/></a:prstGeom><a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>'
            f'<a:ln><a:noFill/></a:ln></p:spPr>{body}</p:sp>')  # fmt: skip

    def image(self, path, x, y, w, h, align="ctr"):
        """Fit the picture inside the box, keeping its aspect ratio."""
        path = Path(path)
        pw, ph = png_size(path)
        s = min(w / pw, h / ph)
        iw, ih = pw * s, ph * s
        ox = x + (w - iw) / 2 if align == "ctr" else x
        rid = f"rId{10 + len(self.images)}"
        self.images.append((rid, self.deck.media(path)))
        i = self._nid()
        self.shapes.append(
            f'<p:pic><p:nvPicPr><p:cNvPr id="{i}" name="Picture {i}" descr="{escape(path.stem)}"/><p:cNvPicPr>'
            f'<a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr><p:blipFill><a:blip r:embed="{rid}"/>'
            f'<a:stretch><a:fillRect/></a:stretch></p:blipFill><p:spPr><a:xfrm><a:off x="{emu(ox)}" y="{emu(y)}"/>'
            f'<a:ext cx="{emu(iw)}" cy="{emu(ih)}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>')  # fmt: skip
        return ih

    def table(self, x, y, widths, rows, size=11, row_h=0.3, highlight=(), header_fill="1F2A44", col_align=None):
        def cell(t, fill, color, bold, align):
            ln = "".join(f'<a:{s} w="6350"><a:solidFill><a:srgbClr val="D5D8DC"/></a:solidFill></a:{s}>'
                         for s in ("lnL", "lnR", "lnT", "lnB"))  # fmt: skip
            return (f'<a:tc><a:txBody><a:bodyPr/><a:lstStyle/>{para(t, size, color, bullet=False, bold=bold, align=align, space_before=0)}'
                    f'</a:txBody><a:tcPr marL="54864" marR="54864" marT="27432" marB="27432" anchor="ctr">{ln}'
                    f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill></a:tcPr></a:tc>')  # fmt: skip

        trs = []
        for r, row in enumerate(rows):
            cells = []
            for c, t in enumerate(row):
                align = (col_align[c] if col_align else ("l" if c == 0 else "ctr"))
                if r == 0:
                    cells.append(cell(t, header_fill, "FFFFFF", True, align))
                elif r in highlight:
                    cells.append(cell(t, "FADBD8", "222222", True, align))
                else:
                    cells.append(cell(t, "F4F6F7" if r % 2 == 0 else "FFFFFF", "222222", False, align))
            trs.append(f'<a:tr h="{emu(row_h)}">{"".join(cells)}</a:tr>')
        grid = "".join(f'<a:gridCol w="{emu(w)}"/>' for w in widths)
        i = self._nid()
        self.shapes.append(
            f'<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="{i}" name="Table {i}"/><p:cNvGraphicFramePr>'
            f'<a:graphicFrameLocks noGrp="1"/></p:cNvGraphicFramePr><p:nvPr/></p:nvGraphicFramePr>'
            f'<p:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(sum(widths))}" cy="{emu(row_h * len(rows))}"/></p:xfrm>'
            f'<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/table"><a:tbl>'
            f'<a:tblPr firstRow="1" bandRow="1"/><a:tblGrid>{grid}</a:tblGrid>{"".join(trs)}</a:tbl></a:graphicData>'
            f'</a:graphic></p:graphicFrame>')  # fmt: skip

    def xml(self) -> str:
        return (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sld {NS}><p:cSld><p:spTree>{GRP}'
                f'{"".join(self.shapes)}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>')

    def notes_xml(self) -> str:
        ps = "".join(para(t, 12, "000000", bullet=False, space_before=3 if t else 0) for t in self.notes.strip().split("\n"))
        return (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:notes {NS}><p:cSld><p:spTree>{GRP}'
                '<p:sp><p:nvSpPr><p:cNvPr id="2" name="Slide Image Placeholder 1"/><p:cNvSpPr><a:spLocks noGrp="1" '
                'noRot="1" noChangeAspect="1"/></p:cNvSpPr><p:nvPr><p:ph type="sldImg"/></p:nvPr></p:nvSpPr><p:spPr/></p:sp>'
                '<p:sp><p:nvSpPr><p:cNvPr id="3" name="Notes Placeholder 2"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr>'
                f'<p:nvPr><p:ph type="body" idx="1"/></p:nvPr></p:nvSpPr><p:spPr/><p:txBody><a:bodyPr/><a:lstStyle/>{ps}'
                '</p:txBody></p:sp></p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:notes>')  # fmt: skip


THEME = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Study"><a:themeElements>
<a:clrScheme name="Study"><a:dk1><a:srgbClr val="222222"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>
<a:dk2><a:srgbClr val="1F2A44"/></a:dk2><a:lt2><a:srgbClr val="F4F6F7"/></a:lt2><a:accent1><a:srgbClr val="1F2A44"/></a:accent1>
<a:accent2><a:srgbClr val="C0392B"/></a:accent2><a:accent3><a:srgbClr val="27AE60"/></a:accent3><a:accent4><a:srgbClr val="16A085"/></a:accent4>
<a:accent5><a:srgbClr val="E67E22"/></a:accent5><a:accent6><a:srgbClr val="95A5A6"/></a:accent6><a:hlink><a:srgbClr val="2980B9"/></a:hlink>
<a:folHlink><a:srgbClr val="8E44AD"/></a:folHlink></a:clrScheme>
<a:fontScheme name="Study"><a:majorFont><a:latin typeface="Calibri"/><a:ea typeface=""/><a:cs typeface=""/></a:majorFont>
<a:minorFont><a:latin typeface="Calibri"/><a:ea typeface=""/><a:cs typeface=""/></a:minorFont></a:fontScheme>
<a:fmtScheme name="Study"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
<a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="6350"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln>
<a:ln w="12700"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln><a:ln w="19050"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst>
<a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>
<a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
<a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme></a:themeElements><a:objectDefaults/><a:extraClrSchemeLst/></a:theme>"""

CLRMAP = ('<p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" '
          'accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>')  # fmt: skip


class Deck:
    def __init__(self, title: str, author: str):
        self.title, self.author, self.slides, self._media = title, author, [], {}

    def add(self) -> Slide:
        s = Slide(self)
        self.slides.append(s)
        return s

    def media(self, path: Path) -> str:
        key = str(path.resolve())
        if key not in self._media:
            self._media[key] = (f"image{len(self._media) + 1}.png", path)
        return self._media[key][0]

    def save(self, out: Path) -> None:
        n = len(self.slides)
        x = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        files: dict[str, str | bytes] = {}
        ov = [("/ppt/presentation.xml", f"{CT}.presentation.main+xml"), ("/ppt/slideMasters/slideMaster1.xml", f"{CT}.slideMaster+xml"),
              ("/ppt/slideLayouts/slideLayout1.xml", f"{CT}.slideLayout+xml"), ("/ppt/notesMasters/notesMaster1.xml", f"{CT}.notesMaster+xml"),
              ("/ppt/theme/theme1.xml", "application/vnd.openxmlformats-officedocument.theme+xml"),
              ("/ppt/theme/theme2.xml", "application/vnd.openxmlformats-officedocument.theme+xml"),
              ("/ppt/presProps.xml", f"{CT}.presProps+xml"), ("/ppt/viewProps.xml", f"{CT}.viewProps+xml"),
              ("/ppt/tableStyles.xml", f"{CT}.tableStyles+xml"),
              ("/docProps/core.xml", "application/vnd.openxmlformats-package.core-properties+xml"),
              ("/docProps/app.xml", "application/vnd.openxmlformats-officedocument.extended-properties+xml")]  # fmt: skip
        ov += [(f"/ppt/slides/slide{i}.xml", f"{CT}.slide+xml") for i in range(1, n + 1)]
        ov += [(f"/ppt/notesSlides/notesSlide{i}.xml", f"{CT}.notesSlide+xml") for i in range(1, n + 1)]
        files["[Content_Types].xml"] = (
            f'{x}<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/>'
            + "".join(f'<Override PartName="{p}" ContentType="{c}"/>' for p, c in ov) + "</Types>")  # fmt: skip

        def rels(items):
            return (f'{x}<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                    + "".join(f'<Relationship Id="{i}" Type="{t}" Target="{g}"/>' for i, t, g in items) + "</Relationships>")

        files["_rels/.rels"] = rels([("rId1", f"{REL}/officeDocument", "ppt/presentation.xml"),
                                     ("rId2", "http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties", "docProps/core.xml"),
                                     ("rId3", f"{REL}/extended-properties", "docProps/app.xml")])  # fmt: skip
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        files["docProps/core.xml"] = (
            f'{x}<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            f"<dc:title>{escape(self.title)}</dc:title><dc:creator>{escape(self.author)}</dc:creator>"
            f'<dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created></cp:coreProperties>')  # fmt: skip
        files["docProps/app.xml"] = (f'{x}<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">'
                                     f"<Application>Microsoft Office PowerPoint</Application><Slides>{n}</Slides><Notes>{n}</Notes></Properties>")
        pres_rels = [("rId1", f"{REL}/slideMaster", "slideMasters/slideMaster1.xml"), ("rId2", f"{REL}/notesMaster", "notesMasters/notesMaster1.xml"),
                     ("rId3", f"{REL}/theme", "theme/theme1.xml"), ("rId4", f"{REL}/presProps", "presProps.xml"),
                     ("rId5", f"{REL}/viewProps", "viewProps.xml"), ("rId6", f"{REL}/tableStyles", "tableStyles.xml")]  # fmt: skip
        pres_rels += [(f"rId{100 + i}", f"{REL}/slide", f"slides/slide{i}.xml") for i in range(1, n + 1)]
        files["ppt/_rels/presentation.xml.rels"] = rels(pres_rels)
        files["ppt/presentation.xml"] = (
            f'{x}<p:presentation {NS} saveSubsetFonts="1"><p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/>'
            '</p:sldMasterIdLst><p:notesMasterIdLst><p:notesMasterId r:id="rId2"/></p:notesMasterIdLst><p:sldIdLst>'
            + "".join(f'<p:sldId id="{255 + i}" r:id="rId{100 + i}"/>' for i in range(1, n + 1))
            + '</p:sldIdLst><p:sldSz cx="12192000" cy="6858000"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>')  # fmt: skip
        files["ppt/presProps.xml"] = f"{x}<p:presentationPr {NS}/>"
        files["ppt/viewProps.xml"] = f"{x}<p:viewPr {NS}/>"
        files["ppt/tableStyles.xml"] = (f'{x}<a:tblStyleLst xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
                                        'def="{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}"/>')
        files["ppt/theme/theme1.xml"] = THEME
        files["ppt/theme/theme2.xml"] = THEME
        bg = '<p:bg><p:bgRef idx="1001"><a:schemeClr val="bg1"/></p:bgRef></p:bg>'
        files["ppt/slideMasters/slideMaster1.xml"] = (
            f'{x}<p:sldMaster {NS}><p:cSld>{bg}<p:spTree>{GRP}</p:spTree></p:cSld>{CLRMAP}'
            '<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst><p:txStyles>'
            '<p:titleStyle><a:lvl1pPr><a:defRPr sz="3200"/></a:lvl1pPr></p:titleStyle><p:bodyStyle><a:lvl1pPr>'
            '<a:defRPr sz="1800"/></a:lvl1pPr></p:bodyStyle><p:otherStyle><a:lvl1pPr><a:defRPr sz="1800"/></a:lvl1pPr>'
            "</p:otherStyle></p:txStyles></p:sldMaster>")  # fmt: skip
        files["ppt/slideMasters/_rels/slideMaster1.xml.rels"] = rels([("rId1", f"{REL}/slideLayout", "../slideLayouts/slideLayout1.xml"),
                                                                      ("rId2", f"{REL}/theme", "../theme/theme1.xml")])  # fmt: skip
        files["ppt/slideLayouts/slideLayout1.xml"] = (f'{x}<p:sldLayout {NS} type="blank" preserve="1"><p:cSld name="Blank">'
                                                      f"<p:spTree>{GRP}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>")
        files["ppt/slideLayouts/_rels/slideLayout1.xml.rels"] = rels([("rId1", f"{REL}/slideMaster", "../slideMasters/slideMaster1.xml")])

        def ph(i, name, typ, idx, off, ext, body=""):
            return (f'<p:sp><p:nvSpPr><p:cNvPr id="{i}" name="{name}"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr>'
                    f'<p:nvPr><p:ph type="{typ}"{idx}/></p:nvPr></p:nvSpPr><p:spPr><a:xfrm><a:off x="{off[0]}" y="{off[1]}"/>'
                    f'<a:ext cx="{ext[0]}" cy="{ext[1]}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr>{body}</p:sp>')

        nbody = '<p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:endParaRPr lang="en-US"/></a:p></p:txBody>'
        files["ppt/notesMasters/notesMaster1.xml"] = (
            f"{x}<p:notesMaster {NS}><p:cSld>{bg}<p:spTree>{GRP}"
            + ph(2, "Slide Image Placeholder 1", "sldImg", "", (381000, 685800), (6096000, 3429000))
            + ph(3, "Notes Placeholder 2", "body", ' idx="1"', (685800, 4343400), (5486400, 4114800), nbody)
            + f'</p:spTree></p:cSld>{CLRMAP}<p:notesStyle><a:lvl1pPr><a:defRPr sz="1200"/></a:lvl1pPr></p:notesStyle></p:notesMaster>')  # fmt: skip
        files["ppt/notesMasters/_rels/notesMaster1.xml.rels"] = rels([("rId1", f"{REL}/theme", "../theme/theme2.xml")])
        for i, s in enumerate(self.slides, 1):
            files[f"ppt/slides/slide{i}.xml"] = s.xml()
            items = [("rId1", f"{REL}/slideLayout", "../slideLayouts/slideLayout1.xml"),
                     ("rId2", f"{REL}/notesSlide", f"../notesSlides/notesSlide{i}.xml")]  # fmt: skip
            items += [(rid, f"{REL}/image", f"../media/{m}") for rid, m in s.images]
            files[f"ppt/slides/_rels/slide{i}.xml.rels"] = rels(items)
            files[f"ppt/notesSlides/notesSlide{i}.xml"] = s.notes_xml()
            files[f"ppt/notesSlides/_rels/notesSlide{i}.xml.rels"] = rels(
                [("rId1", f"{REL}/notesMaster", "../notesMasters/notesMaster1.xml"), ("rId2", f"{REL}/slide", f"../slides/slide{i}.xml")])
        for name, path in self._media.values():
            files[f"ppt/media/{name}"] = path.read_bytes()
        out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("[Content_Types].xml", files.pop("[Content_Types].xml"))
            for k, v in files.items():
                z.writestr(k, v)
