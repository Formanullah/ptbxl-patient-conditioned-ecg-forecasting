from __future__ import annotations

import html
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "presentations" / "ptbxl_ecg_his_project_03_updated.pptx"
OUTPUT = ROOT / "presentations" / "ptbxl_ecg_his_project_03_with_training_agentic_ai_final.pptx"
SOURCE_SLIDE = 10
NEW_SLIDE = 90

NEW_TEXT = [
    "Model training loop and checkpoint selection",
    "Training workflow",
    "10A",
    "Batch preparation",
    "Train-only scaling; move batch to device",
    "Forward pass",
    "Encoder + context produces [B, 12, 150]",
    "Forecast loss",
    "MSE across batch, lead and forecast time",
    "Contrastive branch",
    "0.1 × NT-Xent on two views; temperature 0.1",
    "Total objective",
    "L = forecast MSE + 0.1 Lcontrastive",
    "Backpropagation",
    "BPTT; clip global gradient norm at 1.0",
    "Validation",
    "Evaluation mode; no gradient updates; compute MSE",
    "Checkpoint rule",
    "Lowest validation MSE; patience 3; 5 epochs maximum",
    "Reload the best checkpoint. Invert scaling to mV and evaluate the locked test split once.",
]


def replace_slide_text(xml_bytes: bytes) -> bytes:
    source = xml_bytes.decode("utf-8")
    matches = list(re.finditer(r"<a:t>(.*?)</a:t>", source, flags=re.DOTALL))
    if len(matches) != len(NEW_TEXT):
        raise RuntimeError(f"Expected {len(NEW_TEXT)} text runs, found {len(matches)}")
    out = []
    cursor = 0
    for match, replacement in zip(matches, NEW_TEXT):
        out.append(source[cursor : match.start(1)])
        out.append(html.escape(replacement, quote=False))
        cursor = match.end(1)
    out.append(source[cursor:])
    return "".join(out).encode("utf-8")


def next_relationship_id(root: ET.Element) -> str:
    values = []
    for rel in root:
        match = re.fullmatch(r"rId(\d+)", rel.attrib.get("Id", ""))
        if match:
            values.append(int(match.group(1)))
    return f"rId{max(values, default=0) + 1}"


with zipfile.ZipFile(SOURCE, "r") as src:
    entries = {info.filename: (info, src.read(info.filename)) for info in src.infolist()}

slide_name = f"ppt/slides/slide{SOURCE_SLIDE}.xml"
new_slide_name = f"ppt/slides/slide{NEW_SLIDE}.xml"
new_rels_name = f"ppt/slides/_rels/slide{NEW_SLIDE}.xml.rels"
entries[new_slide_name] = (None, replace_slide_text(entries[slide_name][1]))
entries[new_rels_name] = (
    None,
    entries[f"ppt/slides/_rels/slide{SOURCE_SLIDE}.xml.rels"][1],
)

REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
ET.register_namespace("", REL_NS)

rels_name = "ppt/_rels/presentation.xml.rels"
rels_root = ET.fromstring(entries[rels_name][1])
new_rid = next_relationship_id(rels_root)
ET.SubElement(
    rels_root,
    f"{{{REL_NS}}}Relationship",
    {
        "Id": new_rid,
        "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide",
        "Target": f"slides/slide{NEW_SLIDE}.xml",
    },
)
entries[rels_name] = (None, ET.tostring(rels_root, encoding="utf-8", xml_declaration=True))

ET.register_namespace("a", "http://schemas.openxmlformats.org/drawingml/2006/main")
ET.register_namespace("p", P_NS)
ET.register_namespace("r", R_NS)
presentation_name = "ppt/presentation.xml"
presentation_root = ET.fromstring(entries[presentation_name][1])
sld_id_list = presentation_root.find(f"{{{P_NS}}}sldIdLst")
if sld_id_list is None:
    raise RuntimeError("Presentation has no slide list")
max_id = max(int(item.attrib["id"]) for item in sld_id_list)
new_sld_id = ET.Element(
    f"{{{P_NS}}}sldId",
    {"id": str(max_id + 1), f"{{{R_NS}}}id": new_rid},
)
sld_id_list.insert(SOURCE_SLIDE, new_sld_id)
entries[presentation_name] = (
    None,
    ET.tostring(presentation_root, encoding="utf-8", xml_declaration=True),
)

ET.register_namespace("", CT_NS)
content_name = "[Content_Types].xml"
content_root = ET.fromstring(entries[content_name][1])
ET.SubElement(
    content_root,
    f"{{{CT_NS}}}Override",
    {
        "PartName": f"/ppt/slides/slide{NEW_SLIDE}.xml",
        "ContentType": "application/vnd.openxmlformats-officedocument.presentationml.slide+xml",
    },
)
entries[content_name] = (
    None,
    ET.tostring(content_root, encoding="utf-8", xml_declaration=True),
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(OUTPUT, "w") as dst:
    for name, (info, payload) in entries.items():
        if info is None:
            dst.writestr(name, payload, compress_type=zipfile.ZIP_DEFLATED)
        else:
            dst.writestr(info, payload)

print(OUTPUT)
