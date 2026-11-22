from __future__ import annotations

from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = ROOT / "outputs" / "pilot"
OUTPUT_DIR = ROOT / "presentations"
OUTPUT_PATH = OUTPUT_DIR / "ptbxl_ecg_his_project_03_expanded_questions.pptx"

SLIDE_W = Inches(13.333333)
SLIDE_H = Inches(7.5)
NAVY = RGBColor(18, 31, 48)
BLUE = RGBColor(44, 123, 182)
ORANGE = RGBColor(255, 127, 14)
WHITE = RGBColor(255, 255, 255)
PALE = RGBColor(246, 248, 251)
MUTED = RGBColor(167, 181, 197)
FONT = "Aptos"


MODEL_NAMES = {
    "cnn_lstm_demographics": "CNN-LSTM / demographics",
    "cnn_lstm_ecg_only": "CNN-LSTM / ECG only",
    "cnn_lstm_patient_history": "CNN-LSTM / patient history",
    "cnn_lstm_patient_history_contrastive": "CNN-LSTM / patient history + contrastive",
    "lstm_demographics": "LSTM / demographics",
    "lstm_ecg_only": "LSTM / ECG only",
    "lstm_patient_history": "LSTM / patient history",
    "lstm_patient_history_contrastive": "LSTM / patient history + contrastive",
    "persistence": "Persistence baseline",
    "repeat_window": "Repeated-window baseline",
}

FIGURE_NAMES = {
    "forecast_lowest_id": "Lowest-ID test forecast",
    "forecast_median_error": "Median-error test forecast",
    "forecast_worst_error": "Worst-error test forecast",
    "horizon_error": "Forecast-horizon error",
    "learning_curves": "Learning curves",
    "per_lead_errors": "Per-lead errors",
    "subgroup_errors": "Patient subgroup errors",
    "metadata_distributions": "Pilot cohort metadata distributions",
    "model_comparison": "Held-out model comparison",
}


def set_text_style(shape, size: float, color: RGBColor, *, bold: bool = False,
                   align=PP_ALIGN.LEFT) -> None:
    frame = shape.text_frame
    frame.clear()
    frame.margin_left = 0
    frame.margin_right = 0
    frame.margin_top = 0
    frame.margin_bottom = 0
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    run = paragraph.add_run()
    run.text = ""
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color


def add_text(slide, text: str, left, top, width, height, size: float,
             color: RGBColor, *, bold: bool = False,
             align=PP_ALIGN.LEFT) -> None:
    shape = slide.shapes.add_textbox(left, top, width, height)
    set_text_style(shape, size, color, bold=bold, align=align)
    run = shape.text_frame.paragraphs[0].runs[0]
    run.text = text


def add_full_rect(slide, color: RGBColor, left=0, top=0,
                  width=SLIDE_W, height=SLIDE_H) -> None:
    rect = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    rect.fill.solid()
    rect.fill.fore_color.rgb = color
    rect.line.fill.background()


def add_cover(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_full_rect(slide, NAVY)
    add_full_rect(slide, BLUE, Inches(0.72), Inches(0.84), Inches(0.12), Inches(4.9))
    add_full_rect(slide, ORANGE, Inches(0.72), Inches(5.84), Inches(0.12), Inches(0.42))
    add_text(
        slide,
        "Patient-conditioned ECG forecasting",
        Inches(1.16), Inches(1.33), Inches(10.9), Inches(1.5),
        34, WHITE, bold=True,
    )
    add_text(
        slide,
        "High Integrity Systems Project · TSA Tasks 03",
        Inches(1.18), Inches(3.02), Inches(8.0), Inches(0.56),
        21, MUTED,
    )
    add_text(
        slide,
        "PTB-XL small-data pilot · Winter Semester 2026/27",
        Inches(1.18), Inches(3.75), Inches(10.4), Inches(0.55),
        18, WHITE,
    )
    add_text(
        slide,
        "0.6 s observed · 0.3 s forecast · 12 leads · 70 figures",
        Inches(1.18), Inches(5.53), Inches(10.8), Inches(0.52),
        17, MUTED,
    )


def add_header(slide, title: str, section: str, slide_number: int) -> None:
    add_full_rect(slide, PALE)
    add_full_rect(slide, NAVY, 0, 0, SLIDE_W, Inches(0.78))
    add_text(slide, title, Inches(0.55), Inches(0.12), Inches(9.2), Inches(0.48),
             22, WHITE, bold=True)
    add_text(slide, section, Inches(9.45), Inches(0.16), Inches(3.22), Inches(0.42),
             12.5, MUTED, align=PP_ALIGN.RIGHT)
    add_text(slide, f"{slide_number:02d}", Inches(12.50), Inches(7.16),
             Inches(0.37), Inches(0.19), 8.5, RGBColor(102, 116, 133),
             align=PP_ALIGN.RIGHT)


def add_rule(slide, left, top, width, color=BLUE, height=Inches(0.055)) -> None:
    add_full_rect(slide, color, left, top, width, height)


def add_paragraphs(slide, paragraphs, left, top, width, *, size=18,
                   color=NAVY, gap=0.58, bold_first=False) -> None:
    for index, text in enumerate(paragraphs):
        add_text(slide, text, left, top + Inches(gap * index), width, Inches(0.46),
                 size, color, bold=(bold_first and index == 0))


def add_label_value(slide, label: str, value: str, left, top, width) -> None:
    add_text(slide, label.upper(), left, top, width, Inches(0.28), 10.5,
             RGBColor(94, 111, 130), bold=True)
    add_text(slide, value, left, top + Inches(0.31), width, Inches(0.65), 20,
             NAVY, bold=True)


def add_task_coverage(prs: Presentation, n: int) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, "HIS Project 03 task coverage", "Assignment requirements", n)
    rows = [
        ("1", "State of the art", "Literature comparison and protocol caveats"),
        ("2", "Preprocessing and loaders", "Official patient folds; train-only scaling"),
        ("3-5", "LSTM and CNN-LSTM", "Architecture, training, metrics and comparison"),
        ("6", "Literature benchmark", "Butt et al. values reported without ranking"),
        ("7-8", "Mechanics and mathematics", "CNN/LSTM operations and metric equations"),
        ("9", "Agentic AI", "Definition and minimum agent components"),
        ("10-11", "Delivery", "Report artifacts plus a 25-minute presentation"),
    ]
    y = 1.12
    for task, requirement, evidence in rows:
        add_text(slide, task, Inches(0.62), Inches(y), Inches(0.72), Inches(0.43),
                 17, BLUE, bold=True)
        add_text(slide, requirement, Inches(1.42), Inches(y), Inches(3.46), Inches(0.43),
                 17, NAVY, bold=True)
        add_text(slide, evidence, Inches(5.05), Inches(y), Inches(7.43), Inches(0.43),
                 16.5, RGBColor(68, 83, 101))
        add_rule(slide, Inches(0.62), Inches(y + 0.49), Inches(11.86), RGBColor(218, 225, 233), Inches(0.018))
        y += 0.78
    add_text(slide, "Source: HIS_Project_03.pdf", Inches(0.62), Inches(6.82), Inches(4.0), Inches(0.28),
             10.5, RGBColor(102, 116, 133))


def add_required_questions(prs: Presentation, n: int) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, "Required technical questions", "HIS Project 03", n)
    add_text(slide, "1", Inches(0.72), Inches(1.32), Inches(0.62), Inches(0.72), 31, BLUE, bold=True)
    add_text(slide, "How do CNNs and LSTMs work in this project, including their layers, gates and model training?",
             Inches(1.42), Inches(1.24), Inches(10.75), Inches(1.25), 25, NAVY, bold=True)
    add_rule(slide, Inches(0.72), Inches(2.85), Inches(11.55), RGBColor(210, 219, 229), Inches(0.025))
    add_text(slide, "2", Inches(0.72), Inches(3.36), Inches(0.62), Inches(0.72), 31, ORANGE, bold=True)
    add_text(slide, "What is agentic AI, and which components are required to build a simple AI agent?",
             Inches(1.42), Inches(3.28), Inches(10.75), Inches(1.25), 25, NAVY, bold=True)
    add_text(slide, "Slides 7-13 answer question 1. Slides 18-19 answer question 2.",
             Inches(0.72), Inches(6.22), Inches(11.55), Inches(0.48), 16, RGBColor(94, 111, 130), align=PP_ALIGN.CENTER)


def add_problem_cohort(prs: Presentation, n: int) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, "Forecasting task and active cohort", "PTB-XL pilot", n)
    add_text(slide, "Observed window", Inches(0.66), Inches(1.18), Inches(3.0), Inches(0.42), 16, BLUE, bold=True)
    add_text(slide, "12 leads × 300 samples\n0.6 s at 500 Hz", Inches(0.66), Inches(1.67), Inches(3.0), Inches(1.05), 24, NAVY, bold=True)
    add_text(slide, "Forecast window", Inches(4.10), Inches(1.18), Inches(3.0), Inches(0.42), 16, ORANGE, bold=True)
    add_text(slide, "12 leads × 150 samples\nnext 0.3 s", Inches(4.10), Inches(1.67), Inches(3.0), Inches(1.05), 24, NAVY, bold=True)
    add_text(slide, "Task", Inches(7.55), Inches(1.18), Inches(4.7), Inches(0.42), 16, RGBColor(94, 111, 130), bold=True)
    add_text(slide, "Within-record voltage forecasting\nin original mV units", Inches(7.55), Inches(1.67), Inches(4.7), Inches(1.05), 24, NAVY, bold=True)
    add_rule(slide, Inches(0.66), Inches(3.10), Inches(11.65))
    add_label_value(slide, "Training", "515 records · 450 patients", Inches(0.66), Inches(3.55), Inches(3.7))
    add_label_value(slide, "Validation", "108 records · 100 patients", Inches(4.55), Inches(3.55), Inches(3.7))
    add_label_value(slide, "Test", "116 records · 100 patients", Inches(8.42), Inches(3.55), Inches(3.7))
    add_text(slide, "Prior history exists for 65 training, 8 validation and 16 test records.",
             Inches(0.66), Inches(5.05), Inches(11.6), Inches(0.48), 18, NAVY)
    add_text(slide, "Research question: do demographics, prior diagnostic superclasses or contrastive training reduce short-horizon forecast error?",
             Inches(0.66), Inches(5.78), Inches(11.6), Inches(0.78), 19, NAVY, bold=True)


def add_preprocessing(prs: Presentation, n: int) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, "Preprocessing and patient isolation", "Data integrity", n)
    items = [
        ("Patient split", "PTB-XL official folds 1-8 / 9 / 10 prevent the same patient from crossing train, validation and test."),
        ("Waveforms", "records500 stays at 500 Hz. Inputs use samples 0:300 and targets use 300:450."),
        ("Scaling", "Per-lead mean and standard deviation fit on training inputs only. Targets reuse the same transform."),
        ("Age and sex", "Age outside 0-120 becomes missing. Training median imputes age. Unknown sex uses 0.5 plus a missingness flag."),
        ("History", "Only strictly earlier timestamps contribute NORM, MI, STTC, CD and HYP indicators. A history mask records availability."),
    ]
    y = 1.15
    for label, body in items:
        add_text(slide, label, Inches(0.64), Inches(y), Inches(2.0), Inches(0.48), 18, BLUE, bold=True)
        add_text(slide, body, Inches(2.62), Inches(y), Inches(9.72), Inches(0.72), 17, NAVY)
        y += 1.08
    add_text(slide, "Random window splitting would leak patient identity and inflate validation performance.",
             Inches(0.64), Inches(6.48), Inches(11.6), Inches(0.45), 17.5, ORANGE, bold=True)


def add_dataloaders(prs: Presentation, n: int) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, "Dataloader construction", "Training, validation and test", n)
    columns = [
        ("TRAIN", "515 records\n450 patients\nshuffle = true\nbatch = 64\naugment contrastive views only"),
        ("VALIDATION", "108 records\n100 patients\nshuffle = false\ncheckpoint on forecast MSE\nno augmentation"),
        ("TEST", "116 records\n100 patients\nshuffle = false\noriginal-mV metrics\npatient-level bootstrap"),
    ]
    for i, (label, body) in enumerate(columns):
        x = 0.65 + i * 4.15
        add_text(slide, label, Inches(x), Inches(1.28), Inches(3.55), Inches(0.42), 15, BLUE, bold=True)
        add_rule(slide, Inches(x), Inches(1.78), Inches(3.55), BLUE)
        add_text(slide, body, Inches(x), Inches(2.10), Inches(3.55), Inches(2.82), 20, NAVY)
    add_text(slide, "Each record contributes one fixed window. The loader returns waveform input, static features, target waveform, patient ID and subgroup labels.",
             Inches(0.65), Inches(5.72), Inches(11.85), Inches(0.84), 18, NAVY, bold=True)


def add_architecture(slide, title: str, section: str, n: int, blocks) -> None:
    add_header(slide, title, section, n)
    top = Inches(2.14)
    block_w = Inches(2.18)
    gap = Inches(0.30)
    total_w = len(blocks) * block_w + (len(blocks) - 1) * gap
    left = int((SLIDE_W - total_w) / 2)
    for i, (heading, body, accent) in enumerate(blocks):
        x = left + i * (block_w + gap)
        rect = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, top, block_w, Inches(2.18))
        rect.fill.solid(); rect.fill.fore_color.rgb = WHITE
        rect.line.color.rgb = accent; rect.line.width = Pt(2)
        add_text(slide, heading, x + Inches(0.15), top + Inches(0.18), block_w - Inches(0.30), Inches(0.52),
                 17, accent, bold=True, align=PP_ALIGN.CENTER)
        add_text(slide, body, x + Inches(0.14), top + Inches(0.82), block_w - Inches(0.28), Inches(1.05),
                 15.5, NAVY, align=PP_ALIGN.CENTER)
        if i < len(blocks) - 1:
            add_text(slide, "›", x + block_w + Inches(0.04), top + Inches(0.73), gap - Inches(0.08), Inches(0.7),
                     34, RGBColor(120, 134, 150), bold=True, align=PP_ALIGN.CENTER)


def add_lstm_architecture(prs: Presentation, n: int) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_architecture(slide, "LSTM forecast architecture", "Model mechanics", n, [
        ("ECG input", "[B, 12, 300]\ntranspose to\n[B, 300, 12]", BLUE),
        ("Two-layer LSTM", "hidden size 128\ndropout 0.2\nfinal state [B, 128]", BLUE),
        ("Static branch", "0, 4 or 10 features\nembedding [B, 32]", ORANGE),
        ("Forecast head", "concat [B, 160]\n256 ReLU\n1,800 outputs", ORANGE),
        ("Prediction", "reshape to\n[B, 12, 150]", BLUE),
    ])
    add_text(slide, "Static context enters after the recurrent encoder, so the LSTM state represents the observed ECG sequence.",
             Inches(0.72), Inches(5.28), Inches(11.9), Inches(0.62), 18, NAVY, bold=True, align=PP_ALIGN.CENTER)


def add_lstm_gates(prs: Presentation, n: int) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, "LSTM gates and backpropagation through time", "Model mechanics", n)
    formulas = [
        "zₜ = [hₜ₋₁, xₜ]",
        "fₜ = σ(Wf zₜ + bf)     iₜ = σ(Wi zₜ + bi)",
        "gₜ = tanh(Wg zₜ + bg)",
        "cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ gₜ",
        "oₜ = σ(Wo zₜ + bo)     hₜ = oₜ ⊙ tanh(cₜ)",
    ]
    add_paragraphs(slide, formulas, Inches(0.70), Inches(1.20), Inches(6.6), size=20, gap=0.74)
    add_rule(slide, Inches(7.52), Inches(1.20), Inches(0.055), RGBColor(210, 219, 229), Inches(4.85))
    points = [
        "Sigmoid gates retain or suppress information in the range [0, 1].",
        "The tanh candidate supplies signed content for the cell state.",
        "The direct memory path contains products of forget gates across 300 time steps.",
        "Gradient clipping at norm 1.0 limits exploding gradients during BPTT.",
    ]
    add_paragraphs(slide, points, Inches(7.88), Inches(1.30), Inches(4.65), size=17, gap=1.12)
    add_text(slide, "⊙ denotes elementwise multiplication.", Inches(0.70), Inches(5.78), Inches(5.7), Inches(0.42), 14, RGBColor(94, 111, 130))


def add_lstm_step_detail(prs: Presentation, n: int) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, "LSTM computation across the ECG window", "Detailed working", n)
    steps = [
        ("1", "Input at t", "xₜ contains the 12 lead voltages at one sample."),
        ("2", "Gate values", "The forget, input and output gates combine xₜ with hₜ₋₁."),
        ("3", "Cell update", "fₜ retains prior memory. iₜ controls how much candidate gₜ enters cₜ."),
        ("4", "Hidden state", "oₜ exposes part of tanh(cₜ) as hₜ for the next sample and next layer."),
        ("5", "Second layer", "Layer 2 receives the full layer-1 hidden sequence for all 300 samples."),
        ("6", "Forecast state", "The final layer-2 state [B,128] enters the forecast head after static fusion."),
    ]
    y = 1.08
    for number, label, body in steps:
        add_text(slide, number, Inches(0.66), Inches(y), Inches(0.52), Inches(0.46), 20, BLUE, bold=True, align=PP_ALIGN.CENTER)
        add_text(slide, label, Inches(1.28), Inches(y), Inches(2.10), Inches(0.46), 17, NAVY, bold=True)
        add_text(slide, body, Inches(3.42), Inches(y), Inches(8.82), Inches(0.60), 16.5, NAVY)
        add_rule(slide, Inches(1.28), Inches(y + 0.67), Inches(10.96), RGBColor(218, 225, 233), Inches(0.018))
        y += 0.92
    add_text(slide, "Backpropagation through time unrolls both recurrent layers across all 300 input samples.",
             Inches(0.70), Inches(6.62), Inches(11.55), Inches(0.38), 15.5, ORANGE, bold=True)


def add_cnn(prs: Presentation, n: int) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, "Temporal CNN operation", "Model mechanics", n)
    add_text(slide, "Y[o,t] = b[o] + Σc Σk W[o,c,k] X[c,t+k−padding]",
             Inches(0.70), Inches(1.22), Inches(11.9), Inches(0.72), 24, NAVY, bold=True, align=PP_ALIGN.CENTER)
    add_rule(slide, Inches(1.20), Inches(2.18), Inches(10.9))
    entries = [
        ("Channels", "Each filter combines all input leads and produces one output feature map."),
        ("Kernel", "Widths 7, 5 and 3 capture local voltage morphology with shared weights across time."),
        ("Padding", "Same padding preserves temporal length before the pooling layer."),
        ("Pooling", "Average pooling by 2 changes 300 raw samples into 150 recurrent feature steps."),
        ("Activation", "ReLU follows each convolution. The final receptive field spans 16 raw samples or 32 ms."),
    ]
    y = 2.64
    for label, body in entries:
        add_text(slide, label, Inches(0.75), Inches(y), Inches(1.55), Inches(0.42), 17, BLUE, bold=True)
        add_text(slide, body, Inches(2.35), Inches(y), Inches(9.80), Inches(0.55), 17, NAVY)
        y += 0.82
    add_text(slide, "PyTorch Conv1d computes cross-correlation rather than reversing the kernel.",
             Inches(0.75), Inches(6.60), Inches(11.3), Inches(0.35), 15, ORANGE, bold=True)


def add_cnn_lstm_architecture(prs: Presentation, n: int) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_architecture(slide, "CNN-LSTM forecast architecture", "Model mechanics", n, [
        ("ECG input", "[B, 12, 300]", BLUE),
        ("Temporal CNN", "12→32 k7\n32→64 k5\npool by 2\n64→64 k3", BLUE),
        ("LSTM", "transpose to\n[B, 150, 64]\nfinal [B, 128]", BLUE),
        ("Static branch", "0, 4 or 10 features\nembedding [B, 32]", ORANGE),
        ("Forecast head", "256 ReLU\n1,800 outputs\n[B, 12, 150]", ORANGE),
    ])
    add_text(slide, "The CNN shortens the recurrent sequence but can lose timing precision around sharp QRS peaks.",
             Inches(0.72), Inches(5.28), Inches(11.9), Inches(0.62), 18, NAVY, bold=True, align=PP_ALIGN.CENTER)


def add_training(prs: Presentation, n: int) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, "Training objective and hyperparameters", "Experiment design", n)
    left = [
        ("Forecast loss", "Normalized waveform MSE on clean inputs"),
        ("Contrastive loss", "0.1 × NT-Xent, temperature 0.1"),
        ("Optimizer", "AdamW, learning rate 0.001"),
        ("Regularization", "weight decay 0.0001, dropout 0.2"),
    ]
    right = [
        ("Batch", "64 records"),
        ("Gradient clipping", "global norm 1.0"),
        ("Checkpoint", "minimum validation forecast MSE"),
        ("Budget", "maximum 5 epochs, patience 3, seed 42"),
    ]
    for col, entries in enumerate((left, right)):
        x = 0.72 + col * 6.15
        y = 1.20
        for label, value in entries:
            add_text(slide, label, Inches(x), Inches(y), Inches(2.05), Inches(0.35), 14.5, BLUE, bold=True)
            add_text(slide, value, Inches(x + 2.12), Inches(y), Inches(3.75), Inches(0.55), 17, NAVY)
            add_rule(slide, Inches(x), Inches(y + 0.62), Inches(5.75), RGBColor(218, 225, 233), Inches(0.018))
            y += 1.15
    add_text(slide, "Eight neural runs use identical folds and optimization settings. Persistence and repeated-window forecasts provide non-trained baselines.",
             Inches(0.72), Inches(6.05), Inches(11.9), Inches(0.65), 17.5, NAVY, bold=True)


def add_training_loop(prs: Presentation, n: int) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, "Training loop for one minibatch", "Detailed training", n)
    steps = [
        ("1", "Load batch", "ECG input [B,12,300], target [B,12,150], static context and patient IDs."),
        ("2", "Reset gradients", "optimizer.zero_grad() clears gradients left from the previous batch."),
        ("3", "Forecast forward pass", "The clean input produces the waveform forecast and normalized MSE."),
        ("4", "Contrastive branch", "Contrastive variants create two augmented input views and compute masked NT-Xent."),
        ("5", "Combine losses", "loss = forecast MSE + 0.1 × NT-Xent. Plain variants use forecast MSE only."),
        ("6", "Update weights", "loss.backward(), clip_grad_norm_(1.0), then optimizer.step()."),
        ("7", "Validate", "No augmentation. Validation forecast MSE selects the checkpoint and drives patience 3."),
    ]
    y = 1.02
    for number, label, body in steps:
        add_text(slide, number, Inches(0.62), Inches(y), Inches(0.44), Inches(0.40), 17, BLUE, bold=True, align=PP_ALIGN.CENTER)
        add_text(slide, label, Inches(1.18), Inches(y), Inches(2.08), Inches(0.40), 16.5, NAVY, bold=True)
        add_text(slide, body, Inches(3.34), Inches(y), Inches(9.00), Inches(0.54), 15.8, NAVY)
        y += 0.78
    add_text(slide, "The test set runs once after checkpoint selection. Test errors do not tune the optimizer or architecture.",
             Inches(0.66), Inches(6.52), Inches(11.65), Inches(0.42), 15.5, ORANGE, bold=True)


def add_metrics(prs: Presentation, n: int) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, "Forecast metrics", "Mathematical definitions", n)
    rows = [
        ("MAE", "(1/N) Σ |ŷᵢ − yᵢ|", "mV", "Average absolute voltage error"),
        ("MSE", "(1/N) Σ (ŷᵢ − yᵢ)²", "mV²", "Squared error used for checkpoint selection"),
        ("RMSE", "√MSE", "mV", "Square root restores the voltage unit"),
        ("R²", "1 − Σ(ŷᵢ−yᵢ)² / Σ(yᵢ−ȳ)²", "unitless", "Negative values mean worse than the global mean"),
        ("Pearson", "cov(ŷ,y) / (σŷ σy)", "unitless", "Calculated per lead, then averaged"),
        ("Patient-macro MAE", "(1/P) Σp MAEₚ", "mV", "Each patient receives equal weight"),
    ]
    y = 1.12
    for name, formula, unit, meaning in rows:
        add_text(slide, name, Inches(0.62), Inches(y), Inches(2.05), Inches(0.44), 17, BLUE, bold=True)
        add_text(slide, formula, Inches(2.65), Inches(y), Inches(4.0), Inches(0.44), 17, NAVY, bold=True)
        add_text(slide, unit, Inches(6.78), Inches(y), Inches(1.1), Inches(0.44), 16, ORANGE, bold=True)
        add_text(slide, meaning, Inches(8.00), Inches(y), Inches(4.45), Inches(0.55), 15.5, NAVY)
        y += 0.93
    add_text(slide, "N spans test records, leads and forecast samples. Reported metrics use inverse-transformed voltages.",
             Inches(0.62), Inches(6.72), Inches(11.7), Inches(0.32), 13.5, RGBColor(94, 111, 130))


def add_results_table(prs: Presentation, n: int) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, "Held-out test results", "Original mV units", n)
    data = [
        ("Model", "MAE", "RMSE", "R²", "Patient-macro MAE"),
        ("Persistence", "0.141546", "0.358149", "0.329855", "0.129724"),
        ("Repeated window", "0.168642", "0.336686", "0.407766", "0.161466"),
        ("LSTM / ECG only", "0.151706", "0.430543", "0.031552", "0.148435"),
        ("LSTM / demographics", "0.151759", "0.431238", "0.028427", "0.148413"),
        ("LSTM / history", "0.150113", "0.430582", "0.031379", "0.146684"),
        ("LSTM / history + contrastive", "0.151782", "0.431546", "0.027037", "0.148438"),
        ("CNN-LSTM / ECG only", "0.154296", "0.433948", "0.016176", "0.151576"),
        ("CNN-LSTM / demographics", "0.155650", "0.434724", "0.012654", "0.152838"),
        ("CNN-LSTM / history", "0.154251", "0.433902", "0.016385", "0.151215"),
        ("CNN-LSTM / history + contrastive", "0.155719", "0.435588", "0.008725", "0.152772"),
    ]
    table = slide.shapes.add_table(len(data), len(data[0]), Inches(0.55), Inches(1.05), Inches(12.25), Inches(5.70)).table
    widths = [4.15, 1.65, 1.65, 1.55, 2.75]
    for i, width in enumerate(widths): table.columns[i].width = Inches(width)
    for r, row in enumerate(data):
        for c, value in enumerate(row):
            cell = table.cell(r, c); cell.text = value
            cell.margin_left = Inches(0.08); cell.margin_right = Inches(0.08)
            cell.fill.solid(); cell.fill.fore_color.rgb = NAVY if r == 0 else (RGBColor(232, 241, 249) if r in (1, 5) else WHITE)
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
                for run in p.runs:
                    run.font.name = FONT; run.font.size = Pt(12.5 if r else 13); run.font.bold = r == 0 or r in (1, 5)
                    run.font.color.rgb = WHITE if r == 0 else NAVY
    add_text(slide, "Lowest observed MAE: persistence. Lowest observed neural MAE: LSTM with patient history.",
             Inches(0.58), Inches(6.82), Inches(11.8), Inches(0.30), 13.5, ORANGE, bold=True)


def add_results_findings(prs: Presentation, n: int) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, "Measured model comparisons", "Patient-level uncertainty", n)
    findings = [
        ("Persistence vs best neural", "0.141546 vs 0.150113 mV MAE", "Persistence produced the lowest record-weighted MAE."),
        ("LSTM history vs demographics", "−0.001729 mV patient-macro MAE", "95% interval [−0.003275, −0.000242]."),
        ("LSTM contrastive vs history", "+0.001754 mV patient-macro MAE", "95% interval [+0.000283, +0.003351]."),
        ("CNN-LSTM vs LSTM, ECG only", "+0.003141 mV patient-macro MAE", "95% interval [+0.001435, +0.005009]."),
    ]
    y = 1.16
    for label, value, note in findings:
        add_text(slide, label, Inches(0.68), Inches(y), Inches(3.25), Inches(0.42), 16.5, BLUE, bold=True)
        add_text(slide, value, Inches(3.85), Inches(y), Inches(4.20), Inches(0.42), 18, NAVY, bold=True)
        add_text(slide, note, Inches(8.05), Inches(y), Inches(4.45), Inches(0.55), 15.5, NAVY)
        add_rule(slide, Inches(0.68), Inches(y + 0.68), Inches(11.82), RGBColor(218, 225, 233), Inches(0.018))
        y += 1.25
    add_text(slide, "Intervals use 1,000 patient bootstrap resamples and describe this fitted seed only.",
             Inches(0.68), Inches(6.38), Inches(11.7), Inches(0.44), 16, ORANGE, bold=True)


def add_literature(prs: Presentation, n: int) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, "Literature comparison", "Butt et al. 2025 manuscript", n)
    add_text(slide, "Reported experiments 1-5", Inches(0.65), Inches(1.14), Inches(4.2), Inches(0.42), 17, BLUE, bold=True)
    add_text(slide, "MAE: 0.0504-0.0725\nRMSE: 0.1135-0.1466\nSMAPE: 0.8791-1.0104",
             Inches(0.65), Inches(1.72), Inches(4.2), Inches(1.85), 25, NAVY, bold=True)
    add_text(slide, "Project protocol", Inches(4.86), Inches(1.14), Inches(3.25), Inches(0.42), 17, BLUE, bold=True)
    add_text(slide, "500 Hz PTB-XL\nofficial patient folds\ntrain-input z-scores\nerrors inverted to mV",
             Inches(4.86), Inches(1.72), Inches(3.25), Inches(2.15), 20, NAVY)
    add_text(slide, "Manuscript protocol", Inches(8.42), Inches(1.14), Inches(3.75), Inches(0.42), 17, BLUE, bold=True)
    add_text(slide, "80/10/10 split\nrobust TorchNormalizer\npatient isolation unclear\nlead aggregation unclear",
             Inches(8.42), Inches(1.72), Inches(3.75), Inches(2.15), 20, NAVY)
    add_rule(slide, Inches(0.65), Inches(4.26), Inches(11.65))
    add_text(slide, "The paper abstract reports RMSE 0.061 and MAE 0.128, which conflicts with its Table 1. Protocol and scaling differences prevent a numerical ranking.",
             Inches(0.65), Inches(4.68), Inches(11.65), Inches(1.05), 19, NAVY, bold=True)
    add_text(slide, "Context source: F. S. Butt et al., Temporal Fusion Transformers for Forecasting ECG Signals, pp. 5-6, Table 1.",
             Inches(0.65), Inches(6.34), Inches(11.65), Inches(0.42), 13, RGBColor(94, 111, 130))


def add_agentic_ai(prs: Presentation, n: int) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, "Agentic AI and a minimal ECG analysis agent", "HIS Project 03 task 9", n)
    add_text(slide, "Agentic AI uses a model inside a controlled loop that selects actions, uses tools and checks progress toward a stated goal.",
             Inches(0.68), Inches(1.10), Inches(11.85), Inches(0.78), 21, NAVY, bold=True)
    components = [
        ("Goal", "Example: evaluate one trained checkpoint on the locked test split."),
        ("Model", "Interprets the request and selects the next permitted operation."),
        ("Tools", "Read configuration, run evaluation and retrieve saved metrics."),
        ("State", "Stores the checkpoint, dataset version, completed steps and outputs."),
        ("Policy", "Restricts tool access, patient data exposure and irreversible actions."),
        ("Feedback", "Checks schema, metric units, failures and completion conditions."),
    ]
    y = 2.08
    for i, (label, body) in enumerate(components):
        col = i % 2; row = i // 2
        x = 0.68 + col * 6.15; yy = y + row * 1.35
        add_text(slide, label, Inches(x), Inches(yy), Inches(1.25), Inches(0.40), 17, BLUE, bold=True)
        add_text(slide, body, Inches(x + 1.30), Inches(yy), Inches(4.58), Inches(0.78), 16, NAVY)
    add_text(slide, "For clinical data, human review remains the final authority for interpretation and release.",
             Inches(0.68), Inches(6.40), Inches(11.8), Inches(0.42), 16, ORANGE, bold=True)


def add_agent_execution(prs: Presentation, n: int) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, "Simple AI agent execution loop", "Agentic AI components in operation", n)
    stages = [
        ("1. Receive goal", "Evaluate checkpoint X on the locked PTB-XL test split."),
        ("2. Read state", "Load the approved configuration, dataset version and prior tool results."),
        ("3. Select action", "The model chooses one allowed tool call that advances the goal."),
        ("4. Use tool", "Run evaluation or read saved metrics within the configured permissions."),
        ("5. Observe", "Record the result, validate units and identify tool or data errors."),
        ("6. Stop or continue", "Stop after the required metrics exist and checks pass. Otherwise select another action."),
    ]
    y = 1.10
    for i, (label, body) in enumerate(stages):
        col = i % 2; row = i // 2
        x = 0.68 + col * 6.18; yy = y + row * 1.58
        add_text(slide, label, Inches(x), Inches(yy), Inches(5.60), Inches(0.42), 17, BLUE if col == 0 else ORANGE, bold=True)
        add_text(slide, body, Inches(x), Inches(yy + 0.49), Inches(5.60), Inches(0.78), 16, NAVY)
    add_rule(slide, Inches(0.68), Inches(6.02), Inches(11.78))
    add_text(slide, "Minimum controls: tool allowlist, patient-data boundary, action log, failure handling, completion test and human approval for clinical interpretation.",
             Inches(0.68), Inches(6.24), Inches(11.78), Inches(0.66), 15.8, NAVY, bold=True)


def add_conclusion(prs: Presentation, n: int) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, "Findings and limits", "PTB-XL small-data pilot", n)
    items = [
        ("Measured", "Persistence achieved the lowest MAE at 0.141546 mV."),
        ("Measured", "LSTM with patient history achieved the lowest neural MAE at 0.150113 mV."),
        ("Measured", "The contrastive objective increased LSTM patient-macro MAE by 0.001754 mV in this run."),
        ("Limit", "Five epochs and one seed do not establish architecture superiority."),
        ("Limit", "Only 16 test records contain prior history, and diagnostic groups overlap."),
        ("Limit", "A 0.6 s context may contain less than one beat, so the task does not measure long-term clinical forecasting."),
    ]
    y = 1.10
    for tag, body in items:
        color = BLUE if tag == "Measured" else ORANGE
        add_text(slide, tag.upper(), Inches(0.68), Inches(y), Inches(1.25), Inches(0.36), 12.5, color, bold=True)
        add_text(slide, body, Inches(1.95), Inches(y), Inches(10.45), Inches(0.68), 18, NAVY, bold=True if tag == "Measured" else False)
        y += 0.90
    add_rule(slide, Inches(0.68), Inches(6.48), Inches(11.7))
    add_text(slide, "Next experiment: larger patient cohort, repeated seeds, longer contexts and a predeclared training budget.",
             Inches(0.68), Inches(6.64), Inches(11.7), Inches(0.34), 15.5, NAVY, bold=True)


def add_appendix_divider(prs: Presentation, n: int) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_full_rect(slide, NAVY)
    add_text(slide, "Figure appendix", Inches(0.85), Inches(2.15), Inches(11.6), Inches(0.92), 34, WHITE, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, "70 generated plots across eight neural runs and two baselines", Inches(0.85), Inches(3.38), Inches(11.6), Inches(0.58), 19, MUTED, align=PP_ALIGN.CENTER)
    add_text(slide, f"{n:02d}", Inches(12.50), Inches(7.16), Inches(0.37), Inches(0.19), 8.5, MUTED, align=PP_ALIGN.RIGHT)


def image_box(path: Path, max_left, max_top, max_width, max_height):
    with Image.open(path) as image:
        img_w, img_h = image.size
    scale = min(max_width / img_w, max_height / img_h)
    width = int(img_w * scale)
    height = int(img_h * scale)
    left = int(max_left + (max_width - width) / 2)
    top = int(max_top + (max_height - height) / 2)
    return left, top, width, height


def title_for(path: Path) -> tuple[str, str]:
    stem = FIGURE_NAMES[path.stem]
    if path.parent == FIGURE_ROOT:
        return stem, "PTB-XL small-data pilot"
    return stem, MODEL_NAMES[path.parent.name]


def add_figure_slide(prs: Presentation, path: Path, slide_number: int) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_full_rect(slide, PALE)
    add_full_rect(slide, NAVY, 0, 0, SLIDE_W, Inches(0.78))
    title, model = title_for(path)
    add_text(slide, title, Inches(0.55), Inches(0.12), Inches(8.8), Inches(0.48),
             22, WHITE, bold=True)
    add_text(slide, model, Inches(9.15), Inches(0.16), Inches(3.52), Inches(0.42),
             12.5, MUTED, align=PP_ALIGN.RIGHT)

    max_left = Inches(0.42)
    max_top = Inches(0.94)
    max_width = Inches(12.49)
    max_height = Inches(6.18)
    left, top, width, height = image_box(path, max_left, max_top, max_width, max_height)
    slide.shapes.add_picture(str(path), left, top, width=width, height=height)

    add_text(slide, f"{slide_number:02d}", Inches(12.50), Inches(7.16),
             Inches(0.37), Inches(0.19), 8.5, RGBColor(102, 116, 133),
             align=PP_ALIGN.RIGHT)


def main() -> None:
    figures = sorted(FIGURE_ROOT.rglob("*.png"))
    if len(figures) != 70:
        raise RuntimeError(f"Expected 70 pilot figures, found {len(figures)}")

    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    prs.core_properties.title = "Patient-conditioned ECG forecasting"
    prs.core_properties.subject = "PTB-XL small-data pilot with all generated graphs"
    prs.core_properties.author = "ECG TSA project"
    add_cover(prs)
    content_builders = [
        add_task_coverage,
        add_required_questions,
        add_problem_cohort,
        add_preprocessing,
        add_dataloaders,
        add_lstm_architecture,
        add_lstm_gates,
        add_lstm_step_detail,
        add_cnn,
        add_cnn_lstm_architecture,
        add_training,
        add_training_loop,
        add_metrics,
        add_results_table,
        add_results_findings,
        add_literature,
        add_agentic_ai,
        add_agent_execution,
        add_conclusion,
        add_appendix_divider,
    ]
    for index, builder in enumerate(content_builders, start=2):
        builder(prs, index)
    figure_start = 2 + len(content_builders)
    for index, figure in enumerate(figures, start=figure_start):
        add_figure_slide(prs, figure, index)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prs.save(OUTPUT_PATH)
    print(OUTPUT_PATH)
    print(f"slides={len(prs.slides)} content_slides={1 + len(content_builders)} figures={len(figures)}")


if __name__ == "__main__":
    main()
