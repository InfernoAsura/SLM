"""Build report/writeup.docx (<= 5 pages, answers Q1-Q11) and report/project_report.docx
(a longer plain-language account of the project). All numbers are read from the result
files written by the other scripts, so the documents always match the runs.

python scripts/build_reports.py
"""
import csv
import json
import os
import re

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *p: os.path.join(ROOT, *p)  # noqa: E731
FIG = lambda name: R("report", "figures", name)  # noqa: E731


def load(path):
    with open(R(*path.split("/")), encoding="utf-8") as f:
        return json.load(f)


A = load("report/results/analysis.json")
KV = load("report/results/kv_estimate.json")
TOK = load("artifacts/tokenizer/train_stats.json")
ENC_TRAIN = load("data/train.npy.stats.json")
ENC_VALID = load("data/valid.npy.stats.json")
MAIN = load("runs/main/summary.json")
SWEEP = load("runs/lr_sweep/summary.json")
SWEEP32 = load("runs/lr_sweep_fp32/summary.json")
BENCH = load("runs/bench_compile.json")
STORIES = load("report/stories.json")

rows = list(csv.DictReader(open(R("runs", "main", "log.csv"))))
gn100 = [float(r["grad_norm"]) for r in rows if r["grad_norm"] and 1 <= int(r["step"]) <= 100]
gn_late = [float(r["grad_norm"]) for r in rows if r["grad_norm"] and int(r["step"]) > 1000]
skipped = sum(int(r["skipped"]) for r in rows if r["skipped"])
scales = sorted({float(r["loss_scale"]) for r in rows if r["loss_scale"]})
dts, prev = [], 0.0
for r in rows[1:]:
    dts.append((int(r["step"]), float(r["elapsed_s"]) - prev))
    prev = float(r["elapsed_s"])
standby = sum(d for s, d in dts if d > 60 and s > 1)
active_s = MAIN["train_seconds"] - standby
compile_s = dts[0][1]
train_last100 = [float(r["train_loss"]) for r in rows if r["train_loss"]][-100:]
train_last100 = sum(train_last100) / len(train_last100)

sb = list(csv.DictReader(open(R("runs", "single_batch", "log.csv"))))
sb_first, sb_last = float(sb[1]["train_loss"]), float(sb[-1]["train_loss"])
sb_val0 = float(sb[0]["val_loss"])
sb_val_end = float(sb[-1]["val_loss"])
sb_100 = next(float(r["train_loss"]) for r in sb if r["step"] == "100")

rf = {r["step"]: r for r in csv.DictReader(open(R("runs", "resume_full", "log.csv")))}
rs = {r["step"]: r for r in csv.DictReader(open(R("runs", "resume_split", "log.csv")))}
res_diff = max(abs(float(rf[s]["train_loss"]) - float(rs[s]["train_loss"])) for s in rf if rf[s]["train_loss"])

V, NP = 10_000, MAIN["parameters"]
q5, q6, q7, q8 = A["q5"], A["q6"], A["q7_gpt2_small_swiglu_dff2048"], A["q8"]
q3, q4 = A["q3"], A["q4"]["dots"]
W = A["warmups"]
T = A["tokenizer"]
P1v, P1t = A["q1_profile_valid"], A["q1_profile_train_first_200MB"]
nat = STORIES["natural_lengths"]


def f(x, n=2):
    return f"{x:,.{n}f}"


def i(x):
    return f"{int(x):,}"


# ----------------------------------------------------------------------------- docx helpers
class Doc:
    def __init__(self, base_pt=10.5, margins=1.0):
        self.d = Document()
        sec = self.d.sections[0]
        sec.page_width, sec.page_height = Inches(8.5), Inches(11)
        for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
            setattr(sec, side, Inches(margins))
        self.width = 8.5 - 2 * margins
        st = self.d.styles
        st["Normal"].font.name = "Calibri"
        st["Normal"].font.size = Pt(base_pt)
        st["Normal"].paragraph_format.space_after = Pt(4)
        st["Normal"].paragraph_format.line_spacing = 1.08
        for name, size, color in (("Heading 1", base_pt + 7, "1F3864"), ("Heading 2", base_pt + 3, "1F3864"), ("Heading 3", base_pt + 1, "2E5597")):
            h = st[name]
            h.font.name = "Calibri"
            h.font.size = Pt(size)
            h.font.bold = True
            h.font.color.rgb = RGBColor.from_string(color)
            h.paragraph_format.space_before = Pt(8 if name != "Heading 1" else 0)
            h.paragraph_format.space_after = Pt(3)
            h.paragraph_format.keep_with_next = True
            rpr = h.element.get_or_add_rPr()
            fonts = rpr.find(qn("w:rFonts"))
            if fonts is None:
                fonts = OxmlElement("w:rFonts")
                rpr.append(fonts)
            for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
                fonts.set(qn(a), "Calibri")
        for name in ("List Bullet", "List Number"):
            st[name].font.size = Pt(base_pt)
            st[name].paragraph_format.space_after = Pt(2)

    def h(self, text, level):
        return self.d.add_heading(text, level)

    def p(self, *parts, style=None, align=None, size=None, italic=False):
        """parts: strings; a string starting with '**' and ending with '**' is bold."""
        para = self.d.add_paragraph(style=style)
        # **text** anywhere in the joined parts is bold
        for k, piece in enumerate(re.split(r"\*\*(.+?)\*\*", "".join(parts))):
            if not piece:
                continue
            run = para.add_run(piece)
            run.bold = k % 2 == 1
            run.italic = italic
            if size:
                run.font.size = Pt(size)
        if align == "center":
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        return para

    def bullet(self, *parts):
        return self.p(*parts, style="List Bullet")

    def mono(self, text, size=8.5):
        para = self.d.add_paragraph()
        para.paragraph_format.left_indent = Inches(0.2)
        para.paragraph_format.space_after = Pt(4)
        run = para.add_run(text)
        run.font.name = "Consolas"
        run.font.size = Pt(size)
        return para

    def table(self, header, data, widths, size=None):
        t = self.d.add_table(rows=1, cols=len(header))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for j, htext in enumerate(header):
            c = t.rows[0].cells[j]
            c.text = ""
            r = c.paragraphs[0].add_run(htext)
            r.bold = True
            if size:
                r.font.size = Pt(size)
            shd = OxmlElement("w:shd")
            shd.set(qn("w:val"), "clear")
            shd.set(qn("w:color"), "auto")
            shd.set(qn("w:fill"), "D9E2F3")
            c._tc.get_or_add_tcPr().append(shd)
        for row in data:
            cells = t.add_row().cells
            for j, val in enumerate(row):
                cells[j].text = ""
                r = cells[j].paragraphs[0].add_run(str(val))
                if size:
                    r.font.size = Pt(size)
        last = len(t.rows) - 1
        for k, row in enumerate(t.rows):
            for j, w in enumerate(widths):
                row.cells[j].width = Inches(w)
            for c in row.cells:
                for para in c.paragraphs:
                    para.paragraph_format.space_after = Pt(1)
                    # keep the table on one page (and never strand the header row)
                    para.paragraph_format.keep_with_next = k < last
        self.d.add_paragraph().paragraph_format.space_after = Pt(0)
        return t

    def figure(self, path, width, caption, cap_size=9):
        self.d.add_picture(path, width=Inches(width))
        self.d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        self.d.paragraphs[-1].paragraph_format.keep_with_next = True
        self.p(caption, align="center", size=cap_size, italic=True)

    def save(self, path):
        cp = self.d.core_properties
        cp.author = cp.last_modified_by = "Aditya"
        cp.comments = cp.keywords = cp.subject = ""
        cp.title = self.d.paragraphs[0].text
        self.d.save(path)
        print("wrote", path)


# ----------------------------------------------------------------------------- shared text
pilot_rows = []
for lr in sorted(SWEEP, key=float):
    s = SWEEP[lr]
    if s["steps_completed"] < 300:
        fp16 = f"NaN at step {s['steps_completed']}"
    else:
        fp16 = f(s["final_val_loss"], 3)
    fp32 = f(SWEEP32[lr]["final_val_loss"], 3) if lr in SWEEP32 else "-"
    mg = "inf" if s["max_grad_norm"] == float("inf") else f(s["max_grad_norm"], 1)
    pilot_rows.append((f"{float(lr):g}", fp16, mg, fp32))

story_runs = STORIES["required"]


def story_excerpt(r, n=330):
    t = r["text"].replace("\n", " ")
    return t[:n].rsplit(" ", 1)[0] + " ..."


# ============================================================================= WRITEUP (<= 5 pages)
def build_writeup():
    D = Doc(base_pt=9.5, margins=0.7)
    D.h("Small Language Model from Scratch: Writeup", 1)
    D.p(f"TinyStories V2 (GPT-4). Model: {i(NP)} parameters, 4 layers, d_model 512, 16 heads, d_ff 1,344, RoPE, SwiGLU, RMSNorm, "
        f"10,000-token byte-level BPE. Trained on a single GPU in fp16 (autocast + GradScaler, no bf16, so the "
        f"same code runs on a Colab/Kaggle T4). Final validation loss {f(MAIN['final_val_loss'], 4)}, perplexity {f(MAIN['final_val_perplexity'], 2)}. "
        f"All {48} pytest tests pass. Every number below comes from the scripts in scripts/ and the logs in runs/.")

    D.h("Part 1-2: Tokenizer", 2)
    D.p("**Warm-ups. **",
        f"(1) chr(0) is the NUL character: repr shows {W['chr0_repr']} (so 'a'+chr(0)+'b' is shown as {W['chr0_in_string_repr']}) while print writes an invisible "
        f"control character, so the string looks like 'ab' although its length is 3. "
        f"(2) UTF-8 is the most compact for our text and needs only 256 base symbols: \"hello\" is 5 bytes in UTF-8 vs 10/20 in UTF-16/32 (12/24 with BOM). "
        f"It is not always the smallest: \"नमस्ते\" is 18 bytes in UTF-8 vs 12 in UTF-16. "
        f"(3) Multi-byte characters cannot be decoded byte by byte: 'héllo' is b'h\\xc3\\xa9llo' and bytes([0xc3]).decode() raises \"unexpected end of data\". "
        f"(4) b'\\xc3\\x28' is invalid: 0xC3 starts a 2-byte character, so the next byte must look like 10xxxxxx, but 0x28 is 00101000.")
    D.p("**Training. **",
        f"I trained on the full {f(ENC_TRAIN['bytes'] / 1e9, 2)} GB training file (no need to cut to a quarter): pre-tokenisation in {TOK['workers']} processes "
        f"(chunks cut at <|endoftext|>) took {f(TOK['pretokenize_seconds'], 1)} s and the {i(TOK['num_merges'])} merges {f(TOK['merge_seconds'], 1)} s, "
        f"{f(TOK['total_seconds'] / 60, 1)} min in total, peak memory {i(TOK['peak_memory_mb'])} MB (all processes). "
        f"Longest token: \"{TOK['longest_token']}\" ({TOK['longest_token_bytes']} bytes). It makes sense: a long, fairly common word in "
        f"moral-of-the-story sentences, with its leading space. The vocabulary is exactly 10,000 (256 bytes + <|endoftext|> + 9,743 merges). "
        f"Tests: the low/lower/widest/newest example gives s t, e st, o w, l ow, w est, n e; no token mixes two stories; "
        f"\"the cat ate\" encodes to [9, 7, 1, 5, 10, 3]; English/Hindi/emoji/empty strings round-trip; encode_iterable equals encode.")
    D.p("**Q1 (cProfile). ** ",
        f"On the validation file (single process) merging takes {f(P1v['cumulative_s']['merge_pairs'], 1)} s of the profiled "
        f"{f(P1v['profiled_total_s'], 1)} s, pre-tokenisation {f(P1v['cumulative_s']['pretokenize_file'], 1)} s. Almost all merge time is the "
        f"max() that finds the best pair ({f(P1v['cumulative_s']['builtins.max (finding the best pair)'], 1)} s), not the incremental updates "
        f"(_merge_word {f(P1v['cumulative_s']['_merge_word'], 2)} s). But pre-tokenisation grows with the file size while merging grows with the number "
        f"of unique pre-tokens: on the first 200 MB they are already close ({f(P1t['cumulative_s']['pretokenize_file'], 1)} s vs "
        f"{f(P1t['cumulative_s']['merge_pairs'], 1)} s), so on the full 2.2 GB file a single process would spend most of its time splitting. "
        f"That is why I parallelised pre-tokenisation.")
    D.p("**Encoding. ** ",
        f"Compression on 10 random validation stories: {f(T['compression_10_random_stories_bytes_per_token'], 2)} bytes/token "
        f"({f(ENC_TRAIN['bytes_per_token'], 2)} on the whole training file). Speed: {f(T['single_process_throughput_bytes_per_s'] / 1e6, 2)} MB/s in one "
        f"process; the training file ({i(ENC_TRAIN['tokens'])} tokens) took {f(ENC_TRAIN['seconds'], 0)} s with {ENC_TRAIN['workers']} processes; "
        f"validation has {i(ENC_VALID['tokens'])} tokens. "
        f"**Q2.** IDs go up to 9,999 and uint16 holds 0-65,535, so 2 bytes per token is enough: train.npy is 1.08 GB instead of 2.2 GB "
        f"(int32) or 4.3 GB (int64), and the memory-mapped batches read half as much. It stops being enough when the vocabulary passes 65,536 tokens "
        f"(e.g. a 100k-token tokenizer needs uint32).")

    D.h("Part 3-4: Model", 2)
    D.p("Tests: my hand-written softmax(QKᵀ/√d_k)V with a -inf causal mask matches F.scaled_dot_product_attention to 1e-5 for 3-D and 4-D inputs; "
        "RoPE matches a slow version that builds the full rotation matrix; all pieces keep their input shape with extra batch dims; logits are "
        "(2, 17, 10000); changing tokens after position t leaves outputs up to t bit-identical; the loss at initialisation is ln(10,000) ≈ 9.21 "
        "(nearly uniform predictions); the parameter count equals my hand count.")
    D.p("**Q3. ** ",
        f"With d_k = 32 and unit-variance q, k, the raw scores have std {f(q3['score_std_raw'], 2)} (≈√32) vs {f(q3['score_std_scaled'], 2)} after dividing. "
        f"Without the division the softmax is nearly one-hot (mean largest weight {f(q3['mean_max_prob_raw'], 2)}, entropy {f(q3['mean_entropy_raw_nats'], 2)} "
        f"nats out of {f(q3['max_entropy_nats_8_keys'], 2)}); with it, attention stays spread out ({f(q3['mean_max_prob_scaled'], 2)}, "
        f"{f(q3['mean_entropy_scaled_nats'], 2)} nats). A saturated softmax also has tiny gradients: the mean p(1-p) was "
        f"{f(q3['mean_softmax_jacobian_diag_raw'], 3)} vs {f(q3['mean_softmax_jacobian_diag_scaled'], 3)}.")
    D.p("**Q4. ** ",
        f"Same q and k (d_head 32) placed at different positions: the dot product is {q4['(3,0) i-j=3']} for (i, j) = (3,0), (10,7), (100,97) and (250,247); "
        f"{q4['(5,0) i-j=5']} for both pairs with i-j = 5; {q4['(7,7) i-j=0']} (the unrotated value) when i = j; and {q4['(0,3) i-j=-3']} for i-j = -3. "
        f"RoPE rotates q by angle iθ and k by jθ per pair, and the dot product of two rotated vectors only sees the difference (i-j)θ.")
    D.p("**Q5. ** ",
        f"Embedding {i(q5['token_embedding'])}; per layer {i(q5['per_layer_total'])} = attention {i(q5['per_layer_attention_qkvo'])} (4·512²) + SwiGLU "
        f"{i(q5['per_layer_ffn_swiglu'])} (3·512·1344) + 2 RMSNorm {i(q5['per_layer_rmsnorms'])}; 4 layers {i(q5['all_layers'])}; final norm 512; output layer "
        f"{i(q5['lm_head'])}. Total {i(q5['total'])}, equal to sum(p.numel()). Embedding + output = {f(100 * q5['embedding_plus_output_share'], 1)}% of the model.")
    D.table(["Matmul (one forward pass, T tokens)", "FLOPs, ours (T=256)", "FLOPs, GPT-2 small (T=1024)"], [
        ["q, k, v, o projections, per layer: 4·2·T·d²", i(q6["per_layer_flops"]["q,k,v,o projections (4 x 2*T*d*d)"]), i(q7["per_layer_flops"]["q,k,v,o projections (4 x 2*T*d*d)"])],
        ["QKᵀ scores, per layer: 2·T²·d", i(q6["per_layer_flops"]["attention scores QK^T (2*T*T*d)"]), i(q7["per_layer_flops"]["attention scores QK^T (2*T*T*d)"])],
        ["attention × V, per layer: 2·T²·d", i(q6["per_layer_flops"]["attention-weighted values AV (2*T*T*d)"]), i(q7["per_layer_flops"]["attention-weighted values AV (2*T*T*d)"])],
        ["feed-forward, per layer: 3·2·T·d·d_ff", i(q6["per_layer_flops"]["feed-forward (3 x 2*T*d*d_ff)"]), i(q7["per_layer_flops"]["feed-forward (3 x 2*T*d*d_ff)"])],
        ["output layer: 2·T·d·V", i(q6["lm_head (2*T*d*V)"]), i(q7["lm_head (2*T*d*V)"])],
        ["total (×4 layers / ×12 layers + output)", f"{f(q6['total_forward_flops'] / 1e9, 2)} G", f"{f(q7['total_forward_flops'] / 1e9, 1)} G"],
        ["share: FFN / output / projections / attention", "44.4% / 27.5% / 22.5% / 5.6%", "39.8% / 27.1% / 19.9% / 13.3%"],
    ], [3.3, 1.6, 2.0], size=8.5)
    D.p("**Q6. ** ", f"The feed-forward layers do the most work (44%), then the output layer (27.5%); the attention score/value products are only 5.6%. "
        "**Q7. ** ", f"GPT-2 small (12 layers, d 768, 12 heads, V 50,257, T 1,024; I used a SwiGLU FFN with d_ff 2,048, which costs the same as GPT-2's 2·768·3,072 GELU MLP) "
        f"needs {f(q7['total_forward_flops'] / 1e9, 1)} GFLOPs. Attention scores grow with T², so inside a layer their share rises from "
        f"{f(100 * q6['attention_scores_share_within_a_layer'], 1)}% to {f(100 * q7['attention_scores_share_within_a_layer'], 1)}%, while the FFN share falls. "
        "The output layer stays at about 27% because the vocabulary grew 5× while the model width grew only 1.5×.")

    D.h("Part 5: Training setup", 2)
    D.p("**Learning-rate warm-up (SGD on mean(w²), w = 5·N(0,1) 10×10). ** ",
        f"The gradient is w/50, so each step multiplies w by (1 − lr/50). lr 1: ×0.98, loss falls slowly (26.3 → 18.3). lr 10: ×0.8, loss ×0.64 per step "
        f"(26.3 → 0.47). lr 100: ×(−1), w just flips sign, loss stays exactly 26.27. lr 1000: ×(−19), loss ×361 per step, 2.7e24 after 10 steps.")
    D.p(f"Tests: loss = hand-computed −log softmax; the two AdamW groups (weight matrices with decay 0.1, RMSNorm weights without) hold every "
        f"parameter exactly once; the schedule matches hand values (e.g. 5.5e-4 half way through the cosine); after clipping the norm is ≤ 1.0; "
        f"targets = inputs shifted by one and no slice runs past the end; save → 5 steps → reload → same 5 steps gives losses within 1e-3 "
        f"(CPU fp32 and GPU fp16); the fp16 loss is within 0.01 of fp32, and the gradient norm train_step reports equals the fp32 norm "
        f"(within 2%) although the loss was scaled by {i(scales[0])}, which proves scaler.unscale_ runs before clipping. "
        f"**torch.compile:** {f(BENCH['eager_ms_per_step'], 1)} → {f(BENCH['compiled_ms_per_step'], 1)} ms per step "
        f"({f(BENCH['speedup'], 2)}× faster, 300 timed steps, fp16) after ~{f(BENCH['first_steps_seconds_compiled_incl_compilation'] - BENCH['first_steps_seconds_eager'], 0)} s of compilation.")
    D.p("**Q8. ** ", f"fp32 parameters {f(q8['params_MB'], 1)} MB + gradients {f(q8['grads_MB'], 1)} MB + AdamW state {f(q8['adamw_m_and_v_MB'], 1)} MB "
        f"= {f(q8['total_MB_no_activations'], 1)} MB before activations. AdamW keeps two running averages per weight (first moment m and second "
        "moment v), each the same size as the weights, so its state is twice the weights.")

    D.h("Part 6: Training script checks and learning rate", 2)
    D.figure(FIG("single_batch.png"), 3.6, f"Single batch: training loss {f(sb_first, 2)} → {f(sb_last, 4)} in 300 steps while held-out loss rises to {f(sb_val_end, 2)}.")
    D.figure(FIG("resume_check.png"), 6.2, f"Stopped at step 300 and resumed: at step 600 validation loss is {rf['600']['val_loss']} (uninterrupted) vs "
             f"{rs['600']['val_loss']} (resumed); the learning rate and batches are identical (the checkpoint stores model, optimizer, scheduler, scaler and RNG states).")
    D.p(f"The curves overlap with no jump. The small differences (largest single-step training-loss gap {f(res_diff, 3)}) already exist before step 300 "
        f"({rf['300']['train_loss']} vs {rs['300']['train_loss']}), so they come from non-deterministic GPU kernels, not from resuming. "
        f"**Gradient norm:** in the first 100 steps of the main run it stayed between {f(min(gn100), 2)} and {f(max(gn100), 2)} (peak during warm-up, "
        f"then ~0.4-0.7), never NaN and not growing; after step 1,000 it stays in {f(min(gn_late), 2)}-{f(max(gn_late), 2)}. GradScaler skipped {skipped} steps.")
    D.figure(FIG("lr_sweep.png"), 6.6, "Seven 300-step pilots (batch 32, context 256, 5% warm-up, cosine to 10% of the peak), fp16.")
    D.table(["Peak lr", "Val. loss at 300 (fp16)", "Max grad norm", "Same run in fp32"], pilot_rows, [1.0, 1.9, 1.3, 1.5], size=8.5)
    D.p("I spaced rates evenly on a log scale (×3 or ×3.3 apart) to cover 1e-4 to 1e-2 as asked, and added 3e-2 and 1e-1 because 1e-2 was unstable "
        "but did not blow up. **Choice: 3e-3** (lowest pilot loss). "
        "**Q9. ** ", f"The first rate that blew up (3e-2, NaN at step 22) is 10× my best rate; 1e-2, only 3.3× above it, already had gradient-norm spikes "
        "up to 47 and a much worse loss. So the best rate does sit just below the region where training breaks, roughly one ×3 step below the first "
        "unstable rate. I re-ran the broken rates in fp32: there they spike (1e-1 reaches a loss of 21.7, far above the 9.2 of a random guess) but do not "
        "turn into NaN and recover to a poor loss. So the instability is real; the NaN itself is the fp16 range (max 65,504) overflowing once the weights "
        "grow, with the GradScaler unable to help because the inf appears in the forward pass.")

    D.h("Part 7: Main run and stories", 2)
    D.figure(FIG("main_run.png"), 6.6, "Main run: 5,000 steps × 32 × 256 = 40,960,000 tokens, peak lr 3e-3, 250 warm-up steps, cosine to 3e-4 exactly at the last step.")
    D.p(f"**Final validation loss {f(MAIN['final_val_loss'], 4)}, perplexity {f(MAIN['final_val_perplexity'], 2)}** (50 fixed validation batches = 409,600 held-out tokens; "
        f"mean training loss over the last 100 steps {f(train_last100, 3)}, so no overfitting: the model saw only 7.6% of the training tokens once). "
        f"Training took {f(active_s, 0)} s (7.0 min) including {f(compile_s, 0)} s of compilation. The raw log says {f(MAIN['train_seconds'], 0)} s because my computer "
        f"went to sleep for {f(standby, 0)} s during one step; the run continued correctly afterwards.")
    D.p("**Stories** (prompt \"Once upon a time\"; full texts in report/stories.txt). Left alone the model ends its stories after a median of "
        f"{f(nat['T=0.7,p=0.9']['median'], 0)} tokens ({nat['T=0.7,p=0.9']['min']}-{nat['T=0.7,p=0.9']['max']} over 20 seeds), so for the required ≥256-token stories "
        "<|endoftext|> was blocked until 256 tokens (--min-new-tokens).")
    for r in story_runs:
        D.bullet(f"**T={r['temperature']}, top-p={r['top_p']}, seed {r['seed']} (would stop at {r['natural_stop_tokens']}): **", story_excerpt(r, 260))
    D.p("**Q10. ** ", "Sentence-level fluency is very good: correct grammar, punctuation and dialogue, and character names stay consistent. "
        "Two things change the quality. (1) Length: once the story is forced past its natural ending, the model repeats \"The moral of the story is...\" "
        "lines (seeds 1 and 2 at T=0.7) because a TinyStories story usually ends there. (2) Temperature/top-p: at 0.7/0.9 the plots are safe but generic "
        "(Lily, a toy, a park); at 1.0/0.95 there is more variety but also more slips (\"a man selling ice cream in the house\"). "
        "World logic is the weakest part at every setting (\"Lily gave the cake to her mom\" right after eating it).")

    D.h("Part 8: Packaging", 2)
    D.p("release/ holds model.pt (weights only, 90.8 MB), config.json, vocab.json and merges.txt. inference.py imports only model/ and tokenizer/, rebuilds "
        "the model from config.json, loads the weights with map_location, and prints the story and a tokens/s line. Acceptance test: I copied exactly the "
        "files git would commit into a new folder, created a fresh venv, installed only requirements.txt (torch, regex, einops, numpy) and ran the README "
        "command on the CPU. It printed the same story as my development environment, and two runs with --seed 42 were identical.")
    D.p("**Q11. ** ", f"On the CPU, generating 256 tokens runs at 25.3-26.3 tokens/s "
        f"(30-34 tokens/s for the natural 148-token story). Without a cache, step k re-runs the model over all min(4+k, 256) tokens: "
        f"{i(KV['positions_nocache'])} token positions for 256 new tokens instead of {KV['positions_cache']}, so {f(100 * (1 - KV['positions_cache'] / KV['positions_nocache']), 1)}% "
        f"of the work is repeated. I measured a forward pass at {f(KV['forward_ms_by_len']['1'], 1)} ms for 1 token and {f(KV['forward_ms_by_len']['256'], 1)} ms for 256 tokens; "
        f"summing these predicts {f(KV['est_s_nocache'], 1)} s without a cache (measured ~9.8 s) and about {f(KV['est_s_cache'], 1)} s with one, roughly 3.8× faster (~75% of the time saved). "
        "It is not 99% because each step still has a fixed cost (reading all weights, the 10,000-way output layer, Python overhead). "
        "This is an estimate; I did not build the cache.")

    D.h("Bugs I found vs. limits of the model", 2)
    D.p("**Bugs (fixed):** (1) my first story-boundary test passed even without splitting on <|endoftext|>, which I found by running the training without the "
        "split; I rewrote it. (2) Resuming on the GPU crashed because map_location='cuda' moved the saved RNG states to the GPU (torch.set_rng_state needs a CPU "
        "tensor); the fp16 resume test caught it. (3) torch.compile failed on Windows without Triton; fixed with triton-windows (Colab/Kaggle on Linux need nothing). "
        "**Not bugs:** the NaNs at lr ≥ 3e-2 (fp16 overflow after real divergence, see Q9); the 631 s gap in the main-run log (the computer went to sleep); "
        "repeated morals, generic plots and logic slips (limits of a 22.7M model trained on 41M tokens for 7 minutes).")
    D.save(R("report", "writeup.docx"))


# ============================================================================= PROJECT REPORT (like the example)
def build_project_report():
    D = Doc(base_pt=11, margins=1.0)
    D.h("My Understanding of the Small Language Model Assignment", 1)
    D.p("Everything I built is in this repository: the tokenizer, the model, the training code, the tests, the release folder and this report.")

    D.h("What this project was", 2)
    D.p("In this assignment I built a small language model, a tiny relative of ChatGPT, completely from scratch. It learns from TinyStories, a collection "
        "of very simple stories written for young children, and then writes new stories on its own.")
    D.p("The main parts are the tokenizer (which turns text into numbers), the Transformer (the model itself), the training script, and the code that "
        "generates text. I was allowed to use PyTorch for the basic maths, but not ready-made tokenizers or ready-made Transformer models.")

    D.h("The short result", 2)
    D.bullet(f"My model has {i(NP)} learned numbers (parameters), about 22.7 million.")
    D.bullet(f"It trained for about 7 minutes on a single GPU, using fp16 (half precision), the same setting a free Colab or Kaggle T4 GPU uses.")
    D.bullet(f"Its final score on stories it never trained on (validation loss) is {f(MAIN['final_val_loss'], 4)}. The stretch target was 2.0 or lower, and lower is better, so I beat it.")
    D.bullet("It writes stories that read well sentence by sentence, but the plots are simple and it repeats itself when a story is forced to be long.")
    D.bullet("Anyone can run it on a normal laptop with one command, no GPU needed. I checked this in a brand new Python environment.")
    D.p("The rest of this document explains what I understood at each step, what I measured, and what went wrong along the way.")

    D.h("How I worked", 2)
    D.p("**Where I did what. **", "I wrote and tested all the code on my own computer in a Python virtual environment and trained the model on its GPU. "
        "That GPU supports bf16, a newer number format, but I deliberately did not use it, because the free T4 GPUs on Colab and Kaggle do not support it well. Everything uses fp16 with a gradient scaler, or plain fp32, so the same code runs "
        "on those free GPUs too. I also added a small notebook (notebooks/colab_kaggle.ipynb) that only runs commands; all the real code lives in .py files, as the assignment requires.")
    D.p("**Tests first. **", "For every piece I wrote a pytest test before moving on. The tests check things I can predict in advance, such as shapes, a hand-worked "
        "answer, or that the model cannot see the future. By the end I had 48 passing tests (two of them need a GPU and are skipped on a computer without one).")
    D.p("**The data. **", "I used TinyStories V2 (GPT-4). The training file is about 2.2 GB and the validation file about 22 MB. Stories are separated by the "
        "text <|endoftext|>. I did all my early debugging on the small validation file and only moved to the big file once everything worked.")

    D.h("Key numbers in one table", 2)
    D.table(["What", "My result"], [
        ["Vocabulary size", "10,000 tokens (256 bytes + <|endoftext|> + 9,743 merges)"],
        ["Tokenizer training", f"full 2.2 GB file, {f(TOK['total_seconds'] / 60, 1)} minutes with {TOK['workers']} processes, peak memory {f(TOK['peak_memory_mb'] / 1024, 1)} GB"],
        ["Longest token", f"\"{TOK['longest_token']}\" ({TOK['longest_token_bytes']} bytes)"],
        ["Compression", f"{f(ENC_TRAIN['bytes_per_token'], 2)} bytes per token ({f(T['compression_10_random_stories_bytes_per_token'], 2)} on 10 random stories)"],
        ["Training tokens in the file", i(ENC_TRAIN["tokens"])],
        ["Validation tokens in the file", i(ENC_VALID["tokens"])],
        ["Model size", f"{i(NP)} parameters"],
        ["Model shape", "4 layers, 512 wide, 16 heads, feed-forward size 1,344"],
        ["Best learning rate in pilots", "3e-3"],
        ["torch.compile speed-up", f"{f(BENCH['speedup'], 2)}× ({f(BENCH['eager_ms_per_step'], 0)} → {f(BENCH['compiled_ms_per_step'], 0)} ms per step)"],
        ["Main run", f"5,000 steps, about 41 million tokens, {f(active_s, 0)} seconds of training"],
        ["Final validation loss", f(MAIN["final_val_loss"], 4)],
        ["Perplexity", f(MAIN["final_val_perplexity"], 2)],
        ["Speed on a CPU", "about 26 tokens per second for 256 tokens"],
        ["Size of the saved model", "about 91 MB"],
    ], [2.2, 4.3])

    D.h("What I understood, step by step", 2)
    D.h("1. The tokenizer: turning text into numbers", 3)
    D.p("A model can only work with numbers, so text has to be cut into pieces and every piece gets a number. If each letter is a piece, sentences become very "
        "long. If each word is a piece, the list of words is huge and any new word breaks everything.")
    D.p("BPE (byte pair encoding) sits in the middle. I start with 256 pieces, one for every possible byte, plus the special end-of-story token. Then I repeat one "
        "simple step: find the two neighbouring pieces that appear together most often and glue them into one new piece. After 9,743 rounds I had 10,000 pieces. "
        "The very first merges were \" t\", \"he\", \" a\", \" s\" and \" w\", and soon whole words like \" the\" and \" and\" became single pieces.")
    D.p("Three things I learned here. First, I must split the text at the end-of-story marker before counting, or the tokenizer learns pieces of the marker itself "
        "and pieces that join two stories. Second, recounting every pair after every merge is far too slow, so after each merge I only fix the counts for the words "
        "that actually contained that pair. Third, splitting 2.2 GB of text into words is slow in one Python process, so I cut the file into chunks (always at an "
        "end-of-story marker) and counted them in 12 processes at the same time. That let me use the whole training file instead of only a quarter of it.")
    D.p("I checked my code with the small example from the assignment (low, lower, widest, newest). It gave the expected first merges: s t, e st, o w, l ow, w est, "
        "n e. The first one is a tie between (s, t) and (e, s), and the rule is to pick the bigger pair, which is (s, t).")
    D.h("2. Using the tokenizer and preparing the data", 3)
    D.p("Encoding replays the learned merges, in the order they were learned, on any new text. Decoding does the reverse. The best test is that encoding and then "
        "decoding gives back exactly the same text, including Hindi, emoji and an empty string. That worked. The end-of-story marker always becomes exactly one number.")
    D.p(f"I then turned both story files into long lists of numbers. The training file became {i(ENC_TRAIN['tokens'])} tokens and the validation file "
        f"{i(ENC_VALID['tokens'])}. I stored them as 2-byte numbers (uint16), because my largest ID is 9,999 and uint16 goes up to 65,535. "
        f"The training file took {f(ENC_TRAIN['seconds'], 0)} seconds to encode with 12 processes.")
    D.h("3. The building blocks of the model", 3)
    D.bullet("**Position (RoPE). **", "The model reads all the words at once, so by itself it does not know their order. RoPE gives each word a position by rotating "
             "pairs of its numbers by an angle that depends on where the word is. The nice result is that the match between two words then depends only on how far apart they are.")
    D.bullet("**Attention. **", "Each word looks back at the earlier words and decides which ones matter for predicting the next word. A mask stops it from looking at "
             "future words, otherwise it could just copy the answer. I first wrote this the slow, plain way and checked that it matched PyTorch's fast version to 0.00001.")
    D.bullet("**Feed-forward layer (SwiGLU). **", "After attention mixes information between words, this layer works on each word separately. It uses three weight "
             "matrices and the SiLU function, which is the design most modern models use.")
    D.h("4. The full model", 3)
    D.p("A block is attention plus feed-forward, each one added back to its input (so the original information is never lost), with a normalising step (RMSNorm) "
        "before each. I stacked 4 blocks. At the start there is a table that turns each token number into 512 numbers, and at the end a layer that gives a score "
        "for each of the 10,000 possible next tokens.")
    D.p("Two checks mattered most. The model must not see the future: when I changed the later tokens, the earlier outputs stayed exactly the same, bit for bit. "
        "And before any training the loss should be about 9.21, because guessing evenly between 10,000 tokens gives exactly ln(10,000). It was.")
    D.h("5. Setting up training", 3)
    D.p("This part had many small pieces, and a mistake in any of them can stay hidden because the loss still goes down a little. So I tested each one:")
    for b in ["the loss against a hand calculation", "the optimizer, so every weight is in exactly one group (weight decay only on the big weight matrices)",
              "the learning rate, which climbs for the first 5% of steps and then falls along a cosine curve", "gradient clipping, which stops one huge update from wrecking the model",
              "mixed precision (fp16), which is faster but needs care", "batches, where the target is the input shifted by one token",
              "saving and loading, so a stopped run can continue exactly where it stopped"]:
        D.bullet(b)
    D.p("One detail I learned: with fp16 the loss is multiplied by a big number (65,536 at the start) so that small gradients do not round to zero. The gradients "
        "must be divided back down (unscaled) before clipping, otherwise the clipping works on numbers that are 65,536 times too big. My test checks that the gradient "
        "size reported during fp16 training equals the real fp32 gradient size.")

    D.h("The tests and checks, in detail", 2)
    D.p("A test is only useful if it would fail when the code is wrong. This table lists what my tests prove for each part.")
    D.table(["Part", "What the tests check"], [
        ["Tokenizer training", "The hand-worked example gives s t, e st, o w, l ow, w est, n e. No learned token contains bytes from two stories or from the end-of-story marker (I confirmed the test fails if the split is removed). The vocabulary has exactly the size asked for, with no repeated IDs. Parallel and single-process training give the same result."],
        ["Tokenizer use", "\"the cat ate\" gives 9, 7, 1, 5, 10, 3. Encode then decode returns the same text for English, Hindi, emoji, spaces and an empty string. The end-of-story marker is always one ID; the longer special token wins when two overlap. Reading a file in small pieces gives the same IDs as reading it whole."],
        ["Attention", "My slow version matches PyTorch's within 0.00001 for 3-D and 4-D inputs. Changing later keys never changes earlier outputs."],
        ["RoPE", "Matches a slow version that builds the full rotation matrix. Keeps the length of vectors. Position 0 is unchanged. Dot products depend only on the distance."],
        ["Full model", "Output shape is (2, 17, 10000). The model cannot see the future. The starting loss is close to 9.21. The parameter count matches my hand count."],
        ["Loss and optimizer", "The loss matches a hand calculation. Every parameter is in exactly one group, and the no-decay group is exactly the RMSNorm weights."],
        ["Learning rate", "The schedule matches values I worked out by hand at several steps, and LambdaLR follows it."],
        ["Clipping", "After clipping, the total gradient size is at most 1.0."],
        ["Batches", "Targets are the inputs shifted by one, and no slice runs past the end of the data."],
        ["Checkpoints", "Save, train 5 steps, reload, train the same 5 steps: losses match within 0.001, on CPU and on the GPU in fp16."],
        ["fp16", "The fp16 loss is within 0.01 of the fp32 loss, and gradients are unscaled before clipping (GPU only)."],
        ["Generation", "Greedy decoding gives the same output every time; top-p keeps the smallest set that reaches p; the same seed gives the same story; generation stops at the end-of-story token."],
    ], [1.6, 4.9], size=9.5)

    D.h("Three checks before the real run", 2)
    D.p("A training script that is slightly wrong can waste a whole day, so I ran three checks with the real model first.")
    D.p("**1. Learning one batch. **", f"I trained on the same single batch again and again. A working model should memorise it. The loss went from {f(sb_first, 2)} "
        f"to {f(sb_100, 4)} in 100 steps and {f(sb_last, 4)} after 300. At the same time the loss on the validation stories rose to {f(sb_val_end, 2)}, which is "
        "expected, because the model was learning one batch by heart. This also shows that the validation data really is separate from the training data.")
    D.figure(FIG("single_batch.png"), 5.0, "The model memorises one batch (log scale).")
    D.p("**2. Stopping and resuming. **", f"I trained 600 steps without stopping, and separately stopped a run at step 300 and resumed it from the saved checkpoint. "
        f"The checkpoint also stores the random number generators, so the resumed run picks exactly the same batches. At step 600 the validation loss was "
        f"{rf['600']['val_loss']} for the uninterrupted run and {rs['600']['val_loss']} for the resumed one, and the learning rate was identical at every step. "
        "The tiny differences were already there before the stop, so they come from the GPU, not from resuming.")
    D.figure(FIG("resume_check.png"), 6.3, "The resumed run (dashed) lies on top of the uninterrupted run.")
    D.p("**3. Gradient size. **", f"In the first 100 steps of the main run the gradient norm stayed between {f(min(gn100), 2)} and {f(max(gn100), 2)}. It rose while "
        "the learning rate was warming up and then settled at about 0.5. It was never invalid (NaN) and did not keep growing. Over the whole run the fp16 "
        "gradient scaler never had to skip a step.")
    D.figure(FIG("grad_norm.png"), 6.3, "Gradient norm before clipping.")

    D.h("Finding a good learning rate", 2)
    D.p("The learning rate is the size of each step the model takes while learning. Too small and it learns very slowly; too big and it jumps around and breaks. "
        "A tiny warm-up experiment showed this clearly. Minimising the average square of 100 numbers, a rate of 1 made slow progress, 10 worked well, 100 made the "
        "numbers flip sign every step without the loss changing at all, and 1,000 made the loss explode to about 10²⁴ in 10 steps.")
    D.p("For the real model I ran seven short tests of 300 steps each. To make sure at least one would break, I went beyond the 1e-4 to 1e-2 range the "
        "assignment asked for.")
    D.figure(FIG("lr_sweep.png"), 6.5, "My 300-step pilot runs, batch 32, context 256, fp16.")
    D.table(["Peak learning rate", "Validation loss after 300 steps (fp16)", "Largest gradient norm", "Same run in fp32"], pilot_rows, [1.3, 2.3, 1.4, 1.5], size=9.5)
    D.p("The best rate was 3e-3, with 1e-3 close behind (0.09 apart). At 1e-2 training was already shaky: the gradient norm jumped to 47 and the loss ended much "
        "worse. 3e-2 and 1e-1 broke completely, and the loss turned into NaN (not a number).")
    D.p("I wanted to know whether the NaN was caused by fp16 or by the rate itself, so I ran the broken rates again in fp32. In fp32 there was no NaN, but the "
        "loss still shot up (to 21.7 for 1e-1, much worse than a random guess) and then crawled back to a poor value. So the rate really is too high in both "
        "formats. fp16 just turns the blow-up into NaN, because its largest number is 65,504 and the activations go past it once the weights get large.")
    D.figure(FIG("lr_fp16_vs_fp32.png"), 5.0, "The same high learning rates in fp16 (solid, ending in NaN) and fp32 (dashed).")

    D.h("The main training run", 2)
    D.p("I trained for 5,000 steps with the rate 3e-3. Each step used 32 pieces of text of 256 tokens, so the model saw 40,960,000 tokens in total. That is "
        "only about 7.6% of all the training text, so it never saw the same text twice. The learning rate climbed for the first 250 steps and then came down in a "
        "smooth curve, ending exactly at its lowest value (3e-4) on the last step. I used torch.compile, which made each step about 1.23 times faster.")
    D.figure(FIG("main_run.png"), 6.5, "Main run against steps and against time.")
    D.p(f"The final validation loss was {f(MAIN['final_val_loss'], 4)}, which is a perplexity of {f(MAIN['final_val_perplexity'], 2)}. Perplexity roughly means "
        "how many tokens the model is unsure between at each step, so it is about as unsure as choosing between 5 or 6 options. The training loss at the end "
        f"({f(train_last100, 3)}) was almost the same as the validation loss, so the model was not just memorising.")
    D.p(f"The training itself took {f(active_s, 0)} seconds (7 minutes), including about 30 seconds for torch.compile to prepare the model. My log first said "
        f"{f(MAIN['train_seconds'], 0)} seconds. When I looked, one single step had taken {f(standby, 0)} seconds. The system log showed that my computer "
        "had gone to sleep because I had left it idle, and woke up 11 minutes later. Training simply continued, and the loss was not affected.")

    D.h("Writing stories", 2)
    D.p("To write a story the model predicts one token, adds it to the text, and repeats. Two settings control it. Temperature controls how adventurous it is "
        "(low is safe, high is random). Top-p keeps only the most likely tokens that together cover a chosen share of the probability.")
    D.p(f"Left alone, the model ends its stories after {nat['T=0.7,p=0.9']['min']} to {nat['T=0.7,p=0.9']['max']} tokens (median "
        f"{f(nat['T=0.7,p=0.9']['median'], 0)} over 20 tries). The assignment asked for stories of at least 256 tokens, so for those I blocked the end-of-story token "
        "until 256 tokens were written. To be honest, this forces the model to keep going past where it wanted to stop.")

    D.h("The stories I got", 2)
    D.p("All stories start from \"Once upon a time\". Here are pieces of the real output, with what I noticed. The full texts are in report/stories.txt.")
    notes = [
        "The first part is a clear little story about a mysterious box with a magic toy, and the name Mia stays consistent. After its natural ending at 220 tokens it adds two \"The moral of the story is...\" lines, and one of them makes no sense.",
        "Lily finds a cake at the park and shares it. Each sentence is fine, but the facts drift: she \"gave the cake to her mom\" after eating it, and her mom says \"I'm glad you ate the cake\". After the natural end at 125 tokens it repeats the moral three times.",
        "More unusual words (\"a modest little town\", \"a funny man named Tom\") and a more varied plot, but also odd details such as \"a man selling ice cream in the house\" and a dog that appears from nowhere. It ends with \"The End.\" and then a repeated moral.",
    ]
    for r, note in zip(story_runs, notes):
        D.p(f"**Temperature {r['temperature']}, top-p {r['top_p']}, seed {r['seed']} ({r['new_tokens']} tokens). **", f"It begins: \"{story_excerpt(r, 190)}\" {note}")
    D.p("**The story from my release test. **", "With temperature 0.8, top-p 0.9 and seed 42 the model wrote a 148-token story about a boy named Tim and a red ball "
        "that starts to talk, and it ended by itself. This is the story the fresh-environment test printed, word for word the same both times.")
    D.p("**What I learned from reading them. **", "Lower temperature gives safer but more repetitive text. Higher temperature gives more variety but more mistakes. "
        "The model is good at the shape of a children's story (a name, a small problem, a happy ending, a moral) and weak at keeping the facts consistent. "
        "For 22.7 million parameters and 7 minutes of training, that seems normal to me, but I did not test a bigger model.")

    D.h("Packaging it so anyone can use it", 2)
    D.p("The release folder holds only four files: the weights (model.pt, about 91 MB, no optimizer state), config.json with the model settings, and the "
        "tokenizer files vocab.json and merges.txt. inference.py rebuilds the model from config.json, loads the weights so they work on a CPU even though they "
        "were trained on a GPU, and prints the story followed by one line with the number of tokens and the speed. It only imports my model, tokenizer and "
        "generation code, nothing from the training code.")
    D.p("For the acceptance test I copied exactly the files that git would commit into a brand new folder, made a fresh Python environment, installed only "
        "requirements.txt and ran the one command from the README:")
    D.mono('python inference.py --model-dir release --prompt "Once upon a time" --max-new-tokens 256 --temperature 0.8 --top-p 0.9 --seed 42 --device cpu')
    D.p("It worked and printed the same story as my development setup. Two small problems came up. The first try failed because Windows cannot install PyTorch "
        "in a folder whose path is too long; a shorter folder fixed it, so the problem was the computer, not the code. PyTorch also printed a warning that numpy "
        "was missing, so I added numpy to requirements.txt to keep the output clean.")

    D.h("My answers to the assignment questions", 2)
    D.h("The four warm-up questions", 3)
    D.bullet("**What is chr(0)? **", f"It is the null character. repr shows it as {W['chr0_repr']}, but print shows nothing, so 'a' + chr(0) + 'b' prints like \"ab\" although it has 3 characters.")
    D.bullet("**Why UTF-8? **", "\"hello\" is 5 bytes in UTF-8, 10 in UTF-16 and 20 in UTF-32 (12 and 24 with the byte-order mark). The others waste space on zeros, and "
             "UTF-8 always starts from the same 256 bytes, so no text is ever unknown. It is not always the smallest: \"नमस्ते\" is 18 bytes in UTF-8 and 12 in UTF-16.")
    D.bullet("**Why is decoding one byte at a time wrong? **", "Many characters use several bytes, and one byte from the middle of a character means nothing alone. "
             "\"héllo\" breaks, because é is the two bytes C3 A9 and C3 alone gives \"unexpected end of data\".")
    D.bullet("**An invalid two-byte sequence: **", "C3 28. The first byte says one more byte must follow that starts with the bits 10, but 28 starts with 00.")
    D.h("Q1. Where does the BPE training time go?", 3)
    D.p(f"On the small validation file, mostly in merging: {f(P1v['cumulative_s']['merge_pairs'], 1)} seconds out of {f(P1v['profiled_total_s'], 1)} in a profiled run, "
        f"against {f(P1v['cumulative_s']['pretokenize_file'], 1)} seconds for splitting into words. Inside merging almost all the time is spent finding the most common "
        f"pair with max() ({f(P1v['cumulative_s']['builtins.max (finding the best pair)'], 1)} seconds), not in updating the counts. But splitting grows with the size of "
        f"the file. On the first 200 MB of the training file the two were already close ({f(P1t['cumulative_s']['pretokenize_file'], 1)} versus "
        f"{f(P1t['cumulative_s']['merge_pairs'], 1)} seconds), so on the full file splitting would win. That is why I split the file into chunks and counted them in "
        "parallel. The profiler itself slows things down: the same validation run took 23 seconds without it.")
    D.h("Q2. Why store token IDs as uint16?", 3)
    D.p("My IDs are at most 9,999 and uint16 goes up to 65,535, so 2 bytes per token is enough. The training file became 1.08 GB instead of 2.2 GB (int32) or "
        "4.3 GB (int64). It would stop being enough if the vocabulary grew past 65,536 tokens.")
    D.h("Q3. Why does attention divide by the square root of d_k?", 3)
    D.p(f"With 32 numbers per head, the scores had a spread of {f(q3['score_std_raw'], 2)} without dividing and {f(q3['score_std_scaled'], 2)} after. Without dividing, "
        f"softmax gave on average {f(100 * q3['mean_max_prob_raw'], 0)}% of the attention to a single word; with it, {f(100 * q3['mean_max_prob_scaled'], 0)}%. When "
        "softmax nearly picks one word, very little learning signal passes back through it. Dividing keeps the scores in a healthy range.")
    D.h("Q4. Why does RoPE only care about distance?", 3)
    D.p(f"I put the same query and key at many positions. Whenever the two positions were 3 apart the answer was {q4['(3,0) i-j=3']}, at positions as different as "
        f"(3, 0) and (250, 247). Positions 5 apart always gave {q4['(5,0) i-j=5']}, and equal positions gave the plain unrotated value {q4['(7,7) i-j=0']}. Rotating both "
        "vectors by angles that depend on their positions leaves only the difference between the angles.")
    D.h("Q5. How many parameters does my model have?", 3)
    D.p(f"{i(NP)}. The embedding table and the output layer hold 5,120,000 each, so together they are {f(100 * q5['embedding_plus_output_share'], 0)}% of the model. "
        f"Each of the 4 layers holds {i(q5['per_layer_total'])}. My count by hand matched the count from the code.")
    D.h("Q6. How much work is one forward pass?", 3)
    D.p(f"About {f(q6['total_forward_flops'] / 1e9, 2)} billion operations for 256 tokens. The feed-forward layers do the most (44%), then the output layer (27.5%), "
        "then the attention projections (22.5%). The attention scores themselves are only 5.6%.")
    D.h("Q7. What changes for a GPT-2 sized model?", 3)
    D.p(f"About {f(q7['total_forward_flops'] / 1e9, 1)} billion operations. Attention scores become a much bigger share inside a layer (from 7.8% to 18.2%), "
        "because they grow with the square of the text length. The feed-forward share goes down a little. The output layer stays at about 27%, because the "
        "vocabulary grows 5 times while the width grows only 1.5 times.")
    D.h("Q8. How much memory does training need?", 3)
    D.p(f"In 32-bit numbers: {f(q8['params_MB'], 1)} MB for the weights, {f(q8['grads_MB'], 1)} MB for the gradients and {f(q8['adamw_m_and_v_MB'], 1)} MB for the "
        f"optimizer, about {f(q8['total_MB_no_activations'], 0)} MB without activations. AdamW keeps two extra numbers for every weight (an average of the gradient "
        "and an average of its square), which is why it needs twice the memory of the weights.")
    D.h("Q9. The learning rate that broke versus the best one", 3)
    D.p("The best was 3e-3 and the first one that broke (NaN) was 3e-2, ten times bigger. 1e-2, only about three times bigger than the best, was already "
        "unstable. So yes, the best rate sits just below where training starts to break. The fp32 re-runs showed the breaking is real and not only an fp16 effect.")
    D.h("Q10. How good are the stories?", 3)
    D.p("Sentence by sentence they are good: correct grammar, steady character names and proper quotes. Two things changed the quality. The first is length: "
        "when I forced long stories, they repeated morals. The second is temperature and top-p: at 0.7 the text was safe but generic, at 1.0 odd details appeared. "
        "Keeping facts straight (who ate the cake) is the weakest part. I think that comes from the small model and short training.")
    D.h("Q11. How fast is it on a CPU, and what would a cache save?", 3)
    D.p(f"About 26 tokens per second (25.3 to 26.3) for 256 tokens on the CPU. Without a cache every new token makes the model read "
        f"the whole text again: {i(KV['positions_nocache'])} token positions for 256 new tokens instead of {KV['positions_cache']}, so about 99% of the work is "
        f"repeated. From timing single forward passes ({f(KV['forward_ms_by_len']['1'], 1)} ms for 1 token, {f(KV['forward_ms_by_len']['256'], 1)} ms for 256) I "
        f"estimate {f(KV['est_s_nocache'], 1)} seconds without a cache and about {f(KV['est_s_cache'], 1)} with one, roughly 4 times faster. It is not 99% faster "
        "because each step still has a fixed cost. This is an estimate; I did not build the cache.")

    D.h("Mistakes I made and what each one taught me", 2)
    D.p("The assignment asks me to separate real mistakes from the normal limits of a small model. These are the real mistakes I found and fixed.")
    D.bullet("**A test that could not fail. **", "My first test for \"no token crosses two stories\" failed for the wrong reason (both stories shared the space "
             "character). When I fixed that, it passed, but then I removed the end-of-story split on purpose and it still passed. I rewrote it so that it fails on "
             "broken code. Lesson: check that a test can fail.")
    D.bullet("**Resuming crashed on the GPU. **", "Loading a checkpoint straight onto the GPU also moved the saved random number states there, and PyTorch only "
             "accepts them on the CPU. The fp16 resume test caught it before any real run. Lesson: test the same path the real run uses.")
    D.bullet("**torch.compile did not work at first. **", "On Windows it needs an extra package (triton-windows). Colab and Kaggle run Linux and do not need it.")
    D.bullet("**My timings were wrong twice. **", "The main run looked like it took 17.5 minutes, and a profiling run looked very slow, because my computer went to "
             "sleep while idle. I found it in the system log and now report only the real training time. Lesson: when a number looks odd, find out why before reporting it.")
    D.bullet("**The fresh install failed once. **", "Windows could not install PyTorch in a folder with a very long path. A short folder fixed it.")

    D.h("What I could not check, and what I would try next", 2)
    D.p("These are limits of my setup and results, not bugs.")
    D.bullet("The pilot runs were only 300 steps, so 3e-3 and 1e-3 (0.09 apart) might rank differently in a 5,000-step run.")
    D.bullet(f"The validation loss comes from 50 fixed batches (409,600 tokens, about 7.5% of the validation file), so it is a close but not exact estimate of the loss on the whole file.")
    D.bullet("The stories needed the end-of-story token blocked to reach 256 tokens.")
    D.bullet("GPU runs are not bit-for-bit repeatable, so two identical runs differ slightly (in the 3rd or 4th decimal of the loss).")
    D.bullet("The cache speed-up is an estimate from timings, not a measurement.")
    D.p("I did not do the optional extras, since they were not required: the Docker image, the architecture experiment, the KV cache, weight sharing and GPT-2 initialisation.")

    D.h("What I take away", 2)
    D.p("The biggest lesson for me is that the model itself is not the hardest part. Most of the real work was checking that every small piece is correct, because "
        "many mistakes do not crash anything; they only make the results quietly worse. Writing a test for each piece, making sure each test can fail, and "
        "measuring before guessing saved me several times.")

    D.h("Words I had to learn", 2)
    D.table(["Word", "What it means in simple terms"], [
        ["Token", "One piece of text that has its own number, such as a word or part of a word"],
        ["Vocabulary", "The full list of tokens the model knows (10,000 for mine)"],
        ["BPE", "The method that builds the vocabulary by gluing together the most common neighbouring pieces"],
        ["Parameter", "One number the model learns. Mine has about 22.7 million"],
        ["Attention", "The step where each word looks at earlier words and decides which matter"],
        ["RoPE", "A way to tell the model the position of each word by rotating its numbers"],
        ["Feed-forward layer", "The step that processes each word on its own after attention"],
        ["Loss", "A score for how wrong the model's guesses are. Lower is better"],
        ["Validation loss", "The loss on stories the model never trained on, so it shows real learning"],
        ["Perplexity", "The loss turned into \"how many options the model is unsure between\""],
        ["Learning rate", "The size of each learning step"],
        ["Optimizer (AdamW)", "The part that changes the weights a little after each step"],
        ["Gradient", "The direction and size in which each weight should change"],
        ["Gradient clipping", "Capping the gradient so one huge step cannot damage the model"],
        ["fp16 and fp32", "Smaller and larger number formats. fp16 is faster and uses less memory but overflows more easily"],
        ["Gradient scaler", "Multiplies the loss before the backward pass so small fp16 gradients do not become zero, then divides them back"],
        ["torch.compile", "A PyTorch tool that turns the model into faster GPU code"],
        ["Checkpoint", "A saved copy of the training state so a stopped run can continue"],
        ["Temperature", "A setting that makes the generated text safer (low) or more random (high)"],
        ["Top-p", "A setting that only allows the most likely tokens that together add up to a chosen share"],
        ["KV cache", "Remembering past work while generating, so the model does not read everything again for each new word"],
    ], [1.7, 4.8], size=10)
    D.p("All the code, tests, the model files and the instructions to run it are in the repository; the one command to run it is in the README.")
    D.save(R("report", "project_report.docx"))


if __name__ == "__main__":
    build_writeup()
    build_project_report()
