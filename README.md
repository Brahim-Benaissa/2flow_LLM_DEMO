# BoilDesc — boiling simulations in descriptor space (a feasibility demonstration)

Five notebooks that turn a small corpus of 2-D boiling simulations into a complete
surrogate pipeline: physical indicators are extracted from raw simulation frames,
decoded back into full frames (by retrieval and by a trained CNN generator),
forecast forward in time, and finally served to a user through an LLM-driven
tool-calling service.

The data is a subset of **BubbleML 2.0**, the boiling-simulation dataset published
by the [Bubbleformer](https://hpcforge.github.io/Bubbleformer/) project
([code](https://github.com/HPCForge/bubbleformer) ·
[paper](https://openreview.net/pdf?id=3TN5My3Xw6) ·
[dataset](https://huggingface.co/datasets/hpcforge/BubbleML_2)).
Everything here is a **feasibility demonstration**: it runs end-to-end from
notebooks, every embedded number is reproducible from the downloaded data, and
every model is small (the largest has 0.92 M parameters).

```
              BubbleML 2.0  (.h5 simulations + .json sidecars)
                              │
  1 · indicator & sensor analysis   ← defines the descriptor θ and validates it
                              │
                              ▼
             θ(t) — 13 live indicators + wall temperature
                              │
  2 · retrieval decoders (M0/M1/M2)   3 · CNN generator (M4)   4 · forecasting (F0–F3)
     θ → nearest real frame            θ → any frame            θ(t) → θ(t+Δt)
                              │
                              ▼
  5 · LLM service — an LLM (Gemini) drives the three trained models as tools
```

| # | Notebook | Role | Trains | Needs |
|---|---|---|---|---|
| 1 | [`1_indicator_sensor_analysis_v2.ipynb`](1_indicator_sensor_analysis_v2.ipynb) | defines and validates the descriptor | — | 1 file, CPU |
| 2 | [`2_M2_KNN_retriever_v4.ipynb`](2_M2_KNN_retriever_v4.ipynb) | first θ→frame decoders + the round-trip leaderboard | M0 / M1 / M2 | 1 file, CPU |
| 3 | [`3_M4_CNN_decoder_v5.ipynb`](3_M4_CNN_decoder_v5.ipynb) | the learned generator | M4 (CNN) | 15 files, GPU |
| 4 | [`4_TCN_forecasting_v3.ipynb`](4_TCN_forecasting_v3.ipynb) | forecasts θ(t) forward | F0–F3 | 15 files, GPU (light) |
| 5 | [`5_BoilDesc_LLM_service_v11.ipynb`](5_BoilDesc_LLM_service_v11.ipynb) | natural-language access to the models | — (loads bundles) | data + bundles + optional LLM key |

---

## The notebooks

### 1 · indicator & sensor analysis

`1_indicator_sensor_analysis_v2.ipynb` — works on a single simulation
(`Twall_107` from *PoolBoiling-Saturated-FC72-2D*) and answers two questions:
*which quantities can an engineer actually measure*, and *can we trust the
extraction from raw frames?*

- Reads the JSON sidecar (wall temperature, heater span, nucleation sites,
  dimensionless groups), streams the HDF5 file once, and computes per frame a
  **20-column indicator panel** — 5 wall-normal sections × (void fraction α,
  mean temperature, Sauter mean diameter D32, hot-spot fraction) — plus a
  **virtual-sensor panel** (thermocouple-like wall statistics, wall vapour
  coverage, near-wall heat flux, boundary-layer thickness, bubble counts).
- Validates the extraction against simulation ground truth: 81 % of 13 835
  near-wall bubble centres fall within 0.16 of a JSON nucleation site, and the
  temperature scale is checked against the sidecar's wall temperature.
- The indicator definitions established here are reused **verbatim** by every
  later notebook.
- Pure NumPy/SciPy/h5py/matplotlib — no torch, no GPU, minutes to run.

### 2 · first decoders — the retrieval ladder

`2_M2_KNN_retriever_v4.ipynb` — still on the single file. Builds the ML frame
cache and a **time-blocked 70/15/15 split**, then a ladder of three
descriptor→frame decoders:

- **M0** mean frame, **M1** PCA eigenframes + ridge regression, **M2** k-NN
  retrieval in descriptor space (k ∈ {1, 3, 5, 9, 15}).
- Introduces the judge used everywhere afterwards — the **round trip (RT)**:
  decode a descriptor into a frame, re-extract the descriptor with the
  notebook-1 extractor, and demand the ask back.
- The leaderboard's takeaway: **k = 1 retrieval is the baseline to beat** —
  perfect pixels, but zero flexibility (it can only return training frames) and
  one full frame in memory per entry.
- Also pure NumPy/SciPy (no torch, no sklearn, no pandas); a few minutes.

### 3 · the CNN generator (descriptor → frame)

`3_M4_CNN_decoder_v5.ipynb` — scales up to the whole corpus: **15 simulations,
30 015 frames**. Trains a deterministic CNN (0.92 M parameters, FiLM-conditioned
upsampling stages) to render *any* descriptor as a full 288 × 192 frame with both
channels (`dfun` and `temperature`).

- Conditioning vector = **13 live indicator columns + wall temperature**
  (z-scored). Splits are **whole files** (8 train / 4 val / 3 test, stratified by
  boiling regime) so every test wall temperature was never trained on — all
  reported results are cross-condition.
- Training optimises a **soft round-trip loss** (differentiable surrogates of
  every indicator, computed from the generated frame itself) with pixel losses
  as anchors. Executed run: 154 epochs ≈ 77 GPU-hours (scale down via
  `CFG['EPOCHS']` / `CFG['FRAME_STRIDE']`).
- Headline: **11.7 % round-trip error on unseen wall temperatures** (9.9 % with
  a per-column affine correction) versus 3.9 % for the real-data information
  floor, with sharp, artifact-free frames (edge energy 0.96 vs 1.0 for real).
- Exports `M4_generator_bundle_*.pt` — weights, normalization statistics and
  column maps — plus, alongside it, the `M2_k1_retriever_bundle_*.pt` index of
  the training frames used as the retrieval reference.

### 4 · forecasting in descriptor space

`4_TCN_forecasting_v3.ipynb` — predicts the **future of the descriptor itself**,
not pixel by pixel. Same 15 files, same file-level splits, same extractor
(identical by design, so decoder and forecaster speak the same θ).

- Four model classes share one rollout and one metric: **F0** persistence (the
  null model), **F1** mechanism hybrid (closed-form wall-superheat accumulation
  driving a quiet→burst switch — the only model with the physics written in),
  **F2** TCN (dilated causal convolutions, 37 k parameters, pushforward-trained),
  **F3** ESN (400-unit random reservoir with ridge readout).
- Task: from a 32-step window (3.2 s) predict the residual over persistence for
  the next step, then roll out autoregressively up to 80 steps (400 on held-out
  conditions).
- Headline: physics beats persistence one step ahead (2.9 % vs 3.9 %); the TCN
  wins at horizon 20 (34.8 % vs 40.2 %); at horizon 80 the void variables reach
  a **chaos floor** (best model = persistence at 41.9 %) while thermal variables
  stay predictable to 0.3–1 %.
- Exports `FC_F2_TCN_bundle_*.pt`; forecast trajectories are written as decoder
  asks, so predicted descriptors can be rendered as frames.

### 5 · the LLM service

`5_BoilDesc_LLM_service_v11.ipynb` — packs the three trained models (M4
generator, M2 retriever, FC TCN forecaster) into a **local FastAPI service**
(`http://localhost:8000`) that an LLM — Gemini via its OpenAI-compatible API —
drives through **tool calling**.

- You ask in plain language ("what does the wall region look like at 21 °C?",
  "what happens next at 98 °C?"); the LLM routes the question through a 7-way
  table (render · forecast · conditions · reliability · sensitivity ·
  nearest-real · refuse).
- `boiling_render` answers with **both** decoders side by side — the M4 render
  and the nearest *real* frame from M2 — on one shared colour scale; the answer
  merges them as a range and ends with a deterministic "In short" summary.
- Everything under `DATA_DIR` is **auto-discovered**: the three bundles and
  every `.h5`/`.hdf5` file, newest bundle timestamp wins. If nothing is found
  the notebook switches to **fixture mode** (a tiny synthetic corpus) so the
  service still runs end-to-end for testing.
- Configuration lives in `.env` next to the notebook. Without an LLM key the
  chat runs a scripted placeholder loop, but `POST /tools/render` keeps working.

---

## The whole demonstration

One question runs through all five notebooks: **is a compact, physically
meaningful descriptor space enough to stand in for a boiling simulation?** The
answer is built up piece by piece:

- **The interface** (1). What an engineer can measure — void fractions,
  temperatures, bubble sizes, hot spots — is compressed into the descriptor θ:
  five wall-normal regions × (void fraction, mean temperature, hot-wall
  fraction) + Sauter mean diameter + bubble count, plus wall temperature as the
  case scalar. The extraction is validated against the simulation's own
  ground truth.
- **Decoding** (2 + 3). Two decoders map θ back to full frames: retrieval
  returns the nearest *real* frame (pixels perfect, flexibility zero), while the
  CNN generator renders *any* θ, including wall temperatures it never trained
  on. Their trade-off is the honest framing of the demo: generated frames are
  approximate (11.7 % round-trip vs the 3.9 % information floor) but θ is a
  *control* input, not a lookup key.
- **Forecasting** (4). θ(t) can be rolled forward in time: physics and the TCN
  beat persistence in the short range, thermal variables stay predictable over
  long horizons, and the void variables demonstrably hit a chaos floor — stated,
  not hidden.
- **The two-stage surrogate** (3 + 4). The forecaster evolves θ(t); the decoder
  renders any θ as a frame. Chained, they form a cheap, inspectable stand-in
  for the simulator for design-style questions.
- **Access** (5). None of this helps if only an ML engineer can use it, so the
  models become tools an LLM drives, answering natural-language questions with
  both decoders as a range and the reliability numbers attached.

Two design decisions hold it all together. **One extractor** for real and
generated frames makes the round trip a well-defined judge — the generator is
optimised and judged on the same criterion. **File-level splits** (whole
simulations, stratified by boiling regime, dealt across the temperature range)
make every validation and test number cross-condition by construction.

This is feasibility, not production: the 15 simulations span two disjoint
wall-temperature islands (10–26 °C single-bubble and 75–107 °C developed
boiling), nothing between 26 and 75 °C is supported by data, and the notebooks
are explicit about it.

---

## How to run

### Requirements

```bash
pip install torch h5py numpy scipy pandas matplotlib jupyter
# notebook 5 additionally:
pip install fastapi uvicorn requests pillow
```

Notebooks 1–2 are CPU-only. Notebook 3 needs a CUDA GPU (the executed run was
≈ 77 GPU-hours). Notebook 4 trains the TCN on GPU but is far lighter (the F1/F3
models are closed-form and fit in seconds). Notebook 5 is inference-only.

### 1 · Download the data

The results embedded in the notebooks are based on two subsets of BubbleML 2.0
([CC BY-NC-SA 4.0](https://huggingface.co/datasets/hpcforge/BubbleML_2)) —
about **106 GB** in total:

| Subset | Files used | Per file | Total |
|---|---|---|---|
| [SingleBubble-Saturated-R515B-2D](https://huggingface.co/datasets/hpcforge/BubbleML_2/tree/main/SingleBubble-Saturated-R515B-2D) | all 11 — `Twall_{10,13,14,15,17,18,19,21,22,23,26}` | 3.5 GB | 39 GB |
| [PoolBoiling-Saturated-FC72-2D](https://huggingface.co/datasets/hpcforge/BubbleML_2/tree/main/PoolBoiling-Saturated-FC72-2D) | 4 of 20 — `Twall_{75,98,100,107}` | 16.8 GB | 67 GB |

Download with the Hugging Face CLI (keep the `.json` sidecars next to the
`.hdf5` files — notebooks 1 and 2 read them):

```bash
pip install -U huggingface_hub

# the 11 single-bubble simulations
huggingface-cli download hpcforge/BubbleML_2 --repo-type dataset \
  --include "SingleBubble-Saturated-R515B-2D/*" \
  --local-dir /path/to/boildesc/data

# the 4 pool-boiling simulations used in the notebooks
huggingface-cli download hpcforge/BubbleML_2 --repo-type dataset \
  --include "PoolBoiling-Saturated-FC72-2D/Twall_75.*" \
            "PoolBoiling-Saturated-FC72-2D/Twall_98.*" \
            "PoolBoiling-Saturated-FC72-2D/Twall_100.*" \
            "PoolBoiling-Saturated-FC72-2D/Twall_107.*" \
  --local-dir /path/to/boildesc/data
```

**Other files are included automatically.** All notebooks discover every
`.h5`/`.hdf5` under the data folder by recursive glob, so you can drop in more
BubbleML simulations (the rest of *PoolBoiling-Saturated-FC72-2D*, the LN2 or
subcooled studies, …) with **zero code changes**: the corpus, the splits and the
conditioning vector widen by themselves. Results will then differ from the
embedded ones, which were produced with exactly the 15 files above.

### 2 · Get the models

Notebook 5 runs no training — it needs the three model bundles **in the same
repo** as the data (any subfolder works; discovery is recursive and the newest
filename timestamp wins):

| Bundle | Model | Exported by |
|---|---|---|
| `M4_generator_bundle_*.pt` | CNN generator, descriptor → frame | notebook 3 |
| `M2_k1_retriever_bundle_*.pt` | k = 1 retriever index over the training frames | notebook 3 |
| `FC_F2_TCN_bundle_*.pt` | TCN forecaster | notebook 4 |

Recommended layout — one root folder holding data and models, pointed at by
`DATA_DIR`:

```
boildesc/                              <- DATA_DIR
├── data/
│   ├── PoolBoiling-Saturated-FC72-2D/      Twall_75/98/100/107  .hdf5 + .json
│   └── SingleBubble-Saturated-R515B-2D/    Twall_10..26         .hdf5 + .json
└── models/
    ├── M4_generator_bundle_f15files_*.pt
    ├── M2_k1_retriever_bundle_f15files_*.pt
    └── FC_F2_TCN_bundle_f15files_*.pt
```

Alternatively, retrain: running notebooks 2–4 on the downloaded data exports the
bundles automatically. With no bundles and no data present, notebook 5 falls
back to fixture mode and still runs end-to-end.

### 3 · Point the notebooks at your folders

- **Notebooks 1–4:** set `CFG['DATA_DIR']` in the first cell (default
  `G:\BoilDesc`, the author's machine; Windows paths are fine).
- **Notebook 5:** set `DATA_DIR` in the `.env` file created next to the notebook
  on first run, plus your Gemini key:

```ini
DATA_DIR=/path/to/boildesc
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_API_KEY=            # Google AI Studio key (optional — placeholder loop without it)
LLM_MODEL=gemini-3.1-flash-lite
PORT=8000
```

### 4 · Run in order

Run the notebooks **1 → 2 → 3 → 4 → 5**, cells top to bottom. 1–2 finish in
minutes on CPU; 3 is the expensive training; 4 is lighter; 5 starts the service
in a background thread and prints the URL to open in your browser
(`http://localhost:8000`).

---

## Credits

- **Data:** [BubbleML 2.0](https://huggingface.co/datasets/hpcforge/BubbleML_2)
  (CC BY-NC-SA 4.0), published by the
  [Bubbleformer](https://hpcforge.github.io/Bubbleformer/) /
  [HPCForge](https://github.com/HPCForge/BubbleML) project.
- **Papers:** Bubbleformer
  ([OpenReview](https://openreview.net/pdf?id=3TN5My3Xw6), NeurIPS 2025) and
  BubbleML ([arXiv:2307.14623](https://arxiv.org/abs/2307.14623), NeurIPS 2023
  Datasets & Benchmarks).
