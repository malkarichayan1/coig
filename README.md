# Entity Collapse (EC) Benchmark & Chain-of-Image Generation Pipeline

The Entity Collapse (EC) Benchmark and evaluation pipeline from the paper **"Chain-of-Image Generation: Toward Monitorable and Controllable Image Generation"** (arXiv:2512.08645).

Entity collapse is a failure mode in compositional text-to-image generation where a prompt specifies *n* semantically similar entities with distinct attributes, but the generated image (i) depicts fewer than *n* instances (merge), (ii) misassigns attributes (swap/leak), or (iii) applies one entity's attributes to all (homogenization). This benchmark is designed to induce and measure that failure mode.

## Repository layout

The four folders mirror the pipeline stages. Each script's default paths point at the previous stage's outputs, so they can be run in order without arguments.

```
create_dataset/   1. Build the benchmark prompts
create_prompt/    2. Decompose each prompt into sequential generation steps (CSP)
create_images/    3. Generate images: CoIG step-by-step and single-pass baseline
evaluate/         4. Score images with an MLLM evaluator
```

## Setup

```bash
pip install -r requirements.txt
export GOOGLE_AI_API_KEY=your_key_here
```

## 1. The benchmark (`create_dataset/`)

**`generated_prompts.csv`** is the benchmark used in the paper (320 rows; rows with a non-empty `error` column failed during generation and should be skipped, leaving 306 valid prompts). Each row has:

| Column | Description |
|---|---|
| `prompt` | The image generation prompt (e.g., "Four Bartenders. Bartender A wears leather gloves. ...") |
| `subject` | The shared job/profession of the four entities |
| `attributes` | The four sampled attributes, one per entity |
| `interactions` | The two sampled pairwise interactions (A–B and C–D) |
| `question_count` | Evaluation question: are all four entities present? |
| `question_attr_1..4` | Evaluation questions: is each attribute correctly bound? |
| `question_interaction_1..2` | Evaluation questions: is each interaction rendered between the right pair? |

Each prompt describes **four entities of the same profession** (to enforce inter-entity similarity), **four distinct attributes** (one per entity), and **two pairwise interactions**. The seven questions per prompt support MLLM-based scoring of Entity Count (out of 1), Attribute Binding (out of 4), and Interaction (out of 2) — a total score out of 7.

To regenerate or extend the benchmark (samples from `subject.txt` / `attributes.txt` / `interactions.txt` and phrases prompts + questions with the `dataset_prompt.txt` template):

```bash
cd create_dataset
python generate_dataset.py --out my_prompts.csv --num_samples 320 --model gemini-2.5-flash
```

Note: sampling is random and generation uses temperature 0.7, so regenerated prompts will differ from `generated_prompts.csv`. To compare against the paper, use the released CSV directly.

## 2. Step decomposition (`create_prompt/`)

The Compositional Strategy Planner (CSP): an LLM decomposes each benchmark prompt into six sequential sub-prompts (placeholders → per-entity details/interactions → background), following the decomposition rules in `base_prompt.txt`.

```bash
cd create_prompt
python create_prompt_sbs.py --model gemini-2.5-pro
```

`sbs_prompts/` and `sbs_prompts_results.csv` contain the exact decompositions used in the paper — one folder per prompt with `step_01.txt` … `step_06.txt`.

## 3. Image generation (`create_images/`)

The Autoregressive Refinement Model (ARM): step 1 generates the initial image, steps 2–6 iteratively edit it. The baseline generates each image from the full prompt in a single pass with the same image model.

```bash
cd create_images
python generate_multi_step_image_genai_simple.py   # CoIG (ours) -> multi_step_out/
python generate_baseline_genai_simple.py           # single-pass baseline -> baseline_out/
```

Both default to `gemini-2.5-flash-image-preview` (Nano Banana).

## 4. Evaluation (`evaluate/`)

An MLLM evaluator (default `gemini-2.5-pro`) performs a "visual census" of each image — enumerating visible entities and binding attributes/interactions to them using only visual evidence — and scores Entity Count, Attribute Binding, and Interaction per prompt.

```bash
cd evaluate
python evaluate_images.py                     # baseline images  -> evaluation_results.csv
python evaluate_sbs_images.py --last          # CoIG final step images -> evaluation_sbs_results.csv
```

`evaluate_sbs_images.py` can also score every intermediate step (`--all`, the default) or specific steps (`--steps 1,3`), which is useful for monitoring the step-by-step trajectory.

## Citation

```bibtex
@article{coig2025,
  title={Chain-of-Image Generation: Toward Monitorable and Controllable Image Generation},
  journal={arXiv preprint arXiv:2512.08645},
  year={2025}
}
```
