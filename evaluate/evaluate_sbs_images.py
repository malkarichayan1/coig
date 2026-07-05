#!/usr/bin/env python3
"""
Evaluate SBS multi-step images using Gemini Vision.

This script reads the SBS results CSV (from create_prompt_sbs.py), the original
prompts CSV (to get question columns), and evaluates generated step images
located in the multi-step output directory.

Usage examples:
  python evaluate_sbs_images.py
  python evaluate_sbs_images.py --sbs_csv ../create_prompt/sbs_prompts_results.csv --prompts_csv ../create_dataset/generated_prompts.csv --image_base multi_step_out --last
  python evaluate_sbs_images.py --steps 1,3 --limit 10

"""

import os
import sys
import argparse
import pathlib
import time
import json
import re
import pandas as pd
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig
from tqdm import tqdm
from PIL import Image as PILImage

# Reuse the same evaluation prompt template as evaluate_images.py
EVALUATION_PROMPT_TEMPLATE = """You are an AI quality auditor for text-to-image generation.

Your task is to analyze the given image and answer a yes/no question based solely on its visual content. The question may relate to the presence of a specific object, its attributes, or relationships between multiple elements in the image.

You will also be given the original prompt used to generate the image. The prompt may provide additional context to help interpret the question, but it must never be used to supply or assume visual details.
Your judgment must rely entirely on the image itself. The image must contain clear, unmistakable visual evidence to justify a "yes" answer — the prompt cannot compensate for missing or ambiguous content.

Respond with:
- "yes" only if the answer is **clearly and unambiguously** yes based solely on the visual content. The visual evidence must be **strong, definitive, and require no assumptions or guesses**.
- "no" in **all other cases** — including if the relevant visual detail is missing, unclear, ambiguous, partially shown, obscured, or only suggested.

Even if the image closely matches what is described in the prompt, you must rely on **visible evidence** alone. If the relevant detail cannot be confirmed visually with certainty, answer "no".  
**Ambiguity equals no.**

For conditional questions, answer "yes" only if **both** the condition and the main clause are **clearly and unambiguously true** in the image. If **either part** is false or uncertain, respond "no".

Do **not** provide any explanation, justification, or extra text.
Only return a single word: either "yes" or "no".

Now, evaluate this image:

Prompt: {prompt}
Question: {question}"""


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate SBS multi-step images using Gemini Vision")
    p.add_argument("--sbs_csv", type=str, default="../create_prompt/sbs_prompts_results.csv",
                   help="CSV produced by create_prompt_sbs.py containing output_dir and original_prompt")
    p.add_argument("--prompts_csv", type=str, default="../create_dataset/generated_prompts.csv",
                   help="Original prompts CSV that contains the question columns")
    p.add_argument("--image_base", type=str, default="../create_images/multi_step_out",
                   help="Base output directory where generated step images are stored (item subfolders)")
    p.add_argument("--output_csv", type=str, default="evaluation_sbs_results.csv",
                   help="Output CSV path")
    p.add_argument("--model", type=str, default="gemini-2.5-pro",
                   help="Gemini model name (vision-capable)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=None)
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--all", action="store_true", default=True,
                     help="Evaluate all available steps (default)")
    grp.add_argument("--last", action="store_true", default=False,
                     help="Evaluate only the last step available for each item")
    grp.add_argument("--steps", type=str, default="",
                     help="Comma-separated list of step numbers to evaluate (e.g. 1,3,5)")
    p.add_argument("--delay", type=float, default=1.0)
    p.add_argument("--retry_errors", action="store_true", default=True)
    p.add_argument("--no_retry_errors", action="store_false", dest="retry_errors")
    return p.parse_args()


def get_api_key():
    api_key = os.getenv("GOOGLE_AI_API_KEY")
    if not api_key:
        raise ValueError("Google AI API Key is required")
    return api_key


def initialize_gemini_client():
    api_key = get_api_key()
    return genai.Client(api_key=api_key)


def create_generation_config(temperature: float = 0.0) -> Dict[str, Any]:
    return {"temperature": temperature, "top_p": 0.95, "top_k": 40, "max_output_tokens": 1000}


def evaluate_image_with_gemini(client, image_path: str, prompt: str, question: str, model_name: str, config: Dict[str, Any], max_retries: int = 3) -> str:
    evaluation_prompt = EVALUATION_PROMPT_TEMPLATE.format(prompt=prompt, question=question)
    try:
        pil_image = PILImage.open(image_path)
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return "error"

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[evaluation_prompt, pil_image],
                config=GenerateContentConfig(**config)
            )
            if response.candidates and response.candidates[0].content.parts:
                response_text = response.candidates[0].content.parts[0].text.strip().lower()
                if "yes" in response_text:
                    return "yes"
                elif "no" in response_text:
                    return "no"
                else:
                    print(f"Unexpected response: {response_text}")
                    return "error"
            else:
                print("Empty response from Gemini")
                return "error"
        except Exception as e:
            print(f"Error on attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return "error"

    return "error"


def find_step_image(base_dir: pathlib.Path, item_index: int, step_num: int) -> pathlib.Path:
    path = base_dir / str(item_index) / f"step_{step_num:02d}.png"
    return path


def load_existing_results(output_path: str) -> pd.DataFrame:
    out_path = pathlib.Path(output_path)
    if not out_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(out_path)
    except Exception:
        return pd.DataFrame()


def main():
    load_dotenv()
    args = parse_args()

    client = initialize_gemini_client()
    config = create_generation_config(temperature=args.temperature)


    # Load SBS CSV
    sbs_df = pd.read_csv(args.sbs_csv)
    sbs_df = sbs_df[sbs_df['status'] == 'SUCCESS'].copy()
    # ...existing code...

    # Load prompts CSV (questions)
    prompts_df = pd.read_csv(args.prompts_csv)
    prompts_df = prompts_df[prompts_df['prompt'].notna() & (prompts_df['prompt'] != '')].copy()
    # ...existing code...

    # Question columns (same as evaluate_images)
    question_columns = [
        'question_count',
        'question_attr_1',
        'question_attr_2',
        'question_attr_3',
        'question_attr_4',
        'question_interaction_1',
        'question_interaction_2'
    ]

    # Build evaluations: (item_index, step_num, question_col, question_text)
    evaluations = []

    # Parse steps option
    steps_list: List[int] = []
    if args.steps:
        steps_list = [int(x) for x in re.split(r'\s*,\s*', args.steps.strip()) if x]

    for _, srow in sbs_df.iterrows():
        item_index = srow['item_index']
        original_prompt = srow.get('original_prompt', '')
        output_dir = srow.get('output_dir', '')

        # Find available step images in image_base/<item_index>
        item_image_dir = pathlib.Path(args.image_base) / str(item_index)
    # ...existing code...
        available_steps = []
        if item_image_dir.exists():
            for p in sorted(item_image_dir.glob('step_*.png')):
                m = re.match(r'step_(\d{2})\.png$', p.name)
                if m:
                    available_steps.append(int(m.group(1)))

        if not available_steps:
            # ...existing code...
            continue

        target_steps = []
        if args.last:
            target_steps = [max(available_steps)]
        elif steps_list:
            # only keep steps that actually exist
            target_steps = [s for s in steps_list if s in available_steps]
        else:
            # default: all
            target_steps = available_steps

        # find matching row in prompts_df to get questions
        prow = prompts_df[prompts_df['item_index'] == item_index]
        if prow.empty:
            # no questions available, skip
            continue
        prow = prow.iloc[0]

        for step_num in target_steps:
            for qcol in question_columns:
                if qcol in prow.index and pd.notna(prow[qcol]) and prow[qcol] != '':
                    image_path = find_step_image(pathlib.Path(args.image_base), item_index, step_num)
                    evaluations.append((item_index, step_num, prow['prompt'], qcol, prow[qcol], srow))

    print(f"Prepared {len(evaluations)} evaluations")
    if args.limit:
        evaluations = evaluations[:args.limit]

    # Load existing results to avoid re-evaluating successful pairs
    existing_df = load_existing_results(args.output_csv)
    completed = set()
    if not existing_df.empty:
        for _, r in existing_df.iterrows():
            if r['answer'] != 'error':
                completed.add((r['item_index'], int(r['step_num']), r['question_type']))

    # Evaluate
    results = []
    yes_count = 0
    no_count = 0
    error_count = 0

    for item_index, step_num, prompt, qcol, question, srow in tqdm(evaluations, desc='Evaluating'):
        if (item_index, step_num, qcol) in completed and not args.retry_errors:
            continue

        image_path = find_step_image(pathlib.Path(args.image_base), item_index, step_num)
        if not image_path.exists():
            results.append({'item_index': item_index, 'step_num': step_num, 'question_type': qcol, 'question': question, 'image_path': str(image_path), 'answer': 'error', 'answer_binary': -1})
            error_count += 1
            continue

        answer = evaluate_image_with_gemini(client, str(image_path), prompt, question, args.model, config)
        if answer == 'yes':
            answer_binary = 1
            yes_count += 1
        elif answer == 'no':
            answer_binary = 0
            no_count += 1
        else:
            answer_binary = -1
            error_count += 1

        results.append({'item_index': item_index, 'step_num': step_num, 'question_type': qcol, 'question': question, 'image_path': str(image_path), 'answer': answer, 'answer_binary': answer_binary})

        time.sleep(args.delay)

    # Save
    out_df = pd.DataFrame(results)
    if not existing_df.empty:
        # combine, removing retried pairs
        retried = set((r['item_index'], int(r['step_num']), r['question_type']) for r in results)
        existing_keep = existing_df[~existing_df.apply(lambda row: (row['item_index'], int(row['step_num']), row['question_type']) in retried, axis=1)]
        combined = pd.concat([existing_keep, out_df], ignore_index=True)
    else:
        combined = out_df
    print(combined)
    combined = combined.sort_values(['item_index', 'step_num', 'question_type']).reset_index(drop=True)
    combined.to_csv(args.output_csv, index=False)

    print(f"Done. Yes: {yes_count}, No: {no_count}, Errors: {error_count}. Results saved to {args.output_csv}")


if __name__ == '__main__':
    main()
