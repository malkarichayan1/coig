#!/usr/bin/env python3
"""
Multi-Step Image Generator using Google Gemini for SBS prompts.

This script reads from sbs_prompts_results.csv (created by create_prompt_sbs.py),
finds the generated step-by-step prompt directories, and generates images for each step.

For each successful sequence in the CSV:
1. Reads step_01.txt through step_06.txt from the output_dir
2. Generates initial image from step 1
3. Edits the image iteratively using steps 2-6
4. Saves each step's image

File structure expected (from sbs_prompts_results.csv):
  output_dir: sbs_prompts/item_X_Subject/
    step_01.txt  - First image generation prompt
    step_02.txt  - Image editing prompt
    step_03.txt  - Image editing prompt
    ...
    step_06.txt  - Final editing prompt

Outputs for each sequence:
  outdir/<item_index>/step_01.png         - initial generated image
  outdir/<item_index>/step_01_prompt.txt  - prompt used for step 01
  outdir/<item_index>/step_02.png         - edited image after step 02
  outdir/<item_index>/step_02_prompt.txt  - prompt used for step 02
  ...
  outdir/<item_index>/step_06.png         - final image
  outdir/<item_index>/metadata.json       - processing metadata

Usage example:
  python generate_multi_step_image_genai_simple.py
  python generate_multi_step_image_genai_simple.py --csv ../create_prompt/sbs_prompts_results.csv
  python generate_multi_step_image_genai_simple.py --limit_items 3 --model gemini-2.5-pro-image-preview

Environment:
  Requires GOOGLE_AI_API_KEY environment variable.
  Get one from: https://aistudio.google.com/app/apikey
"""

import os
import re
import argparse
import pathlib
import time
import json
import pandas as pd
from typing import Any, List, Tuple
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

# Google Gen AI SDK imports
from google import genai
from google.genai.types import GenerateContentConfig, Part

# Global Gemini client instance
gemini_client = None

# -------------------- CLI --------------------

def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Generate multi-step images from SBS prompt sequences.")
    # Input/Output settings
    p.add_argument("--csv", type=str, 
                   default="../create_prompt/sbs_prompts_results.csv",
                   help="Path to CSV file containing SBS prompt results")
    p.add_argument("--sbs_base_dir", type=str, 
                   default="../create_prompt",
                   help="Base directory where sbs_prompts folders are located")
    p.add_argument("--outdir", type=str, 
                   default="multi_step_out",
                   help="Output directory for generated step images")
    p.add_argument("--model", type=str, default="gemini-2.5-flash-image-preview",
                   help="Gemini image model to use")
    p.add_argument("--limit_items", type=int, default=None,
                   help="Limit number of sequences to process (for testing)")
    p.add_argument("--safety_level", type=str, default="BLOCK_NONE",
                   choices=["BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_LOW_AND_ABOVE"],
                   help="Content safety filter level")
    p.add_argument("--allow_all_ages", action="store_true",
                   help="Enable content for all age groups")
    p.add_argument("--delay", type=float, default=2.0,
                   help="Delay between API calls in seconds")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip sequences that have already been processed")
    return p.parse_args()

# -------------------- Gemini Client Initialization --------------------

def initialize_gemini_client():
    """Initialize the Gemini API client using environment variables."""
    load_dotenv()
    
    api_key = os.getenv("GOOGLE_AI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing GOOGLE_AI_API_KEY environment variable. "
            "Get one from: https://aistudio.google.com/app/apikey"
        )
    
    return genai.Client(api_key=api_key)

def create_safety_configuration(safety_level="BLOCK_NONE", allow_all_ages=False):
    """Create safety settings for content generation."""
    harm_categories = [
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_CIVIC_INTEGRITY"
    ]
    
    threshold = "BLOCK_NONE" if (allow_all_ages or safety_level == "BLOCK_NONE") else safety_level
    
    return [{"category": category, "threshold": threshold} for category in harm_categories]

def create_generation_configuration(safety_level="BLOCK_NONE", allow_all_ages=False):
    """Create complete generation configuration for Gemini API calls."""
    safety_settings = create_safety_configuration(safety_level, allow_all_ages)
    
    return GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
        candidate_count=1,
        safety_settings=safety_settings
    )

# -------------------- CSV Data Processing --------------------

def load_and_validate_csv(csv_path: pathlib.Path) -> pd.DataFrame:
    """Load and validate CSV with SBS prompt results."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required_columns = ['item_index', 'output_dir', 'status']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Filter only successful sequences
    df_success = df[df['status'] == 'SUCCESS'].copy()
    
    print(f"📊 CSV Data Summary:")
    print(f"   Total rows in CSV: {len(df)}")
    print(f"   Successful sequences: {len(df_success)}")
    
    return df_success

def load_step_prompts(prompt_dir: pathlib.Path) -> List[Tuple[int, str]]:
    """Discover and load any step_NN.txt files in `prompt_dir`.

    Returns list of (step_number, prompt_text) sorted by step_number.
    """
    steps: List[Tuple[int, str]] = []

    # Find files matching step_XX.txt
    for p in sorted(prompt_dir.glob('step_*.txt')):
        name = p.name
        m = re.match(r'step_(\d{2})\.txt$', name)
        if not m:
            continue
        step_num = int(m.group(1))
        try:
            prompt_text = p.read_text(encoding='utf-8').strip()
            # If the file includes the USER'S COMPLEX PROMPT header, remove it
            if prompt_text.startswith("USER'S COMPLEX PROMPT TO DECONSTRUCT:"):
                # remove first line and any following blank line
                parts = prompt_text.split('\n')
                # drop header line
                parts = parts[1:]
                # drop leading blank line
                if parts and parts[0].strip() == '':
                    parts = parts[1:]
                prompt_text = '\n'.join(parts).strip()

            if prompt_text:
                steps.append((step_num, prompt_text))
                print(f"      ✓ Loaded step {step_num}: {len(prompt_text)} chars")
        except Exception as e:
            print(f"      ⚠️ Error loading {name}: {e}")

    # Sort by step number
    steps.sort(key=lambda x: x[0])
    return steps

# -------------------- Image Generation --------------------

def ensure_dir(path: pathlib.Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)

def save_image_bytes(image_bytes: bytes, output_path: pathlib.Path) -> None:
    """Save image bytes as PNG file."""
    ensure_dir(output_path.parent)
    image = Image.open(BytesIO(image_bytes))
    image.save(str(output_path), format="PNG")

def generate_initial_image(prompt: str, *, model: str, config: GenerateContentConfig) -> tuple[bytes, Any]:
    """Generate initial image from text prompt. Returns (image_bytes, response_object)."""
    print(f"      🎨 Generating initial image...")
    print(f"         Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    
    response = gemini_client.models.generate_content(
        model=model,
        contents=[prompt],
        config=config
    )
    
    # Extract image data from response
    if response.candidates and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if part.text:
                print(f"         💬 Model: {part.text}")
            if part.inline_data:
                print(f"         ✅ Image generated")
                return part.inline_data.data, response
    
    raise RuntimeError("No image data found in initial generation response")

def edit_image(image_bytes: bytes, edit_prompt: str, *, model: str, config: GenerateContentConfig) -> tuple[bytes, Any]:
    """Edit an existing image using a text prompt. Returns (image_bytes, response_object)."""
    print(f"      ✏️  Editing image...")
    print(f"         Prompt: {edit_prompt[:100]}{'...' if len(edit_prompt) > 100 else ''}")
    
    # Create image part from bytes
    image_part = Part.from_bytes(data=image_bytes, mime_type="image/png")
    
    response = gemini_client.models.generate_content(
        model=model,
        contents=[image_part, edit_prompt],
        config=config
    )
    
    # Extract image data from response
    if response.candidates and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if part.text:
                print(f"         💬 Model: {part.text}")
            if part.inline_data:
                print(f"         ✅ Image edited")
                return part.inline_data.data, response
    
    raise RuntimeError("No image data found in edit response")

# -------------------- Core Processing --------------------

def process_single_sequence(row: pd.Series, sbs_base_dir: pathlib.Path, output_dir: pathlib.Path, 
                           model: str, config: GenerateContentConfig, delay: float) -> dict:
    """Process a single image sequence from CSV row."""
    item_index = row['item_index']
    prompt_dir_rel = row['output_dir']
    subject = row.get('subject', '')
    original_prompt = row.get('original_prompt', '')
    
    print(f"\n{'='*80}")
    print(f"🔍 Item {item_index} - {subject}")
    print(f"   Original: {original_prompt[:80]}{'...' if len(original_prompt) > 80 else ''}")
    
    # Resolve prompt directory path
    prompt_dir = sbs_base_dir / prompt_dir_rel
    if not prompt_dir.exists():
        print(f"   ❌ Prompt directory not found: {prompt_dir}")
        return {"status": "ERROR", "error": "Prompt directory not found"}
    
    print(f"   📁 Prompt dir: {prompt_dir}")
    
    # Setup output directory
    item_dir = output_dir / str(item_index)
    ensure_dir(item_dir)
    
    # Load step prompts
    print(f"   📝 Loading step prompts...")
    steps = load_step_prompts(prompt_dir)
    
    if not steps:
        print(f"   ❌ No valid step prompts found")
        return {"status": "ERROR", "error": "No valid step prompts found"}
    
    print(f"   ✓ Loaded {len(steps)} steps")

    # Determine final step and skip if already processed
    last_step_num = steps[-1][0]
    final_image = item_dir / f"step_{last_step_num:02d}.png"
    if final_image.exists():
        print(f"   ⏭️  Skipping: already processed (final image exists: {final_image.name})")
        return {"status": "SKIPPED"}
    
    # Process each step
    try:
        current_image_bytes = None
        current_response = None
        
        for step_num, step_prompt in steps:
            print(f"\n   📍 Step {step_num}")
            
            # Setup output paths
            image_path = item_dir / f"step_{step_num:02d}.png"
            prompt_path = item_dir / f"step_{step_num:02d}_prompt.txt"
            
            # Save prompt text
            prompt_path.write_text(step_prompt, encoding="utf-8")
            
            try:
                if step_num == 1 or current_image_bytes is None:
                    # Generate initial image
                    current_image_bytes, current_response = generate_initial_image(
                        step_prompt, model=model, config=config
                    )
                else:
                    # Edit previous image
                    current_image_bytes, current_response = edit_image(
                        current_image_bytes, step_prompt, model=model, config=config
                    )
                
                # Save current step image
                save_image_bytes(current_image_bytes, image_path)
                print(f"      💾 Saved: {image_path.name}")
                
                # Delay between steps
                time.sleep(delay)
                
            except Exception as e:
                print(f"      ❌ Step {step_num} failed: {type(e).__name__}: {e}")
                
                # Log error
                error_log = item_dir / f"error_step_{step_num:02d}.log"
                with open(error_log, 'w', encoding='utf-8') as f:
                    f.write(f"Error at step {step_num}\n")
                    f.write(f"Item: {item_index}\n")
                    f.write(f"Subject: {subject}\n")
                    f.write(f"Error: {type(e).__name__}: {str(e)}\n\n")
                    f.write(f"Prompt:\n{'-'*80}\n{step_prompt}\n{'-'*80}\n")
                
                return {"status": "ERROR", "error": str(e), "failed_step": step_num}
        
        # Save metadata
        metadata = {
            "item_index": item_index,
            "subject": subject,
            "original_prompt": original_prompt,
            "model": model,
            "total_steps": len(steps),
            "prompt_dir": str(prompt_dir),
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        metadata_file = item_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ Completed all {len(steps)} steps")
        return {"status": "SUCCESS", "total_steps": len(steps)}
        
    except Exception as e:
        print(f"   ❌ Critical error: {type(e).__name__}: {e}")
        return {"status": "ERROR", "error": str(e)}

# -------------------- Main Processing --------------------

def main():
    """Main function to process CSV and generate multi-step images."""
    global gemini_client
    
    args = parse_args()

    # Initialize Gemini client
    gemini_client = initialize_gemini_client()

    # Setup paths
    csv_path = pathlib.Path(args.csv)
    sbs_base_dir = pathlib.Path(args.sbs_base_dir)
    output_dir = pathlib.Path(args.outdir)
    ensure_dir(output_dir)
    
    # Load CSV
    df = load_and_validate_csv(csv_path)
    
    if len(df) == 0:
        print("⚠️ No successful sequences found in CSV")
        return
    
    # Optionally limit dataset for testing
    if args.limit_items:
        df = df.head(args.limit_items)
        print(f"🔬 Limited to {args.limit_items} sequences")
    
    # Create generation configuration
    config = create_generation_configuration(
        safety_level=args.safety_level,
        allow_all_ages=args.allow_all_ages
    )
    
    print(f"\n🎨 Starting multi-step image generation")
    print(f"   Model: {args.model}")
    print(f"   SBS Base Dir: {sbs_base_dir}")
    print(f"   Output: {output_dir}")
    print(f"   Sequences: {len(df)}")
    
    # Process each sequence
    results = []
    success_count = 0
    error_count = 0
    skip_count = 0
    
    for idx, row in df.iterrows():
        result = process_single_sequence(
            row, sbs_base_dir, output_dir, args.model, config, args.delay
        )
        
        results.append({
            "item_index": row['item_index'],
            "subject": row.get('subject', ''),
            **result
        })
        
        if result["status"] == "SUCCESS":
            success_count += 1
        elif result["status"] == "SKIPPED":
            skip_count += 1
        else:
            error_count += 1
    
    # Save processing results
    if results:
        results_df = pd.DataFrame(results)
        results_path = output_dir / "multi_step_generation_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\n💾 Results saved: {results_path}")
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"📊 Final Summary:")
    print(f"   ✅ Success: {success_count}")
    print(f"   ⏭️  Skipped: {skip_count}")
    print(f"   ❌ Errors: {error_count}")
    print(f"   📁 Total: {len(df)}")
    
    print(f"\n✅ Complete! Output: {output_dir}")

if __name__ == "__main__":
    main()
