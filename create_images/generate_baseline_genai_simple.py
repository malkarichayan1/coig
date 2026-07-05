#!/usr/bin/env python3
"""
Baseline Image Generator using Google Gemini for generated prompts.

This script reads from generated_prompts.csv and generates a baseline image
for each prompt using Gemini's image generation.

Outputs for each item:
  outdir/<item_index>/baseline.png       - generated baseline image
  outdir/<item_index>/baseline.txt       - prompt used
  outdir/<item_index>/metadata.json      - processing metadata

Usage example:
  python generate_baseline_genai_simple.py \
    --csv ../create_dataset/generated_prompts.csv \
    --outdir baseline_out \
    --model gemini-2.5-flash-image-preview \
    --limit_items 3

Environment:
  Requires GOOGLE_AI_API_KEY environment variable.
  Get one from: https://aistudio.google.com/app/apikey
"""

import os
import argparse
import pathlib
import time
import json
import pandas as pd
from typing import Any
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

# Google Gen AI SDK imports
from google import genai
from google.genai.types import GenerateContentConfig

# Global Gemini client instance
gemini_client = None

# -------------------- CLI --------------------

def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Create baseline images using Gemini from generated prompts.")
    # Input/Output settings
    p.add_argument("--csv", type=str, 
                   default="../create_dataset/generated_prompts.csv",
                   help="Path to CSV file containing prompts")
    p.add_argument("--outdir", type=str, 
                   default="baseline_out",
                   help="Output directory for generated baseline images")
    p.add_argument("--model", type=str, default="gemini-2.5-flash-image-preview",
                   help="Gemini image model to use")
    p.add_argument("--limit_items", type=int, default=None,
                   help="Limit number of items to process (for testing)")
    p.add_argument("--safety_level", type=str, default="BLOCK_NONE",
                   choices=["BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_LOW_AND_ABOVE"],
                   help="Content safety filter level")
    p.add_argument("--allow_all_ages", action="store_true",
                   help="Enable content for all age groups")
    p.add_argument("--delay", type=float, default=2.0,
                   help="Delay between API calls in seconds")
    p.add_argument("--prompt_prefix", type=str, 
                   default="",
                   help="Prefix to add before prompt")
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
    """Load and validate CSV with prompts."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required_columns = ['item_index', 'prompt']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Filter out rows with errors or missing prompts
    df_valid = df[df['prompt'].notna() & (df['prompt'] != '')].copy()
    
    # Check if there's an error column and filter those out
    if 'error' in df.columns:
        df_valid = df_valid[df_valid['error'].isna()].copy()
    
    print(f"📊 CSV Data Summary:")
    print(f"   Total rows in CSV: {len(df)}")
    print(f"   Valid prompts to process: {len(df_valid)}")
    
    return df_valid

# -------------------- Image Generation --------------------

def ensure_dir(path: pathlib.Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)

def save_image_bytes(image_bytes: bytes, output_path: pathlib.Path) -> None:
    """Save image bytes as PNG file."""
    ensure_dir(output_path.parent)
    image = Image.open(BytesIO(image_bytes))
    image.save(str(output_path), format="PNG")

def generate_baseline_image(prompt: str, *, model: str, config: GenerateContentConfig) -> tuple[bytes, Any]:
    """Generate baseline image from text prompt. Returns (image_bytes, response_object)."""
    print(f"🎨 Generating baseline image: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    
    response = gemini_client.models.generate_content(
        model=model,
        contents=[prompt],
        config=config
    )
    
    # Extract image data from response
    if response.candidates and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if part.text:
                print(f"   💬 Model response: {part.text}")
            if part.inline_data:
                print(f"   ✅ Baseline image generated successfully")
                return part.inline_data.data, response
    
    raise RuntimeError("No image data found in generation response")

# -------------------- Core Processing --------------------

def process_single_item(item_row: pd.Series, output_dir: pathlib.Path, 
                        model: str, config: GenerateContentConfig, 
                        prompt_prefix: str, delay: float) -> dict:
    """Process a single item and generate baseline image."""
    item_index = item_row['item_index']
    prompt = str(item_row['prompt']).strip()
    
    print(f"\n🔍 Item {item_index}")
    print(f"   🎯 Prompt: {prompt}")
    
    # Setup output directory
    item_dir = output_dir / str(item_index)
    ensure_dir(item_dir)
    
    # Check if already processed
    baseline_image_path = item_dir / "baseline.png"
    if baseline_image_path.exists():
        print(f"   ⏭️  Skipping: already processed")
        return {"status": "SKIPPED"}
    
    try:
        # Create full prompt
        full_prompt = f"{prompt_prefix}{prompt}"
        
        # Save prompt text
        prompt_file = item_dir / "baseline.txt"
        prompt_file.write_text(full_prompt, encoding="utf-8")
        
        # Generate baseline image
        image_bytes, response = generate_baseline_image(
            full_prompt, model=model, config=config
        )
        
        # Save baseline image
        save_image_bytes(image_bytes, baseline_image_path)
        print(f"   ✅ Saved: {baseline_image_path.name}")
        
        # Save metadata
        metadata = {
            "item_index": str(item_index),
            "prompt": prompt,
            "prompt_used": full_prompt,
            "model": model,
            "image_path": str(baseline_image_path),
            "prompt_file": str(prompt_file),
            "timestamp": pd.Timestamp.now().isoformat(),
            "mode": "baseline"
        }
        
        # Add additional metadata if available
        if 'subject' in item_row.index:
            metadata['subject'] = str(item_row['subject'])
        if 'attribute_1' in item_row.index:
            metadata['attributes'] = [
                str(item_row.get(f'attribute_{i}', '')) 
                for i in range(1, 5) 
                if f'attribute_{i}' in item_row.index
            ]
        if 'interaction_1' in item_row.index:
            metadata['interactions'] = [
                str(item_row.get(f'interaction_{i}', '')) 
                for i in range(1, 3) 
                if f'interaction_{i}' in item_row.index
            ]
        
        metadata_file = item_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Delay before next request
        time.sleep(delay)
        
        return {"status": "SUCCESS"}
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        
        # Log error details
        error_log_path = item_dir / "error.log"
        with open(error_log_path, 'w', encoding='utf-8') as f:
            f.write(f"Error occurred for item {item_index}\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Error type: {type(e).__name__}\n")
            f.write(f"Error message: {str(e)}\n\n")
            f.write(f"Full prompt used:\n")
            f.write(f"{'-'*80}\n")
            f.write(full_prompt)
            f.write(f"\n{'-'*80}\n")
        
        print(f"   📝 Error details saved to: {error_log_path.name}")
        return {"status": "ERROR", "error": str(e)}

# -------------------- Main Processing --------------------

def main():
    """Main function to process CSV and generate baseline images."""
    global gemini_client
    
    args = parse_args()

    # Initialize Gemini client
    gemini_client = initialize_gemini_client()

    # Setup paths
    csv_path = pathlib.Path(args.csv)
    output_dir = pathlib.Path(args.outdir)
    ensure_dir(output_dir)
    
    # Load CSV
    df = load_and_validate_csv(csv_path)
    
    # Optionally limit dataset for testing
    if args.limit_items:
        df = df.head(args.limit_items)
        print(f"🔬 Limited to {args.limit_items} items")
    
    # Create generation configuration
    config = create_generation_configuration(
        safety_level=args.safety_level,
        allow_all_ages=args.allow_all_ages
    )
    
    print(f"\n🎨 Starting baseline image generation")
    print(f"   Model: {args.model}")
    print(f"   Output: {output_dir}")
    print(f"   Items: {len(df)}")
    print(f"   Prompt prefix: '{args.prompt_prefix}'")
    
    # Process each item
    results = []
    success_count = 0
    error_count = 0
    skip_count = 0
    
    for idx, row in df.iterrows():
        result = process_single_item(
            row, output_dir, args.model, config, args.prompt_prefix, args.delay
        )
        
        results.append({"item_index": row['item_index'], **result})
        
        if result["status"] == "SUCCESS":
            success_count += 1
        elif result["status"] == "SKIPPED":
            skip_count += 1
        else:
            error_count += 1
    
    # Save processing results
    if results:
        results_df = pd.DataFrame(results)
        results_path = output_dir / "baseline_generation_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\n💾 Results saved: {results_path}")
    
    # Final summary
    print(f"\n📊 Final Summary:")
    print(f"   ✅ Success: {success_count}")
    print(f"   ⏭️  Skipped: {skip_count}")
    print(f"   ❌ Errors: {error_count}")
    print(f"   📁 Total: {len(df)}")
    
    print(f"\n✅ Complete! Output: {output_dir}")

if __name__ == "__main__":
    main()
