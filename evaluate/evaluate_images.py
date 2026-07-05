#!/usr/bin/env python3
"""
Image Evaluation Script using Gemini Vision (JSON Extraction & Grading v4)

This script:
1. Loads prompts, subjects, attributes, and interactions from generated_prompts.csv
2. For each item, loads the baseline image (baseline.png)
3. Sends the image, prompt, subject, and required lists to Gemini API.
4. Grades the resulting JSON output using the 7-point custom rubric.
5. Saves results (scores and raw JSON) to a CSV file.

Usage example:
  python evaluate_images.py \
    --prompts_csv ../create_dataset/generated_prompts.csv \
    --image_dir ../create_images/baseline_out \
    --output_csv evaluation_results.csv \
    --limit 3
"""

import os
import sys
import argparse
import pathlib
import time
import pandas as pd
import json
import re
import ast  # <-- NEW: For safely parsing string lists
from typing import Dict, Any, Set, Tuple, List, Optional
from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig
from tqdm import tqdm
from PIL import Image as PILImage

# =====================================================================================
# CONFIGURATION & SETUP
# =====================================================================================

# Evaluation prompt template (Updated to accept lists + subject)
EVALUATION_PROMPT_TEMPLATE = """You are an AI quality auditor for text-to-image generation.

Your task is to meticulously analyze the provided image and bind entities to a pre-defined list of attributes and interactions.

**Inputs:**
1.  **Prompt:** General context for the image.
2.  **Required Subject:** The *only* type of person you should identify (e.g., "policeman", "nurse").
3.  **Required Attributes:** The *only* attribute labels you are allowed to use.
4.  **Required Interactions:** The *only* interaction labels you are allowed to use.

Your judgment must rely **entirely on clear visual evidence**.

**Instructions:**
1.  **Identify Entities:** Scan the image and identify every person matching the **Required Subject** (e.g., "policeman").
2.  **Assign IDs:** Assign each *matching* person a unique ID, like "P1", "P2", "P3", etc.
3.  **Assign Attributes:** Look at the `REQUIRED_ATTRIBUTES` list. For each person (P1, P2, etc.), list which attributes from this *exact* list are clearly visible on them. **Do not use any attribute words *not* in this list.**
4.  **Assign Interactions:** Look at the `REQUIRED_INTERACTIONS` list. Find all interactions in the image *between the identified entities (P1, P2, etc.)* that match an item from this list.

You must return **only** a single, valid JSON object and no other text.
Ambiguity means you do not include the data.

**JSON Format Example:**
(If Required Subject was "policeman" and attributes were "sunglasses", "radio")
{{
  "entities": [
    {{ "id": "P1", "attributes": ["sunglasses"] }},
    {{ "id": "P2", "attributes": ["radio"] }}
  ],
  "interactions": [
    {{ "type": "talking", "participants": ["P1", "P2"] }}
  ]
}}

Now, analyze this image using the provided lists.

Prompt: {prompt}
REQUIRED_SUBJECT: {subject}
REQUIRED_ATTRIBUTES: {attributes_list}
REQUIRED_INTERACTIONS: {interactions_list}
"""


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate generated images using Gemini Vision")
    parser.add_argument(
        "--prompts_csv",
        type=str,
        default="../create_dataset/generated_prompts.csv",
        help="Path to generated prompts CSV file (must include subject, attributes, interactions)"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="../create_images/baseline_out",
        help="Base directory containing images (with item_index subdirectories)"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="evaluation_results.csv",
        help="Path to output CSV file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-pro",
        help="Gemini model name (use vision-capable model)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation (0.0 for deterministic)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of items to process (for testing)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between API calls in seconds (to avoid rate limiting)"
    )
    parser.add_argument(
        "--retry_errors",
        action="store_true",
        default=True,
        help="Retry evaluations that previously returned errors (default: True)"
    )
    parser.add_argument(
        "--no_retry_errors",
        action="store_false",
        dest="retry_errors",
        help="Do not retry error evaluations"
    )
    return parser.parse_args()


def get_api_key():
    """Get Google AI API key from environment."""
    api_key = os.getenv("GOOGLE_AI_API_KEY")
    
    if not api_key:
        raise ValueError("Google AI API Key is required")
    
    return api_key


def initialize_gemini_client():
    """Initialize and return Gemini client."""
    api_key = get_api_key()
    return genai.Client(api_key=api_key)


def create_generation_config(temperature: float = 0.0) -> Dict[str, Any]:
    """Create generation configuration for Gemini."""
    return {
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 2048, # Increased for JSON
    }

def find_baseline_image(image_dir: str, item_index: int) -> str:
    """
    Find the baseline.png image for a given item_index.
    """
    possible_paths = [
        os.path.join(image_dir, str(item_index), "baseline.png"),
        os.path.join(image_dir, f"{item_index}", "baseline.png"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    raise FileNotFoundError(f"Could not find baseline.png for item_index {item_index}. Tried: {possible_paths}")

# =====================================================================================
# GEMINI EVALUATION (Updated)
# =====================================================================================

def evaluate_image_with_gemini(
    client,
    image_path: str,
    prompt: str,
    subject: str,              # <-- NEW
    attributes_list: str,      # <-- NEW
    interactions_list: str,  # <-- NEW
    model_name: str,
    config: Dict[str, Any],
    max_retries: int = 3
) -> Optional[Dict[str, Any]]:
    """
    Evaluate an image using Gemini Vision API and return a parsed JSON dictionary.
    """
    # Format the evaluation prompt
    evaluation_prompt = EVALUATION_PROMPT_TEMPLATE.format(
        prompt=prompt,
        subject=subject,                # <-- NEW
        attributes_list=attributes_list,
        interactions_list=interactions_list
    )
    
    # Read the image file using PIL
    try:
        pil_image = PILImage.open(image_path)
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None
    
    # Try evaluation with retries
    for attempt in range(max_retries):
        try:
            # Create the request with both text and image using PIL Image
            response = client.models.generate_content(
                model=model_name,
                contents=[evaluation_prompt, pil_image],
                config=GenerateContentConfig(**config)
            )
            
            # Extract the response text
            if not (response.candidates and response.candidates[0].content.parts):
                print(f"Empty response from Gemini")
                continue # Retry

            response_text = response.candidates[0].content.parts[0].text.strip()
            
            # Clean and parse JSON
            json_match = re.search(r'```(?:json)?\n(.*?)\n```', response_text, re.DOTALL | re.IGNORECASE)
            if json_match:
                json_str = json_match.group(1)
            elif response_text.startswith('{'):
                json_str = response_text
            else:
                print(f"Unexpected response format (no JSON): {response_text[:200]}...")
                continue # Retry

            try:
                parsed_json = json.loads(json_str)
                return parsed_json
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError: {e}\nRaw string: {json_str[:200]}...")
                continue # Retry
                
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            return None
    
    return None

# =====================================================================================
# GRADING LOGIC (Updated to match new prompt)
# =====================================================================================

def grade_json_output(
    json_output: Dict[str, Any], 
    ground_truth: Dict[str, Any]
) -> Dict[str, int]:
    """
    Grades the parsed JSON output based on the 7-point rubric.
    """
    scores = { "entity": 0, "attribute": 0, "interaction": 0, "total": 0 }
    
    # Ensure JSON is well-formed
    if not json_output or "entities" not in json_output or "interactions" not in json_output:
        return { "entity": -1, "attribute": -1, "interaction": -1, "total": -1 } # Error code
        
    entities = json_output.get("entities", [])
    interactions = json_output.get("interactions", [])
    gt_count = ground_truth["entity_count"]
    gt_attrs = ground_truth["attributes"]
    gt_interactions_count = ground_truth["interaction_count"]

    # 1. Entity Count (1 Point)
    # This now checks if the MLLM found the correct number of *subjects*
    found_entities = len(entities)
    if found_entities == gt_count:
        scores["entity"] = 1

    # 2. Attribute Spread (4 Points)
    participants_with_attributes = set()
    for entity in entities:
        entity_id = entity.get("id")
        # Attributes from JSON should be simple, exact strings now
        entity_attrs = set(attr.lower().strip() for attr in entity.get("attributes", []))

        # Simple intersection is all we need now
        if not entity_attrs.isdisjoint(gt_attrs):
            participants_with_attributes.add(entity_id)
                
    scores["attribute"] = len(participants_with_attributes)
    
    # Cap score at ground truth entity count
    if scores["attribute"] > gt_count:
        scores["attribute"] = gt_count

    # 3. Interaction Binding (2 Points)
    found_interactions = len(interactions)
    
    if found_interactions != gt_interactions_count:
        scores["interaction"] = 0 # Total fail (wrong number)
    else:
        # Check for participant reuse
        all_participants = []
        valid_interaction_structure = True
        for interaction in interactions:
            participants = interaction.get("participants", [])
            if not participants: # Malformed interaction
                valid_interaction_structure = False
                break
            all_participants.extend(participants)
        
        if not valid_interaction_structure:
            scores["interaction"] = 0 # Total fail (malformed)
        else:
            has_duplicates = len(all_participants) != len(set(all_participants))
            
            if has_duplicates:
                scores["interaction"] = 1 # Partial fail (reused person)
            else:
                scores["interaction"] = 2 # Perfect

    # Calculate total
    scores["total"] = scores["entity"] + scores["attribute"] + scores["interaction"]
    return scores


# (load_existing_results and get_completed_evaluations are unchanged)
def load_existing_results(output_path: str) -> pd.DataFrame:
    """
    Load existing results from output CSV if it exists.
    """
    out_path = pathlib.Path(output_path)
    
    if not out_path.exists():
        print("📝 No existing output file found. Starting fresh.")
        return pd.DataFrame()
    
    try:
        existing_df = pd.read_csv(out_path)
        print(f"📂 Found existing output: {output_path}")
        print(f"   Loaded {len(existing_df)} existing evaluations")
        return existing_df
    except Exception as e:
        print(f"⚠️  Error loading existing file: {e}. Starting fresh.")
        return pd.DataFrame()


def get_completed_evaluations(existing_results_df: pd.DataFrame) -> Set[int]:
    """
    Get set of (item_index) that have been completed successfully.
    Excludes error rows (score_total == -1).
    """
    if existing_results_df.empty:
        return set()
    
    # Ensure score_total column exists and is numeric
    if 'score_total' not in existing_results_df.columns:
        return set()
        
    try:
        existing_results_df['score_total'] = pd.to_numeric(existing_results_df['score_total'])
        completed = set(
            existing_results_df[existing_results_df['score_total'] != -1]['item_index']
        )
        return completed
    except Exception as e:
        print(f"Error parsing 'score_total' in existing results: {e}. Retrying all.")
        return set()

# =====================================================================================
# HELPER FUNCTIONS (Removed)
# =====================================================================================
# (extract_count_from_question and extract_attribute_from_question are no longer needed)

# =====================================================================================
# MAIN EXECUTION (Updated)
# =====================================================================================

def main():
    """Main execution function."""
    load_dotenv()  # Load environment variables
    args = parse_args()
    
    print("="*80)
    print("IMAGE EVALUATION WITH GEMINI VISION (JSON + GRADING v4)") #<-- Version bump
    print("="*80)
    print(f"🤖 Using model: {args.model}")
    print(f"🌡️ Temperature: {args.temperature}")
    print(f"📊 Prompts CSV: {args.prompts_csv}")
    print(f"🖼️  Image directory: {args.image_dir}")
    print(f"💾 Output CSV: {args.output_csv}")
    
    # Initialize Gemini client
    print("\n🔧 Initializing Gemini client...")
    client = initialize_gemini_client()
    config = create_generation_config(temperature=args.temperature)
    print("✅ Gemini client initialized")
    
    # Load prompts CSV
    print(f"\n📖 Loading prompts from {args.prompts_csv}...")
    try:
        prompts_df = pd.read_csv(args.prompts_csv)
    except FileNotFoundError:
        print(f"Error: Prompts CSV not found at {args.prompts_csv}")
        return
        
    # --- Check for NEW Ground Truth columns ---
    required_gt_columns = ['prompt', 'attributes', 'interactions', 'subject']
    missing_cols = [col for col in required_gt_columns if col not in prompts_df.columns]
    if missing_cols:
        print("\n" + "!"*80)
        print(f"Error: Missing required columns in {args.prompts_csv}")
        print(f"Please ensure at least the following columns exist: {', '.join(missing_cols)}")
        print("!"*80)
        return
    # --------------------------------------

    # Filter out rows with errors
    if 'error' in prompts_df.columns:
        prompts_df = prompts_df[prompts_df['error'].isna()].copy()
    
    # Filter out rows with missing prompts
    prompts_df = prompts_df[prompts_df['prompt'].notna() & (prompts_df['prompt'] != '')].copy()
    
    print(f"   Loaded {len(prompts_df)} valid items")
    
    # Apply limit if specified
    if args.limit is not None:
        prompts_df = prompts_df.head(args.limit)
        print(f"   📊 Limited to {args.limit} items")
    
    # Load existing results (if any)
    print(f"\n🔍 Checking for existing results...")
    existing_results_df = load_existing_results(args.output_csv)
    completed = get_completed_evaluations(existing_results_df)

    # --- Build Ground Truth Map (NEW) ---
    print("\nBuilding ground truth map from prompts CSV...")
    ground_truth_map = {}
    for _, prow in prompts_df.iterrows():
        try:
            item_index = prow['item_index']
            # 1. Parse attributes
            attr_tuples = ast.literal_eval(prow['attributes'])
            gt_attributes_set = {str(attr[1]).lower() for attr in attr_tuples}
            
            # 2. Parse interactions
            gt_interactions_list = ast.literal_eval(prow['interactions'])
            gt_interactions_set = {str(ia).lower() for ia in gt_interactions_list}

            # 3. Get Entity Count (assuming 4 attributes = 4 people)
            gt_entity_count = len(gt_attributes_set) 
            
            # 4. Format lists for the prompt
            attributes_list_str = ", ".join(f'"{attr}"' for attr in gt_attributes_set)
            interactions_list_str = ", ".join(f'"{ia}"' for ia in gt_interactions_set)

            ground_truth_map[item_index] = {
                "prompt": prow['prompt'],
                "subject": prow['subject'],
                "entity_count": gt_entity_count,
                "attributes_set_for_grading": {attr.lower() for attr in gt_attributes_set},
                "interaction_count_for_grading": len(gt_interactions_set),
                "attributes_list_for_prompt": attributes_list_str,
                "interactions_list_for_prompt": interactions_list_str
            }
        except Exception as e:
            print(f"Warning: Could not parse ground truth for item_index {prow.get('item_index')}: {e}")
    
    print(f"   Successfully built ground truth for {len(ground_truth_map)} items.")
    # -------------------------------------------------------------------------
    
    
    # Build list of evaluations to process
    evaluations_to_process = []
    
    for _, row in prompts_df.iterrows():
        item_index = row['item_index']
        
        # Check if already completed
        if item_index not in completed:
            if item_index in ground_truth_map: # Only process if GT was built
                evaluations_to_process.append(row)
    
    # Calculate statistics
    total_possible = len(prompts_df)
    already_completed = len(completed)
    to_process = len(evaluations_to_process)
    
    print(f"   ✓ Already evaluated: {already_completed}/{total_possible}")
    print(f"   → Evaluations to process: {to_process}")
    
    # If nothing to process, exit
    if to_process == 0:
        print("\n🎉 All evaluations already completed! Nothing to do.")
        return
    
    # Start evaluation
    print(f"\n🚀 Starting evaluation of {to_process} images...")
    
    results = []
    success_count = 0
    error_count = 0
    
    # Process each evaluation
    for row in tqdm(evaluations_to_process, desc="Evaluating"):
        item_index = row['item_index']
        
        # Get the ground truth details
        try:
            ground_truth_details = ground_truth_map[item_index]
        except KeyError:
            print(f"Warning: Skipping item {item_index}, failed to build ground truth.")
            continue
            
        prompt = ground_truth_details['prompt']
        
        # Find the baseline image
        try:
            image_path = find_baseline_image(args.image_dir, item_index)
        except FileNotFoundError as e:
            print(f"\n⚠️  {e}")
            error_count += 1
            results.append({
                'item_index': item_index,
                'prompt': prompt,
                'image_path': 'NOT_FOUND',
                'json_output': 'error_img_not_found',
                'score_entity': -1,
                'score_attribute': -1,
                'score_interaction': -1,
                'score_total': -1
            })
            continue
        
        # Evaluate with Gemini
        json_output = evaluate_image_with_gemini(
            client=client,
            image_path=image_path,
            prompt=prompt,
            subject=ground_truth_details['subject'],                                  # <-- PASS IT
            attributes_list=ground_truth_details['attributes_list_for_prompt'],     # <-- PASS IT
            interactions_list=ground_truth_details['interactions_list_for_prompt'], # <-- PASS IT
            model_name=args.model,
            config=config
        )
        
        # Grade the output
        if json_output:
            success_count += 1
            json_output_str = json.dumps(json_output)
            
            # --- Build ground truth for grading function ---
            try:
                grading_gt = {
                    "entity_count": ground_truth_details['entity_count'],
                    "attributes": ground_truth_details['attributes_set_for_grading'],
                    "interaction_count": ground_truth_details['interaction_count_for_grading']
                }
                scores = grade_json_output(json_output, grading_gt)
            except Exception as e:
                print(f"Error grading item {item_index}: {e}")
                scores = { "entity": -2, "attribute": -2, "interaction": -2, "total": -2 } # Grading error code
            # -----------------------------------------------------

        else:  # API error
            error_count += 1
            json_output_str = "error_api"
            scores = { "entity": -1, "attribute": -1, "interaction": -1, "total": -1 } # API error code
        
        # Store result
        results.append({
            'item_index': item_index,
            'subject': ground_truth_details['subject'],
            'prompt': prompt,
            'image_path': image_path,
            'json_output': json_output_str,
            'score_entity': scores['entity'],
            'score_attribute': scores['attribute'],
            'score_interaction': scores['interaction'],
            'score_total': scores['total']
        })
        
        # Add delay to avoid rate limiting
        time.sleep(args.delay)
    
    # (The rest of the script: combining results, saving, and printing stats is unchanged)
    
    # Combine with existing results
    new_results_df = pd.DataFrame(results)
    
    if not existing_results_df.empty:
        # Remove error rows that we just retried from existing results
        retried_items = set(r['item_index'] for r in results)
        
        # Keep only rows that were NOT retried
        existing_to_keep = existing_results_df[
            ~existing_results_df['item_index'].isin(retried_items)
        ]
        
        # Combine kept existing results with new results
        combined_df = pd.concat([existing_to_keep, new_results_df], ignore_index=True)
        
        # Report how many errors were replaced
        num_replaced = len(existing_results_df) - len(existing_to_keep)
        if num_replaced > 0:
            print(f"\n🔄 Replaced {num_replaced} error evaluations with new attempts")
    else:
        combined_df = new_results_df
    
    # Sort by item_index for consistency
    combined_df = combined_df.sort_values('item_index').reset_index(drop=True)
    
    # Save results
    output_path = pathlib.Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Evaluations processed this run: {to_process}")
    print(f"   ✅ Successful API calls: {success_count}")
    print(f"   ❌ Errors (API/Image): {error_count}")
    
    # Overall statistics
    # -1 = API error, -2 = Grading error
    valid_scores_df = combined_df[combined_df['score_total'] >= 0]
    total_evaluated = len(valid_scores_df)
    total_api_errors = len(combined_df[combined_df['score_total'] == -1])
    total_grading_errors = len(combined_df[combined_df['score_total'] == -2])
    
    print(f"\nOverall statistics (all runs):")
    print(f"   Total items in CSV: {len(combined_df)}")
    print(f"   Successfully graded: {total_evaluated}")
    print(f"   API/Image Errors: {total_api_errors}")
    print(f"   Grading Errors: {total_grading_errors}")
    
    if total_evaluated > 0:
        # Calculate max possible scores
        max_entity = 1.0
        try:
            # Get max attribute score from the last processed item's GT
            max_attribute = ground_truth_details['entity_count']
        except NameError: # Handle if loop didn't run
            max_attribute = 4.0 # Default to 4
        
        max_interaction = 2.0 # Fixed at 2 interactions
        max_total = max_entity + max_attribute + max_interaction

        avg_entity = valid_scores_df['score_entity'].mean()
        avg_attribute = valid_scores_df['score_attribute'].mean()
        avg_interaction = valid_scores_df['score_interaction'].mean()
        avg_total = valid_scores_df['score_total'].mean()
        
        print(f"\nAverage Scores (approx. out of {max_total:.1f}):")
        print(f"   Total:       {avg_total:.2f} / {max_total:.1f}")
        print(f"   Entity:      {avg_entity:.2f} / {max_entity:.1f}")
        print(f"   Attribute:   {avg_attribute:.2f} / {max_attribute:.1f}")
        print(f"   Interaction: {avg_interaction:.2f} / {max_interaction:.1f}")

    print(f"\n💾 Results saved to: {output_path}")
    print("✅ Evaluation complete!")


if __name__ == "__main__":
    main()