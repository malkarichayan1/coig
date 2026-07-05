#!/usr/bin/env python3
"""
Generate image generation prompts using Gemini API.
Samples subjects/attributes/interactions and creates prompts for image generation.
"""

import os
import sys
import argparse
import pathlib
import re
import pandas as pd
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig

# Import from sample_generator
from sample_generator import generate_sample


# =====================================================================================
# CONFIGURATION & SETUP
# =====================================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate image prompts using sampled data and Gemini")
    parser.add_argument("--out", type=str, default="generated_prompts.csv",
                       help="Path to output CSV")
    parser.add_argument("--num_samples", type=int, default=320,
                       help="Number of prompt samples to generate")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", 
                       help="Gemini model name")
    parser.add_argument("--temperature", type=float, default=0.7, 
                       help="Temperature for generation")
    parser.add_argument("--prompt_file", type=str, default="dataset_prompt.txt",
                       help="Path to prompt template file")
    return parser.parse_args()


def get_api_key():
    """Get Google AI API key from environment."""
    api_key = os.getenv("GOOGLE_AI_API_KEY")

    if not api_key:
        raise ValueError("No API key found. Set GOOGLE_AI_API_KEY environment variable.")

    return api_key


def initialize_gemini_client():
    """Initialize and return Gemini client."""
    api_key = get_api_key()
    return genai.Client(api_key=api_key)


# =====================================================================================
# PROMPT GENERATION
# =====================================================================================

def create_generation_config(temperature: float = 0.7) -> Dict[str, Any]:
    """Create generation configuration for Gemini."""
    return {
        "temperature": temperature,
        "top_p": 0.9,
        "top_k": 40,
        "max_output_tokens": 2048,
    }


def load_prompt_template(prompt_file: str) -> str:
    """Load prompt template from external file."""
    script_dir = pathlib.Path(__file__).parent
    prompt_path = script_dir / prompt_file
    
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
    except Exception as e:
        raise Exception(f"Error loading prompt template: {e}")


def format_attributes(attributes):
    """Format attributes list into a string."""
    lines = []
    for category, value in attributes:
        lines.append(f"  - {category}: {value}")
    return '\n'.join(lines)


def format_interactions(interactions):
    """Format interactions list into a string."""
    lines = []
    for interaction in interactions:
        lines.append(f"  - {interaction}")
    return '\n'.join(lines)


def create_base_prompt(sample: Dict) -> str:
    """
    Create a simple base prompt from sample data.
    
    Format: Four [Subject]s. [Subject] A has [attribute1]. [Subject] B has [attribute2]. 
            [Subject] C has [attribute3]. [Subject] D has [attribute4]. 
            [Subject] A and [Subject] B are [interaction1]. 
            [Subject] C and [Subject] D are [interaction2].
    """
    subject = sample['subject']
    attributes = sample['attributes']
    interactions = sample['interactions']
    
    # Extract attribute values
    attr_parts = []
    for i in range(4):
        if i < len(attributes):
            category, value = attributes[i]
            attr_parts.append(f"{category}: {value}")
        else:
            attr_parts.append("None")
    
    # Extract interactions
    interaction_1 = interactions[0] if len(interactions) > 0 else "standing"
    interaction_2 = interactions[1] if len(interactions) > 1 else "standing"
    
    # Build the base prompt
    base_prompt = (
        f"Four {subject}s. "
        f"{subject} A has {attr_parts[0]}. "
        f"{subject} B has {attr_parts[1]}. "
        f"{subject} C has {attr_parts[2]}. "
        f"{subject} D has {attr_parts[3]}. "
        f"{subject} A and {subject} B are {interaction_1}. "
        f"{subject} C and {subject} D are {interaction_2}."
    )
    
    return base_prompt


def fill_prompt_template(sample: Dict, prompt_template: str) -> str:
    """
    Fill the prompt template with sampled data.
    
    Args:
        sample: One sample dictionary with subject, 4 attributes, and 2 interactions
        prompt_template: The template loaded from dataset_prompt.txt
    
    Returns:
        Filled prompt text for Gemini
    """
    filled_prompt = prompt_template
    
    # Replace subject
    filled_prompt = filled_prompt.replace('{SUBJECT}', sample['subject'])
    
    # Replace individual attributes
    attributes = sample['attributes']
    for i in range(4):
        if i < len(attributes):
            category, value = attributes[i]
            attr_text = f"{category}: {value}"
        else:
            attr_text = "None"
        filled_prompt = filled_prompt.replace(f'{{ATTRIBUTE_{i+1}}}', attr_text)
    
    # Replace interactions
    interactions = sample['interactions']
    for i in range(2):
        if i < len(interactions):
            interaction_text = interactions[i]
        else:
            interaction_text = "None"
        filled_prompt = filled_prompt.replace(f'{{INTERACTION_{i+1}}}', interaction_text)
    
    return filled_prompt


def generate_image_prompt(prompt_template: str, model_name: str = "gemini-2.5-flash", 
                         temperature: float = 0.7) -> Dict[str, Any]:
    """
    Generate improved image generation prompt and questions using Gemini API with retry logic.
    
    Returns dict with:
        - 'sample': One sample dict with subject, 4 attributes, 2 interactions
        - 'filled_prompt': The prompt sent to Gemini
        - 'generated_prompt': The image generation prompt from Gemini
        - 'questions': Dict of questions
        - 'error': Error message if failed
    """
    # Generate ONE sample (1 subject, 4 attributes, 2 interactions)
    sample = generate_sample()
    
    # Fill the template
    filled_prompt = fill_prompt_template(sample, prompt_template)
    
    # Initialize Gemini client
    client = initialize_gemini_client()
    
    # Create generation config
    config = create_generation_config(temperature=temperature)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"      Attempt {attempt + 1}/{max_retries}...", end=" ")
            
            response = client.models.generate_content(
                model=model_name,
                contents=filled_prompt,
                config=GenerateContentConfig(**config)
            )
            
            if not response or not response.text:
                print(f"Empty response")
                if attempt < max_retries - 1:
                    continue
                return {
                    'sample': sample,
                    'filled_prompt': filled_prompt,
                    'error': 'Empty response from Gemini'
                }
            
            response_text = response.text.strip()
            
            # Parse the response to extract prompt and questions
            prompt_match = re.search(r'Prompt:\s*(.+?)(?=\nquestion_count:|$)', response_text, re.DOTALL)
            
            if not prompt_match:
                print(f"Failed to parse response")
                print(f"Response was: {response_text[:200]}...")
                if attempt < max_retries - 1:
                    continue
                return {
                    'sample': sample,
                    'filled_prompt': filled_prompt,
                    'error': 'Failed to parse Gemini response'
                }
            
            generated_prompt = prompt_match.group(1).strip()
            
            # Extract questions with more flexible regex
            questions = {}
            question_patterns = [
                ('question_count', r'question_count:\s*(.+?)(?=\nquestion_|\n\n|$)'),
                ('question_attr_1', r'question_attr_1:\s*(.+?)(?=\nquestion_|\n\n|$)'),
                ('question_attr_2', r'question_attr_2:\s*(.+?)(?=\nquestion_|\n\n|$)'),
                ('question_attr_3', r'question_attr_3:\s*(.+?)(?=\nquestion_|\n\n|$)'),
                ('question_attr_4', r'question_attr_4:\s*(.+?)(?=\nquestion_|\n\n|$)'),
                ('question_interaction_1', r'question_interaction_1:\s*(.+?)(?=\nquestion_|\n\n|$)'),
                ('question_interaction_2', r'question_interaction_2:\s*(.+?)(?=\n\n|$)'),
            ]
            
            missing_questions = []
            for key, pattern in question_patterns:
                match = re.search(pattern, response_text, re.DOTALL)
                if match:
                    questions[key] = match.group(1).strip()
                else:
                    questions[key] = ''
                    missing_questions.append(key)
            
            # If any questions are missing, retry
            if missing_questions and attempt < max_retries - 1:
                print(f"Missing questions: {', '.join(missing_questions)}, retrying...")
                continue
            
            print(f"✓ Success")
            
            return {
                'sample': sample,
                'filled_prompt': filled_prompt,
                'generated_prompt': generated_prompt,
                'questions': questions
            }
            
        except Exception as e:
            print(f"Error: {str(e)}")
            if attempt < max_retries - 1:
                print(f"      Retrying...")
                continue
            return {
                'sample': sample,
                'filled_prompt': filled_prompt,
                'error': str(e)
            }
    
    return {
        'sample': sample,
        'filled_prompt': filled_prompt,
        'error': f"Failed after {max_retries} attempts"
    }


# =====================================================================================
# CHECKPOINT & RESUME LOGIC
# =====================================================================================

def load_existing_results(output_path: str) -> Tuple[pd.DataFrame, int]:
    """
    Load existing results from output CSV if it exists.
    Returns (existing_df, completed_count).
    """
    out_path = pathlib.Path(output_path)
    
    if not out_path.exists():
        return pd.DataFrame(), 0
    
    try:
        existing_df = pd.read_csv(out_path)
        completed_count = len(existing_df)
        print(f"📂 Found existing results: {completed_count} prompts already generated")
        return existing_df, completed_count
    except Exception as e:
        print(f"⚠️  Error loading existing results: {e}")
        return pd.DataFrame(), 0


# =====================================================================================
# MAIN EXECUTION
# =====================================================================================

def main():
    """Main execution function."""
    load_dotenv()  # Load environment variables
    args = parse_args()
    
    print(f"🎯 Generating {args.num_samples} image prompts")
    print(f"🤖 Using model: {args.model}")
    print(f"🌡️ Temperature: {args.temperature}")
    print(f"💾 Output file: {args.out}")
    
    # Load prompt template
    try:
        prompt_template = load_prompt_template(args.prompt_file)
        print(f"✓ Loaded prompt template from {args.prompt_file}")
    except Exception as e:
        print(f"❌ Error loading prompt template: {e}")
        return
    
    # Load existing results (if any)
    existing_df, completed_count = load_existing_results(args.out)
    
    # Determine how many more to generate
    remaining = args.num_samples - completed_count
    if remaining <= 0:
        print(f"✓ Already have {completed_count} prompts (target: {args.num_samples})")
        print(f"All done! No additional prompts needed.")
        return
    
    print(f"📊 Need to generate {remaining} more prompts")
    print()
    
    # Start with existing results
    out_rows: List[Dict[str, Any]] = existing_df.to_dict('records') if not existing_df.empty else []
    success_count = 0
    error_count = 0
    
    # Generate the remaining prompts
    for i in range(remaining):
        item_index = completed_count + i
        print(f"[{item_index + 1}/{args.num_samples}] Generating prompt...")
        
        try:
            result = generate_image_prompt(prompt_template, args.model, args.temperature)
            
            if 'error' in result:
                print(f"   ❌ Failed: {result['error']}")
                error_count += 1
                
                # Save error row
                row = {
                    'item_index': item_index,
                    'error': result['error'],
                    'filled_prompt': result['filled_prompt'],
                    'subject': result['sample']['subject'],
                    
                    # --- EDITED FOR CONVENIENCE ---
                    'attributes': str(result['sample']['attributes']),
                    'interactions': str(result['sample']['interactions']),
                    # ------------------------------
                }
                
                out_rows.append(row)
            else:
                print(f"   ✓ Success")
                success_count += 1
                
                # Save successful row - use questions from Gemini response
                sample = result['sample']
                
                row = {
                    'item_index': item_index,
                    'prompt': result['generated_prompt'],
                    'filled_prompt': result['filled_prompt'],
                    'subject': sample['subject'],

                    # --- EDITED FOR CONVENIENCE ---
                    'attributes': str(sample['attributes']),
                    'interactions': str(sample['interactions']),
                    # ------------------------------
                }
                
                # Add questions from Gemini (still useful for yes/no eval)
                row.update(result['questions'])
                
                out_rows.append(row)

            
            # Save checkpoint after each iteration
            if out_rows:
                df_temp = pd.DataFrame(out_rows)
                df_temp.to_csv(args.out, index=False)
                
        except Exception as e:
            print(f"   ❌ Unexpected error: {str(e)}")
            error_count += 1
    
    # Print summary statistics
    print(f"\n📊 Generation Summary:")
    print(f"   Successful: {success_count}")
    print(f"   Failed: {error_count}")
    print(f"   Total generated: {completed_count + success_count}")
    print(f"   Output saved to: {args.out}")


if __name__ == "__main__":
    main()