#!/usr/bin/env python3
"""
Sequential Prompt Generator for Baseline Images

This script reads from generated_prompts.csv (created by generate_dataset.py),
processes each prompt to generate step-by-step sequences using the six-step
texture prompt template with Google's Gemini API, and saves the results.

Usage:
    python create_prompt_sbs.py
    python create_prompt_sbs.py --csv ../create_dataset/generated_prompts.csv
    python create_prompt_sbs.py --model gemini-2.5-pro --limit_items 5
"""

import os
import sys
import re
import json
import time
import argparse
import pathlib
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig
import pandas as pd

# =====================================================================================
# CONFIGURATION & SETUP
# =====================================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate step-by-step prompts from baseline prompts")
    parser.add_argument("--csv", type=str, default="../create_dataset/generated_prompts.csv",
                       help="Input CSV file containing prompts (default: ../create_dataset/generated_prompts.csv)")
    parser.add_argument("--output", type=str, default="./sbs_prompts_results.csv",
                       help="Output CSV file path (default: ./sbs_prompts_results.csv)")
    parser.add_argument("--model", type=str, default="gemini-flash-latest",
                       choices=["gemini-flash-latest", "gemini-2.5-flash", "gemini-2.5-pro"],
                       help="Gemini model name (default: gemini-flash-latest, which the free tier serves; "
                            "gemini-2.5-flash is 404 for new keys and gemini-2.5-pro requires paid billing)")
    parser.add_argument("--temperature", type=float, default=0.3, 
                       help="Temperature for generation (default: 0.3)")
    parser.add_argument("--max_tokens", type=int, default=None,
                       help="Override max_output_tokens (Flash: up to 8192, Pro: up to 32000)")
    parser.add_argument("--skip_existing", action="store_true",
                       help="Skip prompts that have already been processed")
    parser.add_argument("--limit_items", type=int, default=None,
                       help="Limit number of items to process (for testing)")
    parser.add_argument("--delay", type=float, default=2.0,
                       help="Delay between API calls in seconds")
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

def load_base_prompt_template() -> str:
    """Load the six-step prompt template from base_prompt.txt."""
    script_dir = pathlib.Path(__file__).parent
    template_path = script_dir / "base_prompt.txt"
    
    if not template_path.exists():
        raise FileNotFoundError(f"Base prompt template not found: {template_path}")
    
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

# =====================================================================================
# CSV PROCESSING
# =====================================================================================

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
        # This line is commented out to re-run items that previously had errors
        # df_valid = df_valid[df_valid['error'].isna()].copy()
        pass
    
    print(f"📊 CSV Data Summary:")
    print(f"   Total rows in CSV: {len(df)}")
    print(f"   Valid prompts to process: {len(df_valid)}")
    
    return df_valid

# =====================================================================================
# PROMPT PROCESSING
# =====================================================================================

def create_sequential_prompt(base_template: str, complex_prompt: str) -> str:
    """Create the full prompt for the Sequential Prompt Architect."""
    # Replace placeholder in template
    return base_template.replace("[PLACEHOLDER FOR THE COMPLEX PROMPT]", complex_prompt)

def validate_steps(steps: List[Dict], original_prompt: str) -> List[Dict]:
    """
    Validates that steps contain placement and action.
    Fills 'final_goal' from original prompt if missing.
    Raises ValueError to trigger retry if 'placement' or 'step_action' are empty.
    """
    validated_steps = []
    missing_fields = False
    for i, step_data in enumerate(steps):
        # Extract data from the step, which might be in various formats
        if isinstance(step_data, dict):
            fg = step_data.get("final_goal", "")
            pl = step_data.get("placement", "")
            ac = step_data.get("step_action", "")
        else:
            # Fallback if step_data isn't the expected dict
            fg, pl, ac = "", "", ""

        # Use original prompt as fallback for Final Goal
        if not fg:
            fg = original_prompt

        # Check for missing critical fields
        if not pl:
            print(f"      ⚠️ Validation Warning: Step {i+1} is missing 'Placement'.")
            missing_fields = True
        if not ac:
            print(f"      ⚠️ Validation Warning: Step {i+1} is missing 'This Step's Action'.")
            missing_fields = True
        
        validated_steps.append({"final_goal": fg, "placement": pl, "step_action": ac})

    if missing_fields:
        # User requested a rerun, so we raise an error to trigger the retry loop
        raise ValueError("Parsed steps are missing critical 'Placement' or 'Action' fields.")
    
    return validated_steps

# -------------------- Optional OpenRouter text backend --------------------
# When USE_OPENROUTER=1, the CSP decomposition call is routed through OpenRouter
# (default model google/gemini-2.5-pro) instead of the free Gemini tier, which
# produced malformed decompositions (empty Placement/Action) on complex prompts.
# Requires OPENROUTER_API_KEY. Default is OFF.

def _use_openrouter_text() -> bool:
    return os.getenv("USE_OPENROUTER") == "1"

def _openrouter_chat(prompt: str, temperature: float = 0.3, max_tokens: int = None) -> str:
    import json as _json
    import urllib.request as _urlreq
    import urllib.error as _urlerr

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("USE_OPENROUTER=1 but OPENROUTER_API_KEY is not set")
    url = "https://openrouter.ai/api/v1/chat/completions"
    model = os.getenv("OPENROUTER_TEXT_MODEL", "google/gemini-2.5-pro")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if max_tokens:
        payload["max_tokens"] = int(max_tokens)
    data = _json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    req = _urlreq.Request(url, data=data, headers=headers)
    try:
        with _urlreq.urlopen(req, timeout=float(os.getenv("OPENROUTER_TIMEOUT", "180"))) as resp:
            body = _json.loads(resp.read().decode("utf-8"))
    except _urlerr.HTTPError as e:
        raise RuntimeError(f"OpenRouter HTTP {e.code}: {e.read().decode('utf-8','replace')[:300]}")
    if isinstance(body, dict) and body.get("error"):
        raise RuntimeError(f"OpenRouter error: {body['error']}")
    return body["choices"][0]["message"]["content"]

def generate_sequential_steps(complex_prompt: str, base_template: str, model_name: str = "gemini-2.5-flash",
                            temperature: float = 0.3, max_tokens: int = None) -> Dict[str, Any]:
    """Generate step-by-step prompts using Gemini API (or OpenRouter if enabled)."""

    # Initialize Gemini client (only needed for the Gemini path)
    client = None if _use_openrouter_text() else initialize_gemini_client()
    
    # Create generation config with higher output limits
    if model_name == "gemini-2.5-pro":
        default_tokens = 20000 if max_tokens is None else max_tokens
        config = {
            "temperature": temperature,
            "top_p": 0.85,
            "top_k": 30,
            "max_output_tokens": min(default_tokens, 32000),
            "response_mime_type": "application/json",  # Request JSON response
        }
    else:  # gemini-2.5-flash
        default_tokens = 8000 if max_tokens is None else max_tokens
        config = {
            "temperature": temperature,
            "top_p": 0.75,
            "top_k": 25,
            "max_output_tokens": min(default_tokens, 8192),
            "response_mime_type": "application/json",  # Request JSON response
        }
    
    # Create the full prompt
    full_prompt = create_sequential_prompt(base_template, complex_prompt)
    
    print(f"      📝 Prompt length: {len(full_prompt)} characters")
    print(f"      🎯 Source: {complex_prompt[:80]}...")
    
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            print(f"      🔄 Attempt {attempt + 1}/{max_retries}")
            if _use_openrouter_text():
                # OpenRouter ignores Gemini's response_mime_type=json flag, so
                # Pro emits the template as prose (which the parser can't segment).
                # Force a strict JSON array keyed exactly as validate_steps reads.
                json_instruction = (
                    "\n\n---\nOUTPUT FORMAT (STRICT): Respond with ONLY a JSON array and nothing "
                    "else -- no prose, no explanations, no markdown code fences. Each element is one "
                    'step object with EXACTLY these three keys: "final_goal", "placement", '
                    '"step_action". Include every step, in order.'
                )
                response_text = _openrouter_chat(
                    full_prompt + json_instruction,
                    temperature=temperature,
                    max_tokens=config.get("max_output_tokens"),
                ).strip()
            else:
                response = client.models.generate_content(
                    model=model_name,
                    contents=full_prompt,
                    config=GenerateContentConfig(**config)
                )
                if not response or not response.text:
                    raise ValueError("Empty response from Gemini API")
                response_text = response.text.strip()
            
            # --- START: ADDED DEBUG PRINT ---
            print("\n" + "="*30 + " RAW API RESPONSE (Attempt " + str(attempt + 1) + ") " + "="*30)
            print(response_text)
            print("="*80 + "\n")
            # --- END: ADDED DEBUG PRINT ---

            print(f"      📦 Response length: {len(response_text)} characters")
            
            steps = []
            parsed_json = None
            
            # Try to parse as JSON first
            try:
                json_text = response_text
                if "```json" in json_text:
                    match = re.search(r'```json\s*(\[.*?\]|\{.*?\})\s*```', json_text, re.DOTALL)
                    if match:
                        json_text = match.group(1)
                elif "```" in json_text:
                    match = re.search(r'```\s*(\[.*?\]|\{.*?\})\s*```', json_text, re.DOTALL)
                    if match:
                        json_text = match.group(1)
                parsed_json = json.loads(json_text)
                
                # --- JSON LIST PARSING ---
                if isinstance(parsed_json, list):
                    json_steps_list = parsed_json # It's a root list
                
                # --- JSON DICT PARSING ---
                elif isinstance(parsed_json, dict):
                    json_steps_list = None
                    # First, check for the common 'steps' key
                    if "steps" in parsed_json and isinstance(parsed_json["steps"], list):
                         json_steps_list = parsed_json["steps"]
                    
                    # --- START: THIS IS THE FIX ---
                    # If not found, find the *first value* in the dict that is a list
                    if json_steps_list is None:
                        for key, value in parsed_json.items():
                            if isinstance(value, list):
                                print(f"      ℹ️ Found step list under unexpected key: '{key}'")
                                json_steps_list = value
                                break
                    # --- END: THIS IS THE FIX ---
                    
                    # Fallback: check if the root dict is just { "Step 1": {...}, "Step 2": {...} }
                    if json_steps_list is None and all(k.lower().startswith('step') for k in parsed_json.keys()):
                        json_steps_list = [ {k: v} for k,v in parsed_json.items() ] # Convert dict to list of dicts

                # --- PROCESS THE FOUND LIST (if any) ---
                if 'json_steps_list' in locals() and json_steps_list is not None:
                    for item in json_steps_list:
                        payload = item
                        if isinstance(item, dict) and len(item) == 1 and list(item.keys())[0].lower().startswith('step'):
                            payload = list(item.values())[0]

                        if isinstance(payload, dict):
                            fg = payload.get("Final Goal (Context)", payload.get("Final Goal", payload.get("final_goal", "")))
                            pl = payload.get("Placement (Compositional Blueprint)", payload.get("Placement", payload.get("placement", "")))
                            # Check for both straight ' and curly ’
                            ac = payload.get("This Step's Action", payload.get("This Step’s Action", payload.get("step_action", "")))
                            steps.append({"final_goal": fg or "", "placement": pl or "", "step_action": ac or ""})
                        elif isinstance(item, str):
                            # Fallback for item-is-string
                            match_fg = re.search(r'Final Goal \(Context\):\s*(.*?)(?:Placement|This Step[\'’]s Action:|$)', item, re.DOTALL) or \
                                    re.search(r'Final Goal:\s*(.*?)(?:Placement|This Step[\'’]s Action:|$)', item, re.DOTALL)
                            match_pl = re.search(r'Placement \(Compositional Blueprint\):\s*(.*?)(?:This Step[\'’]s Action:|$)', item, re.DOTALL) or \
                                    re.search(r'Placement:\s*(.*?)(?:This Step[\'’]s Action:|$)', item, re.DOTALL)
                            # Check for both ' and ’ in regex
                            match_ac = re.search(r'This Step[\'’]s Action:\s*(.*)', item, re.DOTALL)
                            steps.append({
                                "final_goal": match_fg.group(1).strip() if match_fg else "",
                                "placement": match_pl.group(1).strip() if match_pl else "",
                                "step_action": match_ac.group(1).strip() if match_ac else item.strip()
                            })
                    
                    if len(steps) >= 4:
                        steps = validate_steps(steps, complex_prompt) # Validate and raise on empty
                        while len(steps) < 6:
                            steps.append({"final_goal": steps[0]["final_goal"], "placement": steps[0]["placement"], "step_action": f"Continue building the scene (step {len(steps) + 1})"})
                        print(f"      ✅ Successfully parsed and validated JSON with {len(steps)} steps.")
                        return {"success": True, "title": parsed_json.get("title", "") if isinstance(parsed_json, dict) else "", "steps": steps, "raw_response": response_text}
                    else:
                        print(f"      ⚠️ JSON has {len(steps)} steps, expected at least 4")
                
                else:
                    print(f"      ⚠️ JSON missing a list of steps or valid step structure")

            except json.JSONDecodeError as e:
                print(f"      ⚠️ JSON parsing failed: {e}")
            except Exception as e:
                # This will catch validation errors and other parsing issues
                print(f"      ⚠️ Error processing JSON: {e}")
                if "Parsed steps" in str(e): # If it's our validation error, trigger retry
                    raise
            
            # --- FALLBACK TO TEXT PARSING ---
            try:
                steps_from_text = parse_sequential_response(response_text)
                if len(steps_from_text) >= 4:
                    # Convert text-parser format to the simple dict format for validation
                    validation_dict = [{"final_goal": s.get("final_goal"), "placement": s.get("placement"), "step_action": s.get("step_action")} for s in steps_from_text]
                    validated_steps_data = validate_steps(validation_dict, complex_prompt) # Validate and raise on empty
                    
                    # Re-map validated data back to full 'steps_from_text' structure
                    for i, s in enumerate(steps_from_text):
                        s['final_goal'] = validated_steps_data[i]['final_goal']
                        s['placement'] = validated_steps_data[i]['placement']
                        s['step_action'] = validated_steps_data[i]['step_action']

                    print(f"      ✅ Successfully parsed and validated {len(steps_from_text)} steps from text.")
                    
                    while len(steps_from_text) < 6:
                        steps_from_text.append({
                            "step_number": len(steps_from_text) + 1,
                            "final_goal": steps_from_text[0]["final_goal"] if steps_from_text else complex_prompt,
                            "placement": steps_from_text[0]["placement"] if steps_from_text and "placement" in steps_from_text[0] else "",
                            "step_action": f"Continue from step {len(steps_from_text)}",
                            "full_text": f"Continue from step {len(steps_from_text)}"
                        })
                    return {
                        "success": True,
                        "title": "",
                        "steps": steps_from_text, # Return the original structure
                        "raw_response": response_text
                    }
                else:
                    print(f"      ⚠️ Only found {len(steps_from_text)} steps via text parsing, need at least 4")
            except Exception as e:
                print(f"      ⚠️ Text parsing also failed: {e}")
                if "Parsed steps" in str(e): # If it's our validation error, trigger retry
                    raise

            # If all parsing failed, raise error to trigger retry
            raise ValueError("All parsing methods (JSON, Text) failed or returned incomplete data.")

        except Exception as e:
            print(f"      ❌ Error on attempt {attempt + 1}: {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                print(f"      ⏳ Retrying...")
                time.sleep(3 + attempt * 2) # Incremental backoff
            else:
                return {"success": False, "error": f"Failed after {max_retries} attempts: {e}", "steps": []}
                
    return {"success": False, "error": f"Failed after {max_retries} attempts", "steps": []}


def parse_sequential_response(response_text: str) -> List[Dict[str, str]]:
    """Parse response text to extract individual steps with improved robustness."""
    steps = []
    
    # Clean up the response text
    response_text = response_text.strip()
    
    # Try multiple splitting strategies
    # Strategy 1: Look for step markers
    step_patterns = [
        r'\n(?=Step \d+:)',
        r'\n(?=### Step \d+:)',
        r'\n(?=\*\*Step \d+:)',
        r'(?=Step \d+:)',
    ]
    
    step_sections = []
    for pattern in step_patterns:
        step_sections = re.split(pattern, response_text)
        if len(step_sections) > 1:
            break
    
    # If no splits found, try to extract from array-like structure
    if len(step_sections) <= 1:
        # Look for quoted strings that might be steps
        quoted_steps = re.findall(r'"([^"]{100,})"', response_text)
        if quoted_steps:
            for i, step_text in enumerate(quoted_steps[:6], 1):
                steps.append({
                    "step_number": i,
                    "final_goal": "",
                    "placement": "",
                    "step_action": step_text,
                    "full_text": step_text
                })
            # We will let the validation logic in generate_sequential_steps handle this
            return steps
    
    for section in step_sections:
        section = section.strip()
        if not section or len(section) < 50:
            continue
        
        step_match = re.search(r'Step (\d+):', section)
        step_number = int(step_match.group(1)) if step_match else (len(steps) + 1)
        
        # Extract Final Goal, Placement, and This Step's Action
        # Check for both ' and ’ in regex
        goal_patterns = [
            r'Final Goal \(Context\):\s*(.+?)(?=Placement \(Compositional Blueprint\):|Placement:|This Step[\'’]s Action:|$)',
            r'Final Goal:\s*(.+?)(?=Placement \(Compositional Blueprint\):|Placement:|This Step[\'’]s Action:|Action:|$)',
            r'Context:\s*(.+?)(?=Placement \(Compositional Blueprint\):|Placement:|This Step[\'’]s Action:|Action:|$)',
        ]
        placement_patterns = [
            r'Placement \(Compositional Blueprint\):\s*(.+?)(?=This Step[\'’]s Action:|$)',
            r'Placement:\s*(.+?)(?=This Step[\'’]s Action:|$)',
        ]
        action_patterns = [
            r'This Step[\'’]s Action:\s*(.+?)(?=\n\s*Step \d+:|$)',
            r'Action:\s*(.+?)(?=\n\s*Step \d+:|$)',
            r'This Step[\'’]s Action:\s*(.+?)$',
        ]
        
        final_goal = ""
        placement = ""
        step_action = ""
        
        for pattern in goal_patterns:
            goal_match = re.search(pattern, section, re.DOTALL)
            if goal_match:
                final_goal = goal_match.group(1).strip()
                break
                
        for pattern in placement_patterns:
            placement_match = re.search(pattern, section, re.DOTALL)
            if placement_match:
                placement = placement_match.group(1).strip()
                break
                
        for pattern in action_patterns:
            action_match = re.search(pattern, section, re.DOTALL)
            if action_match:
                step_action = action_match.group(1).strip()
                break
                
        if not (final_goal or placement or step_action):
            # Fallback: assume content after "Step X:" is the action
            content = re.sub(r'^Step \d+:.*?\n', '', section, flags=re.MULTILINE)
            step_action = content.strip()
            
        final_goal = re.sub(r'\s+', ' ', final_goal).strip()
        placement = re.sub(r'\s+', ' ', placement).strip()
        step_action = re.sub(r'\s+', ' ', step_action).strip()
        
        if final_goal or placement or step_action:
            full_text = f"Final Goal (Context): {final_goal} Placement (Compositional Blueprint): {placement} This Step's Action: {step_action}"
            steps.append({
                "step_number": step_number,
                "final_goal": final_goal,
                "placement": placement,
                "step_action": step_action,
                "full_text": full_text
            })
            
    steps.sort(key=lambda x: x["step_number"])
    return steps

# =====================================================================================
# FILE MANAGEMENT
# =====================================================================================

def create_output_structure(item_index: int, subject: str = "") -> pathlib.Path:
    """Create the output directory structure for a prompt."""
    base_dir = pathlib.Path("./sbs_prompts")
    base_dir.mkdir(exist_ok=True)
    
    # Create individual prompt directory
    safe_subject = re.sub(r'[^\w\-_\.]', '_', subject) if subject else ""
    prompt_dir_name = f"item_{item_index}_{safe_subject}" if safe_subject else f"item_{item_index}"
    prompt_dir = base_dir / prompt_dir_name
    prompt_dir.mkdir(exist_ok=True)
    
    return prompt_dir

def save_sequential_steps(prompt_dir: pathlib.Path, steps: List[str], 
                         original_prompt: str, item_index: int, 
                         title: str = "", metadata: Dict[str, str] = None,
                         raw_response: str = "") -> List[str]:
    """Save individual step files and return list of file paths."""
    file_paths = []
    
    # Save the original prompt file (single-line)
    original_file = prompt_dir / "00_original_prompt.txt"
    with open(original_file, 'w', encoding='utf-8') as f:
        f.write(f"{original_prompt}\n")
    file_paths.append(str(original_file))

    # Save raw API response for debugging if provided
    if raw_response:
        raw_response_file = prompt_dir / "raw_api_response.txt"
        try:
            with open(raw_response_file, 'w', encoding='utf-8') as f:
                f.write(raw_response)
        except Exception:
            pass

    # Try to parse raw_response JSON into a list of raw steps (authoritative)
    raw_steps = []
    if raw_response:
        try:
            parsed_raw = json.loads(raw_response)
            if isinstance(parsed_raw, dict):
                # Check for 'steps' key
                if "steps" in parsed_raw and isinstance(parsed_raw["steps"], list):
                    raw_steps = parsed_raw["steps"]
                # --- START: THIS IS THE FIX ---
                # If not, find the first value that is a list
                else:
                    for key, value in parsed_raw.items():
                        if isinstance(value, list):
                            raw_steps = value
                            break
                # --- END: THIS IS THE FIX ---
            elif isinstance(parsed_raw, list):
                raw_steps = parsed_raw # Handle root list
            else:
                raw_steps = [parsed_raw]
        except Exception:
            raw_steps = []
    
    # Helper: extract the three required parts from a raw step entry
    def extract_from_raw(step_obj, raw_obj=None):
        # Defaults
        fg = original_prompt or ""
        pl = ""
        ac = ""
        
        authoritative_obj = raw_obj or step_obj
        
        if isinstance(authoritative_obj, dict):
            # Handle nested formats like {"Step 1": {...}}
            if len(authoritative_obj) == 1 and list(authoritative_obj.keys())[0].lower().startswith('step'):
                payload = list(authoritative_obj.values())[0]
            else:
                payload = authoritative_obj
            
            if isinstance(payload, dict):
                fg = payload.get('Final Goal (Context)', 
                                 payload.get('Final Goal', 
                                 payload.get('final_goal', fg))) or fg
                
                pl = payload.get('Placement (Compositional Blueprint)', 
                                 payload.get('Placement', 
                                 payload.get('placement', pl))) or pl
                
                # Check for both straight ' and curly ’
                ac = payload.get("This Step's Action", 
                                 payload.get("This Step’s Action",
                                 payload.get('step_action', 
                                 payload.get('full_text', ac)))) or ac
            else:
                ac = str(payload)

        elif isinstance(authoritative_obj, str):
            # Check for both ' and ’ in regex
            fg_match = re.search(r'Final Goal \(Context\):\s*(.*?)(?:Placement|This Step[\'’]s Action:|$)', authoritative_obj, re.DOTALL) or \
                       re.search(r'Final Goal:\s*(.*?)(?:Placement|This Step[\'’]s Action:|$)', authoritative_obj, re.DOTALL)
            if fg_match:
                fg = fg_match.group(1).strip() or fg
                
            pl_match = re.search(r'Placement \(Compositional Blueprint\):\s*(.*?)(?:This Step[\'’]s Action:|$)', authoritative_obj, re.DOTALL) or \
                       re.search(r'Placement:\s*(.*?)(?:This Step[\'’]s Action:|$)', authoritative_obj, re.DOTALL)
            if pl_match:
                pl = pl_match.group(1).strip() or pl
                
            ac_match = re.search(r'This Step[\'’]s Action:\s*(.*)', authoritative_obj, re.DOTALL)
            if ac_match:
                ac = ac_match.group(1).strip() or ac
                
            if not (fg or pl or ac):
                ac = authoritative_obj.strip()
        else:
            ac = str(authoritative_obj)

        return (fg or original_prompt), pl, ac

    # Helper to format a step like the example in base_prompt.txt
    def format_step(i, final_goal, placement, action):
        return (
            f"Step {i}:\n"
            f"Final Goal (Context): {final_goal}\n\n"
            f"Placement (Compositional Blueprint):\n{placement}\n\n"
            f"This Step's Action: {action}\n"
        )

    # If raw_steps are available, treat them as authoritative and format from them.
    authoritative = []
    source_list = raw_steps if raw_steps else steps
    
    for item in source_list:
        final_goal, placement, action = extract_from_raw(item)
        authoritative.append({'final_goal': final_goal, 'placement': placement, 'step_action': action})

    # Save each step as individual file
    total_steps = len(authoritative)
    for i in range(total_steps):
        step_idx = i + 1
        step_filename = f"step_{step_idx:02d}.txt"
        step_file = prompt_dir / step_filename
        
        entry = authoritative[i]
        final_goal = entry['final_goal']
        placement = entry['placement']
        action = entry['step_action']
            
        if isinstance(placement, dict):
            placement_text = '\n'.join(f"{k}: {v}" for k, v in placement.items())
        else:
            placement_text = str(placement or '')
            
        with open(step_file, 'w', encoding='utf-8') as f:
            f.write(format_step(step_idx, final_goal, placement_text, action))
        file_paths.append(str(step_file))

    # Save final complete sequence file, formatted with header + all steps
    final_file = prompt_dir / "99_complete_sequence.txt"
    with open(final_file, 'w', encoding='utf-8') as f:
        for i in range(total_steps):
            entry = authoritative[i]
            final_goal = entry['final_goal']
            placement = entry['placement']
            action = entry['step_action']

            if isinstance(placement, dict):
                placement_text = '\n'.join(f"{k}: {v}" for k, v in placement.items())
            else:
                placement_text = str(placement or '')
                
            f.write(format_step(i+1, final_goal, placement_text, action))
            f.write("\n")
    file_paths.append(str(final_file))

    return file_paths

# =====================================================================================
# MAIN EXECUTION
# =====================================================================================

def main():
    """Main execution function."""
    load_dotenv()
    args = parse_args()
    
    print(f"🎬 Sequential Prompt Generator for Baseline Images")
    print(f"🤖 Using model: {args.model}")
    print(f"🌡️ Temperature: {args.temperature}")
    print(f"📁 Input CSV: {args.csv}")
    print(f"💾 Output CSV: {args.output}")
    
    # Load the base prompt template
    try:
        base_template = load_base_prompt_template()
        print(f"✅ Loaded base prompt template ({len(base_template)} characters)")
    except Exception as e:
        print(f"❌ Error loading template: {e}")
        return
    
    # Load CSV
    try:
        csv_path = pathlib.Path(args.csv)
        df = load_and_validate_csv(csv_path)
        
        if len(df) == 0:
            print("⚠️ No valid prompts found in CSV")
            return
        
        # Optionally limit dataset for testing
        if args.limit_items:
            df = df.head(args.limit_items)
            print(f"🔬 Limited to {args.limit_items} items")
        
        print(f"✅ Processing {len(df)} items")
        
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return
    
    # Load existing results if CSV exists
    existing_results = []
    processed_items = set()
    if pathlib.Path(args.output).exists():
        try:
            existing_df = pd.read_csv(args.output)
            existing_results = existing_df.to_dict('records')
            # Only consider SUCCESSFUL items as 'processed' for skipping
            processed_items = set(existing_df[existing_df['status'] == 'SUCCESS']['item_index'].tolist())
            print(f"📂 Loaded {len(existing_results)} existing results ({len(processed_items)} successful)")
        except Exception as e:
            print(f"⚠️ Could not load existing results: {e}")
    
    results = []
    success_count = 0
    error_count = 0
    skip_count = 0
    
    # Process each prompt
    for idx, row in df.iterrows():
        item_index = row['item_index']
        prompt = str(row['prompt']).strip()
        
        print(f"\n{'='*80}")
        print(f"🔍 Item {item_index} ({idx + 1}/{len(df)})")
        print(f"   Prompt: {prompt[:100]}...")
        
        # Skip if already processed
        if args.skip_existing and item_index in processed_items:
            print(f"   ⏭️  Skipping: already processed successfully")
            skip_count += 1
            # Need to find the existing result to append to all_results
            existing_row = next((item for item in existing_results if item['item_index'] == item_index), None)
            if existing_row:
                results.append(existing_row) # Add to current session's results to avoid dropping it
            continue
        
        try:
            # Generate sequential steps
            result = generate_sequential_steps(
                prompt, base_template, args.model, args.temperature, args.max_tokens
            )
            
            if result.get("success"):
                # Create output directory
                subject = row.get('subject', '')
                prompt_dir = create_output_structure(item_index, subject)
                
                # Prepare metadata
                metadata = {
                    'item_index': str(item_index),
                    'subject': subject if subject else ''
                }
                
                # Add additional metadata if available
                if 'attribute_1' in row.index:
                    for i in range(1, 5):
                        if f'attribute_{i}' in row.index:
                            metadata[f'attribute_{i}'] = str(row[f'attribute_{i}'])
                if 'interaction_1' in row.index:
                    for i in range(1, 3):
                        if f'interaction_{i}' in row.index:
                            metadata[f'interaction_{i}'] = str(row[f'interaction_{i}'])
                
                # Save steps (pass raw_response so saver can prefer authoritative raw data)
                file_paths = save_sequential_steps(
                    prompt_dir, result["steps"], prompt, item_index, 
                    result.get("title", ""), metadata,
                    raw_response=result.get("raw_response", "")
                )
                
                print(f"   ✅ Saved {len(result['steps'])} steps to: {prompt_dir}")
                
                # Record result
                results.append({
                    "item_index": item_index,
                    "subject": subject,
                    "original_prompt": prompt,
                    "title": result.get("title", ""),
                    "num_steps": len(result["steps"]),
                    "output_dir": str(prompt_dir),
                    "status": "SUCCESS",
                    "error": "" # Explicitly add empty error column
                })
                success_count += 1
                
            else:
                error_msg = result.get("error", "Unknown error")
                print(f"   ❌ Failed: {error_msg}")
                results.append({
                    "item_index": item_index,
                    "subject": row.get('subject', ''),
                    "original_prompt": prompt,
                    "title": "",
                    "num_steps": 0,
                    "output_dir": "",
                    "status": "ERROR",
                    "error": error_msg
                })
                error_count += 1
            
            # Adaptive delay - longer delay after errors
            if idx < len(df) - 1:
                delay = args.delay * 2 if result.get("status") == "ERROR" else args.delay
                time.sleep(delay)
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            results.append({
                "item_index": item_index,
                "subject": row.get('subject', ''),
                "original_prompt": prompt,
                "title": "",
                "num_steps": 0,
                "output_dir": "",
                "status": "ERROR",
                "error": str(e)
            })
            error_count += 1
    
    # Save results to CSV
    if results:
        
        # We need to overwrite old entries (especially failed ones) with new ones.
        
        # Create a dictionary of new results, indexed by item_index
        new_results_map = {r['item_index']: r for r in results}
        
        # Create a list of existing results *that were not processed* in this run
        final_results_list = [r for r in existing_results if r['item_index'] not in new_results_map]
        
        # Add all the new results (which includes new successes, new failures, and skipped items)
        final_results_list.extend(new_results_map.values())
        
        # Sort by item_index to maintain order
        final_results_list.sort(key=lambda x: x['item_index'])

        results_df = pd.DataFrame(final_results_list)
        
        # Ensure consistent columns
        all_cols = ["item_index", "subject", "original_prompt", "title", "num_steps", "output_dir", "status", "error"]
        for col in all_cols:
            if col not in results_df.columns:
                results_df[col] = "" if col != "num_steps" else 0
        
        results_df = results_df[all_cols] # Reorder columns
        results_df.to_csv(args.output, index=False)
        print(f"\n💾 Results saved: {args.output}")
    elif existing_results and not results:
         print("\nℹ️ No new results generated. Output file not modified.")
    else:
         print("\nℹ️ No results generated.")

    
    print(f"\n{'='*80}")
    print(f"📊 Final Summary:")
    print(f"   ✅ Success: {success_count}")
    print(f"   ⏭️  Skipped: {skip_count}")
    print(f"   ❌ Errors: {error_count}")
    print(f"   Processed This Run: {success_count + error_count}")
    print(f"   Total in Source: {len(df)}")
    print("✅ Processing complete!")

if __name__ == "__main__":
    main()