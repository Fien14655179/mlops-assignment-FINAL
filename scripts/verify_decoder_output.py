import json
import argparse
import sys

def verify_jsonl(file_path):
    print(f"Checking output types in: {file_path}")
    
    valid_count = 0
    error_count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if not line.strip(): continue
                
                try:
                    # 1. Parse the outer JSONL line
                    item = json.loads(line)
                    
                    # 2. Extract the model's text response
                    # It might be in 'cleaned', 'response', or 'text' depending on the script version
                    model_output = item.get("cleaned") or item.get("response")
                    
                    if model_output is None:
                        print(f"[WARN] Line {line_num}: No 'cleaned' or 'response' field found.")
                        error_count += 1
                        continue

                    # 3. CRITICAL CHECK: Is the string inside actually valid JSON?
                    extracted_data = json.loads(model_output)
                    
                    if not isinstance(extracted_data, dict):
                        print(f"[FAIL] Line {line_num}: Output parses but is not a Dictionary (Found {type(extracted_data)}).")
                        error_count += 1
                    else:
                        valid_count += 1
                        
                except json.JSONDecodeError:
                    print(f"[FAIL] Line {line_num}: Model output is NOT valid JSON.")
                    error_count += 1
                except Exception as e:
                    print(f"[ERR] Line {line_num}: Unexpected error {e}")
                    error_count += 1
                    
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        sys.exit(1)

    total = valid_count + error_count
    if total == 0:
        print("File was empty or had no valid lines.")
    else:
        print("-" * 30)
        print(f"Results for {file_path}:")
        print(f"Valid JSON Outputs: {valid_count}")
        print(f"Invalid Outputs:    {error_count}")
        print(f"Success Rate: {100 * valid_count / total:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to .jsonl output file")
    args = parser.parse_args()
    
    verify_jsonl(args.input)