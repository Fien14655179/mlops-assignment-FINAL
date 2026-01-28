import asyncio
import json
import argparse
import os
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm


def load_system_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


async def process_item(client, item, args, semaphore, system_prompt: str):
    raw_report = item.get("report") or item.get("text") or item.get("response") or ""
    if not isinstance(raw_report, str):
        raw_report = str(raw_report)

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model="Qwen/Qwen3-4B-AWQ",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"/no_think\nRAW REPORT:\n{raw_report}"},
                ],
                temperature=0.0,
                top_p=0.1,
                extra_body={"top_k": 1, "min_p": 0.0},
                max_tokens=args.max_tokens,
                timeout=180,
            )

            item["cleaned"] = response.choices[0].message.content
            item["finish_reason"] = response.choices[0].finish_reason

        except Exception as e:
            item["error"] = str(e)
            item["cleaned"] = None

        return item


async def run_batch(args):
    system_prompt = load_system_prompt(args.prompt)

    client = AsyncOpenAI(
        base_url=f"http://127.0.0.1:{args.port}/v1",
        api_key="vllm",
    )
    semaphore = asyncio.Semaphore(args.concurrency)

    processed_pids = set()
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    res = json.loads(line)
                    if "pid" in res:
                        processed_pids.add(res["pid"])
                except Exception:
                    continue
        print(f"Resuming: Found {len(processed_pids)} items already processed.")

    data = []
    print(f"Reading input from {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("pid") not in processed_pids:
                data.append(item)

    if not data:
        print("All items already processed or input is empty.")
        return

    print(f"Processing {len(data)} items...")
    tasks = [
        process_item(client, item, args, semaphore, system_prompt)
        for item in data
    ]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "a", encoding="utf-8") as f:
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Inferencing Qwen3-4B"):
            result = await future
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

    print(f"Batch complete. Results saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Inference Decoder (Qwen3-4B-AWQ)")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--prompt",
        type=str,
        default="data/processed/text/v1/prompt.txt",
        help="Path to decoder prompt text file",
    )
    args = parser.parse_args()
    asyncio.run(run_batch(args))
