from __future__ import annotations

import json
from pathlib import Path

TCGA_REPORTS = Path("data/tcga/tcga_reports.jsonl")
SPLITS_DIR = Path("data/splits")
OUT_DIR = Path("data/processed/text/v1")
PROMPT_PATH = OUT_DIR / "prompt.txt"


def load_reports(path: Path) -> dict[str, str]:
    reports: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            pid = str(obj["pid"])
            reports[pid] = obj["report"]
    return reports


def load_pids(path: Path) -> list[str]:
    return json.loads(path.read_text())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
    reports = load_reports(TCGA_REPORTS)

    for split in ["train", "val", "test"]:
        pids = load_pids(SPLITS_DIR / f"{split}_pids.json")
        out_path = OUT_DIR / f"decoder_input_{split}.jsonl"

        n_written = 0
        with out_path.open("w", encoding="utf-8") as out:
            for pid in pids:
                rep = reports.get(pid)
                if rep is None:
                    continue
                rec = {"pid": pid, "prompt": prompt, "report": rep}
                out.write(json.dumps(rec) + "\n")
                n_written += 1

        print(f"[{split}] wrote {n_written} records -> {out_path}")


if __name__ == "__main__":
    main()
