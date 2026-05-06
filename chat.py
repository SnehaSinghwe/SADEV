"""
chat.py
Interactive CLI for testing the SADEV local pipeline.

Usage:
    python chat.py                        # uses gemma2 on localhost:11434
    python chat.py --model gemma2:9b
    python chat.py --model llama3.2
    python chat.py --no-rag               # skip RAG (faster if index not built)
    python chat.py --log session.jsonl    # log all turns for evaluation

Controls:
    /quit   — exit
    /reset  — clear session
    /debug  — toggle debug mode (shows emotion, intent, stressor, RAG chunks)
    /eval   — show evaluation summary for this session
    /help   — show all commands
"""
from __future__ import annotations

import argparse
import json
import sys
import os
import datetime

# Ensure local packages are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from local_pipeline.sadev_pipeline import SadevPipeline as Pipeline, PipelineResult
from local_pipeline.ollama_client import OllamaConnectionError


# ── ANSI colour helpers ───────────────────────────────────────────────────

def _col(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m"

def dim(t):    return _col("2", t)
def bold(t):   return _col("1", t)
def green(t):  return _col("32", t)
def yellow(t): return _col("33", t)
def red(t):    return _col("31", t)
def cyan(t):   return _col("36", t)
def magenta(t):return _col("35", t)


# ── Eval log ──────────────────────────────────────────────────────────────

class EvalLogger:
    def __init__(self, path: str | None):
        self.path = path
        self.entries: list[dict] = []

    def log(self, user_text: str, result: PipelineResult) -> None:
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "session_id": result.session_id,
            "user": user_text,
            "response": result.response_text,
            "emotion": result.emotion_detected,
            "intent": result.intent_detected,
            "stressor": result.stressor_detected,
            "urgency": result.urgency_level,
            "risk": result.risk_level,
            "rag_chunks": result.rag_chunks_used,
            "crisis_exit": result.crisis_exit,
            "parse_strategy": "",  # filled by caller if debug
        }
        self.entries.append(entry)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def summary(self) -> str:
        if not self.entries:
            return "No turns logged yet."
        n = len(self.entries)
        crises = sum(1 for e in self.entries if e["crisis_exit"])
        avg_rag = sum(e["rag_chunks"] for e in self.entries) / n
        intents = {}
        for e in self.entries:
            intents[e["intent"]] = intents.get(e["intent"], 0) + 1
        lines = [
            f"Turns: {n}",
            f"Crisis exits: {crises}",
            f"Avg RAG chunks used: {avg_rag:.1f}",
            "Intents detected:",
        ]
        for intent, count in sorted(intents.items(), key=lambda x: -x[1]):
            lines.append(f"  {intent:<40} {count}")
        return "\n".join(lines)


# ── Main CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SADEV local pipeline interactive chat")
    parser.add_argument("--model",      default="gemma2",               help="Ollama model name")
    parser.add_argument("--url",        default="http://localhost:11434",help="Ollama base URL")
    parser.add_argument("--index",      default="data/vector_index",    help="FAISS index path")
    parser.add_argument("--kb",         default="data/knowledge_base_raw.json")
    parser.add_argument("--top-k",      type=int, default=3)
    parser.add_argument("--no-rag",     action="store_true",            help="Disable RAG")
    parser.add_argument("--log",        default=None,                   help="Path to save eval log JSONL")
    parser.add_argument("--debug",      action="store_true",            help="Start in debug mode")
    parser.add_argument("--temperature",type=float, default=0.7)
    parser.add_argument("--no-llm-scoring", action="store_true",  help="Disable LLM risk scorer (regex only)")
    args = parser.parse_args()

    print(bold("\n  सदैव — SADEV local prototype"))
    print(dim(f"  Model: {args.model}  |  RAG: {'off' if args.no_rag else 'on'}  |  Top-k: {args.top_k}"))
    print(dim("  Type /help for commands\n"))

    # Health check
    from local_pipeline.ollama_client import OllamaClient
    client = OllamaClient(model=args.model, base_url=args.url, temperature=args.temperature)
    health = client.check_health()
    if not health["ollama_running"]:
        print(red("  Ollama not running. Start it with: ollama serve"))
        sys.exit(1)
    if not health["model_available"]:
        print(yellow(f"  Model '{args.model}' not found. Pull it: ollama pull {args.model}"))
        print(yellow(f"  Available models: {', '.join(health['available_models'])}"))
        sys.exit(1)
    print(green(f"  Ollama ready — {args.model}"))

    llm_scoring = not getattr(args, 'no_llm_scoring', False)
    pipeline = Pipeline(
        ollama_model=args.model,
        ollama_url=args.url,
        index_path=args.index,
        kb_path=args.kb,
        top_k=args.top_k,
        llm_scoring=llm_scoring,
    )
    print(dim(f'  LLM risk scoring: {"on" if llm_scoring else "off (regex only)"}'))

    # Disable RAG if requested
    if args.no_rag:
        pipeline.rag._index = None

    if pipeline.rag.is_ready:
        print(green(f"  RAG index ready — {pipeline.rag.size} vectors"))
    else:
        print(yellow("  RAG index not found. Run: python -m rag.build_index"))

    eval_log = EvalLogger(args.log)
    if args.log:
        print(dim(f"  Logging to {args.log}"))

    debug_mode = args.debug
    session_id = None

    print()

    while True:
        try:
            user_input = input(cyan("you  > ")).strip()
        except (KeyboardInterrupt, EOFError):
            print("\n" + dim("  Goodbye."))
            break

        if not user_input:
            continue

        # Commands
        if user_input.startswith("/"):
            cmd = user_input.lower().strip()
            if cmd == "/quit" or cmd == "/exit":
                print(dim("  Goodbye."))
                break
            elif cmd == "/reset":
                if session_id:
                    pipeline.reset_session(session_id)
                session_id = None
                print(dim("  Session reset."))
            elif cmd == "/debug":
                debug_mode = not debug_mode
                print(dim(f"  Debug mode: {'on' if debug_mode else 'off'}"))
            elif cmd == "/eval":
                print(bold("\n  Evaluation summary:"))
                print(eval_log.summary())
                print()
            elif cmd == "/help":
                print(dim("""
  /quit   — exit
  /reset  — clear session history
  /debug  — toggle debug output
  /eval   — session evaluation summary
  /help   — this message
"""))
            else:
                print(dim(f"  Unknown command: {user_input}"))
            continue

        # Run pipeline
        try:
            result = pipeline.chat(user_input, session_id=session_id)
            session_id = result.session_id
        except OllamaConnectionError as e:
            print(red(f"  Ollama error: {e}"))
            continue
        except Exception as e:
            print(red(f"  Pipeline error: {e}"))
            if debug_mode:
                import traceback
                traceback.print_exc()
            continue

        # Crisis display
        if result.crisis_exit:
            print()
            source_tag = dim(f'  [risk {result.risk_level} | via {result.risk_source}'
                             + (' | safeguarding' if result.safeguarding else '') + ']')
            print(red("  ⚠ CRISIS DETECTED") + source_tag)
            print()
            print(bold("sadev") + " > " + result.response_text)
            if result.helpline_name:
                print()
                print(red(f"  Helpline: {result.helpline_name} — {result.helpline_number}"))
            print()
        else:
            print()
            print(bold("sadev") + " > " + result.response_text)
            print()

        # Debug output
        if debug_mode:
            print(dim(f"  emotion={result.emotion_detected} | intent={result.intent_detected}"))
            print(dim(f"  stressor={result.stressor_detected} | urgency={result.urgency_level} | risk={result.risk_level} | source={result.risk_source}"))
            print(dim(f"  safeguarding={result.safeguarding} | rag_chunks={result.rag_chunks_used} | model={result.model_used}"))
            print(dim(f"  ── validation: {result.validation_part[:60]}..."))
            print(dim(f"  ── reflection: {result.reflection_part[:60]}..."))
            print(dim(f"  ── question:   {result.question_part}"))
            print()

        eval_log.log(user_input, result)

    # Final summary
    if eval_log.entries:
        print(bold("\n  Session summary:"))
        print(eval_log.summary())


if __name__ == "__main__":
    main()
