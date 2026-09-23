#!/usr/bin/env python3
"""Minimal direct Flash Next chat. No files, system prompts, or tools are loaded."""

import json
import urllib.error
import urllib.request


URL = "http://127.0.0.1:18088/v1/chat/completions"
MODEL = "Qwen3.8-Flash-Next"


def main(url=URL, model=MODEL, headers=None, label="Flash Next"):
    messages = []
    print(f"Direct {label} chat | medium reasoning | no tools or system prompt")
    print("/new clears the conversation. /quit exits. Nothing is saved.")
    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if prompt == "/quit":
            return
        if prompt == "/new":
            messages.clear()
            print("Fresh conversation.")
            continue
        if not prompt:
            continue
        pending = messages + [{"role": "user", "content": prompt}]
        body = {
            "model": model,
            "messages": pending,
            "stream": True,
            "temperature": 1.0,
            "top_p": 0.95,
            "max_tokens": 16384,
            "reasoning_effort": "medium",
            "chat_template_kwargs": {
                "enable_thinking": True,
                "reasoning_effort": "medium",
            },
        }
        request = urllib.request.Request(
            url, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", **(headers or {})},
        )
        answer = []
        finish = None
        print(f"{label} (thinking may take a moment): ", end="", flush=True)
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                for line in response:
                    if not line.startswith(b"data: "):
                        continue
                    raw = line[6:].strip()
                    if raw == b"[DONE]":
                        break
                    event = json.loads(raw)
                    if event.get("error"):
                        raise RuntimeError(str(event["error"]))
                    for choice in event.get("choices", []):
                        text = choice.get("delta", {}).get("content") or ""
                        answer.append(text)
                        print(text, end="", flush=True)
                        finish = choice.get("finish_reason") or finish
            print()
            if finish == "length":
                print("[Response reached the output limit.]")
            if answer and any(answer):
                messages = pending + [{"role": "assistant", "content": "".join(answer)}]
            else:
                print("[No visible answer; conversation unchanged.]")
        except KeyboardInterrupt:
            print("\n[Cancelled; conversation unchanged.]")
        except (urllib.error.URLError, TimeoutError, ValueError, RuntimeError) as error:
            print(f"\n[Request failed: {error}. Conversation unchanged.]")


if __name__ == "__main__":
    main()
