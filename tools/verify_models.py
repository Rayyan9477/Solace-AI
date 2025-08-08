"""
Quick verifier to check STT (Whisper v3 Turbo), TTS (Dia 1.6B and SpeechT5) and core LLM wiring.
Run: python tools/verify_models.py
"""
import asyncio
import os
import sys
from pathlib import Path

# Ensure project root is importable when invoked directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


async def check_voice():
    from src.utils.voice_ai import VoiceAI
    from src.config.settings import AppConfig

    print("[Voice] Initializing...")
    v = VoiceAI()
    ok_stt = await v.initialize_stt()
    print(f"[Voice] Whisper init: {ok_stt}")

    ok_dia = await v.initialize_dia()
    print(f"[Voice] Dia init: {ok_dia}")

    if not ok_dia:
        ok_tts = await v.initialize_tts()
        print(f"[Voice] SpeechT5 fallback init: {ok_tts}")

    # Tiny synthesis smoke test (no playback)
    res = await v.text_to_speech("Hello from the verifier.", voice_style="warm")
    print(f"[Voice] TTS success: {res.get('success', False)}; bytes: {len(res.get('audio_bytes', b''))}")


async def check_llm():
    from src.models.llm import AgnoLLM
    from src.config.settings import AppConfig

    print("[LLM] Initializing...")
    try:
        llm = AgnoLLM()
        out = await llm.generate("Say 'ok'.")
        print(f"[LLM] Success: {out.get('success', False)}; resp: {out.get('response', '')[:50]}")
    except Exception as e:
        print(f"[LLM] Error: {e}")


async def main():
    await check_voice()
    await check_llm()


if __name__ == "__main__":
    asyncio.run(main())


