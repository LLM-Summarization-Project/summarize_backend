# python/runner.py
import argparse, os, json, time, sys
import summary as pipeline

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--youtube_url", required=True)
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--scene_thresh", type=float, default=0.6)
    p.add_argument("--enable_ocr", action="store_true")
    p.add_argument("--whisper_model", default="large-v3-turbo")
    p.add_argument("--language", default="th")
    p.add_argument("--asr_device", default="cpu")
    p.add_argument("--vl_device", default="cpu")
    p.add_argument("--ollama_api", default=None)
    p.add_argument("--ollama_model", default=None)
    p.add_argument("--summary_id", default=str(int(time.time())))
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    prefix = os.path.join(args.out_dir, args.summary_id + '_')

    # map เข้า full.py
    pipeline.YOUTUBE_URL  = args.youtube_url
    pipeline.SCENE_THRESH = args.scene_thresh
    pipeline.ENABLE_OCR   = bool(args.enable_ocr)
    pipeline.WHISPER_MODEL = args.whisper_model
    pipeline.LANGUAGE      = args.language
    pipeline.ASR_DEVICE    = args.asr_device
    pipeline.VL_DEVICE     = args.vl_device

    if args.ollama_api:
        os.environ["OLLAMA_API"] = args.ollama_api
        pipeline.OLLAMA_API = args.ollama_api
    if args.ollama_model:
        os.environ["OLLAMA_MODEL"] = args.ollama_model
        pipeline.OLLAMA_MODEL = args.ollama_model

    pipeline.AUDIO_OUT         = prefix + "audio.wav"
    pipeline.FRAMES_DIR        = prefix + "frames"
    pipeline.SCENES_JSON       = prefix + "scenes.json"
    pipeline.CAPTIONS_JSON     = prefix + "captions.json"
    pipeline.SCENE_FACTS_JSON  = prefix + "scene_facts.json"
    pipeline.TRANSCRIPT_TXT    = prefix + "transcription.txt"
    pipeline.DROPDOWN_JSON     = prefix + "dropdown_items.json"
    pipeline.FINAL_TXT         = prefix + "dropdown_list.txt"
    pipeline.FINAL_ARTICLE_TXT = prefix + "final_article_th.txt"
    pipeline.METRICS_JSON = prefix + "metrics.json"

    # รัน pipeline
    pipeline.main()

    # ส่งผลลัพธ์ JSON ออก stdout
    def _safe_read(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return None
        
    metrics = None
    try:
        import json as _json
        if os.path.exists(pipeline.METRICS_JSON):
            with open(pipeline.METRICS_JSON, "r", encoding="utf-8") as f:
                metrics = _json.load(f)
    except Exception as e:
        print(f"⚠️ Read metrics failed: {e}", file=sys.stderr, flush=True)

    result = {
        "transcript_path": pipeline.TRANSCRIPT_TXT,
        "bullets_path":    pipeline.FINAL_TXT,
        "article_path":    pipeline.FINAL_ARTICLE_TXT,
        "dropdown_path":   pipeline.DROPDOWN_JSON,
        "scene_facts_path":pipeline.SCENE_FACTS_JSON,
        "article_preview": _safe_read(pipeline.FINAL_ARTICLE_TXT)[:800] if os.path.exists(pipeline.FINAL_ARTICLE_TXT) else None,
        "metrics": metrics,
    }
    sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
