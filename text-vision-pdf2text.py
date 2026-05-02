import os
import shutil
import gradio as gr
from openai import OpenAI
from PIL import Image
from io import BytesIO
import base64
import time
import tempfile
import atexit
import httpx
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pdf2image import convert_from_path

# ── Temp-file cleanup ──────────────────────────────────────────────────────────
temp_files_to_clean = []

def cleanup_temp_files():
    if not temp_files_to_clean:
        return
    print(f"\n--- Commencing Cleanup ({len(temp_files_to_clean)} tracked paths) ---")
    for path in temp_files_to_clean:
        try:
            if os.path.isdir(path):
                num_files = sum([len(files) for _, _, files in os.walk(path)])
                shutil.rmtree(path, ignore_errors=True)
                print(f" ✔ Deleted Folder: {path} (contained {num_files} files)")
            else:
                if os.path.exists(path):
                    os.remove(path)
                    print(f" ✔ Deleted File:   {path}")
                else:
                    print(f" ➖ Skipped (Already gone): {path}")
        except Exception as e:
            print(f" ❌ Failed to delete: {path} | Error: {e}")
    print("--- Cleanup complete ---")

atexit.register(cleanup_temp_files)

# ── Config loaders ─────────────────────────────────────────────────────────────
def load_text_file(filepath, fallback=""):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Warning: '{filepath}' not found.")
        return fallback

def load_models(filepath, fallback=None):
    fallback = fallback or ["meta/llama-3.3-70b-instruct"]
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            models = [l.strip() for l in f if l.strip()]
            return models if models else fallback
    except FileNotFoundError:
        print(f"Warning: '{filepath}' not found. Using default model list.")
        return fallback

# ── Load config ────────────────────────────────────────────────────────────────
all_models      = load_models("nvidia-models.txt")
vision_models   = set(load_models("nvidia-models-image.txt"))
combined_models = list(dict.fromkeys(
    [m for m in all_models if m in vision_models] +
    [m for m in all_models if m not in vision_models] +
    list(vision_models - set(all_models))
))

default_model                = combined_models[0] if combined_models else None
initial_system_prompt        = load_text_file("system-prompt.txt", "You are a helpful assistant.")
initial_system_prompt_vision = load_text_file("system-prompt-image.txt", initial_system_prompt)

# ── NVIDIA API client ──────────────────────────────────────────────────────────
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise EnvironmentError("NVIDIA_API_KEY environment variable is not set.")

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Use split timeouts so a slow server doesn't silently block forever:
#   connect  – fail fast if the network is unreachable (10 s)
#   read     – generous window for the server to start streaming (180 s)
#   write    – time to finish uploading the request body (30 s)
#   pool     – time waiting for an httpx connection from the pool (5 s)
_http_client = httpx.Client(
    timeout=httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=5.0)
)
client = OpenAI(
    base_url=NVIDIA_BASE_URL,
    api_key=NVIDIA_API_KEY,
    http_client=_http_client,
)

# Thread-pool used to run the blocking .create() call off the generator thread
# so we can yield heartbeat updates while waiting for the first token.
_executor = ThreadPoolExecutor(max_workers=4)

# ── PDF / image helpers ────────────────────────────────────────────────────────
MAX_IMAGE_DIMENSION = 2240
PDF_DPI = 200

def pdf_to_images(pdf_path: str) -> list[Image.Image]:
    pdf_out_dir = tempfile.mkdtemp(prefix="pdf_extract_")
    temp_files_to_clean.append(pdf_out_dir)
    pages = convert_from_path(pdf_path, dpi=PDF_DPI, output_folder=pdf_out_dir)
    images = []
    for page in pages:
        if page.mode in ("RGBA", "P"):
            page = page.convert("RGB")
        if max(page.size) > MAX_IMAGE_DIMENSION:
            page.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.Resampling.LANCZOS)
        images.append(page)
    return images

def file_to_images(file_obj_list) -> list[Image.Image]:
    if not file_obj_list:
        return []
    if not isinstance(file_obj_list, list):
        file_obj_list = [file_obj_list]

    all_images = []
    for file_obj in file_obj_list:
        path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
        if path not in temp_files_to_clean:
            temp_files_to_clean.append(path)
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            all_images.extend(pdf_to_images(path))
        else:
            with Image.open(path) as img:
                img = img.copy()
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                if max(img.size) > MAX_IMAGE_DIMENSION:
                    img.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.Resampling.LANCZOS)
                all_images.append(img)
    return all_images

def pil_to_b64_jpeg(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ── Reasoning / thinking parameter helpers ─────────────────────────────────────
#
# Per-model rules derived from NVIDIA API examples:
#
#  deepseek-ai/deepseek-v4-pro
#    extra_body={"chat_template_kwargs": {"thinking": False}}   (thinking off by default)
#    extra_body={"chat_template_kwargs": {"thinking": True}}    (thinking on)
#
#  deepseek-ai/deepseek-v4-flash
#    extra_body={"chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}}
#    reasoning streamed via delta.reasoning OR delta.reasoning_content
#
#  qwen/qwen3.5-122b-a10b  &  qwen/qwen3.5-397b-a17b
#    chat_template_kwargs={"enable_thinking": True/False}  (top-level payload key)
#
#  qwen/qwen3-coder-480b-a35b-instruct
#    no thinking params — standard OpenAI style
#
#  google/gemma-4-31b-it
#    chat_template_kwargs={"enable_thinking": True/False}
#
#  z-ai/glm-5.1   (note: models.txt has "z-ai/glm5.1" without dot — match both)
#    extra_body={"chat_template_kwargs": {"enable_thinking": True, "clear_thinking": False}}
#    reasoning via delta.reasoning_content
#
#  moonshotai/kimi-k2.6
#    chat_template_kwargs={"thinking": True/False}  (top-level payload key)
#    no delta.reasoning_content exposure
#
#  mistralai/mistral-medium-3.5-128b
#    top-level payload key: "reasoning_effort": "high" | "low" | "off"
#    no extra_body needed
#
#  mistralai/mistral-large-3-675b-instruct-2512
#    no thinking params — standard
#
#  nvidia/nemotron-3-nano-omni-30b-a3b-reasoning
#    extra_body={"chat_template_kwargs": {"enable_thinking": True}, "reasoning_budget": 16384}
#    reasoning via delta.reasoning_content
#
#  minimaxai/minimax-m2.7
#    no thinking params — standard
#
#  meta/llama-3.3-70b-instruct
#    no thinking params — standard
#
#  openai/gpt-oss-120b
#    extra_body={"reasoning": {"effort": "low"|"medium"|"high"}}
#    reasoning via delta.reasoning_content
#    UI shows low/medium/high radio instead of off/on
#
# "effort" UI radio:
#   gpt-oss models  → "low" | "medium" | "high"
#   all others      → "off" | "on"

def _model_is(model: str, *fragments) -> bool:
    m = model.lower()
    return any(f in m for f in fragments)

def is_gpt_oss(model: str) -> bool:
    return _model_is(model, "openai/gpt-oss", "openai/gpt-5")

def build_reasoning_params(model: str, effort: str) -> tuple[dict | None, dict | None]:
    """
    Returns (extra_body, top_level_extra) where top_level_extra keys are
    merged directly into the request_params dict (not under extra_body).
    Both may be None if no special params are needed.

    effort values:
      gpt-oss models  -> "low" | "medium" | "high"
      all others      -> "off" | "on"
    """
    thinking_on = (effort == "on")

    # ── GPT-OSS (openai/gpt-oss-120b etc.) ───────────────────────────────────
    # effort is already "low"|"medium"|"high" for this model family
    if is_gpt_oss(model):
        return {"reasoning": {"effort": effort}}, None

    # ── DeepSeek V4 Pro ───────────────────────────────────────────────────────
    if _model_is(model, "deepseek-v4-pro"):
        return {"chat_template_kwargs": {"thinking": thinking_on}}, None

    # ── DeepSeek V4 Flash ─────────────────────────────────────────────────────
    if _model_is(model, "deepseek-v4-flash"):
        if thinking_on:
            return {"chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}}, None
        else:
            return {"chat_template_kwargs": {"thinking": False}}, None

    # ── Qwen3.5 models ───────────────────────────────────────────────────
    if _model_is(model, "qwen3.5-122b", "qwen3.5-397b"):
        return {"chat_template_kwargs": {"enable_thinking": thinking_on}}, None

    # ── Qwen3-Coder: standard, no thinking params ─────────────────────────────
    if _model_is(model, "qwen3-coder"):
        return None, None

    # ── Gemma-4 ───────────────────────────────────────────────────────────────
    if _model_is(model, "gemma-4"):
        return {"chat_template_kwargs": {"enable_thinking": thinking_on}}, None

    # ── GLM 5.x ───────────────────────────────────────────────────────────────
    if _model_is(model, "glm-5", "glm5"):
        if thinking_on:
            return {"chat_template_kwargs": {"enable_thinking": True, "clear_thinking": False}}, None
        else:
            return {"chat_template_kwargs": {"enable_thinking": False, "clear_thinking": True}}, None

    # ── Kimi K2 ───────────────────────────────────────────────────────────────
    if _model_is(model, "kimi-k2"):
        return {"chat_template_kwargs": {"thinking": thinking_on}}, None

    # ── Mistral Medium 3.5 ────────────────────────────────────────────────────
    if _model_is(model, "mistral-medium-3.5"):
        effort_str = "high" if thinking_on else "off"
        return None, {"reasoning_effort": effort_str}

    # ── Mistral Large, Meta Llama, MiniMax, Nemotron (non-thinking parts) ─────
    if _model_is(model, "mistral-large"):
        return None, None

    if _model_is(model, "nemotron"):
        if thinking_on:
            return {"chat_template_kwargs": {"enable_thinking": True}, "reasoning_budget": 16384}, None
        else:
            return {"chat_template_kwargs": {"enable_thinking": False}}, None

    # ── Everything else: no special params ───────────────────────────────────
    return None, None


def has_reasoning_ui(model: str) -> bool:
    """True if this model exposes a reasoning/thinking toggle in the UI."""
    return is_gpt_oss(model) or _model_is(
        model,
        "deepseek-v4-flash", "deepseek-v4-pro",
        "qwen3.5-122b", "qwen3.5-397b",
        "gemma-4",
        "glm-5", "glm5",
        "kimi-k2",
        "mistral-medium-3.5",
        "nemotron",
    )


def extract_reasoning(delta) -> str | None:
    """
    Different NVIDIA models put reasoning traces in different delta attributes.
    Try all known attribute names in priority order.
    """
    for attr in ("reasoning_content", "reasoning"):
        val = getattr(delta, attr, None)
        if val:
            return val
    return None

# ── Core chat function ─────────────────────────────────────────────────────────
def chat(message, history, api_history, model_choice, instructions,
         temperature, max_tokens, effort, input_file):

    initial_dl = gr.update(visible=False)

    if not message.strip() and not input_file:
        yield history, api_history, "", "*No reasoning generated yet...*", initial_dl
        return

    # ── Convert upload → list of PIL images ───────────────────────────────────
    MAX_IMAGES = 8  # NVIDIA NIM hard limit per request
    is_vision  = model_choice in vision_models
    images     = file_to_images(input_file) if is_vision else []

    # ── Build user content ─────────────────────────────────────────────────────
    if images:
        n_total   = len(images)
        truncated = n_total > MAX_IMAGES
        if truncated:
            images = images[:MAX_IMAGES]
            warn = f" ⚠️ Only first {MAX_IMAGES} of {n_total} pages sent (NVIDIA limit)."
        else:
            warn = ""
        label = f"📎 [Attached {len(images)} image(s)/page(s)]{warn}"
        user_content_display = f"{label} {message}".strip()

        user_content_api: list[dict] = [
            {
                "type":      "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{pil_to_b64_jpeg(img)}"},
            }
            for img in images
        ]
        user_content_api.append(
            {"type": "text", "text": message or "Describe the content of these images in detail."}
        )
    else:
        user_content_api     = message
        user_content_display = message

    history = history + [
        {"role": "user",      "content": user_content_display},
        {"role": "assistant", "content": ""},
    ]
    api_history = api_history + [
        {"role": "user",      "content": user_content_api},
        {"role": "assistant", "content": ""},
    ]

    # ── Assemble messages for the API ─────────────────────────────────────────
    messages = []
    if instructions.strip():
        messages.append({"role": "system", "content": instructions})

    for m in api_history:
        if m["role"] == "assistant" and m["content"] == "":
            continue
        messages.append({"role": m["role"], "content": m["content"]})

    # ── API call ──────────────────────────────────────────────────────────────
    # FIRST_TOKEN_TIMEOUT: how long we wait for the server to start streaming.
    # The httpx read timeout (180 s) is the hard backstop; this softer deadline
    # lets us show a live "waiting…" counter and abort cleanly via Stop.
    FIRST_TOKEN_TIMEOUT = 180.0
    # How often we poll the Future while waiting for the stream to open (seconds)
    HEARTBEAT_INTERVAL  = 1.0

    try:
        extra_body, top_level_extra = build_reasoning_params(model_choice, effort)

        request_params = {
            "model":       model_choice,
            "messages":    messages,
            "temperature": temperature,
            "max_tokens":  int(max_tokens),
            "stream":      True,
        }

        if extra_body:
            request_params["extra_body"] = extra_body

        if top_level_extra:
            request_params.update(top_level_extra)

        # ── Submit the blocking .create() call to a background thread ─────────
        # This unblocks the generator immediately so Gradio can render heartbeats
        # and honour the Stop button while the server queues / warms up.
        future = _executor.submit(client.chat.completions.create, **request_params)

        wait_start = time.time()
        completion = None

        while completion is None:
            elapsed = time.time() - wait_start
            if elapsed > FIRST_TOKEN_TIMEOUT:
                future.cancel()
                history[-1]["content"]     = (
                    f"❌ Timed out: server did not start responding within "
                    f"{int(FIRST_TOKEN_TIMEOUT)} seconds."
                )
                api_history[-1]["content"] = ""
                yield history, api_history, "", "Timed out waiting for first token.", initial_dl
                return

            # Show a live waiting counter so the user knows we're still alive
            wait_secs = int(elapsed)
            history[-1]["content"] = f"⏳ Waiting for server… {wait_secs}s"
            yield history, api_history, None, "*Waiting for server…*", initial_dl

            try:
                completion = future.result(timeout=HEARTBEAT_INTERVAL)
            except FuturesTimeoutError:
                # Server hasn't replied yet; loop and yield another heartbeat
                pass

        # ── Stream chunks ─────────────────────────────────────────────────────
        full_content      = ""
        reasoning_content = ""
        last_yield_time   = time.time()
        flush_interval_s  = 0.04

        for chunk in completion:
            if not chunk.choices:
                continue

            delta         = chunk.choices[0].delta
            new_content   = getattr(delta, "content", None) or None
            new_reasoning = extract_reasoning(delta)

            if new_content is not None:
                full_content += new_content
                history[-1]["content"]     = full_content
                api_history[-1]["content"] = full_content

            if new_reasoning is not None:
                reasoning_content += new_reasoning

            if time.time() - last_yield_time >= flush_interval_s:
                last_yield_time = time.time()
                yield history, api_history, None, reasoning_content, initial_dl

        # ── Edge-case: stream opened but was immediately empty ────────────────
        if not full_content and not reasoning_content:
            history[-1]["content"]     = "⚠️ Server returned an empty response."
            api_history[-1]["content"] = ""
            yield history, api_history, "", "Empty response from server.", initial_dl
            return

        # ── Save last response to temp file ───────────────────────────────────
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".md", mode="w", encoding="utf-8"
        ) as tf:
            tf.write(full_content)
            output_filepath = tf.name

        temp_files_to_clean.append(output_filepath)
        yield history, api_history, "", reasoning_content, gr.update(visible=True, value=output_filepath)

    except Exception as e:
        history[-1]["content"]     = f"❌ An error occurred: {e}"
        api_history[-1]["content"] = f"❌ An error occurred: {e}"
        yield history, api_history, "", f"An error occurred: {e}", initial_dl


# ── Gradio UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(title="💬 NVIDIA NIM Chatbot") as demo:
    gr.Markdown("# 💬 Chatbot (Powered by NVIDIA NIM API)")

    api_history = gr.State([])

    with gr.Row():
        with gr.Column(scale=3):
            chatbot_ui = gr.Chatbot(height=500, buttons=["copy"])

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type a message...", scale=4, show_label=False
                )
                send_btn = gr.Button("Send", scale=1)

            input_file = gr.File(
                label="📎 Attach Image(s) or PDF(s) (cleared after each send)",
                file_types=["image", ".pdf"],
                file_count="multiple",
                visible=default_model in vision_models,
            )

            with gr.Row():
                stop_btn     = gr.Button("Stop",  scale=1)
                clear_btn    = gr.Button("Clear Chat", scale=1)
                download_btn = gr.DownloadButton(
                    "⬇️ Download Last Response", visible=False, scale=3
                )

        with gr.Column(scale=1):
            model_choice = gr.Dropdown(
                label="Choose a Model",
                choices=combined_models,
                value=default_model,
            )

            effort = gr.Radio(
                ["low", "medium", "high"] if is_gpt_oss(default_model or "") else ["off", "on"],
                value="medium" if is_gpt_oss(default_model or "") else "on",
                label="🧠 Reasoning / Thinking",
                info=(
                    "gpt-oss: low·medium·high effort · "
                    "others: on/off · hidden for models without thinking support"
                ),
                visible=has_reasoning_ui(default_model or ""),
            )

            instructions = gr.Textbox(
                label="System Instructions",
                value=initial_system_prompt_vision if default_model in vision_models else initial_system_prompt,
                lines=3,
            )
            temperature = gr.Slider(
                0.0, 2.0, value=1.0, step=0.1, label="Temperature"
            )
            max_tokens = gr.Slider(
                100, 65535, value=16384, step=256, label="Max Tokens"
            )
            thoughts_box = gr.Markdown(
                label="🧠 Model Thoughts",
                value="*Reasoning will appear here...*",
            )

    def on_model_change(model_name):
        is_vis   = model_name in vision_models
        show_eff = has_reasoning_ui(model_name)
        gpt      = is_gpt_oss(model_name)
        prompt   = initial_system_prompt_vision if is_vis else initial_system_prompt
        effort_choices = ["low", "medium", "high"] if gpt else ["off", "on"]
        effort_value   = "medium" if gpt else "on"
        return (
            gr.update(visible=is_vis),
            gr.update(visible=show_eff, choices=effort_choices, value=effort_value),
            gr.update(value=prompt),
        )

    model_choice.change(
        fn=on_model_change,
        inputs=[model_choice],
        outputs=[input_file, effort, instructions],
        queue=False,
    )

    # ── Wire up send / submit ──────────────────────────────────────────────────
    inputs  = [msg, chatbot_ui, api_history, model_choice, instructions,
               temperature, max_tokens, effort, input_file]
    outputs = [chatbot_ui, api_history, msg, thoughts_box, download_btn]

    e_submit = msg.submit(chat, inputs, outputs)
    e_click  = send_btn.click(chat, inputs, outputs)

    e_submit.then(fn=lambda: gr.update(value=None), outputs=[input_file])
    e_click.then( fn=lambda: gr.update(value=None), outputs=[input_file])

    stop_btn.click(fn=lambda: None, cancels=[e_submit, e_click], queue=False)

    clear_btn.click(
        fn=lambda: (
            [],
            [],
            "*Reasoning will appear here...*",
            gr.update(visible=False),
            gr.update(value=None),
        ),
        outputs=[chatbot_ui, api_history, thoughts_box, download_btn, input_file],
        cancels=[e_submit, e_click],
        queue=False,
    )

demo.queue()

if __name__ == "__main__":
    print("Launching NVIDIA NIM Chatbot... Press Ctrl+C to exit.")
    print("Temporary files for this session will be cleaned up automatically on exit.")
    demo.launch(theme=gr.themes.Default())
