import os
import gradio as gr
from openai import OpenAI
import time
import tempfile
import atexit

# This list will hold the paths of all generated chat logs for this session.
temp_files_to_clean = []

# --- Function to perform cleanup on exit ---
def cleanup_temp_files():
    """Iterates through the global list and deletes the tracked files."""
    if not temp_files_to_clean:
        return
    print(f"\nCleaning up {len(temp_files_to_clean)} temporary files...")
    for file_path in temp_files_to_clean:
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"  - Error removing {file_path}: {e}")
    print("Cleanup complete.")

atexit.register(cleanup_temp_files)

# --- Function to read system prompt from a file ---
def load_system_prompt(filepath="system-prompt.txt"):
    """Loads the system prompt from a text file, with a fallback default."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Warning: '{filepath}' not found. Using a default system prompt.")
        return "You are a helpful assistant."

# --- Function to read the model list from a file ---
def load_models(filepath="nvidia-models.txt"):
    """Loads the list of models from a text file, with a fallback default list."""
    default_models = [
        "moonshotai/kimi-k2.5",
        "minimaxai/minimax-m2.5",
        "z-ai/glm5",
        "deepseek-ai/deepseek-v3.2",
        "deepseek-ai/deepseek-v3.1",
        "qwen/qwen3-coder-480b-a35b-instruct",
        "qwen/qwen3.5-397b-a17b",
        "google/gemma-4-31b-it",
        "mistralai/mistral-small-4-119b-2603",
        "mistralai/mixtral-8x22b-instruct-v0.1",
        "meta/llama-3.1-405b-instruct",
        "meta/llama-3.3-70b-instruct",
    ]
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            # Read non-empty lines and strip whitespace
            models = [line.strip() for line in f if line.strip()]
            if not models:
                print(f"Warning: '{filepath}' was empty. Using default model list.")
                return default_models
            return models
    except FileNotFoundError:
        print(f"Warning: '{filepath}' not found. Using default model list.")
        return default_models

print("Temporary chat download files will be saved in the OS's default temp directory.")

# Initialize the OpenAI client to connect to NVIDIA NIM
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NVIDIA_API_KEY"),
)

# --- Updated function signature to accept model_choice ---
def chat_with_nvidia(message, history, model_choice, instructions,
                     temperature, max_tokens, reasoning_enabled):

    initial_download_update = gr.update(visible=False)

    if not message.strip():
        return history, "", "*No reasoning generated yet...*", initial_download_update

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""}
    ]

    messages = []
    if instructions.strip():
        messages.append({"role": "system", "content": instructions})

    for m in history:
        if m["role"] == "assistant" and m["content"] == "":
            continue
        messages.append({"role": m["role"], "content": m["content"]})

    try:
        # Use a dictionary for request parameters for conditional logic
        request_params = {
            "model": model_choice, # Use the selected model
            "messages": messages,
            "temperature": temperature,
            "top_p": 1.0,
            "max_tokens": int(max_tokens),
            "stream": True,
        }

        # --- Thinking / reasoning parameter logic ---
        # Only models confirmed to support thinking in thinking_parameters.txt are
        # included here. All other models (LLaMA, Mixtral, MiniMax, Qwen-coder)
        # receive no thinking parameters, regardless of the checkbox state.
        #
        # Allowlist by parameter style:
        #   "reasoning_effort"          : mistral-small
        #   enable_thinking (True/False): glm, qwen3.5 (non-coder), gemma
        #   thinking (True/False)       : kimi, deepseek
        #
        # Models with NO thinking support (never send thinking params):
        #   meta/llama-*, mistralai/mixtral-*, minimaxai/minimax-*,
        #   qwen/qwen3-coder-*
        #
        # Gemma note: also requires top_k=64 for its recommended inference settings.

        model_lower = model_choice.lower()

        # Identify models that have NO thinking support — skip all param injection.
        no_thinking_support = (
            "llama" in model_lower
            or "mixtral-8x22b-instruct-v0.1" in model_lower
            or "minimax-m2.5" in model_lower
            or "qwen3-coder-480b-a35b-instruct" in model_lower
        )

        # Gemma requires top_k=64 regardless of reasoning mode (per reference).
        if "gemma-4" in model_lower:
            request_params["extra_body"] = {"top_k": 64}

        if no_thinking_support:
            # Never send thinking params to these models.
            pass

        elif reasoning_enabled:
            if "mistral-small" in model_lower:
                # Uses top-level reasoning_effort, not chat_template_kwargs
                request_params["reasoning_effort"] = "high"
            elif "glm" in model_lower:
                request_params["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": True, "clear_thinking": False}
                }
            elif "gemma-4" in model_lower:
                # Gemma uses enable_thinking; preserve top_k already set above.
                request_params["extra_body"] = {
                    "top_k": 64,
                    "chat_template_kwargs": {"enable_thinking": True},
                }
            elif "qwen" in model_lower:
                # Applies to qwen3.5 variants (non-coder, already excluded above)
                request_params["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": True}
                }
            elif "kimi" in model_lower or "deepseek" in model_lower:
                request_params["extra_body"] = {
                    "chat_template_kwargs": {"thinking": True}
                }
            # Any other unrecognised model: do NOT send thinking params by default.

        else:
            # Reasoning disabled — explicitly turn off thinking only for models
            # that support it (to override any server-side defaults).
            if "mistral-small" in model_lower:
                # Omitting reasoning_effort disables it for Mistral Small.
                pass
            elif "glm" in model_lower:
                request_params["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": False, "clear_thinking": False}
                }
            elif "gemma" in model_lower:
                # Keep top_k; explicitly disable thinking.
                request_params["extra_body"] = {
                    "top_k": 64,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            elif "qwen" in model_lower:
                request_params["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            elif "kimi" in model_lower or "deepseek" in model_lower:
                request_params["extra_body"] = {
                    "chat_template_kwargs": {"thinking": False}
                }
       
        completion = client.chat.completions.create(**request_params)

        full_content = ""
        metadata_reasoning = ""
        
        last_yield_time = time.time()
        flush_interval_s = 0.04

        for chunk in completion:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            new_content = getattr(delta, "content", None)
            new_reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)

            if new_reasoning is not None:
                metadata_reasoning += new_reasoning

            if new_content is not None:
                full_content += new_content
            
            # Determine display reasoning and chat content
            display_reasoning = metadata_reasoning
            display_chat_content = full_content

            # Handle models with <think> tags in content
            if "<think>" in full_content:
                parts = full_content.split("</think>", 1)
                if len(parts) > 1:
                    # Finished thinking
                    tag_reasoning = parts[0].replace("<think>", "").strip()
                    display_chat_content = parts[1].lstrip()
                    if not display_reasoning:
                        display_reasoning = tag_reasoning
                else:
                    # Still thinking
                    tag_reasoning = full_content.replace("<think>", "").strip()
                    display_chat_content = "*Thinking...*"
                    if not display_reasoning:
                        display_reasoning = tag_reasoning

            history[-1]["content"] = display_chat_content
            
            # Final fallback for reasoning display
            if not display_reasoning and reasoning_enabled:
                display_reasoning = "*Reasoning enabled, waiting for output...*"
            elif not reasoning_enabled:
                display_reasoning = "*Reasoning is not enabled.*"

            now = time.time()
            if now - last_yield_time >= flush_interval_s:
                last_yield_time = now
                yield history, None, display_reasoning, initial_download_update

        with tempfile.NamedTemporaryFile(delete=False, suffix=".md", mode="w", encoding="utf-8") as temp_file:
            output_filepath = temp_file.name
            # Write only the final content (without tags) to the file
            temp_file.write(display_chat_content)

        temp_files_to_clean.append(output_filepath)
        print(f"Created and tracking temp file: {output_filepath}")

        final_download_update = gr.update(visible=True, value=output_filepath)

        yield history, "", display_reasoning, final_download_update

    except Exception as e:
        error_message = f"❌ An error occurred: {str(e)}"
        history[-1]["content"] = error_message
        yield history, "", f"An error occurred: {e}", initial_download_update

# Load external configuration before building UI ---
model_list = load_models()
initial_system_prompt = load_system_prompt()
# Set the default model to the first one in the list, or None if the list is empty
default_model = model_list[0] if model_list else None

# --- Gradio UI  ---
with gr.Blocks(title="💬 NVIDIA NIM Chatbot") as demo:
    gr.Markdown("# 💬 Chatbot (Powered by NVIDIA NIM)")
    with gr.Row():
        with gr.Column(scale=3):
            # NEW (Gradio 6.0)
            chatbot = gr.Chatbot(height=500, buttons=["copy"])
            with gr.Row():
                msg = gr.Textbox(placeholder="Type a message...", scale=4, show_label=False)
                send_btn = gr.Button("Send", scale=1)
            with gr.Row():
                stop_btn = gr.Button("Stop", scale=1)
                clear_btn = gr.Button("Clear Chat", scale=1)
                download_btn = gr.DownloadButton("⬇️ Download Last Response", visible=False, scale=3)

        with gr.Column(scale=1):
            # Models are loaded from nvidia-models.txt ---
            model_choice = gr.Dropdown(
                label="Choose a Model",
                choices=model_list,
                value=default_model
            )
            
            # System prompt is loaded from system-prompt.txt ---
            instructions = gr.Textbox(
                label="System Instructions", 
                value=initial_system_prompt, 
                lines=3
            )
            
            reasoning_enabled = gr.Checkbox(
                label="Enable Reasoning (Thinking Models)",
                value=False
            )
            temperature = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Temperature")
            max_tokens = gr.Slider(1024, 262144, value=8192, step=1024, label="Max Tokens")
            thoughts_box = gr.Markdown(label="🧠 Model Thoughts", value="*Reasoning will appear here...*")

    inputs = [msg, chatbot, model_choice, instructions, temperature, max_tokens, reasoning_enabled]
    outputs = [chatbot, msg, thoughts_box, download_btn]

    e_submit = msg.submit(chat_with_nvidia, inputs, outputs)
    e_click = send_btn.click(chat_with_nvidia, inputs, outputs)

    stop_btn.click(fn=lambda: None, cancels=[e_submit, e_click], queue=False)

    clear_btn.click(
        lambda: ([], "*Reasoning will appear here...*", gr.update(visible=False)),
        outputs=[chatbot, thoughts_box, download_btn],
        cancels=[e_submit, e_click],
        queue=False
    )

demo.queue()

if __name__ == "__main__":
    print("Launching Gradio interface... Press Ctrl+C to exit.")
    print("Temporary files for this session will be cleaned up automatically on exit.")
demo.launch(theme=gr.themes.Default())