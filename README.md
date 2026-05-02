# NVIDIA NIM Chat

A sleek, robust, and multimodal Gradio-based web interface designed to chat seamlessly with large language models and vision models hosted on the [NVIDIA NIM API](https://build.nvidia.com/).

This project intelligently handles API idiosyncrasies, particularly around complex "Reasoning" and "Thinking" parameters, dynamically adapting its UI and backend payloads to support a diverse ecosystem of models (DeepSeek, Qwen, Mistral, Gemma, GLM, Llama, and more).

## 🌟 Key Features

*   **Multimodal (Vision) Support:** Chat about uploaded Images or PDF documents. PDFs are automatically converted to images, resized to respect API limits, and attached to your prompt.
*   **Dynamic Reasoning Controls:** The UI automatically adapts based on the selected model's capabilities:
    *   Models with thinking toggles (e.g., DeepSeek, Qwen3.5, Gemma-4) display an `On`/`Off` reasoning switch.
    *   Specific models like `gpt-oss` offer granular effort controls (`Low`, `Medium`, `High`).
    *   Models without reasoning support automatically hide the UI clutter.
*   **Streaming Responses & Thoughts:** Watch both the final response and the underlying model "thoughts" stream in real-time.
*   **Auto-Cleanup:** Safely cleans up temporary files, converted PDFs, and exported markdown logs automatically upon application exit.
*   **Response Export:** One-click download of the model's last response as a `.md` file.

## 📋 Prerequisites

*   Python 3.10+
*   An NVIDIA API Key (Get one from [NVIDIA API Catalog](https://build.nvidia.com/))
*   `poppler` (Required for PDF to Image conversion. See [pdf2image documentation](https://pdf2image.readthedocs.io/en/latest/installation.html) for OS-specific installation instructions).

## 🚀 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/zakcali/nvidia-nim-chat.git
    cd nvidia-nim-chat
    ```

2.  **Install dependencies:**
    It is recommended to use `uv` or a virtual environment.
    ```bash
    pip install gradio openai Pillow pdf2image
    ```

## ⚙️ Configuration

### 1. Environment Variable
You must set your NVIDIA API key as an environment variable before running the application.

**Windows (PowerShell):**
```powershell
$env:NVIDIA_API_KEY="your_api_key_here"
```

**Linux / macOS:**
```bash
export NVIDIA_API_KEY="your_api_key_here"
```

### 2. Model Lists
The application reads available models from two text files in the root directory. You can edit these to add or remove models as they become available on the NVIDIA API.

*   `nvidia-models.txt`: Contains a list of all available text and reasoning models.
*   `nvidia-models-image.txt`: Contains a list of models that support multimodal (vision) input. *If a selected model is in this list, the file upload UI will become visible.*

### 3. System Prompts (Optional)
You can define default system instructions by creating/editing:
*   `system-prompt.txt`: Default instructions for text-only models.
*   `system-prompt-image.txt`: Default instructions for vision models.

## 💻 Usage

Run the script to launch the local Gradio server:

```bash
python text-vision-pdf2text.py
```
*(If you are using `uv`, you can run `uv run text-vision-pdf2text.py`)*

The application will provide a local URL (typically `http://127.0.0.1:7860`) that you can open in your web browser.

## 🧠 Supported Reasoning Implementations

The script natively bridges the gap between different API schemas required by various models on the NVIDIA platform. It handles the following payload keys automatically under the hood:

*   `chat_template_kwargs: {"thinking": True}`
*   `chat_template_kwargs: {"enable_thinking": True}`
*   `reasoning_effort: "high"`
*   `reasoning_budget` allocations
*   ...and correctly extracts reasoning traces whether they are returned in `delta.reasoning` or `delta.reasoning_content`.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/zakcali/nvidia-nim-chat/issues).
