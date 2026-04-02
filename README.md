# 💬 nvidia-nim-chat

A lightweight, feature-rich chatbot interface powered by [NVIDIA NIM](https://build.nvidia.com/explore/discover) (NVIDIA Inference Microservices), built with [Gradio](https://www.gradio.app/). It supports multiple large language models, optional reasoning/thinking mode, streaming responses, and chat log downloads — all from a clean web UI.

---

## ✨ Features

- 🤖 **Multi-model support** — Switch between 10+ frontier LLMs (DeepSeek, Qwen, Mistral, LLaMA, Kimi, MiniMax, GLM, and more) from a dropdown
- 🧠 **Reasoning / Thinking mode** — Enable extended reasoning for supported models with a single checkbox
- ⚡ **Streaming responses** — Tokens are streamed in real time as the model generates them
- 📥 **Download last response** — Save the last assistant reply as a `.md` file
- 🛑 **Stop generation** — Interrupt a running response at any time
- 🗑️ **Clear chat** — Reset the conversation with one click
- 🔧 **Configurable system prompt** — Loaded from `system-prompt.txt` at startup
- 🌡️ **Adjustable parameters** — Temperature and max tokens sliders

---

## 📋 Requirements

- Python 3.8+
- An [NVIDIA NIM API key](https://build.nvidia.com/)

---

## 🚀 Installation

**1. Clone the repository**

```bash
git clone https://github.com/zakcali/nvidia-nim-chat.git
cd nvidia-nim-chat
```

**2. Install dependencies**

```bash
pip install gradio openai
```

**3. Set your NVIDIA API key**

On Linux/macOS:
```bash
export NVIDIA_API_KEY="your_api_key_here"
```

On Windows (Command Prompt):
```cmd
set NVIDIA_API_KEY=your_api_key_here
```

On Windows (PowerShell):
```powershell
$env:NVIDIA_API_KEY="your_api_key_here"
```

**4. Run the app**

```bash
python nvidia-text2text.py
```

The Gradio interface will launch in your browser at `http://localhost:7860`.

---

## 📁 File Structure

```
nvidia-nim-chat/
├── nvidia-text2text.py   # Main application script
├── nvidia-models.txt     # List of available models (one per line)
└── system-prompt.txt     # System prompt loaded at startup
```

### `nvidia-models.txt`

Contains the list of NVIDIA NIM model IDs to populate the dropdown, one per line. Edit this file to add or remove models without changing the code.

```
moonshotai/kimi-k2.5
deepseek-ai/deepseek-v3.2
meta/llama-3.3-70b-instruct
...
```

### `system-prompt.txt`

Plain text file containing the system prompt sent to the model at the start of every conversation. Leave it empty for no system prompt, or fill it with any instructions you like.

---

## 🧠 Reasoning / Thinking Mode

When the **Enable Reasoning** checkbox is checked, the app sends model-specific parameters to activate extended thinking:

| Model family | Parameter used |
|---|---|
| `mistral-small` | `reasoning_effort: "high"` |
| `glm`, `qwen` | `enable_thinking: true` |
| `kimi`, `deepseek` | `thinking: true` |

Reasoning traces appear in the **🧠 Model Thoughts** panel on the right side of the UI.

---

## ⚙️ UI Controls

| Control | Description |
|---|---|
| Model dropdown | Select the active LLM |
| System Instructions | Editable system prompt (pre-loaded from file) |
| Enable Reasoning | Toggle extended thinking mode |
| Temperature | Controls response randomness (0.0 – 2.0) |
| Max Tokens | Maximum response length (100 – 65535) |
| Send / Enter | Submit a message |
| Stop | Cancel the current generation |
| Clear Chat | Reset the conversation history |
| ⬇️ Download Last Response | Save the last reply as a `.md` file |

---

## 🔑 Getting an NVIDIA API Key

1. Go to [https://build.nvidia.com/](https://build.nvidia.com/)
2. Sign in or create a free account
3. Navigate to your profile → **API Keys**
4. Generate a new key and copy it

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [NVIDIA NIM](https://build.nvidia.com/) for providing access to state-of-the-art hosted inference
- [Gradio](https://www.gradio.app/) for the web UI framework
- [OpenAI Python SDK](https://github.com/openai/openai-python) used as the compatible API client
