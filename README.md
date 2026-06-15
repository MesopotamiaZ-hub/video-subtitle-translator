# 🎬 视频字幕翻译工具

基于 **faster-whisper** + **Deepseek API** + **FFmpeg** 的本地视频字幕翻译工具，提供可视化网页界面。

## 功能概览

1. **视频上传**：支持拖拽/点击上传 mp4、mov、mkv、avi 等主流格式
2. **自动语音识别**：使用 faster-whisper 本地模型，自动检测对话语言并生成精确时间轴 SRT 字幕
3. **AI 字幕翻译**：调用 Deepseek 大模型 API 进行高质量字幕翻译
4. **字幕压制**：使用 FFmpeg 将翻译后字幕硬编码到视频中，带美观样式（底部居中、半透明描边）

## 环境要求

| 组件 | 说明 |
|------|------|
| Python 3.10+ | 运行环境 |
| FFmpeg | 音频提取 + 字幕压制（必须安装） |
| faster-whisper | 本地语音识别（首次运行自动下载模型） |
| Deepseek API Key | 翻译功能必需 |

## 安装步骤

### 1. 安装 FFmpeg

**Windows：**
1. 从 https://ffmpeg.org/download.html 下载 Windows 版本
2. 解压到任意目录（如 `C:\ffmpeg`）
3. 将 `bin` 目录（如 `C:\ffmpeg\bin`）添加到系统 PATH：
   - 右键「此电脑」→「属性」→「高级系统设置」→「环境变量」
   - 在「系统变量」中找到 `Path`，添加 FFmpeg 的 `bin` 目录路径
4. 打开新的终端窗口，运行 `ffmpeg -version` 验证

**macOS：**
```bash
brew install ffmpeg
```

**Ubuntu/Debian：**
```bash
sudo apt update && sudo apt install ffmpeg
```

### 2. 安装 Python 依赖

```bash
# 进入项目目录
cd video-subtitle-translator

# 安装依赖（使用项目已有的虚拟环境）
pip install gradio faster-whisper openai
```

### 3. 获取 Deepseek API Key

1. 访问 https://platform.deepseek.com/
2. 注册并登录账号
3. 在「API Keys」页面创建新的 API Key
4. 复制保存 API Key（格式为 `sk-xxxxxxxxxxxxxxxx`）

## 启动工具

```bash
cd video-subtitle-translator
python video_subtitle_translator.py
```

启动后浏览器会自动打开 `http://127.0.0.1:7860`。

### 通过环境变量配置（可选）

```bash
# Windows (PowerShell)
$env:DEEPSEEK_API_KEY="sk-xxxxxxxxxxxxxxxx"

# Linux/macOS
export DEEPSEEK_API_KEY="sk-xxxxxxxxxxxxxxxx"

# 然后启动
python video_subtitle_translator.py
```

也可以在代码顶部直接修改默认值：
```python
DEEPSEEK_API_KEY = "sk-xxxxxxxxxxxxxxxx"  # 直接填你的 Key
```

## 使用流程

1. **上传视频**：在网页界面拖拽或点击上传视频文件
2. **设置语言**：
   - 源语言：默认「自动检测」（推荐），也可手动选择
   - 目标语言：默认「中文」，可切换为英语、日语等
3. **填写 API Key**：在「API 设置」区域输入 Deepseek API Key（如已通过环境变量设置则自动填入）
4. **开始处理**：点击「开始处理」按钮
5. **等待完成**：观察四个阶段的实时进度：
   - 🔊 提取音频轨道
   - 🎙️ 语音识别（faster-whisper）
   - 🌐 AI 翻译字幕（Deepseek API）
   - 🎬 压制字幕到视频（FFmpeg）
6. **下载结果**：
   - 📄 翻译后的 SRT 字幕文件（可单独使用）
   - 🎥 内嵌字幕的成品视频

## 配置说明

代码顶部提供以下配置项，可按需修改：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `DEEPSEEK_API_KEY` | Deepseek API 密钥 | 从环境变量读取 |
| `DEEPSEEK_BASE_URL` | API 接口地址 | `https://api.deepseek.com` |
| `DEEPSEEK_MODEL` | 使用的模型 | `deepseek-chat` |
| `WHISPER_MODEL_SIZE` | Whisper 模型大小 | `medium`（平衡速度和精度）|
| `FFMPEG_PATH` | FFmpeg 路径 | 自动检测 PATH |

Whisper 模型大小选项：
- `tiny`：最快，精度较低（~75MB）
- `base`：快速，基础精度（~145MB）
- `small`：中等速度，较好精度（~488MB）
- `medium`：较慢，高精度（~1.5GB）✅ 推荐
- `large-v3`：最慢，最高精度（~3GB）

## 字幕样式

压制到视频的字幕样式：
- 底部居中显示
- 白色字体，22号字号
- 黑色半透明描边
- 底部 40px 边距

如需自定义样式，修改 `burn_subtitles()` 函数中的 `subtitle_style` 变量。

## 常见问题

**Q: 提示"未检测到 FFmpeg"怎么办？**
A: 请确保已安装 FFmpeg 并添加到系统 PATH。可在终端运行 `ffmpeg -version` 验证。

**Q: 首次运行很久没反应？**
A: 首次运行会自动下载 faster-whisper 模型（约 1.5GB）。请耐心等待，后续启动不需要重新下载。

**Q: 翻译结果格式有问题？**
A: 偶尔大模型可能输出非纯 SRT 格式。代码已做清理处理。如果仍有问题，尝试调低 `temperature` 参数或更换更大 whisper 模型提高识别准确度。

**Q: 处理视频很慢？**
A: 处理时间取决于视频长度和硬件配置。faster-whisper 使用 CPU 推理时速度较慢，有 NVIDIA GPU 可显著加速（自动检测 CUDA）。

**Q: API 调用报 400 错误？**
A: 确认 API Key 正确、账户余额充足。如果看到 `reasoning_effort` 相关错误，请确认使用的是最新版本代码（已去除所有推理强度相关参数）。
