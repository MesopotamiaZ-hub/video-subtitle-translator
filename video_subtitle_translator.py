#!/usr/bin/env python3
"""
视频字幕翻译工具 - Video Subtitle Translator
基于 faster-whisper + Deepseek API + FFmpeg 的本地视频字幕翻译工具
Gradio 网页界面，支持视频上传、语音识别、字幕翻译、字幕压制
"""

import os
import sys
import tempfile
import shutil
import subprocess
import re
from pathlib import Path
from datetime import timedelta

import gradio as gr
from openai import OpenAI
from faster_whisper import WhisperModel

# ============================================================
# 配置项 - 请在此处填写你的 API 密钥和接口地址
# ============================================================

# Deepseek API 密钥（必填）
# 获取方式：访问 https://platform.deepseek.com/ 注册并创建 API Key
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# Deepseek API 接口地址（通常无需修改）
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# Deepseek 模型名称（通常无需修改）
# 可选: deepseek-chat, deepseek-reasoner
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

# faster-whisper 模型大小（可选: tiny, base, small, medium, large-v2, large-v3）
# 模型越大精度越高，但速度越慢。首次运行会自动下载模型。
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "medium")

# FFmpeg 可执行文件路径（通常无需修改，如果在 PATH 中则留空）
FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "")

# 输出文件保存目录
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# ============================================================

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 全局 Whisper 模型实例（懒加载）
_whisper_model = None


def get_ffmpeg():
    """获取 FFmpeg 可执行文件路径"""
    if FFMPEG_PATH:
        return FFMPEG_PATH
    # 尝试常见安装路径
    common_paths = [
        os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "ffmpeg", "bin", "ffmpeg.exe"),
        r"C:\ffmpeg\bin\ffmpeg.exe",
    ]
    for p in common_paths:
        if os.path.isfile(p):
            return p
    # 尝试在 PATH 中查找
    for cmd in ["ffmpeg", "ffmpeg.exe"]:
        path = shutil.which(cmd)
        if path:
            return path
    raise FileNotFoundError(
        "未检测到 FFmpeg！请安装 FFmpeg 并将其添加到系统 PATH 环境变量中。\n"
        "Windows: 从 https://ffmpeg.org/download.html 下载，解压后将 bin 目录添加到 PATH\n"
        "macOS: brew install ffmpeg\n"
        "Ubuntu/Debian: sudo apt install ffmpeg"
    )


def get_whisper_model():
    """懒加载 faster-whisper 模型"""
    global _whisper_model
    if _whisper_model is None:
        # 使用 CPU 还是 GPU 自动检测
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device=device, compute_type=compute_type)
    return _whisper_model


# ---- 音频提取 ----

def extract_audio(video_path: str, output_audio_path: str) -> str:
    """
    使用 FFmpeg 从视频中提取音频轨道
    输出为 16kHz 单声道 WAV，这是 faster-whisper 的最佳输入格式
    """
    ffmpeg = get_ffmpeg()
    cmd = [
        ffmpeg, "-y", "-i", video_path,
        "-vn",                    # 不要视频
        "-acodec", "pcm_s16le",   # PCM 16-bit
        "-ar", "16000",           # 16kHz 采样率
        "-ac", "1",               # 单声道
        output_audio_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"音频提取失败：{result.stderr}")
    if not os.path.exists(output_audio_path) or os.path.getsize(output_audio_path) == 0:
        raise RuntimeError("音频提取失败：输出文件为空或不存在，可能视频中没有有效的音频轨道。")
    return output_audio_path


# ---- 语音识别 ----

def format_timestamp(seconds: float) -> str:
    """将秒数转换为 SRT 时间戳格式 HH:MM:SS,mmm"""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def transcribe_audio(audio_path: str, language: str = None) -> tuple:
    """
    使用 faster-whisper 对音频进行语音识别，生成 SRT 格式字幕
    返回 (srt_content, detected_language)
    """
    model = get_whisper_model()

    # 设置语言参数：None 表示自动检测
    whisper_lang = None if language in (None, "auto", "") else language

    segments, info = model.transcribe(
        audio_path,
        language=whisper_lang,
        beam_size=5,
        vad_filter=True,              # 使用 VAD 过滤静音段
        vad_parameters=dict(
            min_silence_duration_ms=500,
        ),
    )

    detected_lang = info.language
    srt_lines = []
    index = 1

    for segment in segments:
        start_ts = format_timestamp(segment.start)
        end_ts = format_timestamp(segment.end)
        text = segment.text.strip()

        srt_lines.append(f"{index}")
        srt_lines.append(f"{start_ts} --> {end_ts}")
        srt_lines.append(text)
        srt_lines.append("")
        index += 1

    srt_content = "\n".join(srt_lines)
    return srt_content, detected_lang


# ---- 字幕翻译 ----

def build_translation_prompt(target_language: str) -> str:
    """构建翻译系统提示词"""
    lang_names = {
        "zh": "简体中文",
        "en": "English",
        "ja": "日本語",
        "ko": "한국어",
        "fr": "Français",
        "de": "Deutsch",
        "es": "Español",
        "ru": "Русский",
        "pt": "Português",
        "ar": "العربية",
        "th": "ไทย",
        "vi": "Tiếng Việt",
    }
    target_name = lang_names.get(target_language, target_language)

    return (
        f"你是一个专业的视频字幕翻译器。请将用户提供的SRT格式字幕翻译成{target_name}。\n\n"
        "翻译规则（必须严格遵守）：\n"
        "1. 绝对保留原SRT文件的序号和时间戳格式，只翻译对话文本内容，严禁修改任何时间轴和编号\n"
        "2. 翻译风格口语化自然，符合目标语言的日常表达习惯，适配普通视频、影视、教程等通用场景\n"
        "3. 输出必须是纯标准SRT格式内容，开头、结尾不得添加任何解释、说明、问候、标记性文字，直接输出SRT正文\n"
        "4. 保持每句字幕的长度适中，如果原文过长，可以在保持语义的前提下适当拆分或精简\n"
        "5. 对于无法翻译的专有名词、品牌名、人名等，保留原文"
    )


def translate_srt_chunk(client: OpenAI, srt_chunk: str, target_language: str) -> str:
    """
    调用 Deepseek API 翻译单段 SRT 内容
    严格遵守 API 调用规范：使用 OpenAI 兼容接口，不添加 reasoning_effort 等参数
    """
    system_prompt = build_translation_prompt(target_language)

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"请将以下SRT字幕翻译成目标语言，只输出翻译后的SRT内容：\n\n{srt_chunk}"},
            ],
            temperature=0.3,   # 较低温度保证翻译一致性
            max_tokens=8192,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"Deepseek API 调用失败：{str(e)}")


def chunk_srt(srt_content: str, max_entries: int = 50) -> list:
    """
    将 SRT 内容按字幕条目分块
    每块包含完整的字幕条目（序号+时间戳+文本+空行），避免截断
    """
    # 按空行分割字幕条目
    blocks = re.split(r'\n\s*\n', srt_content.strip())
    blocks = [b.strip() for b in blocks if b.strip()]

    chunks = []
    current_chunk = []

    for block in blocks:
        current_chunk.append(block)
        # 每个有效SRT条目包含3行（序号、时间戳、文本）
        if len(current_chunk) >= max_entries:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def translate_srt(srt_content: str, target_language: str, api_key: str, base_url: str) -> str:
    """
    翻译完整的 SRT 字幕文件
    将长字幕分块翻译后合并，确保输出格式正确
    """
    if not api_key:
        raise ValueError("API 密钥不能为空！请在代码顶部配置 DEEPSEEK_API_KEY 或设置环境变量。")

    client = OpenAI(api_key=api_key, base_url=base_url)

    chunks = chunk_srt(srt_content)

    if len(chunks) == 1:
        translated = translate_srt_chunk(client, chunks[0], target_language)
    else:
        # 分批翻译
        translated_parts = []
        for i, chunk in enumerate(chunks):
            part = translate_srt_chunk(client, chunk, target_language)
            translated_parts.append(part)
        translated = "\n\n".join(translated_parts)

    # 清理和修复输出：去除首尾的非SRT内容
    translated = clean_srt_output(translated)

    return translated


def clean_srt_output(text: str) -> str:
    """
    清理翻译输出，确保是纯SRT格式
    去除可能的开头解释文字、结尾标记等
    """
    # 移除 Markdown 代码块标记
    text = re.sub(r'^```(?:srt)?\s*\n?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n?```\s*$', '', text)

    # 尝试找到第一个有效 SRT 序号（数字开头后紧跟换行和时间戳）
    match = re.search(r'(?:^|\n)(\d+\s*\n\d{2}:\d{2}:\d{2},\d{3}\s*-->)', text)
    if match:
        # 从第一个序号开始截取
        start_pos = match.start()
        if text[start_pos] == '\n':
            start_pos += 1
        text = text[start_pos:]

    return text.strip()


# ---- 字幕压制 ----

def burn_subtitles(video_path: str, srt_path: str, output_video_path: str) -> str:
    """
    使用 FFmpeg 将 SRT 字幕硬编码压制到视频中
    字幕样式：底部居中、字号适中、黑色半透明描边
    """
    ffmpeg = get_ffmpeg()

    # 构建字幕样式
    # MarginV=40: 底部边距 40 像素
    # FontSize=22: 字号
    # Outline=2: 描边宽度
    # Shadow=1: 阴影
    # BorderStyle=3: 不透明背景盒
    # OutlineColour=&H80000000: 黑色半透明描边 (ARGB: 80=半透明, 000000=黑色)
    # Alignment=2: 底部居中
    subtitle_style = (
        "FontName=Arial,"
        "FontSize=22,"
        "PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H80000000,"
        "Outline=2.5,"
        "Shadow=1,"
        "BorderStyle=1,"
        "Alignment=2,"
        "MarginV=40"
    )

    # Windows 上需要转义路径中的冒号和反斜杠
    srt_path_escaped = srt_path.replace("\\", "/").replace(":", "\\:")

    cmd = [
        ffmpeg, "-y", "-i", video_path,
        "-vf", f"subtitles='{srt_path_escaped}':force_style='{subtitle_style}'",
        "-c:v", "libx264",          # H.264 编码
        "-crf", "23",                # 质量参数（越小质量越高）
        "-preset", "medium",         # 编码速度
        "-c:a", "aac",               # 音频 AAC 编码
        "-b:a", "128k",              # 音频码率
        output_video_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"字幕压制失败：{result.stderr}")
    if not os.path.exists(output_video_path) or os.path.getsize(output_video_path) == 0:
        raise RuntimeError("字幕压制失败：输出文件不存在或为空。")
    return output_video_path


# ---- 主处理流程 ----

def process_video(
    video_file,
    source_language: str,
    target_language: str,
    api_key: str,
    base_url: str,
    progress=gr.Progress(),
):
    """
    完整的视频字幕翻译处理流程：
    1. 提取音频
    2. 语音识别生成 SRT
    3. 翻译 SRT 字幕
    4. 压制字幕到视频
    返回: (srt_path, output_video_path, status_message)
    """
    # ---- 参数校验 ----
    if video_file is None:
        return None, None, "❌ 错误：请先上传视频文件！"

    if not api_key:
        return None, None, "❌ 错误：API 密钥不能为空！请在代码顶部或环境变量中设置 DEEPSEEK_API_KEY。"

    try:
        get_ffmpeg()
    except FileNotFoundError as e:
        return None, None, f"❌ 错误：{str(e)}"

    # ---- 准备临时文件 ----
    video_path = video_file.name if hasattr(video_file, 'name') else video_file
    base_name = Path(video_path).stem
    # 清理文件名中的特殊字符
    safe_name = re.sub(r'[^\w\-_.]', '_', base_name)

    temp_dir = tempfile.mkdtemp(prefix="vst_")
    audio_path = os.path.join(temp_dir, f"{safe_name}_audio.wav")
    original_srt_path = os.path.join(temp_dir, f"{safe_name}_original.srt")
    translated_srt_path = os.path.join(temp_dir, f"{safe_name}_translated.srt")

    # 最终输出路径
    output_srt_path = os.path.join(OUTPUT_DIR, f"{safe_name}_translated.srt")
    output_video_path = os.path.join(OUTPUT_DIR, f"{safe_name}_subtitled.mp4")

    try:
        # ======== 阶段 1：提取音频 ========
        progress(0.05, desc="🔊 阶段 1/4：正在从视频中提取音频轨道...")
        extract_audio(video_path, audio_path)
        progress(0.25, desc="✅ 阶段 1/4：音频提取完成")

        # ======== 阶段 2：语音识别 ========
        progress(0.28, desc="🎙️ 阶段 2/4：正在进行语音识别（faster-whisper，可能需要几分钟）...")
        lang_param = None if source_language in ("auto", "", None) else source_language
        srt_content, detected_lang = transcribe_audio(audio_path, lang_param)
        progress(0.50, desc=f"✅ 阶段 2/4：语音识别完成（检测到语言：{detected_lang}）")

        # 保存原始识别出的 SRT（供参考）
        with open(original_srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        # ======== 阶段 3：翻译字幕 ========
        progress(0.52, desc="🌐 阶段 3/4：正在调用 Deepseek API 翻译字幕...")
        translated_srt = translate_srt(srt_content, target_language, api_key, base_url)

        # 保存翻译后的 SRT
        with open(translated_srt_path, "w", encoding="utf-8") as f:
            f.write(translated_srt)
        # 同时复制到输出目录
        shutil.copy(translated_srt_path, output_srt_path)

        progress(0.75, desc="✅ 阶段 3/4：字幕翻译完成")

        # ======== 阶段 4：压制字幕 ========
        progress(0.77, desc="🎬 阶段 4/4：正在将字幕压制到视频中（可能需要几分钟）...")
        burn_subtitles(video_path, translated_srt_path, output_video_path)
        progress(1.0, desc="✅ 全部完成！")

        status = (
            f"✅ 处理完成！\n\n"
            f"📊 处理摘要：\n"
            f"  - 检测到的源语言：{detected_lang}\n"
            f"  - 目标语言：{target_language}\n"
            f"  - 字幕条目数：{len(translated_srt.split(chr(10)+chr(10)))} 条\n"
            f"  - 输出字幕：{output_srt_path}\n"
            f"  - 输出视频：{output_video_path}\n"
        )

        return output_srt_path, output_video_path, status

    except Exception as e:
        error_msg = f"❌ 处理失败：{str(e)}"
        return None, None, error_msg

    finally:
        # 清理临时文件
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


# ---- Gradio 界面 ----

def create_ui():
    """创建 Gradio 网页界面"""

    # 语言列表
    LANGUAGES = [
        ("自动检测", "auto"),
        ("中文 (Chinese)", "zh"),
        ("英语 (English)", "en"),
        ("日语 (Japanese)", "ja"),
        ("韩语 (Korean)", "ko"),
        ("法语 (French)", "fr"),
        ("德语 (German)", "de"),
        ("西班牙语 (Spanish)", "es"),
        ("俄语 (Russian)", "ru"),
        ("葡萄牙语 (Portuguese)", "pt"),
        ("阿拉伯语 (Arabic)", "ar"),
        ("泰语 (Thai)", "th"),
        ("越南语 (Vietnamese)", "vi"),
    ]

    TARGET_LANGUAGES = [
        ("中文 (Chinese)", "zh"),
        ("英语 (English)", "en"),
        ("日语 (Japanese)", "ja"),
        ("韩语 (Korean)", "ko"),
        ("法语 (French)", "fr"),
        ("德语 (German)", "de"),
        ("西班牙语 (Spanish)", "es"),
    ]

    with gr.Blocks(title="Video Subtitle Translator") as app:

        gr.Markdown(
            """
            # 🎬 视频字幕翻译工具
            ### 基于 faster-whisper + Deepseek API + FFmpeg

            上传视频 → 自动提取音频 → 语音识别 → AI翻译字幕 → 压制内嵌字幕视频
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # ---- 视频上传 ----
                gr.Markdown("### 📤 视频上传")
                video_input = gr.Video(
                    label="拖拽或点击上传视频",
                    sources=["upload"],
                )

                # ---- 语言设置 ----
                gr.Markdown("### 🌐 语言设置")
                source_lang = gr.Dropdown(
                    choices=LANGUAGES,
                    value="auto",
                    label="源语言（推荐自动检测）",
                    info="自动检测对话语言，也可手动选择",
                )
                target_lang = gr.Dropdown(
                    choices=TARGET_LANGUAGES,
                    value="zh",
                    label="目标语言",
                    info="选择翻译后的字幕语言",
                )

                # ---- API 设置 ----
                gr.Markdown("### 🔑 API 设置")
                api_key_input = gr.Textbox(
                    value=DEEPSEEK_API_KEY,
                    label="Deepseek API Key",
                    placeholder="sk-xxxxxxxxxxxxxxxx",
                    type="password",
                    info="在 https://platform.deepseek.com/ 获取",
                )
                api_url_input = gr.Textbox(
                    value=DEEPSEEK_BASE_URL,
                    label="API 接口地址",
                    placeholder="https://api.deepseek.com",
                    info="通常无需修改",
                )

                # ---- 操作按钮 ----
                gr.Markdown("### 🚀 操作")
                process_btn = gr.Button("开始处理", variant="primary", size="lg")

            with gr.Column(scale=2):
                # ---- 状态显示 ----
                gr.Markdown("### 📊 处理状态")
                status_output = gr.Textbox(
                    label="状态信息",
                    lines=12,
                    interactive=False,
                    placeholder="等待上传视频...",
                )

                # ---- 结果下载 ----
                gr.Markdown("### 📥 结果下载")
                with gr.Row():
                    srt_output = gr.File(
                        label="下载翻译字幕 (SRT)",
                        file_types=[".srt"],
                    )
                    video_output = gr.File(
                        label="下载内嵌字幕视频 (MP4)",
                        file_types=[".mp4"],
                    )

        # ---- 使用说明 ----
        with gr.Accordion("📖 使用说明", open=False):
            gr.Markdown(
                """
                ### 快速开始
                1. **上传视频**：点击或拖拽视频文件到上传区域（支持 mp4、mov、mkv、avi 等格式）
                2. **设置语言**：选择源语言（推荐自动检测）和目标语言
                3. **填写 API Key**：输入你的 Deepseek API 密钥
                4. **开始处理**：点击「开始处理」按钮，等待四个处理阶段完成
                5. **下载结果**：下载翻译后的 SRT 字幕文件或内嵌字幕的成品视频

                ### 环境要求
                - **Python 3.10+**：运行环境
                - **FFmpeg**：用于音频提取和字幕压制（必须安装并添加到系统 PATH）
                  - Windows: 从 [ffmpeg.org](https://ffmpeg.org/download.html) 下载，解压后添加 `bin` 目录到 PATH
                  - macOS: `brew install ffmpeg`
                  - Ubuntu/Debian: `sudo apt install ffmpeg`
                - **faster-whisper**：本地语音识别模型（首次运行自动下载，约 1.5GB）
                - **Deepseek API Key**：在 [platform.deepseek.com](https://platform.deepseek.com/) 注册获取

                ### 安装步骤
                ```bash
                # 1. 创建并激活虚拟环境
                python -m venv .venv
                source .venv/bin/activate      # Linux/macOS
                .venv\\Scripts\\activate       # Windows

                # 2. 安装依赖
                pip install gradio faster-whisper openai

                # 3. 启动工具
                python video_subtitle_translator.py
                ```

                ### 注意事项
                - 首次运行会自动下载 faster-whisper 模型（medium 约 1.5GB），请耐心等待
                - 较长的视频处理时间更长，请耐心等待进度更新
                - 翻译质量取决于 Deepseek API 和源语言识别准确度
                - 处理完成后临时文件会自动清理
                """
            )

        # ---- 绑定处理函数 ----
        def handle_process(video, src_lang, tgt_lang, api_key, api_url, progress=gr.Progress()):
            """包装处理函数，处理 Gradio 的进度参数"""
            # 获取语言代码
            src_code = src_lang
            tgt_code = tgt_lang

            srt_file, video_file, status = process_video(
                video_file=video,
                source_language=src_code,
                target_language=tgt_code,
                api_key=api_key,
                base_url=api_url,
                progress=progress,
            )
            return srt_file, video_file, status

        process_btn.click(
            fn=handle_process,
            inputs=[video_input, source_lang, target_lang, api_key_input, api_url_input],
            outputs=[srt_output, video_output, status_output],
        )

    return app


# ---- 主入口 ----

if __name__ == "__main__":
    # Windows GBK 终端兼容：确保 stdout 使用 UTF-8
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    # 检查 FFmpeg
    try:
        ffmpeg_path = get_ffmpeg()
        print(f"[OK] FFmpeg detected: {ffmpeg_path}")
    except FileNotFoundError as e:
        print(f"[WARN] {e}")
        print("Please install FFmpeg before using this tool.")

    # 检查 API Key
    if not DEEPSEEK_API_KEY:
        print("[INFO] DEEPSEEK_API_KEY not set in environment.")
        print("       You can enter your API Key in the web UI.")

    print(f"\nStarting video subtitle translator...")
    print(f"   Whisper model: {WHISPER_MODEL_SIZE}")
    print(f"   Output dir: {OUTPUT_DIR}")
    print(f"   URL: http://127.0.0.1:7860\n")

    app = create_ui()
    app.launch(
        theme=gr.themes.Soft(),
        server_name="127.0.0.1",
        server_port=7862,
        share=False,
        inbrowser=True,
    )
