from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from intonation_labeler.demo import build_demo_result
from intonation_labeler.export import AnnotationResult, build_results_zip
from intonation_labeler.gemini_client import annotate_audio, create_client
from intonation_labeler.io import FeatureFormatError, load_feature_json, pair_uploaded_files
from intonation_labeler.prompts import build_prompt


PRESET_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "── 自定义 ──",
]


st.set_page_config(
    page_title="音频语调标注工具",
    page_icon="🎙️",
    layout="wide",
)

st.markdown("""
<style>
    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .subtitle {
        color: #888;
        font-size: 1rem;
        margin-top: 0.2rem;
        margin-bottom: 1.5rem;
    }
    section[data-testid="stSidebar"] { background: #181825; }
</style>
""", unsafe_allow_html=True)

load_dotenv(PROJECT_ROOT / ".env")

with st.sidebar:
    st.markdown("## ⚙️ 配置")
    st.divider()

    run_mode = st.radio("Mode", ["Demo / Dry Run", "Live Gemini"], index=0)
    is_demo_mode = run_mode == "Demo / Dry Run"

    env_key_present = bool(os.getenv("GOOGLE_API_KEY", "").strip())
    manual_key = st.text_input(
        "Google API Key",
        value="",
        type="password",
        placeholder="留空则使用环境变量 GOOGLE_API_KEY",
        help="从 Google AI Studio 获取。页面不会回显环境变量中的 key。",
    )
    api_key = manual_key.strip() or os.getenv("GOOGLE_API_KEY", "").strip()
    if env_key_present and not manual_key:
        st.caption("已检测到环境变量 GOOGLE_API_KEY。")
    if is_demo_mode:
        st.caption("Demo mode will not call Gemini or upload audio.")

    st.markdown("**Gemini 模型**")
    model_choice = st.selectbox("预设模型", PRESET_MODELS, index=0, label_visibility="collapsed")
    if model_choice == "── 自定义 ──":
        custom_model = st.text_input("自定义模型名称", placeholder="例如：gemini-2.5-flash")
        model_id = custom_model.strip() or "gemini-2.5-flash"
    else:
        model_id = model_choice
    st.caption(f"当前使用：`{model_id}`")

    st.divider()
    st.markdown("**使用说明**")
    st.markdown("""
1. 选择 Demo 或 Live 模式
2. 上传同名 `.wav` 与 `.json` 文件（支持批量）
3. 确认配对数量后点击 **开始标注**
4. 完成后下载 ZIP 压缩包
""")
    st.divider()
    st.caption("Powered by Google Gemini · Built with Streamlit")

st.markdown('<p class="main-title">🎙️ 音频语调标注工具</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">基于 Google Gemini 的多模态语调自动标注系统</p>', unsafe_allow_html=True)
st.divider()

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 🔊 音频文件（WAV）")
    wav_files = st.file_uploader("支持批量上传", type=["wav"], accept_multiple_files=True, key="wav_uploader", label_visibility="collapsed")
    if wav_files:
        st.success(f"已选 {len(wav_files)} 个音频文件")

with col2:
    st.markdown("#### 📄 声学参数（JSON）")
    json_files = st.file_uploader("支持批量上传", type=["json"], accept_multiple_files=True, key="json_uploader", label_visibility="collapsed")
    if json_files:
        st.success(f"已选 {len(json_files)} 个 JSON 文件")

st.divider()

pairs = []
if wav_files and json_files:
    pairs, missing_wav, missing_json = pair_uploaded_files(wav_files, json_files)
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("✅ 已配对", len(pairs))
    mc2.metric("⚠️ 缺少 WAV", len(missing_wav))
    mc3.metric("⚠️ 缺少 JSON", len(missing_json))

    if missing_wav:
        with st.expander(f"缺少对应 WAV 的 JSON 文件（{len(missing_wav)} 个）"):
            st.write(", ".join(missing_wav))
    if missing_json:
        with st.expander(f"缺少对应 JSON 的 WAV 文件（{len(missing_json)} 个）"):
            st.write(", ".join(missing_json))

    if pairs:
        st.info(f"共 **{len(pairs)}** 对文件匹配成功，未配对文件将自动跳过。")
    else:
        st.error("未找到任何匹配的文件对，请确认 WAV 与 JSON 主文件名完全一致（如 `sample_001.wav` ↔ `sample_001.json`）。")
elif wav_files or json_files:
    st.warning("请同时上传 WAV 和 JSON 文件以进行配对。")

if not is_demo_mode and not api_key:
    st.warning("请在左侧侧边栏填入 Google API Key，或设置环境变量 GOOGLE_API_KEY。")
elif is_demo_mode:
    st.info("Demo / Dry Run 模式不会调用 Gemini，也不会上传音频；输出为 mock annotation。")

can_run = (is_demo_mode or bool(api_key)) and bool(pairs)
run_btn = st.button("🚀 开始标注", disabled=not can_run, use_container_width=True, type="primary")

if run_btn and can_run:
    client = None
    if not is_demo_mode:
        try:
            client = create_client(api_key)
        except Exception as exc:
            st.error(f"Gemini 初始化失败：{exc}")
            st.stop()

    results: list[AnnotationResult] = []
    total = len(pairs)
    st.divider()
    st.markdown("### 📊 标注进度")
    progress_bar = st.progress(0, text=f"0 / {total}")
    status_text = st.empty()
    table_holder = st.empty()
    table_rows = []

    for idx, (sample_id, wav_file, json_file) in enumerate(pairs):
        status_text.markdown(f"⏳ 正在处理 **{sample_id}**　　`{idx + 1} / {total}`")
        progress_bar.progress(idx / total, text=f"{idx + 1} / {total}")

        tmp_path = None
        try:
            feature_data = load_feature_json(json_file, sample_id)
            if is_demo_mode:
                result_text = build_demo_result(feature_data)
            else:
                prompt = build_prompt(feature_data.transcript, feature_data.anchors)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    wav_file.seek(0)
                    tmp.write(wav_file.read())
                    tmp_path = tmp.name
                result_text = annotate_audio(client, model_id, tmp_path, prompt)
            results.append(AnnotationResult(sample_id=sample_id, result=result_text))
            preview = result_text[:80] + "…" if len(result_text) > 80 else result_text
            table_rows.append([sample_id, "✅ 成功", preview])
        except (FeatureFormatError, Exception) as exc:
            results.append(AnnotationResult(sample_id=sample_id, error=str(exc)))
            table_rows.append([sample_id, "❌ 失败", str(exc)[:80]])
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

        table_holder.dataframe(table_rows, use_container_width=True, hide_index=True)

    progress_bar.progress(1.0, text=f"完成！{total} / {total}")
    n_ok = sum(1 for item in results if item.result is not None)
    status_text.markdown(f"**全部完成** — ✅ 成功 **{n_ok}** 条　❌ 失败 **{total - n_ok}** 条")

    st.divider()
    st.markdown("### 💾 下载结果")
    st.download_button(
        label="📦 下载全部标注结果 (.zip)",
        data=build_results_zip(results),
        file_name="annotation_results.zip",
        mime="application/zip",
        use_container_width=True,
        type="primary",
    )
    st.caption("ZIP 包含：`m_results.txt`、`m_results.json`、`individual/<id>.txt`。")
