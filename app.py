import io
import os
import re
from datetime import datetime
import pandas as pd
import streamlit as st
from fpdf import FPDF
from google import genai
from google.genai import types

# -----------------------------------------------------------------------------
# 1. Configuração da Página e CSS Moderno
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Google Ads AI Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    .hero-container {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.3);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        color: #FFFFFF !important;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        color: #E0E7FF;
        max-width: 750px;
        line-height: 1.5;
    }

    div.stButton > button:first-child {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: #FFFFFF;
        font-weight: 700;
        font-size: 1rem;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 14px 0 rgba(37, 99, 235, 0.39);
    }

    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Constantes e Inicialização de Estado
# -----------------------------------------------------------------------------
REQUIRED_FILES = {
    "campaign_settings": "Campaign Settings",
    "keywords": "Keywords",
    "search_terms": "Search Terms",
    "rsas": "RSAs",
}

OPTIONAL_FILES = {
    "assets_extensions": "Assets & Extensions",
    "locations": "Locations",
    "devices_schedule": "Devices & Schedule",
}

ALL_FILES = {**REQUIRED_FILES, **OPTIONAL_FILES}

if "last_report" not in st.session_state:
    st.session_state.last_report = None
if "negative_keywords" not in st.session_state:
    st.session_state.negative_keywords = ""
if "next_steps" not in st.session_state:
    st.session_state.next_steps = ""
if "upload_dataframes" not in st.session_state:
    st.session_state.upload_dataframes = {}

# -----------------------------------------------------------------------------
# 3. Funções Utilitárias e Processamento de Dados
# -----------------------------------------------------------------------------
def get_active_model(client: genai.Client) -> str:
    preferred_models = [
        "gemini-3.6-flash",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
    ]
    try:
        available_models = [m.name for m in client.models.list()]
        for pref in preferred_models:
            for avail in available_models:
                if pref in avail:
                    return avail
    except Exception:
        pass
    return "gemini-3.6-flash"


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    bytes_data = uploaded_file.getvalue()
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    
    for encoding in encodings:
        for skiprows in range(0, 6):
            try:
                df = pd.read_csv(io.BytesIO(bytes_data), encoding=encoding, skiprows=skiprows)
                if df.shape[1] > 1 and len(df) > 0:
                    df = df.dropna(how="all").dropna(axis=1, how="all")
                    return df
            except Exception:
                continue
    return pd.DataFrame()


def format_dataframe_for_prompt(df: pd.DataFrame, max_rows: int = 150) -> str:
    if df.empty:
        return ""
    sample_df = df.head(max_rows)
    return sample_df.to_markdown(index=False)


def show_upload_preview(label: str, file) -> None:
    with st.expander(f"Visualizar: {label}"):
        df = read_uploaded_csv(file)
        if not df.empty:
            st.dataframe(df.head(10), use_container_width=True)
            st.caption(f"Linhas totais detetadas: {len(df)}")
        else:
            st.warning("Não foi possível carregar pré-visualização.")

# -----------------------------------------------------------------------------
# 4. Geração de PDF e Prompts
# -----------------------------------------------------------------------------
class CustomPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            self.set_fill_color(30, 58, 138)
            self.rect(0, 0, 210, 25, "F")
            self.set_text_color(255, 255, 255)
            self.set_font("Helvetica", "B", 14)
            self.set_y(8)
            self.cell(0, 10, "Auditoria Google Ads - Relatorio Estrategico", align="C", new_x="LMARGIN", new_y="NEXT")
            self.set_y(32)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Pagina {self.page_no()}", align="C")


def generate_report_pdf_fpdf(markdown_text: str) -> bytes:
    pdf = CustomPDF()
    pdf.set_margins(left=15, top=30, right=15)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    
    clean_text = markdown_text.encode("latin-1", "replace").decode("latin-1")
    pdf.set_text_color(33, 33, 33)
    
    for line in clean_text.split("\n"):
        line = line.strip()
        pdf.set_x(pdf.l_margin)
        if line.startswith("# "):
            pdf.set_font("Helvetica", "B", 14)
            pdf.multi_cell(pdf.epw, 8, line[2:].strip(), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
        elif line.startswith("## "):
            pdf.set_font("Helvetica", "B", 12)
            pdf.multi_cell(pdf.epw, 7, line[3:].strip(), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
        elif line.startswith("### "):
            pdf.set_font("Helvetica", "B", 10)
            pdf.multi_cell(pdf.epw, 6, line[4:].strip(), new_x="LMARGIN", new_y="NEXT")
        elif line.startswith("- ") or line.startswith("* "):
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(pdf.epw, 5, f"  * {line[2:].strip()}", new_x="LMARGIN", new_y="NEXT")
        elif line:
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(pdf.epw, 5, line, new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.ln(2)
            
    return bytes(pdf.output())


def show_report_download_buttons(report: str) -> None:
    col1, col2 = st.columns(2)
    with col1:
        try:
            pdf_bytes = generate_report_pdf_fpdf(report)
            st.download_button(
                "Descarregar Relatório (PDF)",
                data=pdf_bytes,
                file_name="auditoria_google_ads.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Erro ao gerar PDF: {e}")
    with col2:
        st.download_button(
            "Descarregar Relatório (.md)",
            data=report,
            file_name="auditoria_google_ads.md",
            mime="text/markdown",
            use_container_width=True
        )

# -----------------------------------------------------------------------------
# 5. Prompts e Integração com a IA
# -----------------------------------------------------------------------------
def build_system_instruction(csv_data: dict) -> str:
    return (
        "És um especialista sénior e estratega de auditoria de campanhas Google Ads. "
        "A tua missão é fornecer um diagnóstico aprofundado, cirúrgico e altamente acionável. "
        "Não faças rodeios genéricos. Foca-te no desperdício de orçamento, oportunidades de escala, "
        "discrepâncias de CPA, CTR, Quality Score e canibalização de termos."
    )


def build_analysis_prompt(csv_data: dict) -> str:
    prompt = (
        "Por favor analisa detalhadamente os dados exportados do Google Ads abaixo e gera um relatório estruturado:\n\n"
    )
    for key, text in csv_data.items():
        prompt += f"### Relatório: {ALL_FILES.get(key, key)}\n```markdown\n{text}\n```\n\n"

    prompt += (
        "Estrutura a resposta rigorosamente com as seguintes secções:\n"
        "## 1. Resumo Executivo e Pontuação Geral da Conta\n"
        "## 2. Diagnóstico de Desperdício e Oportunidades (Keywords & Search Terms)\n"
        "## 3. Análise de Anúncios (RSAs, Headlines, CTR)\n"
        "## 4. Eficiência de Configurações, Redes, Dispositivos e Locais\n"
        "## 5. Lista de Palavras-Chave Negativas Recomendadas\n"
        "```negative_keywords\n"
        "(lista de palavras separadas por linha para exclusão imediata)\n"
        "```\n"
        "## 6. Ações Imediatas e Próximos Passos\n"
        "### Próximas 24 horas\n"
        "### Próximas 48 horas\n"
        "### Próximos 15 dias\n"
    )
    return prompt


def analyze_with_gemini(api_key: str, prompt: str, system_instruction: str) -> str:
    client = genai.Client(api_key=api_key)
    model = get_active_model(client)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.3,
            max_output_tokens=8192
        )
    )
    return response.text.strip()


def extract_negative_keywords(report: str) -> str:
    code_match = re.search(r"```negative_keywords?\s*\n(.*?)```", report, re.DOTALL | re.IGNORECASE)
    if code_match:
        return code_match.group(1).strip()
    return ""


def fetch_negative_keywords_from_gemini(api_key: str, report: str) -> str:
    client = genai.Client(api_key=api_key)
    model = get_active_model(client)
    prompt = (
        "Do relatório de auditoria abaixo, extrai apenas a lista limpa de palavras-chave negativas recomendadas, "
        "uma por linha sem números nem travessões:\n\n" f"{report[:12000]}"
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=2048)
    )
    return response.text.strip()


def resolve_negative_keywords(api_key: str, report: str) -> str:
    negatives = extract_negative_keywords(report)
    if negatives.strip():
        return negatives
    try:
        return fetch_negative_keywords_from_gemini(api_key, report)
    except Exception:
        return ""


def extract_next_steps_section(report: str) -> str:
    section_match = re.search(
        r"##\s*(?:6\.\s*)?(?:Ações Imediatas e Próximos Passos|Ações Imediatas|Próximos Passos)[^\n]*\n(.*?)(?=\n##\s|\Z)",
        report,
        re.DOTALL | re.IGNORECASE,
    )
    if section_match:
        return section_match.group(1).strip()
    return ""


def parse_action_windows(section_text: str) -> dict[str, str]:
    def extract_window(pattern: str) -> str:
        match = re.search(pattern, section_text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    return {
        "24h": extract_window(r"###\s*Próximas?\s*24\s*horas?[^\n]*\n(.*?)(?=\n###|\Z)"),
        "48h": extract_window(r"###\s*Próximas?\s*48\s*horas?[^\n]*\n(.*?)(?=\n###|\Z)"),
        "15d": extract_window(r"###\s*Próximos?\s*15\s*dias?[^\n]*\n(.*?)(?=\n###|\Z)"),
    }


def fetch_next_steps_from_gemini(api_key: str, report: str) -> str:
    client = genai.Client(api_key=api_key)
    model = get_active_model(client)
    prompt = (
        "Do relatório de auditoria Google Ads abaixo, extrai APENAS a secção de "
        "Ações Imediatas e Próximos Passos estruturada exatamente com:\n"
        "### Próximas 24 horas\n"
        "### Próximas 48 horas\n"
        "### Próximos 15 dias\n\n"
        f"{report[:12000]}"
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=2048)
    )
    return response.text.strip()


def resolve_next_steps(api_key: str, report: str) -> str:
    next_steps = extract_next_steps_section(report)
    if next_steps.strip():
        return next_steps
    try:
        return fetch_next_steps_from_gemini(api_key, report)
    except Exception:
        return ""


def show_next_steps_tab(next_steps_text: str) -> None:
    st.markdown("Plano de ação prioritário para execução imediata, organizado por janela de tempo.")

    if not next_steps_text.strip():
        st.warning(
            "Não foi possível extrair o plano de ação. "
            "Execute a análise novamente ou consulte a aba de Auditoria AI completa."
        )
        return

    windows = parse_action_windows(next_steps_text)
    has_structured_windows = any(windows.values())

    if windows["24h"]:
        st.success("#### ⚡ Próximas 24 horas — Ações urgentes")
        st.markdown(windows["24h"])

    if windows["48h"]:
        st.info("#### 🕐 Próximas 48 horas — Curto prazo")
        st.markdown(windows["48h"])

    if windows["15d"]:
        st.markdown("#### 📅 Próximos 15 dias — Implementação")
        st.markdown(windows["15d"])

    if not has_structured_windows:
        st.markdown(next_steps_text)


def show_results_section() -> None:
    report = st.session_state.last_report
    negatives = st.session_state.get("negative_keywords", "")
    next_steps = st.session_state.get("next_steps", "")

    tab_audit, tab_negatives, tab_next_steps = st.tabs(
        [
            "🤖 Auditoria AI completa",
            "🛑 Palavras Negativas",
            "🚀 Próximos Passos",
        ]
    )

    with tab_audit:
        st.markdown(report)
        show_report_download_buttons(report)

    with tab_negatives:
        st.markdown(
            "Lista de palavras-chave negativas sugeridas pela IA, prontas para copiar "
            "e importar no Google Ads."
        )
        if negatives.strip():
            st.code(negatives, language=None)
        else:
            st.warning(
                "Não foi possível extrair palavras negativas do relatório. "
                "Execute a análise novamente ou consulte a aba de Auditoria AI completa."
            )

    with tab_next_steps:
        show_next_steps_tab(next_steps)

# -----------------------------------------------------------------------------
# 6. Interface Principal (Sidebar Segura + Uploads)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuração")
    
    # 1. Obtém a chave dos Secrets (se configurada na Cloud ou local)
    secret_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    
    # 2. Input opcional mantido sempre em branco para proteção da chave secreta
    user_api_key = st.text_input(
        "Gemini API Key (Opcional)",
        value="",
        type="password",
        placeholder="Introduza uma chave própria (opcional)",
        help="Deixe em branco para usar a chave padrão pré-configurada nos Secrets."
    )
    
    if st.button("Guardar API Key", use_container_width=True):
        if user_api_key.strip():
            st.session_state["custom_api_key"] = user_api_key.strip()
            st.success("Chave personalizada guardada!")
        else:
            st.session_state.pop("custom_api_key", None)
            st.info("A usar a chave padrão dos Secrets.")

    # 3. Define a chave ativa com prioridade para chave customizada do visitante
    effective_api_key = st.session_state.get("custom_api_key", "").strip() or secret_key
    st.session_state.gemini_api_key = effective_api_key

    if effective_api_key:
        st.caption("✅ API Key pronta a usar")
    else:
        st.caption("⚠️ Nenhuma API Key configurada")

st.markdown("""
<div class="hero-container">
    <div class="hero-title">📊 Google Ads AI Optimizer</div>
    <div class="hero-subtitle">
        Faça upload dos ficheiros exportados da sua conta para obter diagnósticos estratégicos instantâneos, planos de ação táticos e exclusão de termos irrelevantes com Gemini.
    </div>
</div>
""", unsafe_allow_html=True)

st.subheader("Upload de Ficheiros CSV")

st.markdown("#### Ficheiros Principais (Obrigatórios)")
req_col1, req_col2 = st.columns(2)

with req_col1:
    campaign_file = st.file_uploader(
        "Campaign Settings",
        type=["csv"],
        help="No Google Ads: Vá a Campanhas > Campanhas e descarregue em .csv.",
        key="upload_campaign_settings",
    )
    keywords_file = st.file_uploader(
        "Keywords",
        type=["csv"],
        help="No Google Ads: Vá a Palavras-chave de pesquisa e descarregue em .csv.",
        key="upload_keywords",
    )

with req_col2:
    search_terms_file = st.file_uploader(
        "Search Terms",
        type=["csv"],
        help="No Google Ads: Vá a Termos de pesquisa e descarregue em .csv.",
        key="upload_search_terms",
    )
    rsas_file = st.file_uploader(
        "RSAs",
        type=["csv"],
        help="No Google Ads: Vá a Anúncios e recursos > Anúncios e descarregue em .csv.",
        key="upload_rsas",
    )

st.markdown("#### Ficheiros Complementares (Opcionais)")
opt_col1, opt_col2 = st.columns(2)

with opt_col1:
    assets_file = st.file_uploader(
        "Assets & Extensions",
        type=["csv"],
        help="No Google Ads: Vá a Recursos (Assets) e descarregue em .csv.",
        key="upload_assets",
    )
    locations_file = st.file_uploader(
        "Locations",
        type=["csv"],
        help="No Google Ads: Vá a Localizações e descarregue em .csv.",
        key="upload_locations",
    )

with opt_col2:
    devices_file = st.file_uploader(
        "Devices & Schedule",
        type=["csv"],
        help="No Google Ads: Vá a Dispositivos / Horário dos anúncios e descarregue em .csv.",
        key="upload_devices",
    )

required_uploads = {
    "campaign_settings": campaign_file,
    "keywords": keywords_file,
    "search_terms": search_terms_file,
    "rsas": rsas_file,
}

optional_uploads = {
    "assets_extensions": assets_file,
    "locations": locations_file,
    "devices_schedule": devices_file,
}

all_uploads = {**required_uploads, **optional_uploads}

uploaded_files = [
    (ALL_FILES[key], file) for key, file in all_uploads.items() if file is not None
]
if uploaded_files:
    st.markdown("#### Pré-visualização dos dados")
    for label, file in uploaded_files:
        show_upload_preview(label, file)

analyze_clicked = st.button("Analisar Campanhas", type="primary", use_container_width=True)

if analyze_clicked:
    if not st.session_state.gemini_api_key:
        st.error("Configure a Gemini API Key na barra lateral antes de analisar.")
        st.stop()

    missing_required = [
        REQUIRED_FILES[key] for key, file in required_uploads.items() if file is None
    ]
    if missing_required:
        st.error(
            "Faça upload de todos os ficheiros obrigatórios: "
            + ", ".join(missing_required)
        )
        st.stop()

    csv_data = {}
    upload_dataframes = {}
    for key, file in all_uploads.items():
        if file is not None:
            try:
                df = read_uploaded_csv(file)
                if df.empty:
                    st.error(
                        f"{ALL_FILES[key]}: sem dados válidos após processamento do CSV."
                    )
                    st.stop()

                prompt_text = format_dataframe_for_prompt(df)
                if not prompt_text.strip():
                    st.error(
                        f"{ALL_FILES[key]}: não foi possível preparar dados para análise."
                    )
                    st.stop()

                csv_data[key] = prompt_text
                upload_dataframes[key] = df
            except Exception as e:
                st.error(f"Erro ao ler {ALL_FILES[key]}: {e}")
                st.stop()

    system_instruction = build_system_instruction(csv_data)

    with st.spinner("Analisando campanhas com Gemini... Isto pode levar alguns segundos."):
        try:
            prompt = build_analysis_prompt(csv_data)
            report = analyze_with_gemini(
                st.session_state.gemini_api_key, prompt, system_instruction
            )
            st.session_state.last_report = report
            st.session_state.upload_dataframes = upload_dataframes
            st.session_state.negative_keywords = resolve_negative_keywords(
                st.session_state.gemini_api_key, report
            )
            st.session_state.next_steps = resolve_next_steps(
                st.session_state.gemini_api_key, report
            )
        except Exception as e:
            st.error(f"Erro ao contactar a API Gemini: {e}")
            st.stop()

if st.session_state.last_report:
    st.divider()
    st.subheader("Resultados da Análise")
    show_results_section()