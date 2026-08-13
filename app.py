import io
import re
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
from pandas.errors import ParserError
from google import genai
from google.genai import types

st.set_page_config(
    page_title="Google Ads AI Optimizer",
    page_icon="📊",
    layout="wide",
)

def get_default_api_key() -> str:
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    return ""


if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = get_default_api_key()

BASE_SYSTEM_INSTRUCTION = (
    "Tu és um especialista sénior em PPC e Google Ads com mais de 15 anos de experiência. "
    "Analisa os dados de campanhas fornecidos e produz um relatório de diagnóstico detalhado "
    "com recomendações acionáveis. Estrutura o relatório em Markdown com as seguintes secções:\n\n"
    "## Resumo Executivo\n"
    "## Diagnóstico por Área\n"
    "### Configurações de Campanha\n"
    "### Keywords\n"
    "### Search Terms\n"
    "### RSAs (Responsive Search Ads)\n"
    "## Oportunidades de Otimização\n"
    "## Recomendações Prioritárias\n"
    "## Ações Imediatas e Próximos Passos\n"
    "### Próximas 24 horas\n"
    "### Próximas 48 horas\n"
    "### Próximos 15 dias\n"
    "## Palavras-chave Negativas Sugeridas\n\n"
    "Usa dados concretos dos ficheiros para fundamentar cada ponto. "
    "Prioriza recomendações por impacto esperado (alto, médio, baixo). "
    "Na secção **Ações Imediatas e Próximos Passos**, cria um plano de ação destacado "
    "com as três subsecções obrigatórias (Próximas 24 horas, Próximas 48 horas, Próximos 15 dias), "
    "listando 3 a 5 ações concretas e acionáveis em cada período. "
    "Na secção de palavras-chave negativas, lista as recomendações num bloco de código "
    "com o identificador `negatives`, uma palavra por linha, prontas para importar no Google Ads:\n"
    "```negatives\ntermo1\ntermo2\n```\n"
    "Responde sempre em português."
)

GEO_DEVICE_SECTION = (
    "\n\nInclui também a secção:\n"
    "## 4. ANÁLISE GEOGRÁFICA E DISPOSITIVOS\n"
    "Identifica desperdícios por cidades/regiões ou tipos de dispositivo (Mobile vs Desktop) "
    "e sugere ajustes de licitação (Bid Adjustments) concretos, com percentagens quando possível."
)

ASSETS_SECTION = (
    "\n\nInclui também a secção:\n"
    "## 5. EXTENSÕES DE ANÚNCIO (ASSETS)\n"
    "Avalia as extensões atuais e gera textos concretos para 4 novos Sitelinks e 4 Callouts "
    "alinhados com os termos de maior conversão identificados nos dados."
)

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

CSV_DELIMITERS = (",", ";", "\t")
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "latin1")

FALLBACK_MODELS = [
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-pro",
]

PROMPT_MAX_ROWS = 200
PROMPT_MIN_ROWS = 100

REPORT_CSS = """
@page {
    size: A4;
    margin: 2cm 2.2cm;
}
body {
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.65;
    color: #1f2937;
    background: #ffffff;
}
.report-header {
    margin-bottom: 2rem;
    padding-bottom: 1rem;
    border-bottom: 3px solid #1e40af;
}
.report-header h1 {
    margin: 0 0 0.35rem;
    font-size: 26pt;
    color: #1e3a8a;
    letter-spacing: -0.02em;
}
.report-header .subtitle {
    margin: 0;
    color: #4b5563;
    font-size: 12pt;
}
.report-header .meta {
    margin: 0.75rem 0 0;
    color: #6b7280;
    font-size: 10pt;
}
main h1, main h2, main h3, main h4 {
    color: #1e3a8a;
    line-height: 1.3;
    margin-top: 1.6rem;
    margin-bottom: 0.6rem;
}
main h2 {
    font-size: 18pt;
    border-bottom: 2px solid #e5e7eb;
    padding-bottom: 0.35rem;
}
main h3 {
    font-size: 14pt;
    color: #2563eb;
}
main p, main li {
    margin-bottom: 0.65rem;
}
main ul, main ol {
    padding-left: 1.4rem;
}
main table {
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
    font-size: 10pt;
}
main th, main td {
    border: 1px solid #d1d5db;
    padding: 0.45rem 0.6rem;
    text-align: left;
}
main th {
    background: #f3f4f6;
    color: #111827;
    font-weight: 600;
}
main code, main pre {
    font-family: "Consolas", "Monaco", monospace;
    font-size: 9.5pt;
}
main pre {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 0.85rem;
    overflow-x: auto;
}
main blockquote {
    border-left: 4px solid #93c5fd;
    margin: 1rem 0;
    padding: 0.5rem 1rem;
    background: #f8fafc;
    color: #374151;
}
main strong {
    color: #111827;
}
"""


def _decode_csv_content(content: bytes) -> str | None:
    if not content or not content.strip():
        return None
    for encoding in CSV_ENCODINGS:
        try:
            decoded = content.decode(encoding)
            if decoded.strip():
                return decoded
        except (UnicodeDecodeError, UnicodeError):
            continue
    return None


def _line_column_count(line: str) -> int:
    return max(line.count(sep) + 1 for sep in CSV_DELIMITERS)


def _read_with_delimiter(text: str, skiprows: int, sep: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(
            io.StringIO(text),
            skiprows=range(skiprows),
            sep=sep,
            on_bad_lines="skip",
            engine="python",
        )
        if df.empty or len(df.columns) <= 1:
            return None
        return df
    except (ParserError, ValueError):
        return None


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    cleaned = cleaned.dropna(how="all")
    cleaned = cleaned.replace(r"^\s*$", pd.NA, regex=True)
    cleaned = cleaned.dropna(how="all")
    return cleaned


def _dataframe_to_text(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_csv(index=False)


def format_dataframe_for_prompt(df: pd.DataFrame) -> str:
    """Serialize all columns and up to PROMPT_MAX_ROWS rows for the Gemini prompt."""
    cleaned = _clean_dataframe(df)
    if cleaned.empty:
        return ""

    total_rows = len(cleaned)
    preview_rows = min(total_rows, PROMPT_MAX_ROWS)
    if total_rows >= PROMPT_MIN_ROWS:
        preview_rows = min(total_rows, max(PROMPT_MIN_ROWS, preview_rows))

    preview = cleaned.head(preview_rows)
    columns = ", ".join(cleaned.columns.astype(str).tolist())

    parts = [
        f"Total de linhas: {total_rows}",
        f"Total de colunas: {len(cleaned.columns)}",
        f"Colunas: {columns}",
        "",
        _dataframe_to_text(preview),
    ]
    if total_rows > preview_rows:
        parts.append(
            f"\n[... {total_rows - preview_rows} linhas adicionais omitidas para limitar contexto ...]"
        )
    return "\n".join(parts)


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    content = uploaded_file.getvalue()
    return read_google_ads_csv(content)


def read_google_ads_csv(content: bytes) -> pd.DataFrame:
    """Read a Google Ads CSV export, skipping metadata rows before the real header."""
    text = _decode_csv_content(content)
    if text is None:
        return pd.DataFrame()

    lines = text.splitlines()
    if not lines:
        return pd.DataFrame()

    header_candidates = [
        i
        for i, line in enumerate(lines)
        if line.strip() and _line_column_count(line) > 1
    ]
    if not header_candidates:
        return pd.DataFrame()

    for header_idx in header_candidates:
        for sep in CSV_DELIMITERS:
            df = _read_with_delimiter(text, header_idx, sep)
            if df is None:
                continue
            cleaned = _clean_dataframe(df)
            if not cleaned.empty and len(cleaned.columns) > 1:
                return cleaned

    return pd.DataFrame()


def show_upload_preview(label: str, uploaded_file) -> None:
    with st.expander(f"Pré-visualização: {label}", expanded=False):
        try:
            df = read_uploaded_csv(uploaded_file)
        except Exception as e:
            st.error(f"Erro ao ler o ficheiro: {e}")
            return

        if df.empty:
            st.warning("Nenhum dado lido deste ficheiro.")
            return

        st.caption(f"{len(df)} linhas × {len(df.columns)} colunas")
        st.dataframe(df.head(), use_container_width=True)


def build_system_instruction(csv_data: dict[str, str]) -> str:
    instruction = BASE_SYSTEM_INSTRUCTION

    has_geo_device = "locations" in csv_data or "devices_schedule" in csv_data
    if has_geo_device:
        instruction += GEO_DEVICE_SECTION

    if "assets_extensions" in csv_data:
        instruction += ASSETS_SECTION

    return instruction


def build_analysis_prompt(csv_data: dict[str, str]) -> str:
    parts = [
        "Analisa os seguintes dados exportados de Google Ads e produz o relatório de diagnóstico completo.\n"
    ]
    for key, label in ALL_FILES.items():
        if key in csv_data:
            parts.append(f"\n--- {label} ---\n{csv_data[key]}")
    return "\n".join(parts)


def _normalize_model_name(name: str) -> str:
    if name.startswith("models/"):
        return name.split("/", 1)[1]
    if "/" in name:
        return name.rsplit("/", 1)[-1]
    return name


def _model_supports_generate_content(model) -> bool:
    for attr in ("supported_actions", "supported_generation_methods"):
        methods = getattr(model, attr, None)
        if methods:
            return any("generatecontent" in str(m).lower().replace("_", "") for m in methods)

    name = _normalize_model_name(getattr(model, "name", str(model))).lower()
    return "flash" in name or "pro" in name


def _sort_models_by_preference(names: list[str]) -> list[str]:
    def sort_key(name: str) -> tuple:
        lower = name.lower()
        flash_rank = 0 if "flash" in lower else 1
        pro_rank = 0 if "pro" in lower else 1
        return (flash_rank, pro_rank, lower)

    return sorted(names, key=sort_key)


def _try_models_in_order(client: genai.Client, model_names: list[str]) -> str | None:
    test_config = types.GenerateContentConfig(max_output_tokens=1)
    for model_name in model_names:
        try:
            client.models.generate_content(
                model=model_name,
                contents="ping",
                config=test_config,
            )
            return model_name
        except Exception:
            continue
    return None


def _try_fallback_models(client: genai.Client) -> str:
    model = _try_models_in_order(client, FALLBACK_MODELS)
    if model:
        return model
    raise RuntimeError(
        "Nenhum modelo Gemini disponível. Verifique a API Key e os modelos suportados."
    )


def get_active_model(client: genai.Client) -> str:
    candidates: list[str] = []
    try:
        for model in client.models.list():
            if not _model_supports_generate_content(model):
                continue
            name = _normalize_model_name(getattr(model, "name", ""))
            lower = name.lower()
            if "flash" in lower or "pro" in lower:
                candidates.append(name)
        candidates = _sort_models_by_preference(candidates)
    except Exception:
        candidates = []

    if candidates:
        verified = _try_models_in_order(client, candidates)
        if verified:
            return verified

    return _try_fallback_models(client)


def analyze_with_gemini(api_key: str, prompt: str, system_instruction: str) -> str:
    client = genai.Client(api_key=api_key)
    model = get_active_model(client)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.4,
            max_output_tokens=8192,
        ),
    )
    return response.text


def _markdown_to_html_body(markdown_text: str) -> str:
    try:
        import markdown2

        return markdown2.markdown(
            markdown_text,
            extras=["fenced-code-blocks", "tables", "header-ids", "strike"],
        )
    except ImportError:
        try:
            import markdown

            return markdown.markdown(markdown_text, extensions=["extra", "tables"])
        except ImportError:
            escaped = (
                markdown_text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            return f"<pre>{escaped}</pre>"


def build_report_html(markdown_text: str) -> str:
    body_html = _markdown_to_html_body(markdown_text)
    generated_at = datetime.now().strftime("%d/%m/%Y %H:%M")
    return f"""<!DOCTYPE html>
<html lang="pt">
<head>
    <meta charset="utf-8">
    <title>Auditoria Google Ads</title>
    <style>{REPORT_CSS}</style>
</head>
<body>
    <header class="report-header">
        <h1>Auditoria Google Ads</h1>
        <p class="subtitle">Relatório de diagnóstico e recomendações</p>
        <p class="meta">Gerado em {generated_at}</p>
    </header>
    <main>{body_html}</main>
</body>
</html>"""


def generate_report_pdf_weasyprint(markdown_text: str) -> bytes | None:
    html = build_report_html(markdown_text)
    try:
        from weasyprint import HTML

        return HTML(string=html).write_pdf()
    except Exception:
        return None


def _fpdf_font_dir() -> Path:
    import fpdf

    base = Path(fpdf.__file__).parent
    for folder in ("font", "fonts"):
        candidate = base / folder
        if candidate.exists():
            return candidate
    return base / "font"


def _fpdf_sanitize_text(text: str) -> str:
    return text.replace("**", "").replace("*", "").replace("`", "").strip()


def _fpdf_safe_text(text: str, use_unicode: bool) -> str:
    cleaned = _fpdf_sanitize_text(text)
    if use_unicode:
        return cleaned
    return cleaned.encode("latin-1", errors="replace").decode("latin-1")


def _fpdf_reset_x(pdf) -> None:
    pdf.set_x(pdf.l_margin)


def _fpdf_multiline(
    pdf,
    text: str,
    font_family: str,
    style: str,
    size: int,
    color: tuple[int, int, int],
    line_height: float,
    use_unicode: bool,
) -> None:
    _fpdf_reset_x(pdf)
    pdf.set_font(font_family, style, size)
    pdf.set_text_color(*color)
    safe_text = _fpdf_safe_text(text, use_unicode)

    try:
        pdf.multi_cell(
            pdf.epw,
            line_height,
            safe_text,
            new_x="LMARGIN",
            new_y="NEXT",
        )
    except Exception:
        fallback = safe_text.encode("latin-1", errors="replace").decode("latin-1")
        _fpdf_reset_x(pdf)
        pdf.multi_cell(
            pdf.epw,
            line_height,
            fallback,
            new_x="LMARGIN",
            new_y="NEXT",
        )


def _fpdf_draw_line(pdf, color: tuple[int, int, int] = (229, 231, 235), width: float = 0.4) -> None:
    _fpdf_reset_x(pdf)
    y = pdf.get_y()
    pdf.set_draw_color(*color)
    pdf.set_line_width(width)
    pdf.line(pdf.l_margin, y, pdf.l_margin + pdf.epw, y)
    pdf.set_y(y + 5)


def _write_markdown_to_fpdf(pdf, markdown_text: str, font_family: str, use_unicode: bool) -> None:
    for line in markdown_text.splitlines():
        stripped = line.strip()

        if not stripped:
            pdf.ln(4)
            continue

        if stripped.startswith("#### "):
            _fpdf_multiline(
                pdf, stripped[5:], font_family, "B", 11, (37, 99, 235), 6, use_unicode
            )
            pdf.ln(2)
        elif stripped.startswith("### "):
            _fpdf_multiline(
                pdf, stripped[4:], font_family, "B", 12, (37, 99, 235), 7, use_unicode
            )
            pdf.ln(2)
        elif stripped.startswith("## "):
            _fpdf_multiline(
                pdf, stripped[3:], font_family, "B", 14, (30, 58, 138), 8, use_unicode
            )
            pdf.ln(3)
        elif stripped.startswith("# "):
            _fpdf_multiline(
                pdf, stripped[2:], font_family, "B", 16, (30, 58, 138), 9, use_unicode
            )
            pdf.ln(4)
        elif stripped.startswith("- ") or stripped.startswith("* "):
            bullet = "- " if not use_unicode else "• "
            _fpdf_multiline(
                pdf,
                f"{bullet}{stripped[2:]}",
                font_family,
                "",
                10,
                (31, 41, 55),
                6,
                use_unicode,
            )
        elif stripped.startswith("---") or stripped.startswith("***"):
            pdf.ln(2)
            _fpdf_draw_line(pdf)
        else:
            _fpdf_multiline(
                pdf, stripped, font_family, "", 10, (31, 41, 55), 6, use_unicode
            )


def _build_minimal_pdf_fpdf(markdown_text: str) -> bytes:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    safe_text = markdown_text.encode("latin-1", errors="replace").decode("latin-1")
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(
        pdf.epw,
        5,
        safe_text,
        new_x="LMARGIN",
        new_y="NEXT",
    )
    return bytes(pdf.output())


def _build_report_pdf_fpdf(markdown_text: str) -> bytes:
    from fpdf import FPDF

    font_dir = _fpdf_font_dir()
    regular = font_dir / "DejaVuSans.ttf"
    bold = font_dir / "DejaVuSans-Bold.ttf"
    oblique = font_dir / "DejaVuSans-Oblique.ttf"
    use_unicode_fonts = regular.exists() and bold.exists()

    class AuditPDF(FPDF):
        def header(self) -> None:
            if self.page_no() == 1:
                self.set_fill_color(30, 64, 175)
                self.rect(0, 0, 210, 22, style="F")
                self.set_y(6)
                self.set_x(self.l_margin)
                if use_unicode_fonts:
                    self.set_font("DejaVu", "B", 15)
                else:
                    self.set_font("Helvetica", "B", 15)
                self.set_text_color(255, 255, 255)
                self.cell(
                    self.epw,
                    10,
                    "Auditoria Google Ads",
                    align="C",
                    new_x="LMARGIN",
                    new_y="NEXT",
                )
                self.set_y(30)
            else:
                self.set_y(self.t_margin)

            self.set_text_color(31, 41, 55)

        def footer(self) -> None:
            self.set_y(-14)
            if use_unicode_fonts:
                self.set_font("DejaVu", "", 8)
            else:
                self.set_font("Helvetica", "", 8)
            self.set_text_color(107, 114, 128)
            self.set_x(self.l_margin)
            self.cell(
                self.epw,
                8,
                f"Pagina {self.page_no()}",
                align="C",
                new_x="LMARGIN",
                new_y="NEXT",
            )

    pdf = AuditPDF()
    pdf.set_margins(left=15, top=35, right=15)
    pdf.set_auto_page_break(auto=True, margin=20)

    if use_unicode_fonts:
        pdf.add_font("DejaVu", "", str(regular))
        pdf.add_font("DejaVu", "B", str(bold))
        if oblique.exists():
            pdf.add_font("DejaVu", "I", str(oblique))
        body_font = "DejaVu"
    else:
        body_font = "Helvetica"

    pdf.add_page()
    generated_at = datetime.now().strftime("%d/%m/%Y %H:%M")

    if use_unicode_fonts:
        subtitle = "Relatório de diagnóstico e recomendações"
    else:
        subtitle = "Relatorio de diagnostico e recomendacoes"

    _fpdf_multiline(pdf, subtitle, body_font, "", 11, (75, 85, 99), 6, use_unicode_fonts)
    _fpdf_multiline(
        pdf, f"Gerado em {generated_at}", body_font, "", 11, (75, 85, 99), 6, use_unicode_fonts
    )
    pdf.ln(4)
    _fpdf_draw_line(pdf, color=(30, 64, 175), width=0.7)
    pdf.ln(3)

    _write_markdown_to_fpdf(pdf, markdown_text, body_font, use_unicode_fonts)
    return bytes(pdf.output())


def generate_report_pdf_fpdf(markdown_text: str) -> bytes:
    try:
        return _build_report_pdf_fpdf(markdown_text)
    except Exception:
        return _build_minimal_pdf_fpdf(markdown_text)


def generate_report_pdf(markdown_text: str) -> bytes:
    pdf_bytes = generate_report_pdf_weasyprint(markdown_text)
    if pdf_bytes:
        return pdf_bytes

    return generate_report_pdf_fpdf(markdown_text)


def build_report_filename(extension: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    return f"auditoria_google_ads_{timestamp}.{extension}"


def show_report_download_buttons(report: str) -> None:
    pdf_bytes = generate_report_pdf(report)
    md_bytes = report.encode("utf-8")
    pdf_filename = build_report_filename("pdf")
    md_filename = build_report_filename("md")

    col_pdf, col_md = st.columns(2)

    with col_pdf:
        st.download_button(
            label="Descarregar Relatório (PDF)",
            data=pdf_bytes,
            file_name=pdf_filename,
            mime="application/pdf",
            use_container_width=True,
        )

    with col_md:
        st.download_button(
            label="Descarregar Relatório (.md)",
            data=md_bytes,
            file_name=md_filename,
            mime="text/markdown",
            use_container_width=True,
        )


def extract_negative_keywords(report: str) -> str:
    patterns = [
        re.compile(r"```negatives?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE),
        re.compile(
            r"(?:palavras[- ]chave negativas|palavras negativas)[^\n]*\n+(?:.*?\n)*?```(?:\w*\n)?(.*?)```",
            re.DOTALL | re.IGNORECASE,
        ),
    ]

    for pattern in patterns:
        match = pattern.search(report)
        if match:
            text = match.group(1).strip()
            if text:
                return text

    for match in re.finditer(r"```[^\n]*\n(.*?)```", report, re.DOTALL):
        block = match.group(1).strip()
        if not block:
            continue
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) >= 2 and all(len(line.split()) <= 4 for line in lines[:8]):
            return block

    return ""


def fetch_negative_keywords_from_gemini(api_key: str, report: str) -> str:
    client = genai.Client(api_key=api_key)
    model = get_active_model(client)
    prompt = (
        "Do relatório de auditoria Google Ads abaixo, extrai APENAS a lista de "
        "palavras-chave negativas recomendadas. "
        "Responde com uma palavra ou termo por linha, sem numeração, sem markdown "
        "e sem explicações adicionais.\n\n"
        f"{report[:12000]}"
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=2048),
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
        r"##\s*(?:Ações Imediatas e Próximos Passos|Ações Imediatas|Próximos Passos)[^\n]*\n(.*?)(?=\n##\s|\Z)",
        report,
        re.DOTALL | re.IGNORECASE,
    )
    if section_match:
        return section_match.group(1).strip()

    code_match = re.search(r"```next_steps?\s*\n(.*?)```", report, re.DOTALL | re.IGNORECASE)
    if code_match:
        return code_match.group(1).strip()

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
        "Ações Imediatas e Próximos Passos. "
        "Estrutura a resposta em Markdown com exatamente estas três subsecções:\n"
        "### Próximas 24 horas\n"
        "### Próximas 48 horas\n"
        "### Próximos 15 dias\n"
        "Lista ações concretas em bullet points em cada período.\n\n"
        f"{report[:12000]}"
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=2048),
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
    st.markdown(
        "Plano de ação prioritário para execução imediata, organizado por janela de tempo."
    )

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


with st.sidebar:
    st.header("⚙️ Configuração")
    api_key_input = st.text_input(
        "Gemini API Key",
        value=st.session_state.gemini_api_key,
        type="password",
        placeholder="Introduza a sua API Key",
        help=(
            "Carregada automaticamente de .streamlit/secrets.toml se existir. "
            "Obtenha a chave em https://aistudio.google.com/apikey"
        ),
    )
    if st.button("Guardar API Key", use_container_width=True):
        st.session_state.gemini_api_key = api_key_input
        st.success("API Key guardada em memória.")

    if st.session_state.gemini_api_key:
        st.caption("✅ API Key configurada")

st.title("📊 Google Ads AI Optimizer")
st.markdown(
    "Faça upload dos ficheiros CSV exportados do Google Ads e obtenha um diagnóstico "
    "e recomendações de otimização gerados por IA."
)

st.subheader("Upload de Ficheiros CSV")

st.markdown("#### Ficheiros Principais (Obrigatórios)")
req_col1, req_col2 = st.columns(2)

with req_col1:
    campaign_file = st.file_uploader(
        "Campaign Settings",
        type=["csv"],
        help="No Google Ads: Vá a Campanhas > Campanhas e clique no ícone de Descarregar (.csv).",
        key="upload_campaign_settings",
    )
    keywords_file = st.file_uploader(
        "Keywords",
        type=["csv"],
        help="No Google Ads: Vá a Públicos-alvo, palavras-chave e conteúdos > Palavras-chave de pesquisa e descarregue em .csv.",
        key="upload_keywords",
    )

with req_col2:
    search_terms_file = st.file_uploader(
        "Search Terms",
        type=["csv"],
        help="No Google Ads: Vá a Públicos-alvo, palavras-chave e conteúdos > Termos de pesquisa e descarregue em .csv.",
        key="upload_search_terms",
    )
    rsas_file = st.file_uploader(
        "RSAs",
        type=["csv"],
        help="No Google Ads: Vá a Anúncios e recursos > Anúncios, filtre por Anúncio Responsivo de Pesquisa e descarregue em .csv.",
        key="upload_rsas",
    )

st.markdown("#### Ficheiros Complementares (Opcionais)")
opt_col1, opt_col2 = st.columns(2)

with opt_col1:
    assets_file = st.file_uploader(
        "Assets & Extensions",
        type=["csv"],
        help="No Google Ads: Vá a Anúncios e recursos > Recursos (Assets) e descarregue em .csv.",
        key="upload_assets",
    )
    locations_file = st.file_uploader(
        "Locations",
        type=["csv"],
        help="No Google Ads: Vá a Públicos-alvo, palavras-chave e conteúdos > Localizações e descarregue em .csv.",
        key="upload_locations",
    )

with opt_col2:
    devices_file = st.file_uploader(
        "Devices & Schedule",
        type=["csv"],
        help="No Google Ads: Vá a Apresentação de relatórios > Dispositivos (ou Horário dos anúncios) e descarregue em .csv.",
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

    with st.spinner("Analisando campanhas com Gemini... Isto pode levar alguns minutos."):
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

if "last_report" in st.session_state:
    st.divider()
    st.subheader("Resultados da Análise")
    show_results_section()
