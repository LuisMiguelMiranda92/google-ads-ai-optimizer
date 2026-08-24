# 📊 Google Ads AI Optimizer

Uma aplicação web desenvolvida em **Python** e **Streamlit** que automatiza auditorias completas de contas **Google Ads** através de modelos de inteligência artificial de última geração (**Google Gemini 3 Flash**).

A ferramenta processa relatórios brutos exportados do Google Ads (em formato CSV), analisa métricas essenciais de desempenho e gera diagnósticos acionáveis com planos de execução claros.

---
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://(https://app-ads-ai-optimizer-hygt4ctdztgztaekapgqr2.streamlit.app/))

🔗 **Live Demo:** [Aceder ao Google Ads AI Optimizer](https://[https://app-ads-ai-optimizer-hygt4ctdztgztaekapgqr2.streamlit.app/])

## 🚀 Funcionalidades Principais

* **Leitura Dinâmica de CSVs:** Suporte para até 7 relatórios padrão do Google Ads (*Campaign Settings*, *Keywords*, *Search Terms*, *Ad/RSA Report*, *Assets & Extensions*, *Locations*, *Devices & Schedule*).
* **Limpeza e Tratamento Automático:** Filtra metadados e cabeçalhos de exportação, com suporte para relatórios em múltiplos idiomas (Português e Inglês).
* **Auditoria Estratégica com IA:** Análise de discrepâncias de CPA, desperdício em redes de pesquisa/display, taxas de irrelevância de termos e Quality Score.
* **UI Estruturada em 3 Separadores:**
  * `🤖 Auditoria AI completa`: Diagnóstico aprofundado com análise por área (Bidding, Keywords, Anúncios, Localização e Dispositivos).
  * `🛑 Palavras Negativas`: Bloco de código formatado com lista de termos recomendados para exclusão imediata (pronto a copiar com 1 clique).
  * `🚀 Próximos Passos`: Plano de ação tático dividido em prazos (24h, 48h e 15 dias).
* **Exportação Multi-Formato:** Botões nativos para descarregar o relatório final em **PDF estilizado** (`fpdf2`) ou em **Markdown (`.md`)**.

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.10+
* **Interface Web:** Streamlit
* **Processamento de Dados:** Pandas
* **Geração de PDF:** FPDF2 / Markdown2
* **Modelo de IA:** Google Gemini API (`gemini-3-flash-preview` / `gemini-2.5-flash`)

---

## ⚙️ Instalação e Execução Local

### 1. Clonar o repositório
```bash
git clone [https://github.com/SEU_UTILIZADOR/google-ads-ai-optimizer.git](https://github.com/SEU_UTILIZADOR/google-ads-ai-optimizer.git)
cd google-ads-ai-optimizer
