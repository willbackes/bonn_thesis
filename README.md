# Can ChatGPT predict wages? Synthetic data as complement for labour market research

**Master's Thesis, University of Bonn, Economics Department**

William Backes | Advisor: Prof. Dr. Sebastian Kube | 25 March 2026

## Overview

Investigates whether LLMs (GPT-4) can predict gross labour income from SOEP survey and
LinkedIn profile data (2013–2019). Uses measurement error frameworks to validate
predictions at individual and stratum levels.

**Main findings:** LLM reproduces broad stratum-level wage patterns but exhibits
focal-value heaping (€1,000 multiples), systematic underprediction in upper tail
(non-classical error), and compressed distributions. LinkedIn-enriched prompts improve
accuracy but predictions unsuitable as wage substitutes. Sample: 70,378 person-year
observations (21,376 individuals, full-time workers only).

## Research Questions

1. Can LLMs predict labour income conditional on SOEP covariates?
1. Does enriching LinkedIn information (job titles, companies) improve predictions?

## Methodology

**LLM Setup:** Zero-shot prompting, GPT-4.1-nano, temperature=0.7

**Prompt format:** Year | ISCO-08 (3-digit) | 5 education levels | Sex | Federal state |
Experience

**Stratum-level validation:** Groups by education, occupation, location, sex, year.
Compares average log wages: $u_g = \\bar{\\hat{y}}^{LLM}\_g - \\bar{y}^\*\_g$.

**Data preparation:** SOEP restricted to full-time workers; LinkedIn parsed via NLP;
education/location normalized to ISCED/16 states; records harmonized for consistent
coding.

## Key Results

| Finding                   | Details                                                             |
| ------------------------- | ------------------------------------------------------------------- |
| **Focal heaping**         | Top 10 values = 50%+ of predictions; all €1,000 multiples           |
| **Mean prediction error** | €3,105/month (LLM) vs €3,522/month (SOEP)                           |
| **Distribution**          | Compressed wage range; systematic underprediction                   |
| **Measurement error**     | Non-classical: mean-reverting ($\\mathbb{E}[u \\mid Y^\*] \\neq 0$) |
| **LinkedIn enrichment**   | Improves accuracy, reduces heaping, closer SOEP distribution        |
| **Mincer regressions**    | Coefficients closer to SOEP benchmarks with LinkedIn context        |
|                           |                                                                     |

## Project Structure

```
src/bonn_thesis/
├── analysis/              # Notebooks: individual/grouped/LinkedIn comparisons
├── data_management/       # Cleaning, education, location, ISCO, sampling
├── openai_processing/     # Batch managers, fine-tuning, costs

bld/
├── data/                  # Processed outputs & metadata
├── tables/                # Regression results (LaTeX)
├── figures/               # Visualizations

documents/
├── thesis.tex             # Full manuscript
├── paper.tex
└── refs.bib
```

## Data Access & Compliance

⚠️ **Raw data NOT included:**

- **SOEP:** Restricted microdata; apply via [DIW Berlin](https://www.diw.de/soep)
- **LinkedIn:** Proprietary;

**Included:** Processed outputs, metadata, classification results, ISCO validation
metrics

Researchers replicating this work must apply independently for data access.

## Installation & Use

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"

# Run pipeline
pytask                      # Full workflow
pytask -k data_management   # Data prep only
pytask -k "task_clean_*"    # Specific tasks

# Configure (optional)
echo "OPENAI_API_KEY=your_key" > .env
```

## Stack

Python 3.11+ | pytask | OpenAI API | pandas/scipy | statsmodels | SQLAlchemy | Jupyter |
LaTeX

## Key Files

| Component      | File(s)                                                               | Purpose                                  |
| -------------- | --------------------------------------------------------------------- | ---------------------------------------- |
| Data cleaning  | `clean_*.py`                                                          | SOEP/LinkedIn standardization            |
| Sample design  | `sample_selection.py`                                                 | Stratum definition                       |
| LLM processing | `*batch_manager.py`                                                   | OpenAI API interaction                   |
| Validation     | `1_compare_gpt_to_soep_individual.ipynb`, `2_compare_*_grouped.ipynb` | Measurement error analysis               |
| Regression     | `*grouped*.ipynb`                                                     | Mincer equations, coefficient comparison |
| Manuscript     | `documents/thesis.tex`                                                | Full thesis                              |
