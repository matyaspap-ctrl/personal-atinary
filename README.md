# LLM Data Privacy - Observability Challenge

A materials science assistant application demonstrating the challenge of implementing observability for LLM-powered systems while protecting sensitive data.

## Objective

Build an **observability layer** that enables effective debugging and monitoring while protecting sensitive information. The solution should **mask sensitive data** while preserving enough structure for traces to remain useful.

---

## Phase 1: Primary Objective

The main goal is to implement masking that protects sensitive content while maintaining trace structure:

| Data Type | Masking Rule |
|-----------|--------------|
| **Expert Context** | Mask the entire free-text content |
| **Design Space** | Mask parameter names and units, preserve numeric values |
| **Prior Data** | Mask all keys and units, preserve numeric values |

### Design Space: Mask Names & Units, Preserve Values

**Original:**
```
Target Property: ionic conductivity (mS/cm)
Parameters:
  - sintering_temp: 478 °C (range: 400 - 600)
  - pressure: 2.8 atm (range: 1.0 - 5.0)
  - sintering_aid: PLT-X9  (range: 0 - 10.0)
  - aid_loading: 4.8 mol% (range: 1.0 - 10.0)
  - hold_time: 4.0 hours (range: 1.0 - 8.0)
```

**Masked:**
```
Target Property: [MASKED] ([MASKED])
Parameters:
  - [MASKED]: 478 [MASKED] (range: 400 - 600)
  - [MASKED]: 2.8 [MASKED] (range: 1.0 - 5.0)
  - [MASKED]: [MASKED]  (range: 0 - 10.0)
  - [MASKED]: 4.8 [MASKED] (range: 1.0 - 10.0)
  - [MASKED]: 4.0 [MASKED] (range: 1.0 - 8.0)
```

### Prior Data: Mask All Keys & Units, Preserve Values

**Original:**
```
Iteration 1: {"temp": 450, "pressure": 2.3, "aid": 4.5, "conductivity": "2.3 mS/cm", "density": "94.2%"}
Iteration 2: {"temp": 465, "pressure": 2.5, "aid": 4.6, "conductivity": "2.8 mS/cm", "density": "95.1%"}
Iteration 3: {"temp": 478, "pressure": 2.8, "aid": 4.8, "conductivity": "3.1 mS/cm", "density": "96.3%"}
```

**Masked:**
```
Iteration 1: {"[MASKED]": 450, "[MASKED]": 2.3, "[MASKED]": 4.5, "[MASKED]": "2.3 [MASKED]", "[MASKED]": "94.2[MASKED]"}
Iteration 2: {"[MASKED]": 465, "[MASKED]": 2.5, "[MASKED]": 4.6, "[MASKED]": "2.8 [MASKED]", "[MASKED]": "95.1[MASKED]"}
Iteration 3: {"[MASKED]": 478, "[MASKED]": 2.8, "[MASKED]": 4.8, "[MASKED]": "3.1 [MASKED]", "[MASKED]": "96.3[MASKED]"}
```

### Expert Context: Mask Entirely

**Original:**
```
We're synthesizing a garnet-type solid electrolyte for lithium-ion batteries.
The synthesis uses a high-pressure tube furnace under argon atmosphere to
prevent oxidation. Our sintering aid helps with densification but we've
found diminishing returns above 5 mol% loading.
```

**Masked:**
```
[MASKED_EXPERT_CONTEXT]
```

---

## Phase 2: Stretch Goal

Once Phase 1 is complete, explore more sophisticated masking that preserves sentence structure for better debuggability:

### NLP-Based Selective Masking

Instead of masking the entire expert context, use NLP to identify and mask only sensitive elements (nouns) while preserving verbs, articles and adjectives:

**Original:**
```
We're synthesizing a garnet-type solid electrolyte for lithium-ion batteries.
The synthesis uses a high-pressure tube furnace under argon atmosphere to
prevent oxidation.
```

**Selectively Masked:**
```
[MASKED]'re synthesizing a [MASKED]-type [MASKED] [MASKED] for [MASKED]-[MASKED] [MASKED].
The [MASKED] uses a high-[MASKED] [MASKED] [MASKED] under [MASKED] [MASKED] to
prevent [MASKED].
```

---

## The Challenge

Standard observability tools (CloudWatch, LangSmith, Sentry) would expose all data in logs. This application demonstrates typical LLM interactions containing:

- Scientific parameters and experimental values
- Proprietary compound/catalyst identifiers
- Equipment and sample identifiers
- Technical specifications and units

---

## Scenario Examples

### Scenario 1: Solid-State Electrolyte Synthesis

| Section | Contains | Phase 1 Masking |
|---------|----------|-----------------|
| Design Space | `sintering_temp`, `pressure`, `sintering_aid`, `PLT-X9`, `°C`, `atm`, `mol%`, `hours` | Mask names and units |
| Prior Data | Keys: `temp`, `pressure`, `aid`, `conductivity`, `density` | Mask all keys and units |
| Expert Context | Free-text about garnet electrolytes, lithium batteries, tube furnace | Mask entirely |

### Scenario 2: Perovskite Solar Cell Fabrication

| Section | Contains | Phase 1 Masking |
|---------|----------|-----------------|
| Design Space | `annealing_temp`, `spin_speed`, `precursor_conc`, `°C`, `rpm`, `M`, `min`, `sec` | Mask names and units |
| Prior Data | Keys: `temp`, `speed`, `conc`, `efficiency`, `Voc` | Mask all keys and units |
| Expert Context | Free-text about perovskite cells, spin coating, glovebox | Mask entirely |

### Scenario 3: Drug-Target Binding Assay

| Section | Contains | Phase 1 Masking |
|---------|----------|-----------------|
| Design Space | `inhibitor_conc`, `buffer_pH`, `temperature`, `DMSO_percent`, `µM`, `nM` | Mask names and units |
| Prior Data | Keys: `conc`, `pH`, `temp`, `Ki`, `Z_factor` | Mask all keys and units |
| Expert Context | Free-text about fluorescence assay, kinase inhibitor, binding | Mask entirely |

---

## AWS Architecture

<p align="center">
  <img src="./ExpertContextDataFlow.svg" alt="AWS Architecture" width="900">
</p>

---

## Requirements

| Requirement | Description |
|-------------|-------------|
| **Reversible Tokenization** | Authorized users can decode masked data when needed |
| **Trace Correlation** | Request/response pairs must be correlatable across logs |
| **Latency** | Masking overhead should add <100ms to request latency |

---

## Setup & Execution

```bash
# Install dependencies
uv sync

# Copy environment configuration
cp .env.example .env

# Edit .env with your AWS credentials (requires Bedrock access)
# Required: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
# Optional: AWS_SESSION_TOKEN (for SSO/temporary credentials)

# Run the sample scenarios
uv run main.py
```

---

## File Structure

```
llm-data-privacy/
├── main.py          # Sample assistant application with 3 scenarios
├── pyproject.toml   # Dependencies (langchain, boto3, etc.)
├── .env.example     # Environment template
└── README.md        # This file
```

---

## Notes for Implementation

1. **LLM responses may echo sensitive data** — The model may repeat compound names, parameters, etc. in its responses (responses also need masking)

2. **Do not modify application logic** — The observability layer should wrap/intercept, not change the core application
# personal-atinary
