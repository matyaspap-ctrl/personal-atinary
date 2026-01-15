"""
Materials Science Assistant

A simple LLM-powered assistant that helps researchers interpret
and guide their materials science experiments.

This application represents a typical use case where users provide
experimental context and receive AI-generated recommendations.
"""

import os
from dataclasses import dataclass, field

from botocore.config import Config
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate

# [NEW] Import the security modules
from masking_service import MaskingService
from monitoring import SecureMonitoringCallback

load_dotenv()


# =============================================================================
# Bedrock Configuration
# =============================================================================

@dataclass
class BedrockSettings:
    """Configuration for AWS Bedrock client."""

    bedrock_region: str = field(
        default_factory=lambda: os.environ.get("BEDROCK_REGION", "eu-central-1")
    )
    bedrock_model_id: str = field(
        default_factory=lambda: os.environ.get(
            "BEDROCK_MODEL_ID", "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"
        )
    )
    bedrock_read_timeout: int = field(
        default_factory=lambda: int(os.environ.get("BEDROCK_READ_TIMEOUT", "300"))
    )
    bedrock_max_attempts: int = field(
        default_factory=lambda: int(os.environ.get("BEDROCK_MAX_ATTEMPTS", "10"))
    )
    bedrock_retry_mode: str = field(
        default_factory=lambda: os.environ.get("BEDROCK_RETRY_MODE", "standard")
    )

    def get_boto_config(self) -> Config:
        """Create boto3 Config object with retry and timeout settings."""
        return Config(
            read_timeout=self.bedrock_read_timeout,
            retries={
                "max_attempts": self.bedrock_max_attempts,
                "mode": self.bedrock_retry_mode,
            },
        )


# =============================================================================
# Prompt Template
# =============================================================================

PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert materials science assistant specializing in 
experimental design and workflow management. Help the researcher with their question."""),
    ("human", """## Design Space

{design_space}

## Prior Experimental Data

{prior_data}

## Expert Context

{expert_context}"""),
])


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class DesignSpace:
    """
    The parameter space for experimental design.
    Constructed via the UI based on user's experimental setup.
    """

    target_property: str
    parameters: dict[str, dict]  # param_name -> {value, min, max, unit}

    def format(self) -> str:
        lines = [
            f"Target Property: {self.target_property}",
            "Parameters:",
        ]
        for name, spec in self.parameters.items():
            lines.append(
                f"  - {name}: {spec['value']} {spec.get('unit', '')} "
                f"(range: {spec.get('min', '?')} - {spec.get('max', '?')})"
            )
        return "\n".join(lines)


@dataclass
class PriorData:
    """
    Historical experimental observations from previous experiments.
    """

    observations: list[dict]

    def format(self) -> str:
        if not self.observations:
            return "(No prior observations)"
        lines = []
        for i, obs in enumerate(self.observations, 1):
            lines.append(f"Iteration {i}: {obs}")
        return "\n".join(lines)


# =============================================================================
# Assistant
# =============================================================================

class BOAssistant:
    """
    LLM assistant for experimental design and workflow management.

    Users interact with this assistant to get help interpreting their
    experiments and understanding optimization suggestions.
    """

    def __init__(self, settings: BedrockSettings | None = None):
        self.settings = settings or BedrockSettings()
        
        # [NEW] Initialize the Masking Service and Callback
        # We assume masking_config.yaml exists in the same directory
        self.masker = MaskingService("masking_config.yaml")
        self.secure_callback = SecureMonitoringCallback(self.masker)

        self.llm = ChatBedrock(
            model_id=self.settings.bedrock_model_id,
            region_name=self.settings.bedrock_region,
            model_kwargs={"temperature": 0.1},
            config=self.settings.get_boto_config(),
            # [NEW] Attach the secure callback to the LLM
            callbacks=[self.secure_callback]
        )
        self.chain = PROMPT | self.llm

    def ask(
        self,
        design_space: DesignSpace,
        prior_data: PriorData,
        expert_context: str,
    ) -> str:
        """
        Ask a question about the experiment.

        Args:
            design_space: The BO parameter space (from UI)
            prior_data: Historical observations
            expert_context: User-provided context and questions
        """
        response = self.chain.invoke({
            "design_space": design_space.format(),
            "prior_data": prior_data.format(),
            "expert_context": expert_context,
        })
        return response.content


# =============================================================================
# Example Usage Scenarios
# =============================================================================

if __name__ == "__main__":
    # [NEW] Quick check to prevent crashing if config is missing
    if not os.path.exists("masking_config.yaml"):
        print("WARNING: masking_config.yaml not found. Creating default...")
        with open("masking_config.yaml", "w") as f:
            f.write("sensitive_keys: []\nsensitive_terms: []\ntoken_map: {}")

    assistant = BOAssistant()

    # =========================================================================
    # SCENARIO 1: Solid-State Electrolyte Synthesis
    # =========================================================================
    print("\n" + "=" * 70)
    print("Scenario 1: Solid-State Electrolyte Synthesis")
    print("=" * 70)

    design_space_1 = DesignSpace(
        target_property="ionic conductivity (mS/cm)",
        parameters={
            "sintering_temp": {"value": 478, "min": 400, "max": 600, "unit": "°C"},
            "pressure": {"value": 2.8, "min": 1.0, "max": 5.0, "unit": "atm"},
            "sintering_aid": {"value": "PLT-X9", "min": None, "max": None, "unit": ""},
            "aid_loading": {"value": 4.8, "min": 1.0, "max": 10.0, "unit": "mol%"},
            "hold_time": {"value": 4.0, "min": 1.0, "max": 8.0, "unit": "hours"},
        },
    )

    prior_data_1 = PriorData(observations=[
        {"temp": 450, "pressure": 2.3, "aid": 4.5, "conductivity": "2.3 mS/cm", "density": "94.2%"},
        {"temp": 465, "pressure": 2.5, "aid": 4.6, "conductivity": "2.8 mS/cm", "density": "95.1%"},
        {"temp": 478, "pressure": 2.8, "aid": 4.8, "conductivity": "3.1 mS/cm", "density": "96.3%"},
        {"temp": 490, "pressure": 2.2, "aid": 5.0, "conductivity": "2.4 mS/cm", "density": "93.8%"},
        {"temp": 475, "pressure": 3.1, "aid": 4.7, "conductivity": "3.3 mS/cm", "density": "96.8%"},
    ])

    expert_context_1 = """
    We're synthesizing a garnet-type solid electrolyte for lithium-ion batteries. 
    The synthesis uses a high-pressure tube furnace under argon atmosphere to 
    prevent oxidation. Our sintering aid helps with densification but we've 
    found diminishing returns above 5 mol% loading.

    From our experiments, I've noticed that temperature and pressure have a 
    strong interaction. Increasing temperature alone above 480°C leads to 
    lithium volatilization and actually reduces conductivity (see iteration 4). 
    However, when we increase pressure alongside temperature, the lithium is 
    retained and we get better densification (iteration 5 gave our best result).

    The optimizer is now suggesting 485°C with 3.3 atm pressure. We've never 
    operated above 3.0 atm before. Is this a reasonable next step given the 
    temperature-pressure interaction we're seeing, or should we be more 
    conservative and first explore higher pressures at the current temperature?
    """

    response_1 = assistant.ask(design_space_1, prior_data_1, expert_context_1)
    print(f"\nAssistant: {response_1[:600]}...")

    # =========================================================================
    # SCENARIO 2: Perovskite Solar Cell Fabrication
    # =========================================================================
    print("\n" + "=" * 70)
    print("Scenario 2: Perovskite Solar Cell Fabrication")
    print("=" * 70)

    design_space_2 = DesignSpace(
        target_property="power conversion efficiency (%)",
        parameters={
            "annealing_temp": {"value": 150, "min": 100, "max": 200, "unit": "°C"},
            "spin_speed": {"value": 3000, "min": 1000, "max": 5000, "unit": "rpm"},
            "precursor_conc": {"value": 1.2, "min": 0.5, "max": 2.0, "unit": "M"},
            "annealing_time": {"value": 30, "min": 10, "max": 60, "unit": "min"},
            "antisolvent_delay": {"value": 15, "min": 5, "max": 30, "unit": "sec"},
        },
    )

    prior_data_2 = PriorData(observations=[
        {"temp": 140, "speed": 2500, "conc": 1.0, "efficiency": "18.2%", "Voc": "1.08 V"},
        {"temp": 145, "speed": 2800, "conc": 1.1, "efficiency": "19.1%", "Voc": "1.10 V"},
        {"temp": 150, "speed": 3000, "conc": 1.2, "efficiency": "19.8%", "Voc": "1.12 V"},
        {"temp": 160, "speed": 3000, "conc": 1.2, "efficiency": "18.5%", "Voc": "1.06 V"},
        {"temp": 150, "speed": 3500, "conc": 1.3, "efficiency": "19.4%", "Voc": "1.11 V"},
        {"temp": 155, "speed": 3200, "conc": 1.25, "efficiency": "20.1%", "Voc": "1.13 V"},
    ])

    expert_context_2 = """
    We're fabricating mixed-cation perovskite solar cells using spin coating 
    in a nitrogen glovebox. The process involves depositing the precursor 
    solution, applying an antisolvent drip during spinning, then annealing on 
    a hotplate to crystallize the perovskite film.

    We use a stabilizing additive in the precursor (5 mol%) that improves 
    reproducibility but becomes unstable above 160°C based on literature reports 
    for similar compounds.

    The data shows interesting patterns. Higher spin speeds produce thinner 
    films that seem more sensitive to annealing temperature. At iteration 4, 
    jumping to 160°C clearly hurt performance - the Voc dropped significantly, 
    suggesting thermal decomposition. But iteration 6 at 155°C with slightly 
    faster spin and higher concentration gave us our best result (20.1%).

    The optimizer now suggests:
    - Annealing temperature: 158°C
    - Spin speed: 3300 rpm  
    - Precursor concentration: 1.28 M

    I'm nervous about 158°C given the degradation at 160°C. But the model is 
    also increasing spin speed and concentration. Could thinner films from 
    faster spinning tolerate slightly higher temperatures? Or is this too risky 
    given we're so close to the instability threshold?
    """

    response_2 = assistant.ask(design_space_2, prior_data_2, expert_context_2)
    print(f"\nAssistant: {response_2[:600]}...")

    # =========================================================================
    # SCENARIO 3: Drug-Target Binding Assay
    # =========================================================================
    print("\n" + "=" * 70)
    print("Scenario 3: Drug-Target Binding Assay")
    print("=" * 70)

    design_space_3 = DesignSpace(
        target_property="binding affinity Ki (nM)",
        parameters={
            "inhibitor_conc": {"value": 25.0, "min": 1.0, "max": 100.0, "unit": "µM"},
            "buffer_pH": {"value": 7.4, "min": 6.0, "max": 8.5, "unit": ""},
            "temperature": {"value": 37, "min": 25, "max": 45, "unit": "°C"},
            "incubation_time": {"value": 60, "min": 15, "max": 120, "unit": "min"},
            "DMSO_percent": {"value": 1.0, "min": 0.5, "max": 5.0, "unit": "%"},
        },
    )

    prior_data_3 = PriorData(observations=[
        {"conc": 10, "pH": 7.4, "temp": 30, "Ki": "52 nM", "Z_factor": 0.72},
        {"conc": 15, "pH": 7.4, "temp": 37, "Ki": "45 nM", "Z_factor": 0.75},
        {"conc": 20, "pH": 7.2, "temp": 37, "Ki": "38 nM", "Z_factor": 0.78},
        {"conc": 25, "pH": 7.0, "temp": 37, "Ki": "32 nM", "Z_factor": 0.71},
        {"conc": 30, "pH": 7.4, "temp": 40, "Ki": "41 nM", "Z_factor": 0.65},
        {"conc": 25, "pH": 7.2, "temp": 37, "Ki": "29 nM", "Z_factor": 0.79},
    ])

    expert_context_3 = """
    We're running a fluorescence polarization assay to measure binding affinity 
    of our kinase inhibitor. The assay uses a fluorescent tracer at 10 nM and 
    we compete it off with our compound at various concentrations.

    The target kinase has pH-dependent conformational dynamics - at lower pH 
    the binding pocket becomes more accessible, but compound solubility decreases. 
    At physiological pH (7.4) binding should be optimal but we see aggregation 
    at higher compound concentrations.

    I'm seeing some concerning patterns in the data. At iteration 5 (30 µM, 40°C), 
    the Z-factor dropped to 0.65 indicating noisy data, and I noticed slight 
    cloudiness in the wells suggesting precipitation. Mass spec analysis of 
    those samples showed a degradation product consistent with amide hydrolysis - 
    this wasn't present in the 37°C samples.

    Interestingly, lowering pH to 7.0-7.2 seems to improve measured affinity 
    (iterations 3, 4, 6 show lower Ki values). But I'm worried this could be 
    an artifact - the pH might be affecting the tracer fluorescence rather than 
    genuinely improving binding. Also, at pH 7.0 the compound is only ~80% 
    ionized versus ~95% at pH 7.4, which could hurt cell permeability later.

    Should we continue exploring lower pH, or focus on optimizing at 
    physiological pH to ensure the results translate to cellular assays?
    """

    response_3 = assistant.ask(design_space_3, prior_data_3, expert_context_3)
    print(f"\nAssistant: {response_3[:600]}...")

    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("All scenarios complete.")
    print("=" * 70)