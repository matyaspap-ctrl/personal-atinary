import json
import logging
from typing import Any, Dict, List
from langchain_core.callbacks import BaseCallbackHandler
from masking_service import MaskingService

# Setup a logger that mimics CloudWatch/X-Ray
logger = logging.getLogger("CloudWatchSimulator")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('\033[93m%(asctime)s - [CLOUDWATCH] - %(message)s\033[0m')) # Yellow for visibility
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class SecureMonitoringCallback(BaseCallbackHandler):
    """
    LangChain Callback: Intercepts events, sanitizes data, and logs it.
    """
    def __init__(self, masking_service: MaskingService):
        self.masker = masking_service

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Log the PROMPT, but MASK it first."""
        masked_prompts = [self.masker.mask_text(p) for p in prompts]
        logger.info(f"LLM START - Prompts: {json.dumps(masked_prompts)[:2000]}...") # Truncated for demo

    def on_llm_end(self, response: Any, **kwargs: Any) -> Any:
        """Log the RESPONSE, but MASK it first."""
        try:
            text = response.generations[0][0].text
            masked_text = self.masker.mask_text(text)
            logger.info(f"LLM END - Response: {masked_text[:2000]}...")
        except:
            logger.info("LLM END - (Could not parse response text)")

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        """Log the INPUTS (DesignSpace dicts), but MASK keys/values."""
        # Convert inputs to a standard dict if they aren't already
        safe_inputs = inputs.copy()
        masked_inputs = self.masker.mask_structure(safe_inputs)
        logger.info(f"CHAIN START - Inputs: {json.dumps(masked_inputs)[:2000]}...")