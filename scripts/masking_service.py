import re
import yaml
from typing import Any

class MaskingService:
    def __init__(self, config_path: str = "masking_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.sensitive_keys = set(self.config.get("sensitive_keys", []))
        self.sensitive_terms = self.config.get("sensitive_terms", [])
        self.token_map = self.config.get("token_map", {})
        
        # Sort by length descending to ensure longest match first
        sorted_terms = sorted(self.sensitive_terms, key=len, reverse=True)
        if sorted_terms:
            self.pattern = re.compile(
                r'|'.join(re.escape(term) for term in sorted_terms), 
                re.IGNORECASE
            )
        else:
            self.pattern = None

    def mask_text(self, text: str) -> str:
        """Masks sensitive terms within a string using the token map."""
        if not self.pattern or not text:
            return text
            
        def replace_match(match):
            original = match.group(0)
            # Find the specific token for the matched text
            for term, token in self.token_map.items():
                if term.lower() in original.lower():
                    return f"[{token}]"
            return "[MASKED_TERM]"
            
        return self.pattern.sub(replace_match, text)

    def mask_structure(self, data: Any) -> Any:
        """Recursively masks dictionaries, lists, and strings."""
        if isinstance(data, dict):
            new_dict = {}
            for k, v in data.items():
                # 1. If KEY is sensitive, hide the value completely
                if k in self.sensitive_keys:
                    new_dict[k] = "[MASKED_VALUE]" 
                # 2. If key is safe, recurse into the value
                else:
                    new_dict[k] = self.mask_structure(v)
            return new_dict
        elif isinstance(data, list):
            return [self.mask_structure(item) for item in data]
        elif isinstance(data, str):
            return self.mask_text(data)
        else:
            return data