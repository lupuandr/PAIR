import re
from abc import ABC, abstractmethod


class Agent(ABC):
    """Abstract class for all agents"""

    @abstractmethod
    def infer(self, prompts, nums_only=False, skip_special_tokens=True):
        pass

    def process_outputs(self, outputs, nums_only=False):
        return [self.clean_string(out, nums_only) for out in outputs]

    def clean_string(self, text, nums_only=False):
        if nums_only:
            # If nums_only is True, return only numbers from the string
            if "\n" in text:
                text = text.split("\n")[-1]
            return "".join(re.findall(r"\d", text))

        # A list of special tokens to remove
        tokens_to_remove = [
            r"<s>",
            r"</s>",
            r"<unk>",
            r"<<SYS>>",
            r"<</SYS>>",
            r"\[INST\]",  # Escape on square brackets is crucial for regex.
            r"\[/INST\]",
            r"\n",
            # Add any additional tokens you want to remove in the list above
        ]

        # Create a combined regular expression pattern to match any of the tokens
        pattern = "|".join(tokens_to_remove)

        # Substitute the matched tokens with an empty string to remove them
        cleaned_text = re.sub(pattern, "", text)

        return (
            cleaned_text.strip()
        )  # Removing any extra spaces before and after the cleaned string

    def parse_output(self, text):
        """
        Parse output to remove excess. Each split below could be a no-op.
        """

        # Remove input prompt (-hf models only)
        text = text.rsplit("[/INST]", 1)[-1]
        # Remove trailing "Output:" or "Mutated prompt:"
        # (-hf models with prompt injection and all models in case target repeats "Output:"
        text = text.rsplit("Output:", 1)[-1]
        text = text.rsplit("Mutated prompt:", 1)[-1]
        # Remove square brackets (used in conditional mutate)
        text = text.rsplit("[[", 1)[-1]
        text = text.rsplit("]]", 1)[0]

        return text
