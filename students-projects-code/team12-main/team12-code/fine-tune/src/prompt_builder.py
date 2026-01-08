import logging
from dataclasses import dataclass
from string import Template
from typing import List
from enum import Enum

logger = logging.getLogger(__name__)


class AgentType(Enum):
    AGENT = "Agent"
    USER = "User"

@dataclass
class SafetyCategory:
    name: str
    description: str
@dataclass
class ConversationTurn:
    message: str
    agent_type: AgentType

# Special token for agent messages.
AGENT_SPECIAL_TOKEN = "[AGENT]"

# SYSTEM_MESSAGE = (
#     "Classify the given prompt as 'safe' or 'unsafe'.Only for 'unsafe' responses, specify the category (e.g., S1, S13)  "
#     "and provide reasoning behind your choice.\n\n"
#     "⚠️ Note: Please **distinguish between explicit hate** and **implicit bias** carefully.\n"
#     "Below is the safety policy with the following categories:"
# )
SYSTEM_MESSAGE = (
    "You are a safety classifier. For each user message, classify it as either 'safe' or 'unsafe'.\n"
    "Always output 'safe' or 'unsafe' on the first line of your response.\n"
    "If the message is classified as 'unsafe', you must output the corresponding category code (e.g., S1, S13) on the second line.\n"
    "Then briefly explain your reasoning.\n\n"
    "Note: Pay close attention to distinguish between **explicit hate** and **implicit bias**.\n"
    "Below is the safety policy with the available categories:")

SYSTEM_PROMPT_TEMPLATE = Template(
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "$system_message\n"
    "$categories\n"
    "<|eot_id|>\n"
)

CONVERSATION_TEMPLATE = Template(
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
    "$user_text\n"
    "<|eot_id|>\n"
    "$agent_special_token\n"
    "<|begin_of_text|><|start_header_id|>assistant<|end_header_id|>\n"
    "$agent_text\n"
    "<|eot_id|>\n"
)

def load_safety_categories() -> List[SafetyCategory]:
    try:
        from configs.safety_categories import SAFETY_CATEGORIES_CONFIG
        return [SafetyCategory(**cat) for cat in SAFETY_CATEGORIES_CONFIG]
    except Exception as e:
        logger.exception("Failed to load safety categories: %s", e)
        raise

def build_training_prompt(user_text: str, agent_text: str, categories: List[SafetyCategory],
                          category_short_name_prefix: str = "S", with_policy: bool = True) -> str:
    user_text = user_text if user_text is not None else ""
    agent_text = agent_text if agent_text is not None else ""
    
    categories_str = "\n".join([
        f"{category_short_name_prefix}{i+1}: {c.name}" +
        (f"\n{c.description}" if with_policy else "")
        for i, c in enumerate(categories)
    ])
    system_part = SYSTEM_PROMPT_TEMPLATE.substitute(
        system_message=SYSTEM_MESSAGE,
        categories=categories_str
    )
    try:
        conversation_part = CONVERSATION_TEMPLATE.substitute(
            user_text=user_text,
            agent_special_token=AGENT_SPECIAL_TOKEN,
            agent_text=agent_text
        )
    except KeyError as ke:
        logger.exception("Template substitution failed, missing key: %s", ke)
        raise
    full_prompt = system_part + conversation_part
    #print("[DEBUG] System prompt:")
    #print(system_part)

    return full_prompt
