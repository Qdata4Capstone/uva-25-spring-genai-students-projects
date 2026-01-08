import logging
import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
from src.prompt_builder import (
    build_training_prompt,
    ConversationTurn,
    AgentType,
    load_safety_categories
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class LlamaGuardPredictor:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.safety_categories = load_safety_categories()

    def predict(self,
                conversation: list,
                unsloth_categories: dict = None,
                max_new_tokens: int = 200,
                use_custom_prompt: bool = False) -> str:
        try:
            if use_custom_prompt:
                conv_turns = []
                for msg in conversation:
                    role_str = msg.get("role", "").lower()
                    agent_type = AgentType.USER if role_str == "user" else AgentType.AGENT
                    try:
                        text = msg["content"][0]["text"]
                    except (KeyError, IndexError):
                        logger.error("Message format error: %s", msg)
                        continue
                    conv_turns.append(ConversationTurn(message=text, agent_type=agent_type))

                user_text, agent_text = "", ""
                for turn in reversed(conv_turns):
                    if turn.agent_type == AgentType.AGENT and agent_text == "":
                        agent_text = turn.message
                    elif turn.agent_type == AgentType.USER and user_text == "":
                        user_text = turn.message
                    if user_text and agent_text:
                        break

                prompt = build_training_prompt(
                    user_text=user_text,
                    agent_text=agent_text,
                    categories=self.safety_categories,
                    category_short_name_prefix="S",
                    with_policy=True
                )
                logger.info("Using custom training prompt (with system safety policy).")
                input_ids = self.tokenizer(prompt, return_tensors="pt")
                input_ids = input_ids.to(self.model.device) if hasattr(input_ids, 'to') else {k: v.to(self.model.device) for k, v in input_ids.items()}

                prompt_len = input_ids["input_ids"].shape[1]

            else:
                logger.info("Using default unsloth chat template.")
                input_ids = self.tokenizer.apply_chat_template(
                    conversation,
                    return_tensors="pt",
                    categories=unsloth_categories,
                ).to(self.model.device)
            self.model.eval()
            with torch.no_grad():
                output = self.model.generate(
                    input_ids["input_ids"],
                    max_new_tokens=max_new_tokens,
                    pad_token_id=0,
                )
            generated_tokens = output[:, prompt_len:]
            result = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            lines = result.strip().splitlines()
            print("LINES", lines)
            category = ""
            label = ""

            for line in lines:
              line = line.strip().lower()

              if "unsafe" in line:
               label = "unsafe"

            if label == "unsafe":
             for line in lines:
                line = line.strip().lower()
                if line.startswith("s") and line[1:].isdigit():
                   category = line.upper()
                   break 
            return label, category, result
        except Exception as e:
            logger.exception("Error during prediction: %s", e)
            raise

