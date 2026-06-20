from langchain_core.prompts.chat import (
    HumanMessage,
    SystemMessage,
    AIMessage,
)


class ChatHistoryFormatter:
    """Centralized chat history formatter for different model types."""
    
    @staticmethod
    def format_chat_history(chat_history, model_name: str) -> str:
        """Format the chat history for different model types.
        
        Args:
            chat_history: The chat history object containing messages
            model_name: The name of the model to format for
            
        Returns:
            str: Formatted chat history string
        """
        formatted_history = ""
        for message in chat_history.messages:
            if isinstance(message, SystemMessage):
                role = "system"
            elif isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            else:
                continue  # Skip unknown message types
                
            content = message.content
            if not isinstance(content, str):
                content = str(content)

            if "gpt" in model_name.lower():
                formatted_history += f"{role}\n{content}\n"
            elif model_name == "WhiteRabbit-Qwen":
                formatted_history += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            elif model_name == "WhiteRabbit-LLama":
                formatted_history += f"<|start_header_id|>{role}<|end_header_id|>{content}<|eot_id|>"
            else:
                # Default formatting for unknown models
                formatted_history += f"{role}\n{content}\n"

        return formatted_history