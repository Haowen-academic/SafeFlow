"""
ARIA (Agent Risk Identification Assessment) evaluator.
"""

from ....logger import logger
from ..prompts.aria_prompt import ARIA_PROMPT


def aria(logs: str, framework: str = None) -> str:
    """
    ARIA evaluation.
    """
    from ...clients.openai_compatible import get_openai_client_sync, get_llm_config

    system_prompt = ARIA_PROMPT
    prompt = f""" The log content is:\n{logs}\n"""

    try:
        client = get_openai_client_sync()
        config = get_llm_config()
        model_name = config.get("azure_deployment") or config["model"]

        response = client.chat.completions.create(
            model=model_name,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        resp = response.choices[0].message.content
        return resp.split("<answer>")[1].split("</answer>")[0].strip()
    except Exception as e:
        logger.error(f"ARIA evaluation failed: {e}")
        return "3"
