"""
Client factory for creating framework-specific clients.
This module standardizes client creation across different frameworks.
"""

from typing import Any, Dict
from .framework_types import Framework


class ClientFactory:
    """
    Factory class for creating framework-specific clients.
    Standardizes client creation and configuration across frameworks.
    """

    @staticmethod
    def create_client(framework: Framework, llm_config: Dict[str, Any]) -> Any:
        if framework == Framework.AUTOGEN:
            return ClientFactory._create_autogen_client(llm_config)
        elif framework == Framework.LANGGRAPH:
            return ClientFactory._create_langgraph_client(llm_config)
        elif framework == Framework.OPENAI_AGENTS:
            return ClientFactory._create_openai_agents_client(llm_config)
        else:
            raise ValueError(f"Unsupported framework: {framework}")

    @staticmethod
    def _create_autogen_client(llm_config: Dict[str, Any]) -> Any:
        provider = llm_config.get("provider", "azure")
        if provider == "openai_compatible":
            from autogen_ext.models.openai import OpenAIChatCompletionClient
            return OpenAIChatCompletionClient(
                model=llm_config["model"],
                api_key=llm_config["api_key"],
                base_url=llm_config["base_url"],
                temperature=llm_config["temperature"],
                model_info=llm_config["model_info"],
            )

        from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
        return AzureOpenAIChatCompletionClient(**llm_config)

    @staticmethod
    def _create_langgraph_client(llm_config: Dict[str, Any]) -> Any:
        provider = llm_config.get("provider", "azure")
        if provider == "openai_compatible":
            from langchain_openai.chat_models import ChatOpenAI
            return ChatOpenAI(
                model=llm_config["model"],
                temperature=llm_config["temperature"],
                api_key=llm_config["api_key"],
                base_url=llm_config["base_url"],
                timeout=llm_config.get("timeout", 120.0),
                max_retries=llm_config.get("max_retries", 2),
            )

        from langchain_openai.chat_models import AzureChatOpenAI

        langgraph_config = {
            "name": llm_config["model"],
            "temperature": llm_config["temperature"],
            "azure_deployment": llm_config["azure_deployment"],
            "azure_endpoint": llm_config["azure_endpoint"],
            "api_version": llm_config["api_version"],
            "azure_ad_token_provider": llm_config["azure_ad_token_provider"],
        }

        return AzureChatOpenAI(**langgraph_config)

    @staticmethod
    def _create_openai_agents_client(llm_config: Dict[str, Any]) -> Any:
        import os
        from agents import set_default_openai_api, set_default_openai_client, set_tracing_disabled

        provider = llm_config.get("provider", "azure")
        if provider == "openai_compatible":
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=llm_config["api_key"],
                base_url=llm_config["base_url"],
                timeout=llm_config.get("timeout", 120.0),
                max_retries=llm_config.get("max_retries", 2),
            )

            set_default_openai_api("chat_completions")
            set_default_openai_client(client, False)
            set_tracing_disabled(disabled=True)

            os.environ["OPENAI_API_KEY"] = llm_config["api_key"]
            os.environ["OPENAI_BASE_URL"] = llm_config["base_url"]
            return client

        from openai import AsyncAzureOpenAI

        openai_agents_config = {
            "azure_endpoint": llm_config["azure_endpoint"],
            "azure_ad_token_provider": llm_config["azure_ad_token_provider"],
            "api_version": llm_config["api_version"],
        }

        client = AsyncAzureOpenAI(**openai_agents_config)

        set_default_openai_api("chat_completions")
        set_default_openai_client(client, False)
        set_tracing_disabled(disabled=True)

        os.environ["OPENAI_API_TYPE"] = "azure"
        os.environ["OPENAI_API_BASE"] = llm_config["azure_endpoint"]
        os.environ["OPENAI_API_VERSION"] = llm_config["api_version"]

        return client

    @staticmethod
    def bind_tools_for_framework(client: Any, framework: Framework, tools: list,
                                   parallel_tool_calls: bool = True) -> Any:
        if framework == Framework.LANGGRAPH:
            return client.bind_tools(tools, parallel_tool_calls=parallel_tool_calls)
        elif framework == Framework.AUTOGEN:
            return client
        elif framework == Framework.OPENAI_AGENTS:
            return client
        else:
            return client
