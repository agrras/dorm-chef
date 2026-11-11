"""Model backends for the dorm-chef agent.

Three interchangeable providers, all exposing the same async interface:

    anthropic  Claude via the Anthropic API              (ANTHROPIC_API_KEY)
    azure_*    Your fine-tuned gpt-4.1-nano deployments  (AZURE_OPENAI_API_KEY)
               -> base / SFT / DPO from the CIS 5270 experiments
    demo       Offline scripted "model" so the UI + agent loop run with no keys

The agent never cares which one it is talking to.
"""
from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from typing import AsyncIterator, Protocol

# Exact system prompt used for every SFT/DPO training example in this repo.
REPO_SYSTEM = (
    "You are a recipe assistant for college students. Generate quick recipes (under 15 minutes) "
    "with 3-7 clear numbered steps, using only the ingredients provided."
)

# General-purpose models (Claude) get a little extra format guidance. The fine-tuned
# models were trained on REPO_SYSTEM alone, so they get it unchanged.
FORMAT_HINT = (
    "\n\nFormat: a bold title line with the total time, e.g. **Kimchi Fried Rice** (12 min), then "
    "3-7 numbered steps. Water, salt, pepper, cooking oil and butter are always allowed; any other "
    "ingredient must come from the provided list. If a diet or exclusion rule conflicts with an item "
    "in the list, silently skip that item. Output only the recipe."
)

Message = dict  # {"role": "user"|"assistant", "content": str}


class Provider(Protocol):
    id: str
    label: str
    is_real_model: bool
    system_prompt: str

    def available(self) -> bool: ...
    def stream(self, messages: list[Message], temperature: float = 0.7) -> AsyncIterator[str]: ...
    async def complete(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 400) -> str: ...


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------
@dataclass
class AnthropicProvider:
    id: str = "claude"
    label: str = "Claude"
    is_real_model: bool = True
    model: str = ""

    def __post_init__(self) -> None:
        self.model = self.model or os.getenv("DORMCHEF_CLAUDE_MODEL", "claude-sonnet-5-5")
        self.label = f"Claude ({self.model})"
        self.system_prompt = REPO_SYSTEM + FORMAT_HINT

    def available(self) -> bool:
        return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))

    def _client(self):
        from anthropic import AsyncAnthropic

        return AsyncAnthropic()

    async def stream(self, messages: list[Message], temperature: float = 0.7) -> AsyncIterator[str]:
        client = self._client()
        async with client.messages.stream(
            model=self.model,
            max_tokens=700,
            temperature=temperature,
            system=self.system_prompt,
            messages=messages,
        ) as s:
            async for text in s.text_stream:
                yield text

    async def complete(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 400) -> str:
        client = self._client()
        r = await client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")


# ---------------------------------------------------------------------------
# Azure OpenAI (the project's fine-tuned deployments)
# ---------------------------------------------------------------------------
@dataclass
class AzureProvider:
    id: str
    label: str
    deployment_env: str  # env var holding the deployment name
    is_real_model: bool = True

    def __post_init__(self) -> None:
        # Fine-tunes were trained on the bare repo prompt; don't change it.
        self.system_prompt = REPO_SYSTEM

    @property
    def deployment(self) -> str:
        return os.getenv(self.deployment_env, "")

    def available(self) -> bool:
        return bool(os.getenv("AZURE_OPENAI_API_KEY") and self.deployment)

    def _client(self):
        from openai import AsyncAzureOpenAI

        return AsyncAzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://cis-5270-team-14.openai.azure.com"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
        )

    async def stream(self, messages: list[Message], temperature: float = 0.7) -> AsyncIterator[str]:
        client = self._client()
        resp = await client.chat.completions.create(
            model=self.deployment,
            messages=[{"role": "system", "content": self.system_prompt}, *messages],
            temperature=temperature,
            max_completion_tokens=700,
            stream=True,
        )
        async for chunk in resp:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def complete(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 400) -> str:
        # Judge calls use the project's judge deployment (gpt-4.1-mini in the notebooks).
        client = self._client()
        r = await client.chat.completions.create(
            model=os.getenv("DORMCHEF_AZURE_JUDGE", "gpt-4.1-mini"),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        return r.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Demo (offline)
# ---------------------------------------------------------------------------
class DemoProvider:
    """A scripted stand-in so the app runs with no API keys.

    It is NOT a language model. Attempt 1 is deliberately sloppy (an invented
    ingredient, too many steps, or a banned item when the pantry allows it) so
    the verify -> revise loop is visible; once it sees verifier feedback it
    returns a clean recipe built from the pantry.
    """

    id = "demo"
    label = "Demo (offline, scripted — not a model)"
    is_real_model = False
    system_prompt = REPO_SYSTEM

    def available(self) -> bool:
        return True

    @staticmethod
    def _parse(messages: list[Message]) -> tuple[list[str], str]:
        first = messages[0]["content"]
        m = re.search(r"using these ingredients:\s*(.+?)\.(?:\s|$)", first, re.S)
        items = [i.strip().lower() for i in (m.group(1) if m else "").split(",") if i.strip()]
        return items, first

    async def stream(self, messages: list[Message], temperature: float = 0.7) -> AsyncIterator[str]:
        pantry, prompt = self._parse(messages)
        revised = len(messages) > 1
        text = self._recipe(pantry, prompt.lower(), sloppy=not revised)
        for i in range(0, len(text), 6):
            await asyncio.sleep(0.012)
            yield text[i : i + 6]

    async def complete(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 400) -> str:
        return '{"constraint_followed": true, "reason": "demo judge"}'

    # -- recipe builder -----------------------------------------------------
    def _recipe(self, pantry: list[str], prompt: str, sloppy: bool) -> str:
        banned: set[str] = set()
        if "vegan" in prompt:
            banned |= {"chicken", "egg", "eggs", "cheese", "yogurt", "milk", "tuna", "shrimp", "paneer", "butter"}
        elif "vegetarian" in prompt:
            banned |= {"chicken", "tuna", "shrimp", "beef", "bacon", "ham", "salmon"}
        elif "jain" in prompt:
            banned |= {"chicken", "egg", "eggs", "onion", "garlic", "potato", "carrot", "tuna", "shrimp"}
        elif "dairy-free" in prompt:
            banned |= {"cheese", "yogurt", "milk", "cream", "butter", "paneer"}
        m = re.search(r"do not include ([a-z ]+?)\.", prompt)
        if m:
            banned.add(m.group(1).strip())

        usable = [p for p in pantry if not any(b == p or b in p.split() for b in banned)] or pantry[:3]
        base = usable[:5]
        if len(base) < 3:
            base = (base + ["water", "salt", "pepper"])[:3]
        title_item = base[0].title()

        steps = [
            f"Prep the {base[0]}: rinse and chop it into small, even pieces.",
            f"Heat a little cooking oil in a pan over medium heat for 1 minute.",
            f"Add the {base[1] if len(base) > 1 else base[0]} and cook for 3-4 minutes, stirring often.",
            f"Stir in the {base[2] if len(base) > 2 else base[0]} and season with salt and pepper.",
            f"Add the {base[3] if len(base) > 3 else base[0]} and cook 2 more minutes until everything is hot.",
            "Taste, adjust the salt, and serve straight from the pan.",
        ]
        minutes = 11

        if sloppy:
            # Intentionally flawed first draft to show the repair loop.
            invented = next((f for f in ("parmesan", "basil", "lemon", "soy sauce", "cilantro") if f not in " ".join(pantry)), None)
            slip = next((b for b in banned if any(b in p for p in pantry)), None)
            if slip:
                steps[2] = f"Add the {slip} and cook for 3-4 minutes, stirring often."
            elif invented:
                steps[5] = f"Finish with a squeeze of {invented} and serve straight from the pan."
            else:
                steps = steps + ["Wash up.", "Let it cool.", "Enjoy."]
            minutes = 12 if not (slip or invented) else 10

        body = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
        return f"**Quick {title_item} Skillet** ({minutes} min)\n\n{body}"


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
def build_registry() -> dict[str, Provider]:
    providers: list[Provider] = [
        AnthropicProvider(),
        AzureProvider("azure_base", "gpt-4.1-nano · base", "DORMCHEF_AZURE_BASE"),
        AzureProvider("azure_sft", "gpt-4.1-nano · SFT fine-tune", "DORMCHEF_AZURE_SFT"),
        AzureProvider("azure_dpo", "gpt-4.1-nano · DPO fine-tune", "DORMCHEF_AZURE_DPO"),
        DemoProvider(),
    ]
    return {p.id: p for p in providers}
