"""Check which Azure OpenAI deployments still respond with your key.

    export AZURE_OPENAI_API_KEY=...        (never commit this)
    python -m app.check_azure

Tries one tiny chat call per deployment name (from .env.example / env vars) and
also lists the resource's fine-tuning jobs, which a plain API key is allowed to read.
"""
import os
import sys

from openai import AzureOpenAI

ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://cis-5270-team-14.openai.azure.com")
DEPLOYMENTS = {
    "base": os.getenv("DORMCHEF_AZURE_BASE", "gpt-4.1-nano"),
    "sft": os.getenv("DORMCHEF_AZURE_SFT", "1-nano-2025-04-14-supervised-fine-tuning-"),
    "dpo": os.getenv("DORMCHEF_AZURE_DPO", "gpt-4-1-nano-2025-04-14-ft-e456d7607fb3499db3091af24f4a60a5"),
    "dpo (harder run)": "1-nano-2025-04-14-1",
    "judge": os.getenv("DORMCHEF_AZURE_JUDGE", "gpt-4.1-mini"),
}


def main() -> int:
    key = os.getenv("AZURE_OPENAI_API_KEY")
    if not key:
        print("Set AZURE_OPENAI_API_KEY first.")
        return 1
    c = AzureOpenAI(api_key=key, azure_endpoint=ENDPOINT, api_version="2025-04-01-preview")
    print(f"Endpoint: {ENDPOINT}\n")
    for label, name in DEPLOYMENTS.items():
        try:
            r = c.chat.completions.create(model=name, messages=[{"role": "user", "content": "Say ok"}], max_completion_tokens=5)
            print(f"  OK       {label:18} {name}  -> {r.choices[0].message.content!r}")
        except Exception as e:
            msg = str(e).replace("\n", " ")[:130]
            print(f"  FAILED   {label:18} {name}\n           {msg}")
    print("\nFine-tuning jobs visible to this key:")
    try:
        for j in c.fine_tuning.jobs.list(limit=20):
            print(f"  {j.id}  {j.status:10} {j.fine_tuned_model or '-'}")
    except Exception as e:
        print("  could not list:", str(e)[:150])
    return 0


if __name__ == "__main__":
    sys.exit(main())
