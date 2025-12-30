#!/usr/bin/env python3
"""Quick test to verify Gemini integration works."""
import asyncio
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

async def test_gemini():
    """Test basic Gemini generation."""
    from src.llm import GeminiProvider, ModelRouter, TaskType

    print("Testing Gemini Integration")
    print("=" * 40)

    # Check API key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not set in .env file")
        print("Please paste your API key after GOOGLE_API_KEY= in .env")
        return False

    print(f"API Key: {api_key[:8]}...{api_key[-4:]}")

    try:
        # Test direct provider
        print("\n1. Testing GeminiProvider directly...")
        provider = GeminiProvider()

        response = await provider.generate(
            "What is the capital of France? Answer in one word.",
            model="gemini-2.5-flash",
            temperature=0.1,
            max_tokens=100,
        )
        print(f"   Model: {response.model}")
        print(f"   Response: {response.content}")
        print(f"   Usage: {response.usage}")

        # Test model router
        print("\n2. Testing ModelRouter with task types...")
        router = ModelRouter({"google": provider})

        # Fast response
        fast_response = await router.generate(
            "What is 2+2?",
            task_type=TaskType.FAST_RESPONSE,
        )
        print(f"   Fast task ({fast_response.model}): {fast_response.content.strip()}")

        # Code generation
        code_response = await router.generate(
            "Write a Python one-liner to reverse a string",
            task_type=TaskType.CODE_GENERATION,
        )
        print(f"   Code task ({code_response.model}): {code_response.content.strip()[:80]}...")

        print("\n" + "=" * 40)
        print("SUCCESS! Gemini integration is working!")
        return True

    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_gemini())
    exit(0 if success else 1)
