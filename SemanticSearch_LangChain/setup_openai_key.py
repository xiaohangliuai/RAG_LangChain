"""
Helper script to set OpenAI API key for the current session
"""
import os
import sys

print("=" * 60)
print("OpenAI API Key Setup")
print("=" * 60)
print("\nThis will set your OpenAI API key for the current Python session.")
print("Get your API key from: https://platform.openai.com/api-keys\n")

# Check if already set
if os.environ.get("OPENAI_API_KEY"):
    current_key = os.environ.get("OPENAI_API_KEY")
    masked_key = current_key[:7] + "..." + current_key[-4:] if len(current_key) > 11 else "***"
    print(f"✓ OPENAI_API_KEY is already set: {masked_key}\n")
    response = input("Do you want to update it? (y/n): ").strip().lower()
    if response != 'y':
        print("Keeping existing key.")
        sys.exit(0)

# Get new API key
api_key = input("\nEnter your OpenAI API key: ").strip()

if not api_key:
    print("❌ No API key entered. Exiting.")
    sys.exit(1)

if not api_key.startswith("sk-"):
    print("⚠️  Warning: OpenAI API keys typically start with 'sk-'")
    response = input("Continue anyway? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        sys.exit(1)

# Set the environment variable
os.environ["OPENAI_API_KEY"] = api_key

print("\n✅ OpenAI API key set successfully!")
print("\nTo use it permanently, add this to your system environment variables:")
print(f"   Windows (PowerShell): $env:OPENAI_API_KEY=\"{api_key[:7]}...\"")
print(f"   Windows (CMD): set OPENAI_API_KEY={api_key[:7]}...")
print(f"   Linux/Mac: export OPENAI_API_KEY=\"{api_key[:7]}...\"")
print("\nOr create a .env file in this directory with:")
print(f"   OPENAI_API_KEY={api_key[:7]}...")

