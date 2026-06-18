import os

api_key = os.environ.get('ANTHROPIC_API_KEY')

if not api_key:
    print("❌ Variable no encontrada")
    exit()

print("✅ API Key encontrada")
print(f"   Empieza con: {api_key[:15]}...")

try:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=20,
        messages=[{"role": "user", "content": "Say: Working!"}]
    )
    
    print(f"✅ API Response: {message.content[0].text}")
    print("\n🎉 Todo listo para audit")
    
except Exception as e:
    print(f"❌ Error: {e}")