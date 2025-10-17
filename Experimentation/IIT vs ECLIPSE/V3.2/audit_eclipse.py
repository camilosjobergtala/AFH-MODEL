import os
import sys
from pathlib import Path

# Verificar API key
api_key = os.environ.get('ANTHROPIC_API_KEY')
if not api_key:
    print("❌ API key no encontrada en variable de entorno")
    exit()

print("="*80)
print("🤖 LLM CODE AUDIT - ECLIPSE v3.2.0")
print("="*80)

# Path al código
code_file = Path(r"G:\Mi unidad\AFH\GITHUB\AFH-MODEL\Experimentation\IIT vs ECLIPSE\V3.2\IIT vs ECLIPSE v3.2.py")

if not code_file.exists():
    print(f"❌ Archivo no encontrado: {code_file}")
    exit()

print(f"\n📄 Archivo: {code_file.name}")
print(f"   Tamaño: {code_file.stat().st_size / 1024:.1f} KB")

# Leer código
with open(code_file, 'r', encoding='utf-8') as f:
    code_content = f.read()

print(f"   Líneas: {len(code_content.splitlines())}")

# Preparar prompt para Claude
audit_prompt = f"""You are a code auditor for scientific research. Analyze this Python code for protocol violations in a preregistered study.

The code implements ECLIPSE framework for falsifying Integrated Information Theory. Check for:

1. **Holdout Data Access**: Any access to holdout/test data before validation stage
2. **Threshold Manipulation**: Multiple threshold adjustments or p-hacking
3. **Multiple Testing**: Statistical tests without correction (Bonferroni/FDR)
4. **Data Snooping**: Looking at validation data during development
5. **Post-hoc Changes**: Modifying criteria after seeing results

Identifiers to watch for: 'holdout', 'test', 'holdout_data', 'holdout_subjects'

CODE TO AUDIT:
```python
{code_content[:15000]}  # Primeros 15K caracteres para no exceder límite
```

Provide:
1. Overall adherence score (0-100)
2. List of violations (if any) with severity (critical/high/medium/low)
3. Risk level (low/medium/high/critical)
4. Recommendation

Format as JSON:
{{
  "adherence_score": 95,
  "violations": [
    {{"severity": "medium", "description": "...", "line": 123}}
  ],
  "risk_level": "low",
  "passed": true,
  "summary": "..."
}}
"""

print("\n🔍 Enviando a Claude para análisis...")

try:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[
            {"role": "user", "content": audit_prompt}
        ]
    )
    
    audit_response = message.content[0].text
    
    print("\n" + "="*80)
    print("📊 AUDIT RESULTS")
    print("="*80)
    print(audit_response)
    
    # Guardar reporte
    report_file = Path("./eclipse_results_v3_2/IIT_v3_2_2ch_natural_LLM_AUDIT.txt")
    report_file.parent.mkdir(exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("LLM-POWERED CODE AUDIT - ECLIPSE v3.2.0\n")
        f.write("="*80 + "\n\n")
        f.write(f"File: {code_file.name}\n")
        f.write(f"Date: {Path(__file__).stat().st_mtime}\n\n")
        f.write(audit_response)
    
    print(f"\n✅ Reporte guardado: {report_file}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)