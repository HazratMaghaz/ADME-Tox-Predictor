"""Quick test to verify MAX-POOLING is working"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from predictor import ADMEToxPredictor

print("\n🧪 TESTING MAX-POOLING RISK CALCULATION\n")
print("="*60)

predictor = ADMEToxPredictor(models_dir="models", use_extended_descriptors=False)

# Test with Caffeine
smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
result = predictor.predict(smiles)

print(f"\nInput: Caffeine")
print(f"SMILES: {smiles}\n")

print("Individual Endpoint Probabilities:")
endpoint_probs = []
for endpoint, pred in result['predictions'].items():
    if endpoint != 'solubility' and 'probability' in pred:
        prob = pred['probability'] * 100
        endpoint_probs.append((endpoint, prob))
        print(f"  {endpoint.replace('_', ' ').title():20s} = {prob:.1f}%")

max_prob = max([p[1] for p in endpoint_probs])
max_ep = [p[0] for p in endpoint_probs if p[1] == max_prob][0]
avg_prob = sum([p[1] for p in endpoint_probs]) / len(endpoint_probs)

print(f"\n{'='*60}")
print(f"Average (Old Logic):    {avg_prob:.1f}%")
print(f"Maximum (New Logic):    {max_prob:.1f}%")
print(f"{'='*60}")

print(f"\nActual Overall Risk from System:")
print(f"  Score:    {result['overall_risk']['score']}%")
print(f"  Endpoint: {result['overall_risk'].get('max_endpoint', 'N/A')}")
print(f"  Status:   {result['overall_risk']['level']}")

if abs(result['overall_risk']['score'] - max_prob) < 0.1:
    print(f"\n✅ SUCCESS: Using MAX-POOLING (worst-case)")
    print(f"   Risk = {result['overall_risk']['score']}% from {result['overall_risk'].get('max_endpoint')}")
else:
    print(f"\n❌ FAILED: Not using MAX-POOLING")
    print(f"   Expected: {max_prob}%, Got: {result['overall_risk']['score']}%")

print("\n" + "="*60 + "\n")
