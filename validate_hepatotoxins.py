"""
Test known hepatotoxins to assess false negative rate
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from predictor import ADMEToxPredictor

print("\n" + "="*75)
print("  HEPATOTOXIN VALIDATION TEST")
print("="*75)

# Known hepatotoxins with FDA warnings
test_compounds = {
    "Valproic Acid": {
        "smiles": "CCCC(CCC)C(=O)O",
        "status": "FDA Black Box - Hepatotoxin",
        "notes": "Toxic via 4-ene-VPA metabolite"
    },
    "Acetaminophen": {
        "smiles": "CC(=O)Nc1ccc(O)cc1",
        "status": "FDA Warning - Hepatotoxin at high doses",
        "notes": "Toxic via NAPQI metabolite"
    },
    "Isoniazid": {
        "smiles": "NNC(=O)c1ccncc1",
        "status": "FDA Black Box - Hepatotoxin",
        "notes": "Toxic via acetylhydrazine metabolite"
    },
    "Diclofenac": {
        "smiles": "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
        "status": "FDA Warning - Hepatotoxin",
        "notes": "Toxic via quinone imine metabolites"
    }
}

predictor = ADMEToxPredictor(models_dir="models", use_extended_descriptors=False)

results = []

for name, data in test_compounds.items():
    print(f"\n{'='*75}")
    print(f"  {name}")
    print(f"{'='*75}")
    print(f"Clinical Status: {data['status']}")
    print(f"Mechanism: {data['notes']}")
    
    try:
        result = predictor.predict(data['smiles'])
        hepa = result['predictions']['hepatotoxicity']
        prob = hepa['probability'] * 100
        risk = hepa['risk']
        
        print(f"\nModel Prediction:")
        print(f"  Hepatotoxicity Probability: {prob:.1f}%")
        print(f"  Risk Flag: {'HIGH RISK' if risk == 1 else 'LOW RISK'}")
        print(f"  Overall Status: {result['overall_risk']['level']}")
        
        if risk == 0:
            print(f"\n  ❌ FALSE NEGATIVE - Model missed known hepatotoxin")
            verdict = "FALSE NEGATIVE"
        else:
            print(f"\n  ✅ TRUE POSITIVE - Correctly identified")
            verdict = "CORRECT"
            
        results.append({
            'compound': name,
            'probability': prob,
            'risk_flag': risk,
            'verdict': verdict
        })
        
    except Exception as e:
        print(f"\n  ❌ ERROR: {str(e)}")
        results.append({
            'compound': name,
            'probability': None,
            'risk_flag': None,
            'verdict': "ERROR"
        })

print(f"\n{'='*75}")
print(f"  VALIDATION SUMMARY")
print(f"{'='*75}")

false_negatives = [r for r in results if r['verdict'] == 'FALSE NEGATIVE']
true_positives = [r for r in results if r['verdict'] == 'CORRECT']
errors = [r for r in results if r['verdict'] == 'ERROR']

print(f"\nTotal Compounds Tested: {len(results)}")
print(f"  ✅ Correctly Identified: {len(true_positives)}")
print(f"  ❌ False Negatives: {len(false_negatives)}")
print(f"  ⚠️  Errors: {len(errors)}")

if false_negatives:
    print(f"\nFalse Negative Details:")
    for r in false_negatives:
        print(f"  - {r['compound']}: {r['probability']:.1f}% (should be HIGH RISK)")

print(f"\n{'='*75}")
print(f"  CONCLUSION")
print(f"{'='*75}")

if len(false_negatives) > 0:
    print(f"""
⚠️  CRITICAL FINDING: {len(false_negatives)}/{len(results)} known hepatotoxins missed

ROOT CAUSE: These compounds are toxic via METABOLIC ACTIVATION
- Model analyzes parent structure only
- Does NOT predict metabolites or bioactivation
- This is a fundamental QSAR model limitation

IMPACT: False negative rate for metabolically-activated toxins is HIGH

SOLUTION REQUIRED: Phase 2 metabolite prediction module
""")
else:
    print(f"\n✅ All tested hepatotoxins correctly identified")

print(f"\n{'='*75}\n")
