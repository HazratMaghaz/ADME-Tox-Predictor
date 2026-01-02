import sys
sys.path.insert(0, 'src')
from predictor import ADMEToxPredictor

print("Loading predictor...")
predictor = ADMEToxPredictor(models_dir='models')

print("Testing Clozapine...")
result = predictor.predict("CN1CCN(CC1)C2=NC3=CC=CC=C3NC4=C2C=C(C=C4)Cl")

hep = result['predictions']['hepatotoxicity']
print(f"Result: {hep['probability']*100:.1f}% ({hep['risk']})")
print(f"Metabolites: {hep.get('metabolites_generated', 0)}")
print("Done!")
