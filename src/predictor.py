"""
ADME-Tox Prediction Module for Phase 2
Loads models and makes predictions from SMILES
"""

import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

from smiles_processor import SMILESProcessor


class ADMEToxPredictor:
    """Main predictor class that loads models and makes predictions"""
    
    def __init__(self, models_dir: str = "../models", use_extended_descriptors: bool = False):
        """
        Initialize predictor by loading all models
        
        Args:
            models_dir: Path to directory containing trained models
            use_extended_descriptors: Use extended feature set (for future models)
        """
        self.models_dir = Path(models_dir)
        self.processor = SMILESProcessor(use_extended_descriptors=use_extended_descriptors)
        self.use_extended_descriptors = use_extended_descriptors
        self.models = {}
        self.scaler = None
        
        self._load_models()
    
    def _load_models(self):
        """Load all trained models"""
        try:
            # Load scaler
            scaler_path = self.models_dir / 'feature_scaler.pkl'
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                print("[INFO] Loaded feature scaler")
            else:
                print("[WARN] Feature scaler not found")
            
            # Load prediction models
            model_files = {
                'herg': 'hERG_Blocker_model.pkl',
                'solubility': 'Solubility_model.pkl',
                'mutagenicity': 'Mutagenicity_model.pkl',
                'carcinogenicity': 'Carcinogenicity_model.pkl',
                'hepatotoxicity': 'Hepatotoxicity_model.pkl'
            }
            
            for model_name, filename in model_files.items():
                model_path = self.models_dir / filename
                if model_path.exists():
                    self.models[model_name] = joblib.load(model_path)
                    print(f"[INFO] Loaded {model_name} model")
                else:
                    print(f"[WARN] {model_name} model not found")
            
            print(f"\n[SUCCESS] Loaded {len(self.models)}/5 models successfully")
            
        except Exception as e:
            print(f"[ERROR] Error loading models: {str(e)}")
            raise
    
    def predict(self, smiles: str) -> Dict:
        """
        Make predictions for all ADME-Tox endpoints from SMILES
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            Dictionary with all predictions and probabilities
        """
        # Validate SMILES
        is_valid, mol = self.processor.validate_smiles(smiles)
        if not is_valid:
            return {
                'error': 'Invalid SMILES string',
                'smiles': smiles
            }
        
        # Calculate descriptors
        descriptors = self.processor.calculate_descriptors(smiles)
        if descriptors is None:
            return {
                'error': 'Failed to calculate molecular descriptors',
                'smiles': smiles
            }
        
        # Get basic properties
        properties = self.processor.get_basic_properties(smiles)
        lipinski = self.processor.check_lipinski_rules(smiles)
        
        # Phase-1 models were trained on 7 basic features; extended descriptors are for display/future models.
        basic_feature_names = [
            'MolWt', 'LogP', 'TPSA', 'NumHDonors',
            'NumHAcceptors', 'NumRotatableBonds', 'NumAromaticRings'
        ]

        basic_descriptors = {k: descriptors[k] for k in basic_feature_names if k in descriptors}
        mordred_descriptors = {k: v for k, v in descriptors.items() if k.startswith('Mordred_')}
        rdkit_extended_descriptors = {
            k: v for k, v in descriptors.items()
            if (k not in basic_feature_names) and (not k.startswith('Mordred_'))
        }

        # Choose feature set for the *model* based on what the scaler/models expect.
        expected_n_features: Optional[int] = None
        if self.scaler is not None and hasattr(self.scaler, 'n_features_in_'):
            expected_n_features = int(getattr(self.scaler, 'n_features_in_'))
        else:
            for model in self.models.values():
                if hasattr(model, 'n_features_in_'):
                    expected_n_features = int(getattr(model, 'n_features_in_'))
                    break

        if expected_n_features is None or expected_n_features == len(basic_feature_names):
            model_feature_names = basic_feature_names
        else:
            # Future-proofing: if models/scaler were trained on an extended feature set.
            model_feature_names = list(descriptors.keys())
            if expected_n_features != len(model_feature_names):
                raise ValueError(
                    f"Feature count mismatch: model/scaler expects {expected_n_features} features, "
                    f"but descriptor generator produced {len(model_feature_names)}. "
                    "Either retrain models with the same descriptor set or disable extended descriptors."
                )

        # Prepare features for ML models (keep strict ordering)
        missing = [f for f in model_feature_names if f not in descriptors]
        if missing:
            raise ValueError(f"Missing required descriptors for model: {missing[:10]}" + ("..." if len(missing) > 10 else ""))

        feature_vector = np.array([[descriptors[fname] for fname in model_feature_names]])
        
        # Scale features
        if self.scaler:
            feature_vector_scaled = self.scaler.transform(feature_vector)
        else:
            feature_vector_scaled = feature_vector
        
        # Make predictions
        predictions = {
            'input': {
                'smiles': smiles,
                'canonical_smiles': properties['canonical_smiles'],
                'molecular_formula': properties['molecular_formula'],
                'molecular_weight': round(properties['molecular_weight'], 2)
            },
            # Keep `descriptors` as the original 7 for backward compatibility
            'descriptors': {k: round(v, 2) for k, v in basic_descriptors.items()},
            # RDKit extra descriptors (non-Mordred)
            'descriptors_rdkit_extended': {k: round(v, 4) for k, v in rdkit_extended_descriptors.items()} if self.use_extended_descriptors else {},
            # Mordred descriptors (prefixed with Mordred_)
            'descriptors_extended': {k: round(v, 6) for k, v in mordred_descriptors.items()} if self.use_extended_descriptors else {},
            # Model feature count (Phase-1 models should remain 7)
            'num_features': len(model_feature_names),
            # Total descriptor counts produced
            'num_descriptors_total': len(descriptors),
            'num_rdkit_extended': len(rdkit_extended_descriptors) if self.use_extended_descriptors else 0,
            'num_mordred': len(mordred_descriptors) if self.use_extended_descriptors else 0,
            'lipinski_compliance': lipinski,
            'predictions': {}
        }
        
        # Solubility (Regression)
        if 'solubility' in self.models:
            sol_pred = self.models['solubility'].predict(feature_vector_scaled)[0]
            predictions['predictions']['solubility'] = {
                'logS': round(sol_pred, 2),
                'interpretation': self._interpret_solubility(sol_pred)
            }
        
        # hERG Blocker (Classification)
        if 'herg' in self.models:
            herg_pred = self.models['herg'].predict(feature_vector_scaled)[0]
            herg_proba = self.models['herg'].predict_proba(feature_vector_scaled)[0]
            predictions['predictions']['herg_blocker'] = {
                'risk': int(herg_pred),
                'probability': round(herg_proba[1], 3),
                'risk_level': self._get_risk_level(herg_proba[1]),
                'interpretation': 'High cardiotoxicity risk' if herg_pred == 1 else 'Low cardiotoxicity risk'
            }
        
        # Mutagenicity (Classification)
        if 'mutagenicity' in self.models:
            mut_pred = self.models['mutagenicity'].predict(feature_vector_scaled)[0]
            mut_proba = self.models['mutagenicity'].predict_proba(feature_vector_scaled)[0]
            predictions['predictions']['mutagenicity'] = {
                'risk': int(mut_pred),
                'probability': round(mut_proba[1], 3),
                'risk_level': self._get_risk_level(mut_proba[1]),
                'interpretation': 'Mutagenic (Ames positive)' if mut_pred == 1 else 'Non-mutagenic (Ames negative)'
            }
        
        # Carcinogenicity (Classification)
        if 'carcinogenicity' in self.models:
            carc_pred = self.models['carcinogenicity'].predict(feature_vector_scaled)[0]
            carc_proba = self.models['carcinogenicity'].predict_proba(feature_vector_scaled)[0]
            predictions['predictions']['carcinogenicity'] = {
                'risk': int(carc_pred),
                'probability': round(carc_proba[1], 3),
                'risk_level': self._get_risk_level(carc_proba[1]),
                'interpretation': 'Carcinogenic risk' if carc_pred == 1 else 'No carcinogenic risk'
            }
        
        # Hepatotoxicity (Classification)
        if 'hepatotoxicity' in self.models:
            hep_pred = self.models['hepatotoxicity'].predict(feature_vector_scaled)[0]
            hep_proba = self.models['hepatotoxicity'].predict_proba(feature_vector_scaled)[0]
            predictions['predictions']['hepatotoxicity'] = {
                'risk': int(hep_pred),
                'probability': round(hep_proba[1], 3),
                'risk_level': self._get_risk_level(hep_proba[1]),
                'interpretation': 'Hepatotoxic (liver toxic)' if hep_pred == 1 else 'Non-hepatotoxic'
            }
        
        # Overall risk score
        predictions['overall_risk'] = self._calculate_overall_risk(predictions['predictions'])
        
        return predictions
    
    def _interpret_solubility(self, log_s: float) -> str:
        """Interpret solubility value"""
        if log_s < -4:
            return "Poorly soluble"
        elif log_s < -2:
            return "Moderately soluble"
        else:
            return "Highly soluble"
    
    def _get_risk_level(self, probability: float) -> str:
        """Convert probability to risk level"""
        if probability < 0.3:
            return "Low"
        elif probability < 0.7:
            return "Medium"
        else:
            return "High"
    
    def _calculate_overall_risk(self, predictions: Dict) -> Dict:
        """Calculate overall toxicity risk score using MAX-POOLING (worst-case)"""
        max_risk_score = 0
        max_risk_endpoint = None
        total_endpoints = 0
        flagged_count = 0
        
        # Find maximum probability across all toxicity endpoints
        for endpoint, pred_data in predictions.items():
            if endpoint != 'solubility' and 'probability' in pred_data:
                total_endpoints += 1
                prob_percent = pred_data['probability'] * 100
                
                if pred_data.get('risk') == 1:
                    flagged_count += 1
                
                if prob_percent > max_risk_score:
                    max_risk_score = prob_percent
                    max_risk_endpoint = endpoint
        
        if total_endpoints == 0:
            return {'score': 0, 'level': 'Unknown', 'color': 'gray', 'max_endpoint': None, 'flagged_endpoints': 0, 'total_endpoints': 0}
        
        # Traffic light thresholds based on max score
        if max_risk_score >= 80:
            level = "DANGER 🔴"
            color = "red"
        elif max_risk_score >= 40:
            level = "WARNING 🟡"
            color = "orange"
        else:
            level = "SAFE 🟢"
            color = "green"
        
        return {
            'score': round(max_risk_score, 1),
            'level': level,
            'color': color,
            'max_endpoint': max_risk_endpoint.replace('_', ' ').title() if max_risk_endpoint else 'N/A',
            'flagged_endpoints': flagged_count,
            'total_endpoints': total_endpoints
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize predictor
    predictor = ADMEToxPredictor(models_dir="../models")
    
    # Test molecules
    test_molecules = {
        'Aspirin': 'CC(=O)Oc1ccccc1C(=O)O',
        'Caffeine': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
        'Ibuprofen': 'CC(C)Cc1ccc(cc1)C(C)C(O)=O'
    }
    
    print("\n" + "="*80)
    print("TESTING ADME-TOX PREDICTOR")
    print("="*80)
    
    for name, smiles in test_molecules.items():
        print(f"\n{'='*60}")
        print(f"Molecule: {name}")
        print(f"SMILES: {smiles}")
        print("-"*60)
        
        result = predictor.predict(smiles)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"Formula: {result['input']['molecular_formula']}")
            print(f"MW: {result['input']['molecular_weight']} g/mol")
            print(f"Lipinski: {'✓ Pass' if result['lipinski_compliance']['compliant'] else '✗ Fail'}")
            print(f"\nOverall Risk: {result['overall_risk']['level']} ({result['overall_risk']['score']}%)")
            
            print("\nEndpoint Predictions:")
            for endpoint, pred in result['predictions'].items():
                print(f"  • {endpoint.title()}: {pred.get('interpretation', pred)}")
