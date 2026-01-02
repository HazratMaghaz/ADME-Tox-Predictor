"""
Metabolic Activation Risk Detection Module
Identifies compounds with bioactivation liability (Phase 2 metabolism risk)
"""

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from typing import Dict, List, Tuple

class MetabolismRiskDetector:
    """Detect metabolic activation patterns and bioactivation risk"""
    
    def __init__(self):
        # SMARTS patterns for metabolic liability
        self.reactive_metabolite_patterns = {
            'quinone': '[#6]1:[#6]:[#6]:[#6](=[O]):[#6](=[O]):[#6]:1',
            'epoxide': 'C1OC1',
            'aldehyde': '[CX3H1](=O)[#6]',
            'acyl_halide': '[CX3](=[OX1])[F,Cl,Br,I]',
            'isocyanate': 'N=C=O',
            'michael_acceptor': '[CX3]=[CX3]-[CX3]=[OX1]',
            'aromatic_amine': 'c[NX3;H2,H1]',
            'nitro_aromatic': 'c[N+](=O)[O-]',
            'hydrazine': '[NX3][NX3]',
            'thiophene': 'c1ccsc1',  # Can form reactive epoxides
            'furan': 'c1ccoc1',  # Can form reactive epoxides
            'aniline': 'c1ccccc1N',  # Can form reactive quinone imines
            'phenol': 'c1ccccc1O',  # Can form reactive quinones
            'carboxylic_acid': '[CX3](=O)[OX2H1]',  # Can form reactive acyl glucuronides
            'terminal_alkene': 'C=C[CX4]',  # Can form reactive epoxides
            'aliphatic_hydroxyl': '[CX4][OX2H]',  # Can be oxidized
            'nitrogen_mustard': '[N]([CH2][CH2][Cl])[CH2][CH2][Cl]',  # Alkylating agents (cyclophosphamide, ifosfamide)
            'alkyl_chloride': '[CX4][Cl]',  # Can form reactive carbocations
            'phosphoramide': 'P(=O)(N)(N)',  # Phosphoramide chemotherapy agents
        }
        
        # Known hepatotoxic substructures (after metabolism)
        self.hepatotoxic_patterns = {
            'valproic_acid_like': '[CX4]([CX4])([CX4])[CX3](=O)[OX2H]',  # Branched carboxylic acid
            'acetaminophen_like': '[#6]1:[#6]:[#6](O):[#6]:[#6]:[#6]:1NC(=O)[CH3]',
            'isoniazid_like': '[#6]1:[#6]:[#6]:[#7]:[#6]:[#6]:1C(=O)NN',
            'nsaid_like': 'c1ccccc1CC(=O)O',  # Arylacetic acid
            'cyclophosphamide_like': 'P(=O)(N1CCCO1)N([CH2][CH2]Cl)[CH2][CH2]Cl',  # Oxazaphosphorine nitrogen mustards
            'alkylating_agent': '[N,O,S]([CH2][CH2][Cl,Br,I])([CH2][CH2][Cl,Br,I])',  # Bis-haloalkyl compounds (DNA alkylators)
        }
        
        # Compile SMARTS patterns
        self.reactive_patterns = {name: Chem.MolFromSmarts(smarts) 
                                  for name, smarts in self.reactive_metabolite_patterns.items()}
        self.hepatotoxic_patterns_mol = {name: Chem.MolFromSmarts(smarts)
                                         for name, smarts in self.hepatotoxic_patterns.items()}
    
    def detect_metabolic_liability(self, smiles: str) -> Dict:
        """
        Detect metabolic activation risk factors
        Returns dict with risk score and detected patterns
        """
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return {'error': 'Invalid SMILES'}
        
        detected_alerts = []
        risk_score = 0
        
        # Check for reactive metabolite-forming groups
        for name, pattern in self.reactive_patterns.items():
            if pattern and mol.HasSubstructMatch(pattern):
                # Higher risk for alkylating agents and nitrogen mustards
                if name in ['nitrogen_mustard', 'alkyl_chloride']:
                    risk_contribution = 25  # VERY HIGH RISK - DNA alkylators
                else:
                    risk_contribution = 10
                    
                detected_alerts.append({
                    'type': 'reactive_metabolite_precursor',
                    'pattern': name,
                    'risk_contribution': risk_contribution
                })
                risk_score += risk_contribution
        
        # Check for known hepatotoxic scaffolds (HIGHEST PRIORITY)
        for name, pattern in self.hepatotoxic_patterns_mol.items():
            if pattern and mol.HasSubstructMatch(pattern):
                # Maximum risk for alkylating agents (chemotherapy drugs)
                if 'alkylating' in name or 'cyclophosphamide' in name:
                    risk_contribution = 50  # EXTREME RISK
                else:
                    risk_contribution = 30
                    
                detected_alerts.append({
                    'type': 'hepatotoxic_scaffold',
                    'pattern': name,
                    'risk_contribution': risk_contribution
                })
                risk_score += risk_contribution
        
        # Additional risk factors
        risk_factors = self._assess_structural_risk_factors(mol)
        detected_alerts.extend(risk_factors['alerts'])
        risk_score += risk_factors['score']
        
        # Cap risk score at 100
        risk_score = min(risk_score, 100)
        
        # Determine risk level
        if risk_score >= 50:
            risk_level = 'HIGH'
        elif risk_score >= 25:
            risk_level = 'MODERATE'
        else:
            risk_level = 'LOW'
        
        return {
            'metabolic_risk_score': risk_score,
            'risk_level': risk_level,
            'detected_alerts': detected_alerts,
            'alert_count': len(detected_alerts),
            'requires_metabolite_testing': risk_score >= 25
        }
    
    def _assess_structural_risk_factors(self, mol) -> Dict:
        """Assess additional structural risk factors"""
        alerts = []
        score = 0
        
        # Lipophilicity (high LogP can increase metabolic issues)
        logp = Descriptors.MolLogP(mol)
        if logp > 5:
            alerts.append({
                'type': 'high_lipophilicity',
                'pattern': f'LogP={logp:.1f}',
                'risk_contribution': 15
            })
            score += 15
        
        # Molecular complexity (more rings = more metabolic pathways)
        ring_count = rdMolDescriptors.CalcNumRings(mol)
        if ring_count >= 3:
            alerts.append({
                'type': 'complex_structure',
                'pattern': f'{ring_count}_rings',
                'risk_contribution': 5
            })
            score += 5
        
        # Check for carboxylic acid (can form reactive acyl glucuronides)
        if mol.HasSubstructMatch(Chem.MolFromSmarts('[CX3](=O)[OX2H1]')):
            alerts.append({
                'type': 'carboxylic_acid',
                'pattern': 'acyl_glucuronide_former',
                'risk_contribution': 15
            })
            score += 15
        
        # Check for branched aliphatic chains (Valproic Acid pattern)
        if mol.HasSubstructMatch(Chem.MolFromSmarts('[CH1]([CH2,CH3])([CH2,CH3])[CH2,CH3]')):
            alerts.append({
                'type': 'branched_aliphatic',
                'pattern': 'valproate_like_structure',
                'risk_contribution': 20
            })
            score += 20
        
        return {'alerts': alerts, 'score': score}
    
    def adjust_hepatotoxicity_prediction(self, base_probability: float, 
                                        metabolic_risk: Dict) -> Tuple[float, int]:
        """
        Adjust hepatotoxicity prediction based on metabolic risk
        Returns (adjusted_probability, adjusted_risk_flag)
        """
        # If high metabolic risk detected, boost hepatotoxicity
        metabolic_score = metabolic_risk['metabolic_risk_score']
        
        # Add metabolic risk to base prediction
        adjusted_prob = base_probability + (metabolic_score / 100) * 0.5
        adjusted_prob = min(adjusted_prob, 1.0)  # Cap at 100%
        
        # Determine risk flag (threshold: 40%)
        adjusted_risk = 1 if adjusted_prob >= 0.40 else 0
        
        return adjusted_prob, adjusted_risk


# Example usage and testing
if __name__ == "__main__":
    detector = MetabolismRiskDetector()
    
    # Test cases
    test_compounds = {
        "Valproic Acid": "CCCC(CCC)C(=O)O",
        "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "Acetaminophen": "CC(=O)Nc1ccc(O)cc1",
        "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    }
    
    for name, smiles in test_compounds.items():
        print(f"\n{name}:")
        result = detector.detect_metabolic_liability(smiles)
        print(f"  Metabolic Risk: {result['metabolic_risk_score']}% ({result['risk_level']})")
        print(f"  Alerts: {result['alert_count']}")
        if result['detected_alerts']:
            for alert in result['detected_alerts']:
                print(f"    - {alert['pattern']} (+{alert['risk_contribution']})")
