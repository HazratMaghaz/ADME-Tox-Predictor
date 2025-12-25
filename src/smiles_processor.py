"""
SMILES Processing Module for Phase 2
Handles SMILES validation, canonicalization, and feature extraction
"""

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
import numpy as np
from typing import Dict, Tuple, Optional


class SMILESProcessor:
    """Process and validate SMILES strings"""
    
    def __init__(self, use_extended_descriptors=True):
        self.valid_molecules = []
        self.errors = []
        self.use_extended_descriptors = use_extended_descriptors
        
        # Try to import Mordred
        self.mordred_available = False
        if use_extended_descriptors:
            try:
                from mordred import Calculator, descriptors as mordred_desc
                self.mordred_calc = Calculator(mordred_desc, ignore_3D=True)
                self.mordred_available = True
                print("[INFO] Mordred initialized successfully")
            except Exception as e:
                print(f"[ERROR] Mordred initialization failed: {e}")
                self.mordred_available = False
    
    def validate_smiles(self, smiles: str) -> Tuple[bool, Optional[Chem.Mol]]:
        """
        Validate SMILES string and return molecule object
        
        Args:
            smiles: SMILES string to validate
            
        Returns:
            Tuple of (is_valid, molecule_object)
        """
        try:
            smiles = smiles.strip()
            if not smiles:
                return False, None
                
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False, None
                
            return True, mol
            
        except Exception as e:
            self.errors.append(str(e))
            return False, None
    
    def canonicalize(self, smiles: str) -> Optional[str]:
        """
        Convert SMILES to canonical form
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            Canonical SMILES or None if invalid
        """
        is_valid, mol = self.validate_smiles(smiles)
        if is_valid and mol:
            return Chem.MolToSmiles(mol)
        return None
    
    def get_molecular_formula(self, smiles: str) -> Optional[str]:
        """Get molecular formula from SMILES"""
        is_valid, mol = self.validate_smiles(smiles)
        if is_valid and mol:
            return Chem.rdMolDescriptors.CalcMolFormula(mol)
        return None
    
    def calculate_descriptors(self, smiles: str) -> Optional[Dict]:
        """
        Calculate molecular descriptors needed for prediction
        Now includes extended RDKit + Mordred descriptors
        
        Args:
            smiles: SMILES string
            
        Returns:
            Dictionary with descriptors (7 basic + extended if enabled)
        """
        is_valid, mol = self.validate_smiles(smiles)
        if not is_valid or not mol:
            return None
        
        try:
            # Basic 7 descriptors (original Phase 1)
            descriptors = {
                'MolWt': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'TPSA': Descriptors.TPSA(mol),
                'NumHDonors': Lipinski.NumHDonors(mol),
                'NumHAcceptors': Lipinski.NumHAcceptors(mol),
                'NumRotatableBonds': Lipinski.NumRotatableBonds(mol),
                'NumAromaticRings': Lipinski.NumAromaticRings(mol)
            }
            
            # Extended RDKit descriptors (Phase 2 addition)
            if self.use_extended_descriptors:
                # Calculate ALL available RDKit descriptors
                extended_rdkit = {}
                for name, func in Descriptors.descList:
                    if name not in descriptors:  # Don't overwrite basic ones
                        try:
                            extended_rdkit[name] = func(mol)
                        except:
                            pass
                
                descriptors.update(extended_rdkit)
                
                # Mordred descriptors (if available)
                if self.mordred_available:
                    try:
                        mordred_results = self.mordred_calc(mol)
                        # Select key Mordred descriptors (avoid NaN/error)
                        mordred_desc = {}
                        for desc_name, value in zip(self.mordred_calc.descriptors, mordred_results):
                            try:
                                val = float(value)
                                if not np.isnan(val) and not np.isinf(val):
                                    # Add only non-problematic descriptors
                                    mordred_desc[f'Mordred_{desc_name}'] = val
                                    # Limit removed to allow full descriptor set
                                    # if len(mordred_desc) >= 50:
                                    #    break
                            except:
                                continue
                        descriptors.update(mordred_desc)
                    except:
                        pass  # Skip Mordred if calculation fails
            
            return descriptors
            
        except Exception as e:
            self.errors.append(f"Descriptor calculation error: {str(e)}")
            return None
    
    def get_basic_properties(self, smiles: str) -> Optional[Dict]:
        """
        Get basic molecular properties for display
        
        Args:
            smiles: SMILES string
            
        Returns:
            Dictionary with basic properties
        """
        is_valid, mol = self.validate_smiles(smiles)
        if not is_valid or not mol:
            return None
        
        try:
            properties = {
                'molecular_formula': Chem.rdMolDescriptors.CalcMolFormula(mol),
                'molecular_weight': Descriptors.MolWt(mol),
                'num_atoms': mol.GetNumAtoms(),
                'num_bonds': mol.GetNumBonds(),
                'num_rings': Chem.rdMolDescriptors.CalcNumRings(mol),
                'canonical_smiles': Chem.MolToSmiles(mol)
            }
            return properties
            
        except Exception as e:
            self.errors.append(f"Property calculation error: {str(e)}")
            return None
    
    def check_lipinski_rules(self, smiles: str) -> Optional[Dict]:
        """
        Check Lipinski's Rule of Five for drug-likeness
        
        Args:
            smiles: SMILES string
            
        Returns:
            Dictionary with Lipinski compliance
        """
        descriptors = self.calculate_descriptors(smiles)
        if not descriptors:
            return None
        
        violations = 0
        rules = {
            'MW_under_500': descriptors['MolWt'] <= 500,
            'LogP_under_5': descriptors['LogP'] <= 5,
            'HBD_under_5': descriptors['NumHDonors'] <= 5,
            'HBA_under_10': descriptors['NumHAcceptors'] <= 10
        }
        
        violations = sum(1 for passed in rules.values() if not passed)
        
        return {
            'compliant': violations <= 1,  # Lipinski allows 1 violation
            'violations': violations,
            'rules': rules
        }


def validate_and_process_smiles(smiles: str) -> Dict:
    """
    Convenience function to validate and extract all info from SMILES
    
    Args:
        smiles: Input SMILES string
        
    Returns:
        Dictionary with validation status and molecular info
    """
    processor = SMILESProcessor()
    
    # Validate
    is_valid, mol = processor.validate_smiles(smiles)
    
    if not is_valid:
        return {
            'valid': False,
            'error': 'Invalid SMILES string',
            'smiles': smiles
        }
    
    # Get all info
    result = {
        'valid': True,
        'smiles': smiles,
        'canonical_smiles': processor.canonicalize(smiles),
        'properties': processor.get_basic_properties(smiles),
        'descriptors': processor.calculate_descriptors(smiles),
        'lipinski': processor.check_lipinski_rules(smiles)
    }
    
    return result


# Example usage
if __name__ == "__main__":
    # Test with common drugs
    test_molecules = {
        'Aspirin': 'CC(=O)Oc1ccccc1C(=O)O',
        'Caffeine': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
        'Ibuprofen': 'CC(C)Cc1ccc(cc1)C(C)C(O)=O'
    }
    
    processor = SMILESProcessor()
    
    for name, smiles in test_molecules.items():
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"SMILES: {smiles}")
        
        result = validate_and_process_smiles(smiles)
        
        if result['valid']:
            print("✓ Valid SMILES")
            print(f"Formula: {result['properties']['molecular_formula']}")
            print(f"MW: {result['properties']['molecular_weight']:.2f}")
            print(f"Lipinski Compliant: {result['lipinski']['compliant']}")
        else:
            print(f"✗ Invalid: {result['error']}")
