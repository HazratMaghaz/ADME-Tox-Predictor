"""
Metabolite Generator Module
Generates Phase I metabolites using common CYP450 transformation rules
"""

from rdkit import Chem
from rdkit.Chem import AllChem
from typing import List, Dict, Tuple

class MetaboliteGenerator:
    """Generate potential metabolites using CYP450 reaction rules"""
    
    def __init__(self):
        # Define common CYP450 biotransformation reactions (SMARTS patterns)
        # Format: (reaction_name, SMARTS_pattern, description)
        self.reactions = [
            # Aromatic hydroxylation
            ("aromatic_hydroxylation", 
             "[c:1]>>[c:1]O",
             "Aromatic C-H → C-OH"),
            
            # Aliphatic hydroxylation
            ("aliphatic_hydroxylation",
             "[CH3:1]>>[CH2:1]O",
             "Aliphatic CH3 → CH2OH"),
            
            ("aliphatic_hydroxylation_2",
             "[CH2:1]>>[CH1:1]O",
             "Aliphatic CH2 → CHOH"),
            
            # N-dealkylation
            ("N_dealkylation",
             "[N:1][CH3:2]>>[N:1].[CH3:2]O",
             "N-CH3 → N-H + formaldehyde"),
            
            ("N_dealkylation_ethyl",
             "[N:1][CH2:2][CH3:3]>>[N:1].[CH2:2][CH3:3]O",
             "N-Et → N-H + acetaldehyde"),
            
            # O-dealkylation
            ("O_dealkylation",
             "[O:1][CH3:2]>>[O:1].[CH3:2]O",
             "O-CH3 → O-H + formaldehyde"),
            
            # Epoxidation of alkenes
            ("epoxidation",
             "[C:1]=[C:2]>>[C:1]1O[C:2]1",
             "C=C → epoxide"),
            
            # Oxidation of alcohols to aldehydes/ketones
            ("alcohol_oxidation",
             "[CH2:1][OH:2]>>[CH1:1]=[O:2]",
             "Primary alcohol → aldehyde"),
            
            ("alcohol_oxidation_2",
             "[CH1:1]([OH:2])>>[C:1](=[O:2])",
             "Secondary alcohol → ketone"),
            
            # Oxidative deamination
            ("oxidative_deamination",
             "[C:1][NH2:2]>>[C:1][OH]",
             "Primary amine → alcohol"),
            
            # S-oxidation
            ("S_oxidation",
             "[S:1]>>[S:1](=O)",
             "Sulfide → sulfoxide"),
            
            ("S_oxidation_2",
             "[S:1](=O)>>[S:1](=O)(=O)",
             "Sulfoxide → sulfone"),
            
            # N-oxidation
            ("N_oxidation",
             "[n:1]>>[n+:1][O-]",
             "Aromatic N → N-oxide"),
            
            # Aldehyde oxidation to carboxylic acid
            ("aldehyde_oxidation",
             "[CH:1]=O>>[C:1](=O)O",
             "Aldehyde → carboxylic acid"),
            
            # Ester hydrolysis
            ("ester_hydrolysis",
             "[C:1](=O)[O:2][C:3]>>[C:1](=O)O.[C:3]O",
             "Ester → carboxylic acid + alcohol"),
            
            # Amide hydrolysis
            ("amide_hydrolysis",
             "[C:1](=O)[N:2]>>[C:1](=O)O.[N:2]",
             "Amide → carboxylic acid + amine"),
        ]
        
        # Compile reactions
        self.compiled_reactions = []
        for name, smarts, desc in self.reactions:
            try:
                rxn = AllChem.ReactionFromSmarts(smarts)
                if rxn:
                    self.compiled_reactions.append((name, rxn, desc))
            except Exception as e:
                print(f"Warning: Could not compile reaction {name}: {e}")
    
    def generate_metabolites(self, smiles: str, max_depth: int = 1) -> List[Dict]:
        """
        Generate potential metabolites from parent compound
        
        Args:
            smiles: Parent compound SMILES
            max_depth: How many rounds of metabolism to simulate (1 or 2)
        
        Returns:
            List of dicts with metabolite info
        """
        parent_mol = Chem.MolFromSmiles(smiles)
        if not parent_mol:
            return []
        
        metabolites = []
        seen_smiles = {smiles}  # Track to avoid duplicates
        
        # Generate Phase I metabolites
        for name, rxn, desc in self.compiled_reactions:
            try:
                products = rxn.RunReactants((parent_mol,))
                
                for product_tuple in products:
                    for product_mol in product_tuple:
                        try:
                            Chem.SanitizeMol(product_mol)
                            product_smiles = Chem.MolToSmiles(product_mol)
                            
                            # Skip if already seen or invalid
                            if product_smiles in seen_smiles or len(product_smiles) < 3:
                                continue
                            
                            seen_smiles.add(product_smiles)
                            
                            metabolites.append({
                                'smiles': product_smiles,
                                'reaction': name,
                                'description': desc,
                                'generation': 1,
                                'parent': smiles
                            })
                        except Exception:
                            continue
            except Exception:
                continue
        
        # Optional: Second round (metabolites of metabolites)
        if max_depth >= 2:
            first_gen_metabolites = metabolites.copy()
            for metabolite in first_gen_metabolites:
                met_mol = Chem.MolFromSmiles(metabolite['smiles'])
                if not met_mol:
                    continue
                
                for name, rxn, desc in self.compiled_reactions:
                    try:
                        products = rxn.RunReactants((met_mol,))
                        for product_tuple in products:
                            for product_mol in product_tuple:
                                try:
                                    Chem.SanitizeMol(product_mol)
                                    product_smiles = Chem.MolToSmiles(product_mol)
                                    
                                    if product_smiles in seen_smiles or len(product_smiles) < 3:
                                        continue
                                    
                                    seen_smiles.add(product_smiles)
                                    
                                    metabolites.append({
                                        'smiles': product_smiles,
                                        'reaction': name,
                                        'description': desc,
                                        'generation': 2,
                                        'parent': metabolite['smiles']
                                    })
                                except Exception:
                                    continue
                    except Exception:
                        continue
        
        return metabolites


# Test the generator
if __name__ == "__main__":
    generator = MetaboliteGenerator()
    
    # Test with Valproic Acid
    print("Testing Valproic Acid:")
    print("=" * 80)
    valproic_acid = "CCCC(CCC)C(=O)O"
    metabolites = generator.generate_metabolites(valproic_acid, max_depth=1)
    print(f"Generated {len(metabolites)} potential metabolites\n")
    
    for i, met in enumerate(metabolites[:10], 1):
        print(f"{i}. {met['reaction']}: {met['description']}")
        print(f"   SMILES: {met['smiles']}\n")
    
    # Test with Acetaminophen
    print("\nTesting Acetaminophen:")
    print("=" * 80)
    acetaminophen = "CC(=O)Nc1ccc(O)cc1"
    metabolites = generator.generate_metabolites(acetaminophen, max_depth=1)
    print(f"Generated {len(metabolites)} potential metabolites\n")
    
    for i, met in enumerate(metabolites[:10], 1):
        print(f"{i}. {met['reaction']}: {met['description']}")
        print(f"   SMILES: {met['smiles']}\n")
