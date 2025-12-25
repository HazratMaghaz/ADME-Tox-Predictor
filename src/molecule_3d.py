"""
3D Molecular Visualization Module for Phase 2
Creates interactive 3D molecular structures using Py3Dmol
"""

import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem import AllChem


def generate_3d_structure(smiles: str) -> str:
    """
    Generate 3D coordinates for a molecule from SMILES
    
    Args:
        smiles: Input SMILES string
        
    Returns:
        Molecule with 3D coordinates
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Add hydrogens
        mol = Chem.AddHs(mol)
        
        # Generate 3D coordinates
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        
        return mol
    except:
        return None


def mol_to_pdb_string(mol) -> str:
    """
    Convert RDKit molecule to PDB format string
    
    Args:
        mol: RDKit molecule object with 3D coordinates
        
    Returns:
        PDB format string
    """
    if mol is None:
        return None
    
    try:
        pdb_string = Chem.MolToPDBBlock(mol)
        return pdb_string
    except:
        return None


def render_3d_molecule(smiles: str, style: str = 'stick', width: int = 400, height: int = 400):
    """
    Render interactive 3D molecule visualization using py3Dmol
    
    Args:
        smiles: Input SMILES string
        style: Display style ('stick', 'sphere', 'cartoon', 'line')
        width: Viewer width in pixels
        height: Viewer height in pixels
    """
    # Generate 3D structure
    mol = generate_3d_structure(smiles)
    if mol is None:
        st.error("Could not generate 3D structure")
        return
    
    # Convert to PDB
    pdb_string = mol_to_pdb_string(mol)
    if pdb_string is None:
        st.error("Could not convert to PDB format")
        return
    
    # Create HTML with py3Dmol
    html_template = f"""
    <html>
    <head>
        <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
        <style>
            #container {{
                width: {width}px;
                height: {height}px;
                position: relative;
                margin: 0 auto;
            }}
        </style>
    </head>
    <body>
        <div id="container"></div>
        <script>
            let element = document.getElementById('container');
            let viewer = $3Dmol.createViewer(element);
            
            let pdbData = `{pdb_string}`;
            viewer.addModel(pdbData, "pdb");
            
            // Set style
            viewer.setStyle({{}}, {{'{style}': {{}}}});
            
            // Add surface (optional)
            // viewer.addSurface($3Dmol.SurfaceType.VDW, {{opacity: 0.7, color: 'white'}});
            
            viewer.zoomTo();
            viewer.render();
            
            // Enable rotation
            viewer.rotate(10);
        </script>
    </body>
    </html>
    """
    
    # Render in Streamlit
    components.html(html_template, height=height+50, scrolling=False)


def create_3d_viewer_component(smiles: str):
    """
    Create a Streamlit component with 3D viewer and controls
    
    Args:
        smiles: Input SMILES string
    """
    st.subheader("🌐 3D Molecular Structure")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.write("**Display Style:**")
        style = st.radio(
            "Style",
            options=['stick', 'sphere', 'line', 'cross'],
            index=0,
            label_visibility='collapsed'
        )
        
        st.write("**Controls:**")
        st.caption("🖱️ Click & drag to rotate")
        st.caption("🔍 Scroll to zoom")
    
    with col1:
        render_3d_molecule(smiles, style=style, width=450, height=400)


# Alternative: Generate static 3D image (if py3Dmol not available)
def generate_3d_image(smiles: str, size=(400, 400)):
    """
    Generate static 3D image of molecule
    
    Args:
        smiles: Input SMILES string
        size: Image size tuple
        
    Returns:
        PIL Image object
    """
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    
    try:
        mol = generate_3d_structure(smiles)
        if mol is None:
            return None
        
        # Remove hydrogens for cleaner visualization
        mol = Chem.RemoveHs(mol)
        
        # Generate 2D depiction with 3D coordinates
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        
        # Get image
        img_data = drawer.GetDrawingText()
        
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(img_data))
        
        return img
    except:
        return None


# Example usage
if __name__ == "__main__":
    # Test with Aspirin
    test_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    
    print("="*60)
    print("Testing 3D Molecule Visualization")
    print("="*60)
    print(f"SMILES: {test_smiles}")
    
    # Generate 3D structure
    mol = generate_3d_structure(test_smiles)
    if mol:
        print("✓ 3D structure generated")
        
        # Convert to PDB
        pdb = mol_to_pdb_string(mol)
        if pdb:
            print("✓ PDB conversion successful")
            print(f"PDB Preview:\n{pdb[:200]}...")
        else:
            print("✗ PDB conversion failed")
    else:
        print("✗ 3D structure generation failed")
