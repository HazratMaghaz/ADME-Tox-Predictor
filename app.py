"""
ADME-Tox Prediction System - Phase 2
Streamlit Web Application
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from rdkit import Chem
from rdkit.Chem import Draw
import warnings
import pubchempy as pcp
import pubchempy as pcp

# Suppress sklearn version warnings
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from predictor import ADMEToxPredictor
from smiles_processor import SMILESProcessor

# Page config
st.set_page_config(
    page_title="ADME-Tox Predictor",
    page_icon="🧬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-low {
        background-color: #d4edda;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #28a745;
    }
    .risk-medium {
        background-color: #fff3cd;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
    }
    .risk-high {
        background-color: #f8d7da;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor_v2():
    """Load the predictor model (cached) - v2 forces reload for extended descriptors"""
    # Compute extended descriptors for display/export, but the current Phase-1 models
    # will still use only the original 7 features internally.
    return ADMEToxPredictor(models_dir="models", use_extended_descriptors=True)


@st.cache_data(ttl=86400)
@st.cache_data(ttl=86400)
def resolve_name_to_smiles(compound_name: str) -> dict:
    """Resolve compound name to canonical SMILES via PubChem (cached for 24h)"""
    try:
        compounds = pcp.get_compounds(compound_name, 'name')
        if compounds:
            return {
                'success': True,
                'smiles': compounds[0].canonical_smiles,
                'cid': compounds[0].cid,
                'iupac_name': compounds[0].iupac_name if hasattr(compounds[0], 'iupac_name') else compound_name
            }
        else:
            return {'success': False, 'error': 'No compound found in PubChem database'}
    except Exception as e:
        return {'success': False, 'error': f'PubChem API error: {str(e)}'}


def detect_input_type(user_input: str) -> tuple:
    """Detect if input is SMILES or compound name, return (type, canonical_smiles, metadata)"""
    user_input = user_input.strip()
    
    # First try parsing as SMILES
    try:
        mol = Chem.MolFromSmiles(user_input)
        if mol:
            canonical = Chem.MolToSmiles(mol)
            return ('smiles', canonical, {'original_input': user_input})
    except:
        pass
    
    # If not valid SMILES, try resolving as name
    result = resolve_name_to_smiles(user_input)
    if result['success']:
        return ('name', result['smiles'], {
            'original_input': user_input,
            'pubchem_cid': result['cid'],
            'resolved_name': result['iupac_name']
        })
    else:
        return ('unknown', None, {'error': result['error'], 'original_input': user_input})


def mol_to_image(smiles, size=(400, 300)):
    """Convert SMILES to molecular structure image"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            img = Draw.MolToImage(mol, size=size)
            # Convert to format Streamlit can handle
            import io
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            buf.seek(0)
            return buf
    except Exception as e:
        st.warning(f"Could not generate image: {str(e)}")
    return None


def create_radar_chart(predictions):
    """Create radar chart for toxicity predictions"""
    endpoints = []
    probabilities = []
    
    for endpoint, data in predictions.items():
        if endpoint != 'solubility' and 'probability' in data:
            endpoints.append(endpoint.replace('_', ' ').title())
            probabilities.append(data['probability'])
    
    if not endpoints:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=probabilities,
        theta=endpoints,
        fill='toself',
        name='Risk Probability',
        line=dict(color='rgb(255, 99, 71)')
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title=dict(text="Toxicity Risk Profile", x=0.5, xanchor='center'),
        height=400,
        margin=dict(t=80, b=40)
    )
    return fig


def create_risk_gauge(overall_risk):
    """Create gauge chart for overall risk"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=overall_risk['score'],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Overall Risk Score"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': overall_risk['color']},
            'steps': [
                {'range': [0, 25], 'color': "lightgreen"},
                {'range': [25, 50], 'color': "yellow"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': overall_risk['score']
            }
        }
    ))
    fig.update_layout(height=300)
    return fig


def display_prediction_report(result):
    """Display the full prediction report for a single molecule result"""
    # Molecular structure and basic info
    st.header("📊 Molecular Information")
    
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        st.subheader("2D Structure")
        mol_img = mol_to_image(result['input']['canonical_smiles'])
        if mol_img:
            st.image(mol_img, width=400)
        else:
            st.warning("Could not generate structure image")
    
    with col2:
        st.subheader("Basic Properties")
        st.markdown(f"""
        **Formula:** {result['input']['molecular_formula']}  
        **Molecular Weight:** {result['input']['molecular_weight']} g/mol  
        **Canonical SMILES:** `{result['input']['canonical_smiles'][:40]}...`
        """)
        
        # Lipinski
        lipinski = result['lipinski_compliance']
        if lipinski['compliant']:
            st.success("✓ Lipinski Rule of 5: **PASS**")
        else:
            st.warning(f"⚠ Lipinski Rule of 5: **FAIL** ({lipinski['violations']} violations)")
    
    with col3:
        st.subheader("Overall Risk")
        overall_risk = result['overall_risk']
        
        # Determine color based on risk level
        risk_level = overall_risk['level']
        if risk_level == 'High':
            risk_color = 'red'
        elif risk_level == 'Medium':
            risk_color = 'orange'
        else:
            risk_color = 'green'
        
        st.markdown(f"""
        <div style='padding:20px; border-radius:10px; background-color:rgba(255,0,0,0.1); border:2px solid {risk_color};'>
            <h3 style='margin:0; color:{risk_color};'>{overall_risk['level']}</h3>
            <p style='margin:5px 0;'><strong>Worst-Case Risk Score:</strong> {overall_risk['score']}%</p>
            <p style='margin:0; font-size:0.9em;'><strong>Driven by:</strong> {overall_risk.get('max_endpoint', 'N/A')}</p>
            <p style='margin:0; font-size:0.8em; margin-top:5px;'>{overall_risk['flagged_endpoints']}/{overall_risk['total_endpoints']} endpoints flagged</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Descriptors Section with Tabs
    st.header("🧬 Molecular Descriptors")
    
    # Calculate counts
    count_basic = len(result.get('descriptors', {}))
    count_rdkit = result.get('num_rdkit_extended', 0)
    count_mordred = result.get('num_mordred', 0)
    count_total = result.get('num_descriptors_total', count_basic + count_rdkit + count_mordred)
    
    st.info(f"**Total Descriptors Calculated:** {count_total} (Basic: {count_basic} | RDKit: {count_rdkit} | Mordred: {count_mordred})")
    
    desc_tab1, desc_tab2, desc_tab3 = st.tabs([
        f"Basic ({count_basic})", 
        f"RDKit Extended ({count_rdkit})", 
        f"Mordred ({count_mordred})"
    ])
    
    with desc_tab1:
        st.caption("Fundamental physicochemical properties used by the Phase 1 ADME-Tox models.")
        desc_df = pd.DataFrame([result['descriptors']])
        st.dataframe(desc_df, use_container_width=True)
        
    with desc_tab2:
        if count_rdkit > 0:
            st.caption("Extended structural and electronic descriptors from RDKit.")
            rdkit_df = pd.DataFrame([result['descriptors_rdkit_extended']])
            st.dataframe(rdkit_df, use_container_width=True)
        else:
            st.warning("No extended RDKit descriptors available.")

    with desc_tab3:
        if count_mordred > 0:
            st.caption("Comprehensive molecular descriptors from Mordred (2D).")
            mordred_df = pd.DataFrame([result['descriptors_extended']])
            st.dataframe(mordred_df, use_container_width=True)
        else:
            st.error("Mordred descriptors not available.")

    # Predictions
    st.header("🎯 ADME-Tox Predictions")
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Detailed predictions
        predictions = result['predictions']
        
        # Solubility
        if 'solubility' in predictions:
            st.subheader("💧 Solubility")
            sol = predictions['solubility']
            st.metric(
                label="logS (aqueous solubility)",
                value=f"{sol['logS']}",
                help="Higher is more soluble"
            )
            st.caption(sol['interpretation'])
        
        st.markdown("---")
        
        # Toxicity endpoints
        st.subheader("☠️ Toxicity Predictions")
        
        for endpoint in ['herg_blocker', 'mutagenicity', 'carcinogenicity', 'hepatotoxicity']:
            if endpoint in predictions:
                pred = predictions[endpoint]
                
                col_a, col_b, col_c = st.columns([2, 1, 2])
                
                with col_a:
                    st.write(f"**{endpoint.replace('_', ' ').title()}**")
                
                with col_b:
                    if pred['risk'] == 1:
                        st.error("⚠️ HIGH RISK")
                    else:
                        st.success("✓ Low Risk")
                
                with col_c:
                    st.progress(pred['probability'])
                    st.caption(f"Probability: {pred['probability']:.1%}")
                
                st.caption(pred['interpretation'])
                st.markdown("")
    
    with col2:
        # Radar chart
        st.subheader("Risk Profile")
        radar_fig = create_radar_chart(predictions)
        if radar_fig:
            st.plotly_chart(radar_fig)
        
        # Gauge chart
        gauge_fig = create_risk_gauge(overall_risk)
        st.plotly_chart(gauge_fig)


def main():
    # Header
    st.markdown('<div class="main-header">🧬 AI-Based ADME-Tox Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predict Drug Safety & Toxicity from Chemical Structure - Phase 2</div>', unsafe_allow_html=True)
    
    # Load predictor
    try:
        predictor = load_predictor_v2()
    except Exception as e:
        st.error(f"❌ Failed to load models: {str(e)}")
        st.info("Make sure the models directory exists with all trained models.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        
        # Debug Info
        with st.expander("🔧 System Info"):
            import sys
            st.code(f"Python: {sys.version.split()[0]}")
            st.code(f"Env: {sys.prefix}")
            try:
                import mordred
                st.success(f"Mordred: {mordred.__version__}")
            except ImportError:
                st.error("Mordred: Not Found")
        
        st.write("""
        This tool predicts **ADME-Tox properties** from molecular structures (SMILES).
        
        **System Status:**
        - ✅ Models Loaded
        - ✅ Extended Descriptors Enabled
        - ✅ RDKit & Mordred Integration Active
        
        **Predictions:**
        - Solubility (logS)
        - hERG Blocker Risk
        - Mutagenicity
        - Carcinogenicity
        - Hepatotoxicity
        
        **Model Performance:**
        - Average Accuracy: 90.7%
        - Trained on 1,067+ molecules
        """)
        
        st.header("📚 Examples")
        examples = {
            'Aspirin': 'CC(=O)Oc1ccccc1C(=O)O',
            'Caffeine': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
            'Ibuprofen': 'CC(C)Cc1ccc(cc1)C(C)C(O)=O',
            'Paracetamol': 'CC(=O)Nc1ccc(O)cc1',
            'Warfarin': 'CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O'
        }
        
        selected_example = st.selectbox("Try an example:", [''] + list(examples.keys()))
        
        if selected_example:
            st.session_state['example_smiles'] = examples[selected_example]
    
    # Critical Warning Banner
    st.warning("""
    ⚠️ **IMPORTANT LIMITATION - Metabolic Activation Not Modeled**
    
    This model analyzes **parent compound structures only** and does NOT predict metabolites or bioactivation. 
    Many drugs become toxic only after liver metabolism (e.g., Valproic Acid, Acetaminophen). 
    
    **False negatives are possible** for compounds toxic via metabolic activation. Always validate high-risk 
    compounds with additional testing. For metabolite prediction, contact us about Phase 2 enhancements.
    """)
    
    # Main tabs
    tab1, tab2 = st.tabs(["🔬 Single Prediction", "📊 Batch Prediction"])
    
    # TAB 1: Single Prediction
    with tab1:
        st.header("🔬 Enter Compound (SMILES or Name)")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            user_input = st.text_input(
                "Compound Input:",
                value=st.session_state.get('example_smiles', ''),
                placeholder="e.g., Aspirin or CC(=O)Oc1ccccc1C(=O)O",
                help="Enter a compound name (e.g., Aspirin) or SMILES notation"
            )
        
        with col2:
            st.write("")
            st.write("")
            predict_button = st.button("🚀 Predict", type="primary", width="stretch")
        
        # Clear example after use
        if 'example_smiles' in st.session_state and predict_button:
            del st.session_state['example_smiles']
        
        # Make prediction
        if predict_button and user_input:
            with st.spinner('🔄 Resolving input and making predictions...'):
                # Detect input type and resolve to SMILES
                input_type, canonical_smiles, metadata = detect_input_type(user_input)
                
                if input_type == 'unknown':
                    st.error(f"❌ Could not resolve input: {metadata.get('error', 'Unknown error')}")
                    st.info("Please enter a valid compound name or SMILES string.")
                else:
                    # Show resolution info
                    if input_type == 'name':
                        st.info(f"✅ **Input Type:** Compound Name  \n**Resolved Name:** {metadata.get('resolved_name', 'N/A')}  \n**PubChem CID:** {metadata.get('pubchem_cid', 'N/A')}  \n**Canonical SMILES:** `{canonical_smiles}`")
                    else:
                        st.info(f"✅ **Input Type:** SMILES  \n**Canonical SMILES:** `{canonical_smiles}`")
                    
                    # Run prediction
                    result = predictor.predict(canonical_smiles)
                    
                    if 'error' in result:
                        st.error(f"❌ {result['error']}")
                        st.info("Prediction failed. Please check the input.")
                    else:
                        # Store metadata in result for export
                        result['input_metadata'] = {
                            'original_input': metadata.get('original_input', user_input),
                            'input_type': input_type,
                            'resolved_smiles': canonical_smiles
                        }
                        if input_type == 'name':
                            result['input_metadata']['pubchem_cid'] = metadata.get('pubchem_cid')
                            result['input_metadata']['resolved_name'] = metadata.get('resolved_name')
                        
                        # Display results
                        st.success("✅ Prediction completed successfully!")
                        
                        # Model limitation notice
                        st.info("""
                        📌 **Note:** This prediction is based on the parent compound structure only. 
                        If this compound undergoes metabolic activation (bioactivation), actual toxicity 
                        may be higher than predicted. Consider metabolite testing for high-risk decisions.
                        """)
                        
                        display_prediction_report(result)
                
                # Download results
                st.markdown("---")
                st.subheader("💾 Download Results")
                
                # Prepare download data
                download_data = {
                    'Original_Input': result.get('input_metadata', {}).get('original_input', result['input']['smiles']),
                    'Input_Type': result.get('input_metadata', {}).get('input_type', 'smiles'),
                    'Resolved_SMILES': result.get('input_metadata', {}).get('resolved_smiles', result['input']['canonical_smiles']),
                    'Canonical_SMILES': result['input']['canonical_smiles'],
                    'Formula': result['input']['molecular_formula'],
                    'Molecular_Weight': result['input']['molecular_weight'],
                    'Max_Risk_Score': result['overall_risk']['score'],
                    'Max_Risk_Endpoint': result['overall_risk'].get('max_endpoint', 'N/A'),
                    'Risk_Status': result['overall_risk']['level'],
                    'Flagged_Endpoints': result['overall_risk']['flagged_endpoints']
                }
                
                # Add PubChem data if available
                if result.get('input_metadata', {}).get('input_type') == 'name':
                    download_data['PubChem_CID'] = result.get('input_metadata', {}).get('pubchem_cid', '')
                    download_data['Resolved_Name'] = result.get('input_metadata', {}).get('resolved_name', '')
                
                # Add predictions
                for endpoint, pred in result['predictions'].items():
                    if endpoint == 'solubility':
                        download_data['Solubility_logS'] = pred['logS']
                    else:
                        download_data[f'{endpoint}_risk'] = pred['risk']
                        download_data[f'{endpoint}_probability'] = pred['probability']
                
                # Add descriptors
                for desc, value in result['descriptors'].items():
                    download_data[f'Descriptor_{desc}'] = value

                # Add RDKit extended descriptors
                for desc, value in result.get('descriptors_rdkit_extended', {}).items():
                    download_data[f'RDKit_{desc}'] = value

                # Add Mordred descriptors
                for desc, value in result.get('descriptors_extended', {}).items():
                    # keep original Mordred_* name but prefix to avoid collisions
                    download_data[f'{desc}'] = value
                
                df_download = pd.DataFrame([download_data])
                csv = df_download.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"ADME_Tox_Prediction_{result['input']['molecular_formula']}.csv",
                    mime="text/csv"
                )
        
        elif predict_button:
            st.warning("⚠️ Please enter a SMILES string first")
    
    # TAB 2: Batch Prediction
    with tab2:
        st.header("📊 Batch Prediction")
        st.info("Upload a CSV or TXT file with a 'SMILES' column to predict multiple molecules at once.")
        
        # Show format info
        with st.expander("ℹ️ File Format Guide"):
            st.write("""
            **Accepted formats:**
            - CSV files (.csv) - comma-separated
            - TXT files (.txt) - tab or comma-separated
            
            **Required column:** 
            - Must have a column named 'SMILES' or 'smiles'
            
            **Example:**
            ```
            Name,SMILES
            Aspirin,CC(=O)Oc1ccccc1C(=O)O
            Caffeine,CN1C=NC2=C1C(=O)N(C(=O)N2C)C
            ```
            """)
        
        # Download example
        st.download_button(
            label="📥 Download Example CSV",
            data=open("example_batch.csv", 'rb').read() if Path("example_batch.csv").exists() else "SMILES\nCC(=O)Oc1ccccc1C(=O)O\nCN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            file_name="example_batch.csv",
            mime="text/csv"
        )
        
        uploaded_file = st.file_uploader("Choose a CSV or TXT file", type=['csv', 'txt'])
        
        if uploaded_file is not None:
            try:
                # Handle both CSV and TXT files
                if uploaded_file.name.endswith('.txt'):
                    df_input = pd.read_csv(uploaded_file, sep='\t')  # Tab-separated
                    # Try comma-separated if tab fails
                    if len(df_input.columns) == 1:
                        uploaded_file.seek(0)
                        df_input = pd.read_csv(uploaded_file, sep=',')
                else:
                    df_input = pd.read_csv(uploaded_file)
                
                if 'SMILES' not in df_input.columns and 'smiles' not in df_input.columns:
                    st.error("❌ CSV must contain a 'SMILES' column")
                else:
                    smiles_col = 'SMILES' if 'SMILES' in df_input.columns else 'smiles'
                    st.success(f"✓ Loaded {len(df_input)} molecules")
                    
                    st.write("Preview:")
                    st.dataframe(df_input.head(), use_container_width=True)
                    
                    if st.button("🚀 Predict All", type="primary", key="batch_predict"):
                        results_list = []
                        full_results_map = {} # Store full result objects for detailed view
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for idx, row in df_input.iterrows():
                            status_text.text(f"Processing {idx+1}/{len(df_input)}...")
                            result = predictor.predict(row[smiles_col])
                            
                            if 'error' not in result:
                                # Store full result for interactive view
                                mol_id = f"Mol_{idx+1}"
                                if 'Name' in row:
                                    mol_id = f"{row['Name']} ({idx+1})"
                                full_results_map[mol_id] = result

                                result_row = {
                                    'ID': mol_id,
                                    'SMILES': result['input']['smiles'],
                                    'Formula': result['input']['molecular_formula'],
                                    'MW': result['input']['molecular_weight'],
                                    'Overall_Risk': result['overall_risk']['score'],
                                    'Risk_Level': result['overall_risk']['level']
                                }
                                
                                # Add predictions
                                for endpoint, pred in result['predictions'].items():
                                    if endpoint == 'solubility':
                                        result_row[f'{endpoint}_logS'] = pred['logS']
                                    else:
                                        result_row[f'{endpoint}_risk'] = pred['risk']
                                        result_row[f'{endpoint}_prob'] = pred['probability']

                                # Add descriptors (basic + extended) to the batch export
                                for desc, value in result.get('descriptors', {}).items():
                                    result_row[f'Descriptor_{desc}'] = value
                                for desc, value in result.get('descriptors_rdkit_extended', {}).items():
                                    result_row[f'RDKit_{desc}'] = value
                                for desc, value in result.get('descriptors_extended', {}).items():
                                    result_row[f'{desc}'] = value
                                
                                results_list.append(result_row)
                            
                            progress_bar.progress((idx + 1) / len(df_input))
                        
                        status_text.text("✅ Complete!")
                        
                        # Store in session state for persistence
                        st.session_state['batch_results_df'] = pd.DataFrame(results_list)
                        st.session_state['batch_full_results'] = full_results_map
                        
                    # Check if results exist in session state
                    if 'batch_results_df' in st.session_state:
                        df_results = st.session_state['batch_results_df']
                        full_results_map = st.session_state['batch_full_results']

                        st.subheader("Results")
                        st.dataframe(df_results[['ID', 'SMILES', 'Formula', 'MW', 'Overall_Risk', 'Risk_Level']], use_container_width=True)
                        
                        # --- Batch Analysis Visualizations ---
                        st.markdown("---")
                        st.header("📈 Batch Analysis")
                        
                        # 1. Summary Metrics
                        col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
                        with col_sum1:
                            st.metric("Total Molecules", len(df_results))
                        with col_sum2:
                            high_risk_count = df_results[df_results['Risk_Level'] == 'High'].shape[0]
                            st.metric("High Risk Molecules", high_risk_count)
                        with col_sum3:
                            if 'solubility_logS' in df_results.columns:
                                avg_sol = df_results['solubility_logS'].mean()
                                st.metric("Avg Solubility (logS)", f"{avg_sol:.2f}")
                        with col_sum4:
                            if 'MW' in df_results.columns:
                                avg_mw = df_results['MW'].mean()
                                st.metric("Avg Molecular Weight", f"{avg_mw:.2f}")

                        col_viz1, col_viz2 = st.columns(2)
                        
                        with col_viz1:
                            # 2. Risk Distribution
                            st.subheader("Risk Distribution")
                            if 'Risk_Level' in df_results.columns:
                                risk_counts = df_results['Risk_Level'].value_counts().reset_index()
                                risk_counts.columns = ['Risk Level', 'Count']
                                fig_risk = px.pie(risk_counts, values='Count', names='Risk Level', 
                                                title='Overall Risk Distribution',
                                                color='Risk Level',
                                                color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'})
                                st.plotly_chart(fig_risk)

                        with col_viz2:
                            # 3. Chemical Space (MW vs Solubility)
                            st.subheader("Chemical Space")
                            if 'MW' in df_results.columns and 'solubility_logS' in df_results.columns:
                                fig_space = px.scatter(df_results, x='MW', y='solubility_logS', 
                                                    color='Risk_Level',
                                                    hover_data=['SMILES', 'ID'],
                                                    title='MW vs Solubility (Colored by Risk)',
                                                    color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'})
                                st.plotly_chart(fig_space)
                        
                        # 4. Toxicity Breakdown
                        st.subheader("Toxicity Breakdown")
                        tox_cols = [c for c in df_results.columns if c.endswith('_risk') and c != 'Overall_Risk']
                        if tox_cols:
                            tox_data = []
                            for col in tox_cols:
                                endpoint = col.replace('_risk', '').replace('_', ' ').title()
                                high_risk = df_results[df_results[col] == 1].shape[0]
                                low_risk = df_results[df_results[col] == 0].shape[0]
                                tox_data.append({'Endpoint': endpoint, 'Risk': 'High Risk', 'Count': high_risk})
                                tox_data.append({'Endpoint': endpoint, 'Risk': 'Low Risk', 'Count': low_risk})
                            
                            df_tox = pd.DataFrame(tox_data)
                            fig_tox = px.bar(df_tox, x='Endpoint', y='Count', color='Risk', 
                                             title='Risk Counts per Endpoint',
                                             color_discrete_map={'High Risk': 'red', 'Low Risk': 'green'},
                                             barmode='group')
                            st.plotly_chart(fig_tox)
                        
                        st.markdown("---")
                        
                        # --- Individual Molecule Inspector ---
                        st.header("🔍 Individual Molecule Inspector")
                        st.info("Select a molecule from the batch to view its detailed report (Radar Chart, Gauge, Descriptors).")
                        
                        selected_mol_id = st.selectbox("Select Molecule:", list(full_results_map.keys()))
                        
                        if selected_mol_id:
                            st.markdown("---")
                            st.subheader(f"Report for: {selected_mol_id}")
                            display_prediction_report(full_results_map[selected_mol_id])
                            st.markdown("---")

                        # -------------------------------------

                        # Download
                        csv_results = df_results.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv_results,
                            file_name="batch_predictions.csv",
                            mime="text/csv",
                            key="download_batch"
                        )
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🧬 AI-Based ADME-Tox Prediction System - Phase 2</p>
        <p>Built with RDKit, Scikit-learn, and Streamlit</p>
        <p style='font-size: 0.8em;'>⚠️ For research purposes only. Not for clinical use.</p>
        <p style='font-size: 0.75em; color: #999; margin-top: 10px;'>
            <strong>Model Limitation:</strong> Predicts parent compound toxicity only. 
            Does not account for metabolic activation or bioactivation pathways. 
            False negatives possible for compounds toxic via metabolism (e.g., Valproic Acid).
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
