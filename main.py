# Enhanced ESMFold App with Advanced Features
# Credit: Inspired by https://huggingface.co/spaces/osanseviero/esmfold

import streamlit as st
from stmol import showmol
import py3Dmol
import requests
import biotite.structure.io as bsio
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
from io import StringIO
import time
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Protein Lab - Enhanced ESMFold",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",

)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .confidence-high { color: #00ff00; font-weight: bold; }
    .confidence-medium { color: #ffaa00; font-weight: bold; }
    .confidence-low { color: #ff0000; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title('🧬Protein Lab - Enhanced ESMFold')
st.sidebar.markdown("""
[*ESMFold*](https://esmatlas.com/about) is an end-to-end single sequence protein structure predictor based on the ESM-2 language model. 

**New Features:**
- 📊 Advanced confidence analysis
- 🎨 Multiple visualization styles
- 📋 FASTA format support
- 🔍 Structural analysis tools
- 📈 Interactive plots
""")

# Initialize session state
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = {}
if 'current_sequence' not in st.session_state:
    st.session_state.current_sequence = ""


class SequenceValidator:
    """Validate and process protein sequences"""

    VALID_AA = set('ACDEFGHIKLMNPQRSTVWY')
    AMBIGUOUS_AA = set('BJOUXZ')

    @staticmethod
    def parse_fasta(text: str) -> List[Tuple[str, str]]:
        """Parse FASTA format text"""
        sequences = []
        current_header = ""
        current_seq = ""

        for line in text.strip().split('\n'):
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    sequences.append((current_header, current_seq))
                current_header = line[1:]
                current_seq = ""
            else:
                current_seq += line.upper()

        if current_seq:
            sequences.append((current_header, current_seq))

        return sequences

    @staticmethod
    def clean_sequence(seq: str) -> str:
        """Clean sequence by removing invalid characters"""
        # Remove whitespace, numbers, and special characters
        cleaned = re.sub(r'[^A-Za-z]', '', seq.upper())
        return cleaned

    @staticmethod
    def validate_sequence(seq: str) -> Dict:
        """Validate protein sequence"""
        cleaned_seq = SequenceValidator.clean_sequence(seq)

        if not cleaned_seq:
            return {
                'valid': False,
                'cleaned_sequence': '',
                'errors': ['Empty sequence'],
                'warnings': [],
                'stats': {}
            }

        errors = []
        warnings = []

        # Check length
        if len(cleaned_seq) < 10:
            warnings.append(f"Very short sequence ({len(cleaned_seq)} residues)")
        elif len(cleaned_seq) > 1000:
            warnings.append(f"Very long sequence ({len(cleaned_seq)} residues) - prediction may be slow")

        # Check for invalid characters
        invalid_chars = set(cleaned_seq) - SequenceValidator.VALID_AA - SequenceValidator.AMBIGUOUS_AA
        if invalid_chars:
            errors.append(f"Invalid amino acid codes: {', '.join(sorted(invalid_chars))}")

        # Check for ambiguous characters
        ambiguous_chars = set(cleaned_seq) & SequenceValidator.AMBIGUOUS_AA
        if ambiguous_chars:
            warnings.append(f"Ambiguous amino acid codes found: {', '.join(sorted(ambiguous_chars))}")

        # Calculate composition
        composition = {aa: cleaned_seq.count(aa) for aa in SequenceValidator.VALID_AA}
        composition_pct = {aa: (count / len(cleaned_seq)) * 100 for aa, count in composition.items() if count > 0}

        # Calculate molecular weight safely
        try:
            molecular_weight = sum(composition[aa] * mw for aa, mw in AA_WEIGHTS.items())
        except:
            molecular_weight = 0

        stats = {
            'length': len(cleaned_seq),
            'composition': composition_pct,
            'molecular_weight': molecular_weight,
        }

        return {
            'valid': len(errors) == 0,
            'cleaned_sequence': cleaned_seq,
            'errors': errors,
            'warnings': warnings,
            'stats': stats
        }


# Amino acid molecular weights (Da)
AA_WEIGHTS = {
    'A': 89.1, 'R': 174.2, 'N': 132.1, 'D': 133.1, 'C': 121.2,
    'E': 147.1, 'Q': 146.2, 'G': 75.1, 'H': 155.2, 'I': 131.2,
    'L': 131.2, 'K': 146.2, 'M': 149.2, 'F': 165.2, 'P': 115.1,
    'S': 105.1, 'T': 119.1, 'W': 204.2, 'Y': 181.2, 'V': 117.1
}


class StructureAnalyzer:
    """Analyze protein structure properties"""

    @staticmethod
    def calculate_secondary_structure(structure):
        """Calculate secondary structure using DSSP-like algorithm"""
        try:
            # This is a simplified secondary structure assignment
            # Get CA atoms using proper biotite syntax
            ca_mask = (structure.atom_name == "CA")
            ca_atoms = structure[ca_mask]

            if len(ca_atoms) == 0:
                return {'alpha': 0, 'beta': 0, 'coil': 0}

            # Simple secondary structure assignment based on phi/psi angles
            # This is a basic implementation - for full DSSP you'd need biotite.structure.annotate_sse
            ss_counts = {'alpha': 0, 'beta': 0, 'coil': len(ca_atoms)}

            return ss_counts
        except Exception as e:
            return {'alpha': 0, 'beta': 0, 'coil': 0}

    @staticmethod
    def calculate_geometry_stats(structure):
        """Calculate basic geometry statistics"""
        try:
            # Get CA atoms using proper biotite syntax
            ca_mask = (structure.atom_name == "CA")
            ca_atoms = structure[ca_mask]

            if len(ca_atoms) == 0:
                return {
                    'centroid': [0, 0, 0],
                    'radius_gyration': 0,
                    'max_distance': 0,
                    'num_residues': 0
                }

            coords = ca_atoms.coord

            # Calculate centroid
            centroid = np.mean(coords, axis=0)

            # Calculate radius of gyration
            distances = np.linalg.norm(coords - centroid, axis=1)
            radius_gyration = np.sqrt(np.mean(distances ** 2)) if len(distances) > 0 else 0

            # Calculate max distance
            max_distance = np.max(distances) if len(distances) > 0 else 0

            return {
                'centroid': centroid.tolist() if hasattr(centroid, 'tolist') else [0, 0, 0],
                'radius_gyration': float(radius_gyration),
                'max_distance': float(max_distance),
                'num_residues': len(ca_atoms)
            }
        except Exception as e:
            st.warning(f"Could not calculate structure geometry: {str(e)}")
            return {
                'centroid': [0, 0, 0],
                'radius_gyration': 0,
                'max_distance': 0,
                'num_residues': 0
            }


def render_mol_enhanced(pdb: str, style: str = 'cartoon', color_scheme: str = 'spectrum',
                        confidence_data: Optional[np.ndarray] = None):
    """Enhanced molecular visualization with multiple styles and coloring options"""

    pdbview = py3Dmol.view(width=800, height=500)
    pdbview.addModel(pdb, 'pdb')

    # Set background
    pdbview.setBackgroundColor('white')

    # Apply styling based on parameters
    if style == 'cartoon':
        if color_scheme == 'confidence' and confidence_data is not None:
            # Color by confidence (B-factor)
            pdbview.setStyle({'cartoon': {'colorscheme': 'RdYlBu', 'color': 'b'}})
        elif color_scheme == 'spectrum':
            pdbview.setStyle({'cartoon': {'color': 'spectrum'}})
        elif color_scheme == 'chain':
            pdbview.setStyle({'cartoon': {'color': 'chain'}})
        else:
            pdbview.setStyle({'cartoon': {'color': color_scheme}})

    elif style == 'surface':
        pdbview.addSurface(py3Dmol.VDW, {'opacity': 0.7, 'color': color_scheme})
        pdbview.setStyle({'stick': {}})

    elif style == 'ball_stick':
        pdbview.setStyle({'stick': {'radius': 0.2}, 'sphere': {'radius': 0.5}})

    elif style == 'ribbon':
        pdbview.setStyle({'ribbon': {'color': color_scheme}})

    # Set view
    pdbview.zoomTo()
    pdbview.zoom(1.5, 800)

    # Add spin option
    pdbview.spin(True)

    return pdbview


def plot_confidence_analysis(b_factors: np.ndarray, sequence: str) -> go.Figure:
    """Create interactive confidence analysis plot"""

    # Handle length mismatch between B-factors and sequence
    min_length = min(len(b_factors), len(sequence))
    b_factors_trimmed = b_factors[:min_length]
    sequence_trimmed = sequence[:min_length]

    # Convert to 0-1 scale if needed (ESMFold sometimes returns 0-100 scale)
    if np.max(b_factors_trimmed) > 1.0:
        b_factors_trimmed = b_factors_trimmed / 100.0

    residue_numbers = list(range(1, min_length + 1))

    # Create confidence categories based on correct pLDDT thresholds
    confidence_categories = []
    colors = []
    for bf in b_factors_trimmed:
        if bf > 0.9:
            confidence_categories.append('Very High (>0.9)')
            colors.append('#00ff00')
        elif bf > 0.7:
            confidence_categories.append('Confident (0.7-0.9)')
            colors.append('#90EE90')
        elif bf > 0.5:
            confidence_categories.append('Low (0.5-0.7)')
            colors.append('#FFD700')
        else:
            confidence_categories.append('Very Low (<0.5)')
            colors.append('#FF0000')

    fig = go.Figure()

    # Add confidence line
    fig.add_trace(go.Scatter(
        x=residue_numbers,
        y=b_factors_trimmed,
        mode='lines+markers',
        name='pLDDT Score',
        line=dict(color='blue', width=2),
        marker=dict(color=colors, size=4),
        hovertemplate='<b>Residue %{x}</b><br>' +
                      'pLDDT: %{y:.3f}<br>' +
                      'AA: %{text}<br>' +
                      '<extra></extra>',
        text=list(sequence_trimmed)
    ))

    # Add confidence threshold lines with correct values
    fig.add_hline(y=0.9, line_dash="dash", line_color="green",
                  annotation_text="Very High Confidence (>0.9)", annotation_position="bottom right")
    fig.add_hline(y=0.7, line_dash="dash", line_color="orange",
                  annotation_text="Confident (>0.7)", annotation_position="bottom right")
    fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                  annotation_text="Low Confidence (>0.5)", annotation_position="bottom right")

    fig.update_layout(
        title="Per-Residue Confidence (pLDDT) Analysis",
        xaxis_title="Residue Number",
        yaxis_title="pLDDT Score",
        height=400,
        showlegend=True,
        yaxis=dict(range=[0, 1])
    )

    return fig


def create_confidence_histogram(b_factors: np.ndarray) -> go.Figure:
    """Create histogram of confidence scores"""

    fig = go.Figure(data=[go.Histogram(
        x=b_factors,
        nbinsx=20,
        name='plDDT Distribution',
        marker_color='skyblue',
        opacity=0.7
    )])

    fig.update_layout(
        title="Distribution of Confidence Scores",
        xaxis_title="plDDT Score",
        yaxis_title="Number of Residues",
        height=300
    )

    return fig


def analyze_confidence_regions(b_factors: np.ndarray, sequence: str, threshold: float = 0.5) -> pd.DataFrame:
    """Identify low confidence regions"""

    # Handle length mismatch
    min_length = min(len(b_factors), len(sequence))
    b_factors_trimmed = b_factors[:min_length]
    sequence_trimmed = sequence[:min_length]

    # Convert to 0-1 scale if needed
    if np.max(b_factors_trimmed) > 1.0:
        b_factors_trimmed = b_factors_trimmed / 100.0

    low_conf_mask = b_factors_trimmed < threshold
    regions = []

    if not np.any(low_conf_mask):
        return pd.DataFrame(columns=['Start', 'End', 'Length', 'Avg_pLDDT', 'Sequence', 'Interpretation'])

    # Find continuous low confidence regions
    diff = np.diff(np.concatenate(([False], low_conf_mask, [False])).astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    for start, end in zip(starts, ends):
        region_seq = sequence_trimmed[start:end] if end <= len(sequence_trimmed) else sequence_trimmed[start:]
        avg_plddt = np.mean(b_factors_trimmed[start:end])

        # Determine interpretation based on pLDDT ranges
        if avg_plddt > 0.9:
            interpretation = "Very high accuracy expected"
        elif avg_plddt > 0.7:
            interpretation = "Backbone likely correct, side-chains less reliable"
        elif avg_plddt > 0.5:
            interpretation = "Low confidence - interpret with caution"
        else:
            interpretation = "Very low confidence - likely disordered/inaccurate"

        regions.append({
            'Start': start + 1,
            'End': end,
            'Length': end - start,
            'Avg_pLDDT': avg_plddt,
            'Sequence': region_seq,
            'Interpretation': interpretation
        })

    return pd.DataFrame(regions)


# Main app interface
st.markdown('<h1 class="main-header">🧬 Protein Lab (Protein Structure Predictor)</h1>',
            unsafe_allow_html=True)

# Input section
st.sidebar.subheader("📝 Sequence Input")

# Input method selection
input_method = st.sidebar.radio(
    "Input method:",
    ["Paste sequence", "Upload FASTA file", "Example sequences"]
)

# Default sequence
DEFAULT_SEQ = "MGSSHHHHHHSSGLVPRGSHMRGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTRVSDFRTANCSLEDPAANKARKEAELAAATAEQ"

txt = ""
if input_method == "Paste sequence":
    txt = st.sidebar.text_area(
        'Input sequence (FASTA format supported)',
        DEFAULT_SEQ,
        height=275,
        help="Paste your protein sequence here. FASTA format with headers is supported."
    )

elif input_method == "Upload FASTA file":
    uploaded_file = st.sidebar.file_uploader("Choose a FASTA file", type=['fasta', 'fa', 'txt'])
    if uploaded_file is not None:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        txt = stringio.read()
    else:
        txt = DEFAULT_SEQ

elif input_method == "Example sequences":
    examples = {
        "Green Fluorescent Protein": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFGYGLQCFARYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
        "Insulin (Human)": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN",
        "Lysozyme (Human)": "MKALIVLGLVLLSVTVQGKVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"
    }

    selected_example = st.sidebar.selectbox("Choose an example:", list(examples.keys()))
    txt = examples[selected_example]

# Sequence validation
if txt:
    # Parse and validate sequence
    if txt.startswith('>') or '\n>' in txt:
        # FASTA format
        sequences = SequenceValidator.parse_fasta(txt)
        if sequences:
            if len(sequences) > 1:
                seq_names = [f"{name[:50]}..." if len(name) > 50 else name for name, _ in sequences]
                selected_seq_idx = st.sidebar.selectbox("Select sequence:", range(len(sequences)),
                                                        format_func=lambda x: seq_names[x])
                selected_name, selected_seq = sequences[selected_seq_idx]
            else:
                selected_name, selected_seq = sequences[0]
        else:
            st.sidebar.error("No valid sequences found in FASTA format")
            selected_seq = ""
    else:
        # Plain sequence
        selected_seq = txt
        selected_name = "Input sequence"

    # Validate the selected sequence
    validation_result = SequenceValidator.validate_sequence(selected_seq)

    # Display validation results
    if validation_result['errors']:
        st.sidebar.error("❌ Sequence validation failed:")
        for error in validation_result['errors']:
            st.sidebar.error(f"• {error}")

    if validation_result['warnings']:
        st.sidebar.warning("⚠️ Warnings:")
        for warning in validation_result['warnings']:
            st.sidebar.warning(f"• {warning}")

    if validation_result['valid']:
        cleaned_seq = validation_result['cleaned_sequence']
        stats = validation_result['stats']

        # Display sequence stats
        st.sidebar.success("✅ Sequence is valid")
        st.sidebar.info(f"Length: {stats['length']} residues")
        st.sidebar.info(f"Molecular Weight: {stats['molecular_weight']:.1f} Da")

        # Visualization options
        st.sidebar.subheader("🎨 Visualization Options")

        viz_style = st.sidebar.selectbox(
            "Rendering style:",
            ["cartoon", "surface", "ball_stick", "ribbon"]
        )

        color_scheme = st.sidebar.selectbox(
            "Color scheme:",
            ["spectrum", "confidence", "chain", "cyan", "red", "green", "blue"]
        )

        # Advanced options
        with st.sidebar.expander("🔬 Advanced Analysis"):
            show_confidence_plot = st.checkbox("Show confidence analysis", True)
            show_structure_analysis = st.checkbox("Show structure analysis", True)
            show_sequence_analysis = st.checkbox("Show sequence analysis", True)
            confidence_threshold = st.slider("Low confidence threshold", 0.3, 0.8, 0.5, 0.1)


# Prediction function
def update_prediction(sequence: str):
    """Updated prediction function with enhanced features"""

    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("🔄 Sending request to ESMFold...")
        progress_bar.progress(25)

        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        response = requests.post(
            'https://api.esmatlas.com/foldSequence/v1/pdb/',
            headers=headers,
            data=sequence,
            timeout=300  # 5 minute timeout
        )

        if response.status_code != 200:
            st.error(f"API request failed with status {response.status_code}")
            return None

        status_text.text("📥 Processing structure...")
        progress_bar.progress(50)

        pdb_string = response.content.decode('utf-8')

        # Save PDB file
        with open('predicted.pdb', 'w') as f:
            f.write(pdb_string)

        status_text.text("🧬 Analyzing structure...")
        progress_bar.progress(75)

        # Load structure for analysis
        struct = bsio.load_structure('predicted.pdb', extra_fields=["b_factor"])
        b_factors = struct.b_factor

        # Calculate statistics
        b_value_mean = round(np.mean(b_factors), 4)
        b_value_std = round(np.std(b_factors), 4)
        b_value_min = round(np.min(b_factors), 4)
        b_value_max = round(np.max(b_factors), 4)

        # Structure analysis
        structure_stats = StructureAnalyzer.calculate_geometry_stats(struct)

        status_text.text("✅ Analysis complete!")
        progress_bar.progress(100)

        # Clear progress indicators
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()

        return {
            'pdb_string': pdb_string,
            'structure': struct,
            'b_factors': b_factors,
            'confidence_stats': {
                'mean': b_value_mean,
                'std': b_value_std,
                'min': b_value_min,
                'max': b_value_max
            },
            'structure_stats': structure_stats,
            'sequence': sequence
        }

    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. The sequence might be too long or the server is busy.")
        progress_bar.empty()
        status_text.empty()
        return None
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        progress_bar.empty()
        status_text.empty()
        return None


# Predict button
predict = st.sidebar.button('🚀 Predict Structure', type="primary")

if predict and validation_result['valid']:
    cleaned_seq = validation_result['cleaned_sequence']

    with st.spinner('🧬 Predicting protein structure...'):
        prediction_results = update_prediction(cleaned_seq)

        if prediction_results:
            st.session_state.prediction_results = prediction_results
            st.session_state.current_sequence = cleaned_seq

# Display results if available
if st.session_state.prediction_results:
    results = st.session_state.prediction_results

    # Main structure visualization
    st.subheader('🧬 3D Protein Structure')

    col1, col2 = st.columns([3, 1])

    with col1:
        confidence_data = results['b_factors'] if color_scheme == 'confidence' else None
        pdbview = render_mol_enhanced(
            results['pdb_string'],
            viz_style,
            color_scheme,
            confidence_data
        )
        showmol(pdbview, height=500, width=800)

    with col2:
        st.markdown("**🎯 Structure Info**")
        stats = results['confidence_stats']

        # Convert to 0-1 scale if needed for display
        mean_val = stats['mean']
        if mean_val > 1.0:
            display_mean = mean_val / 100.0
            display_std = stats['std'] / 100.0
            display_min = stats['min'] / 100.0
            display_max = stats['max'] / 100.0
        else:
            display_mean = mean_val
            display_std = stats['std']
            display_min = stats['min']
            display_max = stats['max']

        st.metric("Mean pLDDT", f"{display_mean:.3f}")
        st.metric("Std Dev", f"{display_std:.3f}")
        st.metric("Min pLDDT", f"{display_min:.3f}")
        st.metric("Max pLDDT", f"{display_max:.3f}")

        # Confidence level indicator with correct thresholds
        if display_mean > 0.9:
            st.success("🟢 Very High Confidence")
            st.caption("Very high accuracy expected")
        elif display_mean > 0.7:
            st.success("🔵 High Confidence")
            st.caption("Backbone likely correct")
        elif display_mean > 0.5:
            st.warning("🟡 Moderate Confidence")
            st.caption("Interpret with caution")
        else:
            st.error("🔴 Low Confidence")
            st.caption("Likely disordered/inaccurate")

    # Advanced Analysis Sections
    if show_confidence_plot:
        st.subheader('📊 Confidence Analysis')

        col1, col2 = st.columns(2)

        with col1:
            # Per-residue confidence plot
            conf_fig = plot_confidence_analysis(results['b_factors'], results['sequence'])
            st.plotly_chart(conf_fig, use_container_width=True)

        with col2:
            # Confidence distribution histogram
            hist_fig = create_confidence_histogram(results['b_factors'])
            st.plotly_chart(hist_fig, use_container_width=True)

        # Low confidence regions
        low_conf_regions = analyze_confidence_regions(
            results['b_factors'],
            results['sequence'],
            confidence_threshold
        )

        if not low_conf_regions.empty:
            st.subheader(f'⚠️ Low Confidence Regions (pLDDT < {confidence_threshold})')

            # Add interpretation guide
            st.info("""
            **Interpretation Guide:**
            - **>0.9**: Very high accuracy expected for that region
            - **0.7-0.9**: Backbone likely correct, side-chains less reliable
            - **0.5-0.7**: Low confidence, interpret with caution
            - **<0.5**: Likely disordered or very inaccurate
            """)

            st.dataframe(low_conf_regions, use_container_width=True)
        else:
            st.success(f"✅ No regions with pLDDT < {confidence_threshold} found!")

    if show_structure_analysis:
        st.subheader('🔬 Structural Analysis')

        col1, col2, col3 = st.columns(3)

        struct_stats = results['structure_stats']

        with col1:
            st.metric("Number of Residues", struct_stats['num_residues'])
            st.metric("Radius of Gyration", f"{struct_stats['radius_gyration']:.2f} Å")

        with col2:
            st.metric("Max Distance", f"{struct_stats['max_distance']:.2f} Å")
            centroid = struct_stats['centroid']
            st.write(f"**Centroid:** ({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})")

        with col3:
            # Calculate additional metrics
            if struct_stats['num_residues'] > 0 and struct_stats['radius_gyration'] > 0:
                compactness = struct_stats['radius_gyration'] / (struct_stats['num_residues'] ** (1 / 3))
                st.metric("Compactness Index", f"{compactness:.3f}")
            else:
                st.metric("Compactness Index", "N/A")

    if show_sequence_analysis:
        st.subheader('📋 Sequence Composition Analysis')

        # Amino acid composition
        sequence = results['sequence']
        composition = {aa: sequence.count(aa) for aa in set(sequence)}
        composition_pct = {aa: (count / len(sequence)) * 100 for aa, count in composition.items()}

        # Create composition chart
        aa_df = pd.DataFrame([
            {'Amino Acid': aa, 'Count': count, 'Percentage': composition_pct[aa]}
            for aa, count in composition.items()
        ]).sort_values('Percentage', ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            # Bar chart of composition
            fig = px.bar(aa_df.head(10), x='Amino Acid', y='Percentage',
                         title='Top 10 Amino Acid Composition')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Display composition table
            st.dataframe(aa_df, use_container_width=True)

    # Download section
    st.subheader('💾 Download Results')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            label="📄 Download PDB",
            data=results['pdb_string'],
            file_name='predicted_structure.pdb',
            mime='text/plain',
        )

    with col2:
        # Create analysis report
        report_data = {
            'sequence_length': len(results['sequence']),
            'mean_plddt': results['confidence_stats']['mean'],
            'structure_stats': results['structure_stats']
        }

        st.download_button(
            label="📊 Download Analysis Report",
            data=str(report_data),
            file_name='analysis_report.txt',
            mime='text/plain',
        )

    with col3:
        # Confidence data as CSV - handle length mismatch
        b_factors = results['b_factors']
        sequence = results['sequence']

        # Ensure arrays are the same length
        min_length = min(len(b_factors), len(sequence))
        b_factors_export = b_factors[:min_length]

        # Convert to 0-1 scale if needed
        if np.max(b_factors_export) > 1.0:
            b_factors_export = b_factors_export / 100.0

        conf_df = pd.DataFrame({
            'Residue': range(1, min_length + 1),
            'Amino_Acid': list(sequence[:min_length]),
            'pLDDT': b_factors_export
        })

        if len(b_factors) != len(sequence):
            st.warning(
                f"Note: Sequence length ({len(sequence)}) and B-factors length ({len(b_factors)}) don't match. Using first {min_length} residues.")

        st.download_button(
            label="📈 Download Confidence Data",
            data=conf_df.to_csv(index=False),
            file_name='confidence_data.csv',
            mime='text/csv',
        )

elif not txt:
    st.info('👈 Enter a protein sequence in the sidebar to get started!')
elif not validation_result.get('valid', False):
    st.warning('⚠️ Please fix the sequence validation errors shown in the sidebar before predicting.')
else:
    st.info('👈 Click "Predict Structure" to analyze your sequence!')

# Footer
st.markdown("---")
st.markdown("""
Original concept by [osanseviero](https://huggingface.co/osanseviero) 
| Enhanced with advanced analysis features | ESMFold by Meta AI
""")