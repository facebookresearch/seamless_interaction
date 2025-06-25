# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import plotly.express as px

from seamless_interaction.fs import SeamlessInteractionFS
from seamless_interaction.constants import ALL_LABELS, ALL_SPLITS
from seamless_interaction.app.config import CSS, DatasetStats

# Page configuration
st.set_page_config(
    page_title="Seamless Interaction Dataset Browser",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data
def load_metadata(label: str = "improvised", split: str = "dev") -> pd.DataFrame:
    """Load the metadata CSV file."""
    metadata_path = f"filelists/{label}/{split}/metadata.csv"
    if os.path.exists(metadata_path):
        return pd.read_csv(metadata_path)
    else:
        st.error(f"Metadata file not found: {metadata_path}")
        return pd.DataFrame()

@st.cache_resource
def get_fs_instance() -> SeamlessInteractionFS:
    """Initialize the SeamlessInteractionFS instance."""
    return SeamlessInteractionFS()

def calculate_dataset_stats(df: pd.DataFrame) -> DatasetStats:
    """Calculate dataset statistics."""
    if df.empty:
        return DatasetStats(0, 0, [], {}, {})
    
    total_interactions = len(df)
    participants = set(df['participant_0_id'].unique()) | set(df['participant_1_id'].unique())
    total_participants = len(participants)
    vendors = df['vendor'].unique().tolist()
    
    sessions_per_vendor = df.groupby('vendor')['session'].nunique().to_dict()
    interactions_per_session = df.groupby(['vendor', 'session']).size().to_dict()
    
    return DatasetStats(
        total_interactions=total_interactions,
        total_participants=total_participants,
        vendors=vendors,
        sessions_per_vendor=sessions_per_vendor,
        interactions_per_session=interactions_per_session
    )

def display_overview_stats(stats: DatasetStats):
    """Display overview statistics."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Interactions", stats.total_interactions)
    with col2:
        st.metric("Total Participants", stats.total_participants)
    with col3:
        st.metric("Vendors", len(stats.vendors))
    with col4:
        avg_interactions = np.mean(list(stats.interactions_per_session.values())) if stats.interactions_per_session else 0
        st.metric("Avg Interactions/Session", f"{avg_interactions:.1f}")

def create_distribution_plots(df: pd.DataFrame):
    """Create distribution plots for the dataset."""
    if df.empty:
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Interactions per Vendor")
        vendor_counts = df['vendor'].value_counts()
        fig = px.bar(
            x=vendor_counts.index, 
            y=vendor_counts.values,
            labels={'x': 'Vendor', 'y': 'Number of Interactions'},
            color=vendor_counts.values,
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Sessions per Vendor")
        session_counts = df.groupby('vendor')['session'].nunique()
        fig = px.pie(
            values=session_counts.values,
            names=session_counts.index,
            title="Distribution of Sessions"
        )
        st.plotly_chart(fig, use_container_width=True)

def download_archive_files(fs: SeamlessInteractionFS, label: str, split: str, batch: str, archive: str) -> tuple[bool, str]:
    """Download and extract a specific archive from HuggingFace."""
    try:
        with st.spinner(f"Downloading archive {batch}/{archive}..."):
            success, extract_path = fs.download_archive(
                label=label,
                split=split, 
                batch=batch,
                archive=archive,
                extract=True
            )
            
            if success:
                # Update the local directory in fs to point to extracted directory
                fs._local_dir = os.path.dirname(extract_path)
                return True, extract_path
            else:
                return False, ""
                
    except Exception as e:
        st.error(f"Error downloading archive: {e}")
        return False, ""

def display_hf_browser(fs: SeamlessInteractionFS):
    """Browse the HuggingFace dataset structure and download archives."""
    st.subheader("🤗 HuggingFace Dataset Browser")
    st.markdown("Browse and download tar archives from the HuggingFace dataset repository.")
    
    # Dataset structure info
    with st.expander("📁 Dataset Structure", expanded=False):
        st.code("""
        datasets/facebook/seamless-interaction/
        ├── improvised/
        │   ├── train/
        │   │   ├── 0000/
        │   │   │   ├── 0000.tar
        │   │   │   ├── 0001.tar
        │   │   │   └── ...
        │   │   ├── 0001/
        │   │   └── ...
        │   ├── dev/
        │   └── test/
        └── naturalistic/
            ├── train/
            ├── dev/
            └── test/
        
        Each tar file contains:
        - .mp4 files (video)
        - .wav files (audio) 
        - .json files (annotations, metadata)
        - .npz files (movement, smplh, keypoints data)
        """)
    
    # Selection interface
    col1, col2 = st.columns(2)
    
    with col1:
        label = st.selectbox("Label:", ALL_LABELS, key="hf_label")
        split = st.selectbox("Split:", ALL_SPLITS, key="hf_split")
    
    with col2:
        # List available batches
        try:
            batches = fs.list_batches(label, split)
            if batches:
                batch = st.selectbox("Batch:", batches, key="hf_batch")
                
                # List available archives in selected batch
                archives = fs.list_archives(label, split, batch)
                if archives:
                    archive = st.selectbox("Archive:", archives, key="hf_archive")
                    
                    # Download button
                    if st.button(f"📥 Download {batch}/{archive}", use_container_width=True):
                        success, extract_path = download_archive_files(fs, label, split, batch, archive)
                        if success:
                            st.success(f"✅ Archive downloaded and extracted to: {extract_path}")
                            
                            # Show extracted files
                            try:
                                files = os.listdir(extract_path)
                                st.markdown("### 📄 Extracted Files")
                                for file in sorted(files)[:20]:  # Show first 20 files
                                    file_path = os.path.join(extract_path, file)
                                    file_size = os.path.getsize(file_path)
                                    st.text(f"📄 {file} ({file_size:,} bytes)")
                                    
                                if len(files) > 20:
                                    st.text(f"... and {len(files) - 20} more files")
                                    
                            except Exception as e:
                                st.error(f"Error listing extracted files: {e}")
                        else:
                            st.error("❌ Failed to download archive")
                else:
                    st.warning(f"No archives found in batch {batch}")
            else:
                st.warning(f"No batches found for {label}/{split}")
                
        except Exception as e:
            st.error(f"Error browsing HuggingFace dataset: {e}")

def check_local_files(fs: SeamlessInteractionFS, participant_id: str) -> dict:
    """Check what files are available locally for a participant."""
    try:
        paths = fs.get_path_list_for_file_id(participant_id, local=True)
        
        file_status = {
            'video': None,
            'audio': None,
            'annotations': [],
            'metadata': [],
            'movement_smplh': [],
            'total_files': 0,
            'available_files': 0
        }
        
        for path in paths:
            file_status['total_files'] += 1
            if os.path.exists(path):
                file_status['available_files'] += 1
                
                if path.endswith('.mp4'):
                    file_status['video'] = path
                elif path.endswith('.wav'):
                    file_status['audio'] = path
                elif '.json' in path and ('annotation' in path.lower() or 'transcript' in path.lower()):
                    file_status['annotations'].append(path)
                elif '.json' in path or '.jsonl' in path:
                    file_status['metadata'].append(path)
                elif '.npz' in path or '.npy' in path:
                    file_status['movement_smplh'].append(path)
        
        return file_status
        
    except Exception as e:
        return {
            'video': None,
            'audio': None,
            'annotations': [],
            'metadata': [],
            'movement_smplh': [],
            'total_files': 0,
            'available_files': 0,
            'error': str(e)
        }


def display_file_status_widget(participant_id: str, file_status: dict):
    """Display a compact file status widget."""
    if 'error' in file_status:
        st.error(f"Error checking files: {file_status['error']}")
        return
    
    total = file_status['total_files']
    available = file_status['available_files']
    
    if total == 0:
        st.warning("No files expected (participant not in metadata)")
        return
    
    # Status indicator
    if available == 0:
        status_color = "🔴"
        status_text = "No files downloaded"
    elif available == total:
        status_color = "🟢"
        status_text = "All files available"
    else:
        status_color = "🟡"
        status_text = f"{available}/{total} files available"
    
    st.markdown(f"**{status_color} {status_text}**")
    
    # Quick file type status
    file_types = []
    if file_status['video']:
        file_types.append("📹 Video")
    if file_status['audio']:
        file_types.append("🎵 Audio")
    if file_status['annotations']:
        file_types.append(f"📝 Annotations ({len(file_status['annotations'])})")
    if file_status['metadata']:
        file_types.append(f"📊 Metadata ({len(file_status['metadata'])})")
    if file_status['movement_smplh']:
        file_types.append(f"🤖 Movement/SMPLH ({len(file_status['movement_smplh'])})")
    
    if file_types:
        st.markdown("Available: " + " | ".join(file_types))


def suggest_download_options(fs: SeamlessInteractionFS, participant_id: str, label: str, split: str):
    """Suggest which archives might contain the participant's files."""
    st.markdown("### 📥 Download Options")
    
    # Try to get available batches
    try:
        batches = fs.list_batches(label, split)
        if not batches:
            st.warning(f"No batches found for {label}/{split}")
            return
        
        st.markdown(f"**Available batches for {label}/{split}:**")
        
        # Show first few batches with download options
        for i, batch in enumerate(batches[:5]):  # Show first 5 batches
            archives = fs.list_archives(label, split, batch)
            if archives:
                with st.expander(f"📁 Batch {batch} ({len(archives)} archives)"):
                    # Show first few archives
                    for j, archive in enumerate(archives[:3]):  # Show first 3 archives
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.text(f"📦 {archive}")
                        with col2:
                            if st.button(f"Download", key=f"dl_{batch}_{archive}_{participant_id}"):
                                success, extract_path = download_archive_files(fs, label, split, batch, archive)
                                if success:
                                    st.success(f"✅ Downloaded to: {extract_path}")
                                    st.rerun()
                                else:
                                    st.error("❌ Download failed")
                    
                    if len(archives) > 3:
                        st.text(f"... and {len(archives) - 3} more archives")
        
        if len(batches) > 5:
            st.markdown(f"*... and {len(batches) - 5} more batches. Use the HuggingFace Browser tab to see all options.*")
            
    except Exception as e:
        st.error(f"Error getting download options: {e}")
        st.markdown("💡 **Tip**: Use the HuggingFace Browser tab to manually browse and download archives.")


def display_participant_videos(fs: SeamlessInteractionFS, participant_0_id: str, participant_1_id: str):
    """Display videos for both participants side by side with local file checking."""
    st.markdown('<div class="interaction-header">', unsafe_allow_html=True)
    st.markdown(f"<h3>🎭 Interaction: {participant_0_id.split('_')[2]} | Session: {participant_0_id.split('_')[1]}</h3>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Get current label and split from session state or default
    label = st.session_state.get('current_label', 'improvised')
    split = st.session_state.get('current_split', 'dev')
    
    # Check local files for both participants
    status_0 = check_local_files(fs, participant_0_id)
    status_1 = check_local_files(fs, participant_1_id)
    
    cols = st.columns(2)
    
    with cols[0]:
        st.markdown("### 👤 Participant 0")
        st.text(f"ID: {participant_0_id}")
        
        # Display file status
        display_file_status_widget(participant_0_id, status_0)
        
        # Show video if available
        if status_0['video'] and os.path.exists(status_0['video']):
            st.video(status_0['video'])
        else:
            st.warning("Video not available locally")
            
    with cols[1]:
        st.markdown("### 👤 Participant 1") 
        st.text(f"ID: {participant_1_id}")
        
        # Display file status
        display_file_status_widget(participant_1_id, status_1)
        
        # Show video if available
        if status_1['video'] and os.path.exists(status_1['video']):
            st.video(status_1['video'])
        else:
            st.warning("Video not available locally")
    
    # Sync play button - only show if both videos exist
    if (status_0['video'] and os.path.exists(status_0['video']) and 
        status_1['video'] and os.path.exists(status_1['video'])):
        if st.button("🎬 Sync Play All Videos", use_container_width=True):
            st.components.v1.html(
                """<script>
                let videos = parent.document.querySelectorAll("video");
                videos.forEach(v => {
                    v.currentTime = 0;
                    v.play();
                })
                </script>""", 
                width=0, height=0
            )
    
    # Show download suggestions if files are missing
    total_available = status_0['available_files'] + status_1['available_files']
    total_expected = status_0['total_files'] + status_1['total_files']
    
    if total_available < total_expected and total_expected > 0:
        st.markdown("---")
        st.markdown("### 🔍 Missing Files Detected")
        
        missing_0 = status_0['total_files'] - status_0['available_files']
        missing_1 = status_1['total_files'] - status_1['available_files']
        
        if missing_0 > 0 or missing_1 > 0:
            st.info(f"💡 **Missing files**: Participant 0: {missing_0}, Participant 1: {missing_1}")
            
            # Show download suggestions
            with st.expander("📥 Download Missing Files", expanded=False):
                st.markdown("**Suggested approach:**")
                st.markdown("1. Files are organized in batches and archives")
                st.markdown("2. Each archive contains multiple participants")
                st.markdown("3. Download archives that might contain your participants")
                
                # Show download options for the current label/split
                suggest_download_options(fs, participant_0_id, label, split)
    
    elif total_expected > 0:
        st.success("✅ All expected files are available locally!")

def display_multimodal_data(fs: SeamlessInteractionFS, participant_id: str):
    """Display multimodal data for a participant."""
    
    st.info("💡 **Note**: Download archives from the HuggingFace Browser to access multimodal data files.")
    
    try:
        paths = fs.get_path_list_for_file_id(participant_id, local=True)
        
        # Organize paths by modality
        modalities = {
            'Video': [],
            'Audio': [],
            'Annotations': [],
            'Metadata': [],
            'Movement/SMPLH': [],
            'Other': []
        }
        
        for path in paths:
            if '.mp4' in path:
                modalities['Video'].append(path)
            elif '.wav' in path:
                modalities['Audio'].append(path)
            elif '.json' in path and ('annotation' in path.lower() or 'transcript' in path.lower()):
                modalities['Annotations'].append(path)
            elif '.json' in path or '.jsonl' in path:
                modalities['Metadata'].append(path)
            elif '.npz' in path or '.npy' in path:
                modalities['Movement/SMPLH'].append(path)
            else:
                modalities['Other'].append(path)
        
        st.subheader(f"📊 Multimodal Data for {participant_id}")
        
        if not any(modalities.values()):
            st.warning("No local files found. Download archives from the HuggingFace Browser tab.")
            return
        
        for modality, files in modalities.items():
            if files:
                with st.expander(f"{modality} ({len(files)} files)"):
                    for file_path in files:
                        exists = os.path.exists(file_path)
                        status_icon = "✅" if exists else "❌"
                        st.text(f"{status_icon} {os.path.basename(file_path)}")
                        
                        # Try to load and preview some files
                        if exists and modality == 'Metadata' and (file_path.endswith('.json') or file_path.endswith('.jsonl')):
                            try:
                                with open(file_path, 'r') as f:
                                    if file_path.endswith('.jsonl'):
                                        lines = f.readlines()[:3]  # Preview first 3 lines
                                        data = [json.loads(line) for line in lines]
                                    else:
                                        data = json.load(f)
                                    st.json(data)
                            except Exception as e:
                                st.error(f"Error reading {file_path}: {e}")
                                
                        elif exists and modality == 'Movement/SMPLH' and file_path.endswith('.npz'):
                            try:
                                data = np.load(file_path)
                                st.text(f"Arrays: {list(data.keys())}")
                                for key in list(data.keys())[:3]:  # Show first 3 arrays
                                    arr = data[key]
                                    st.text(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
                            except Exception as e:
                                st.error(f"Error reading {file_path}: {e}")
                                
    except Exception as e:
        st.error(f"Error loading multimodal data: {e}")

def main():
    """Main Streamlit application."""
    st.title("🎭 Seamless Interaction Dataset Browser")
    st.markdown("---")
    
    # Initialize filesystem
    fs = get_fs_instance()
    
    fs.fetch_all_filelist()
    
    # Sidebar for navigation
    st.sidebar.title("🔍 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Overview", "Interaction Viewer", "HuggingFace Browser", "Dataset Analysis"]
    )
    
    # Label and split selection
    st.sidebar.markdown("### Dataset Configuration")
    label = st.sidebar.selectbox("Label:", ["improvised", "naturalistic"], index=0)
    split = st.sidebar.selectbox("Split:", ["dev", "train", "test"], index=0)
    
    # Store in session state for use in other functions
    st.session_state['current_label'] = label
    st.session_state['current_split'] = split
    
    # Load metadata
    df = load_metadata(label, split)
    
    if page == "Overview":
        st.header("📊 Dataset Overview")
        
        if not df.empty:
            stats = calculate_dataset_stats(df)
            display_overview_stats(stats)
            
            st.markdown("---")
            create_distribution_plots(df)
            
            st.markdown("---")
            st.subheader("📋 Sample Data")
            st.dataframe(df.head(10), use_container_width=True)
            
        else:
            st.warning("No metadata available for the selected configuration.")
    
    elif page == "Interaction Viewer":
        st.header("🎬 Interaction Viewer")
        
        if not df.empty:
            # Interaction selection
            interaction_options = df.apply(
                lambda row: f"{row['vendor']} - {row['session']} - {row['interaction']}", 
                axis=1
            ).tolist()
            
            selected_idx = st.selectbox(
                "Select an interaction:",
                range(len(interaction_options)),
                format_func=lambda i: interaction_options[i]
            )
            
            if selected_idx is not None:
                selected_row = df.iloc[selected_idx]
                participant_0_id = selected_row['participant_0_id']
                participant_1_id = selected_row['participant_1_id']
                
                # Display participant videos with integrated file checking
                display_participant_videos(fs, participant_0_id, participant_1_id)
                
                st.markdown("---")
                
                # Tabs for additional data
                tab1, tab2, tab3 = st.tabs(["Participant 0", "Participant 1", "Interaction Info"])
                
                with tab1:
                    display_multimodal_data(fs, participant_0_id)
                    
                with tab2:
                    display_multimodal_data(fs, participant_1_id)
                    
                with tab3:
                    st.subheader("Interaction Metadata")
                    info_dict = selected_row.to_dict()
                    st.json(info_dict)
        else:
            st.warning("No interaction data available for the selected configuration.")
    
    elif page == "HuggingFace Browser":
        display_hf_browser(fs)
        
    elif page == "Dataset Analysis":
        st.header("📈 Dataset Analysis")
        
        if not df.empty:
            # Vendor analysis
            st.subheader("Vendor Analysis")
            vendor_stats = df.groupby('vendor').agg({
                'session': 'nunique',
                'interaction': 'nunique',
                'participant_0_id': 'nunique'
            }).rename(columns={
                'session': 'Sessions',
                'interaction': 'Interactions',
                'participant_0_id': 'Unique Participants'
            })
            st.dataframe(vendor_stats, use_container_width=True)
            
            # Session analysis
            st.subheader("Session Analysis")
            session_interaction_counts = df.groupby(['vendor', 'session']).size().reset_index(name='interaction_count')
            
            fig = px.box(
                session_interaction_counts,
                x='vendor',
                y='interaction_count',
                title="Distribution of Interactions per Session by Vendor"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Interaction ID analysis
            st.subheader("Interaction ID Patterns")
            df['interaction_num'] = df['interaction'].str.extract(r'I(\d+)').astype(int)
            
            fig = px.histogram(
                df,
                x='interaction_num',
                nbins=50,
                title="Distribution of Interaction Numbers"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("No data available for analysis.")

if __name__ == "__main__":
    main() 