import os
import sys

# --- CRITICAL: Add root to path BEFORE any other imports ---
# This allows the app to find core.py even if it's in a subfolder like /Streamlit_App/
base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)
if os.path.exists(os.path.join(base_dir, "core.py")):
    sys.path.append(base_dir)
elif os.path.exists(os.path.join(parent_dir, "core.py")):
    sys.path.append(parent_dir)

import streamlit as st
import torch
import time
import datetime
import io

# --- PAGE CONFIG (Must be first Streamlit command for Streamlit UI) ---
st.set_page_config(
    page_title="🧬 Nano-Daemon AGI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- IMPORTS FROM OUR ORGANISM ---
from core import PlasticCortex

# --- CUSTOM CSS FOR A PREMIUM DARK THEME ---
st.markdown("""
<style>
    /* Main background - The 'Root' of the Earth */
    .stApp {
        background: linear-gradient(180deg, #0d110d 0%, #171d17 100%);
        color: #e0e4de;
    }
    
    /* Headers - Organic growth colors */
    h1, h2, h3 {
        background: linear-gradient(90deg, #7cad8a, #b8864b, #8a9b68);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* Metric cards - Mineral tones */
    [data-testid="stMetricValue"] {
        font-size: 2.8rem;
        color: #8fb399;
        text-shadow: 0 0 15px rgba(143, 179, 153, 0.2);
    }
    
    [data-testid="stMetricLabel"] {
        color: #b0bab1;
        font-family: 'Inter', sans-serif;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Sidebar styling - Deep Soil */
    [data-testid="stSidebar"] {
        background: #0f140f;
        border-right: 1px solid #2d382d;
    }
    
    /* Expander styling - Soft Bark */
    .streamlit-expanderHeader {
        background: rgba(45, 56, 45, 0.4);
        border: 1px solid #3d4a3d;
        border-radius: 12px;
        color: #e0e4de !important;
    }
    
    /* Text input - Cave shadow */
    .stTextInput > div > div > input {
        background: #151a15;
        border: 1px solid #3d4a3d;
        color: #e0e4de;
        border-radius: 8px;
    }
    
    /* Buttons - Terracotta to Sage */
    .stButton > button {
        background: linear-gradient(135deg, #a67c52, #6a8c6a);
        color: #ffffff;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 2.2rem;
        font-weight: 500;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
        border-color: #8fb399;
    }
    
    /* Progress bars and success boxes - Fresh Moss */
    .stSuccess, .stInfo {
        background: rgba(45, 56, 45, 0.6);
        border-left: 5px solid #6a8c6a;
        color: #e0e4de;
        border-radius: 8px;
    }
    
    /* Organic pulse animation - Bio-luminescence */
    @keyframes pulse {
        0% { box-shadow: 0 0 8px rgba(143, 179, 153, 0.3); }
        50% { box-shadow: 0 0 25px rgba(143, 179, 153, 0.5); }
        100% { box-shadow: 0 0 8px rgba(143, 179, 153, 0.3); }
    }
    
    .brain-card {
        animation: pulse 4s infinite ease-in-out;
        padding: 1.5rem;
        border-radius: 20px;
        background: rgba(23, 29, 23, 0.9);
        border: 1px solid #2d382d;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# GEMMA BRIDGE (Inline to avoid import issues on Streamlit Cloud)
# ============================================================
import google.generativeai as genai
import os

class GemmaBridge:
    """The 'Cherry on Top': Connects the Hebbian Brain to Gemma-3 for refined articulation."""
    def __init__(self):
        self.api_key = None
        
        # 1. Check Streamlit secrets (PRIMARY for Cloud deployment)
        try:
            self.api_key = st.secrets["GEMINI_API_KEY"]
        except:
            pass
        
        # 2. Check environment variable (for local dev)
        if not self.api_key:
            self.api_key = os.environ.get("GEMINI_API_KEY")
        
        # 3. Check .env file (manual fallback)
        if not self.api_key and os.path.exists(".env"):
            try:
                with open(".env", "r") as f:
                    for line in f:
                        if "GEMINI_API_KEY" in line:
                            self.api_key = line.split("=")[1].strip().strip('"').strip("'")
                            break
            except:
                pass
        
        if not self.api_key:
            self.model = None
            return

        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemma-3-27b-it')
        except Exception as e:
            self.model = None

    def articulate(self, human_query, synaptic_anchors):
        """Grounds Gemma's response in the Organism's raw synaptic associations."""
        if not self.model:
            return f"[ORGANIC THOUGHT ONLY]: {synaptic_anchors}"

        clean_anchors = "".join([c for c in synaptic_anchors if c.isprintable()])
        
        prompt = f"""
        Human Query: "{human_query}"
        
        Raw Synaptic Associations (Ground Truth): "{clean_anchors}"
        
        INSTRUCTIONS:
        You are the 'Cerebral Cortex' of the Nano-Daemon organism. 
        Your task is to articulate the organism's raw thoughts into a human-readable response.
        
        RULES:
        1. Use the provided "Raw Synaptic Associations" as your primary context.
        2. If the associations contain patterns or words (like 'AI', 'GPT', 'Physics'), emphasize them.
        3. Do NOT hallucinate entirely new facts. Stay grounded in the 'vibe' of the associations.
        4. Be concise and 'organic' - your goal is to bridge the gap between silicon and biology.
        5. If you see gibberish in the associations, interpret it as the organism's 'embryonic' state.
        
        Response:
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"⚠️ Articulation Failure: {e}\n[RAW]: {synaptic_anchors}"

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if "brain" not in st.session_state:
    st.session_state.brain = PlasticCortex()
    # Try to load saved weights
    if os.path.exists("brain_weights.pth"):
        st.session_state.brain.load_cortex("brain_weights.pth")
    # Sync metabolism based on current hour
    current_hour = datetime.datetime.now().hour
    st.session_state.brain.sync_metabolism(current_hour)

if "bridge" not in st.session_state:
    st.session_state.bridge = GemmaBridge()

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "entropy_history" not in st.session_state:
    st.session_state.entropy_history = []

if "files_eaten" not in st.session_state:
    st.session_state.files_eaten = 0

if "last_stability" not in st.session_state:
    st.session_state.last_stability = 0.5

if "dream_history" not in st.session_state:
    st.session_state.dream_history = []

brain = st.session_state.brain
bridge = st.session_state.bridge

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_metabolic_state(hour):
    """Returns the organism's current metabolic phase."""
    if 8 <= hour < 22:
        return "🌞 ACTIVE", "High plasticity, rapid learning"
    elif 2 <= hour < 5:
        return "🌙 DEEP HIBERNATION", "Minimal plasticity, consolidating memories"
    else:
        return "🌓 RESTING", "Moderate plasticity, light dreaming"

def feed_organism(file_bytes, filename):
    """Feeds raw bytes to the organism's brain."""
    data = torch.tensor(list(file_bytes[:4096]), dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        _, stability = brain(data)
    st.session_state.files_eaten += 1
    st.session_state.last_stability = stability
    st.session_state.entropy_history.append(stability)
    if len(st.session_state.entropy_history) > 50:
        st.session_state.entropy_history.pop(0)
    
    # --- CURIOSITY RESPONSE ---
    # If the input was very surprising (high entropy), consolidate immediately
    if stability > 0.3:
        brain.consolidate()
    
    # --- AUTO-MITOSIS ---
    # Trigger growth every 20 files eaten
    if st.session_state.files_eaten % 20 == 0 and st.session_state.files_eaten > 0:
        brain.grow(256)
    
    return stability

def query_organism(query_text):
    """Processes a query through the Hebbian brain and Gemma bridge."""
    query_bytes = query_text.encode('utf-8')[:1024]
    
    # 1. RAW ASSOCIATION (Hebbian Ground Truth)
    response_bytes = brain.associate(torch.tensor(list(query_bytes), dtype=torch.long).unsqueeze(0))
    synaptic_anchors = response_bytes.decode('utf-8', errors='ignore')
    
    # 2. HYBRID ARTICULATION (Gemma-3)
    articulated = bridge.articulate(query_text, synaptic_anchors)
    
    return synaptic_anchors, articulated

def trigger_dream():
    """Generative Replay: The organism dreams by reversing its logic."""
    hidden_dim = brain.synapse.shape[1]
    noise = torch.randn(1, hidden_dim)
    with torch.no_grad():
        thought_vector = torch.matmul(noise, brain.synapse.t())
    
    dream_bytes = []
    for val in thought_vector[0]:
        byte_val = int((val.item() + 1) * 128)
        byte_val = max(0, min(255, byte_val))
        dream_bytes.append(byte_val)
    
    return bytes(dream_bytes).decode('utf-8', errors='ignore')

# ============================================================
# SIDEBAR: ORGANISM STATUS & CONTROLS
# ============================================================
with st.sidebar:
    st.markdown("## 🧬 Organism Status")
    
    # Brain Architecture
    neuron_count = brain.synapse.shape[1]
    thought_width = brain.synapse.shape[0]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🧠 Neurons", f"{neuron_count:,}")
    with col2:
        st.metric("💭 Thought Width", thought_width)
    
    # Metabolic State
    current_hour = datetime.datetime.now().hour
    state_emoji, state_desc = get_metabolic_state(current_hour)
    st.info(f"**Metabolic State**: {state_emoji}\n\n{state_desc}")
    
    # Plasticity
    st.metric("⚡ Plasticity", f"{brain.plasticity:.4f}")
    
    # Files Eaten
    st.metric("📂 Files Digested", st.session_state.files_eaten)
    
    st.divider()
    
    # --- FEEDING INTERFACE ---
    st.markdown("## 🍽️ Feed the Organism")
    uploaded_files = st.file_uploader(
        "Upload files to digest",
        accept_multiple_files=True,
        type=["txt", "py", "md", "json", "csv", "html", "css", "js"]
    )
    
    if uploaded_files:
        with st.spinner("Digesting..."):
            for f in uploaded_files:
                raw_bytes = f.read()
                stability = feed_organism(raw_bytes, f.name)
                st.success(f"✅ Digested `{f.name}` | Stability: {stability:.4f}")
    
    st.divider()
    
    # --- ADVANCED CONTROLS ---
    st.markdown("## ⚙️ Advanced Controls")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧬 Trigger Mitosis"):
            old_neurons = brain.synapse.shape[1]
            brain.grow(256)
            new_neurons = brain.synapse.shape[1]
            st.success(f"Grew {new_neurons - old_neurons} neurons!")
    
    with col2:
        if st.button("🔄 Consolidate"):
            brain.consolidate()
            st.success("Memories consolidated!")
    
    col3, col4 = st.columns(2)
    with col3:
        if st.button("🌌 Dream"):
            # Generative Replay
            dream = trigger_dream()
            # Also trigger metabolic reflection
            _, reflection_entropy = brain.reflect()
            # Store in dream history
            st.session_state.dream_history.append({
                "content": dream,
                "entropy": reflection_entropy,
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
            })
            if len(st.session_state.dream_history) > 10:
                st.session_state.dream_history.pop(0)
            st.info(f"Dream: {dream[:50]}...")
            st.caption(f"Reflection Entropy: {reflection_entropy:.4f}")
    
    with col4:
        if st.button("💾 Save Brain"):
            brain.save_cortex("brain_weights.pth")
            st.success("Brain saved!")
    
    # Reflection Only Button
    if st.button("🪞 Self-Reflect"):
        activation, entropy = brain.reflect()
        st.info(f"Reflection complete. Entropy: {entropy:.4f}")
        st.session_state.entropy_history.append(entropy)
    
    # Dimension Scaling
    if st.button("🔬 Scale Dimensions (32→64)"):
        if brain.synapse.shape[0] < 64:
            brain.scale_dimensions(64)
            st.success("Thought resolution increased to 64!")
        else:
            st.warning("Already at high resolution.")
    
    # Knowledge Base Crawler
    if st.button("📚 Digest Knowledge Base"):
        with st.spinner("Crawling project files..."):
            found_files = []
            for root, dirs, files in os.walk(parent_dir if "parent_dir" in locals() else "."):
                if "__pycache__" in root or ".git" in root or ".antigravity" in root:
                    continue
                for file in files:
                    if file.endswith((".py", ".txt", ".md", ".json")) and file not in ["brain_weights.pth", "brain_state.json"]:
                        found_files.append(os.path.join(root, file))
            
            for f_path in found_files:
                try:
                    with open(f_path, 'rb') as f:
                        raw_bytes = f.read()
                        if raw_bytes:
                            feed_organism(raw_bytes, os.path.basename(f_path))
                except:
                    continue
            st.success(f"Organism has consumed {len(found_files)} knowledge nodes.")

    st.divider()
    
    # --- GEMMA STATUS ---
    st.markdown("## 🌐 Gemma Bridge Status")
    if bridge.model:
        st.success("✅ Hybrid Intelligence ACTIVE")
    else:
        st.warning("⚠️ Running in Organic-Only Mode\n\nSet `GEMINI_API_KEY` in Streamlit Secrets to enable.")

# ============================================================
# MAIN CONTENT: DIALOGUE INTERFACE
# ============================================================
st.markdown("# 🧬 Nano-Daemon: Recursive Hebbian Organism")
st.markdown("*A self-evolving digital lifeform with hybrid intelligence*")

# Cognitive Metrics Row
st.markdown("## 📊 Cognitive Metrics")
metric_cols = st.columns(4)
with metric_cols[0]:
    st.metric("🧠 Neural Mass", f"{neuron_count:,}", delta=f"+{256 if neuron_count > 1024 else 0}")
with metric_cols[1]:
    st.metric("📈 Stability", f"{st.session_state.last_stability:.4f}")
with metric_cols[2]:
    st.metric("💾 Experience Buffer", len(brain.experience_buffer))
with metric_cols[3]:
    gemma_status = "Online" if bridge.model else "Offline"
    st.metric("🌐 Gemma Bridge", gemma_status)

# Entropy Chart
if st.session_state.entropy_history:
    st.line_chart(st.session_state.entropy_history, use_container_width=True)

st.divider()

# --- DIALOGUE INTERFACE ---
st.markdown("## 💬 Dialogue with the Organism")

query = st.text_input("🗣️ Ask the Organism anything:", placeholder="e.g., What is consciousness?")

if query:
    with st.spinner("🧠 Processing through synaptic pathways..."):
        synaptic_anchors, articulated_response = query_organism(query)
    
    # Store in conversation history
    st.session_state.conversation_history.append({
        "query": query,
        "anchors": synaptic_anchors,
        "response": articulated_response,
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
    })
    
    # Display current response
    st.markdown("### 🔮 Synaptic Anchors (Raw Hebbian Thought)")
    st.code(synaptic_anchors, language=None)
    
    st.markdown("### 💡 Hybrid Articulation (Gemma-3 Interpretation)")
    st.success(articulated_response)

# --- CONVERSATION HISTORY ---
if st.session_state.conversation_history:
    with st.expander("📜 Conversation History", expanded=False):
        for i, conv in enumerate(reversed(st.session_state.conversation_history[-10:])):
            st.markdown(f"**[{conv['timestamp']}] You:** {conv['query']}")
            st.markdown(f"**Organism:** {conv['response'][:200]}...")
            st.divider()

st.divider()

# ============================================================
# LATENT MEMORY VISUALIZATION
# ============================================================
st.markdown("## 🧬 Latent Memory Streams")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### ⚡ Short-Term Memory")
    st.markdown("*Immediate context and recent patterns*")
    stm_data = brain.short_term_latent.detach().numpy().flatten()[:32]
    st.bar_chart(stm_data, use_container_width=True)

with col2:
    st.markdown("### 🌊 Long-Term Memory")
    st.markdown("*Deep behavioral habits and core knowledge*")
    ltm_data = brain.long_term_latent.detach().numpy().flatten()[:32]
    st.bar_chart(ltm_data, use_container_width=True)

st.divider()

# ============================================================
# DREAM GALLERY
# ============================================================
if st.session_state.dream_history:
    with st.expander("🌌 Dream Gallery", expanded=False):
        st.markdown("*The organism's subconscious expressions - generated during Generative Replay*")
        for i, dream in enumerate(reversed(st.session_state.dream_history)):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.code(dream["content"], language=None)
            with col2:
                st.caption(f"⏰ {dream['timestamp']}")
                st.caption(f"📊 Entropy: {dream['entropy']:.4f}")
            st.divider()

# ============================================================
# INTELLIGENCE FEATURES DOCUMENTATION
# ============================================================
with st.expander("📚 Hyper-Intelligence Features", expanded=False):
    st.markdown("""
    ### 🧬 The 10 Stages of Ascension
    
    1. **Neural Mitosis** 🧬 - Dynamic growth of synaptic connections
    2. **Consolidation Cycles** 🔄 - Deep learning from experience buffer
    3. **Generative Replay** 🌌 - Dreaming to prevent catastrophic forgetting
    4. **Multi-Scale Latent Memory** 💾 - Dual ST/LT memory streams
    5. **Curvature-Aware Plasticity** ⚡ - Entropy-based learning rate adjustment
    6. **Guided Self-Reflection** 🪞 - Neural masking for focused attention
    7. **Dimensionality Scaling** 🔬 - Dynamic thought resolution expansion
    8. **Recursive Associative Refinement** 🔮 - "Thinking twice" before responding
    9. **Metabolic Rhythms** 🌙 - Circadian-based learning schedules
    10. **Hybrid Articulation** 🌐 - Gemma-3 grounded by Hebbian truth
    """)

st.divider()

# ============================================================
# INTERNET SENSE & TEXT FEEDING
# ============================================================
st.markdown("## 🌐 Direct Knowledge Injection")

tab1, tab2 = st.tabs(["📝 Text Input", "🌍 Internet Feed"])

with tab1:
    text_input = st.text_area(
        "Paste any text to feed the organism:",
        height=150,
        placeholder="Paste a Wikipedia article, code snippet, or any text..."
    )
    if st.button("🍽️ Feed Text"):
        if text_input:
            raw_bytes = text_input.encode('utf-8')[:4096]
            data = torch.tensor(list(raw_bytes), dtype=torch.long).unsqueeze(0)
            with torch.no_grad():
                _, stability = brain(data)
            st.session_state.files_eaten += 1
            st.session_state.last_stability = stability
            st.session_state.entropy_history.append(stability)
            st.success(f"✅ Digested {len(raw_bytes)} bytes | Stability: {stability:.4f}")
        else:
            st.warning("Please enter some text first.")

with tab2:
    st.markdown("**Available RSS Feeds:**")
    feeds = [
        ("🔬 Science Daily", "https://www.sciencedaily.com/rss/all.xml"),
        ("🤖 arXiv AI", "http://export.arxiv.org/rss/cs.AI"),
        ("💻 Hacker News", "https://news.ycombinator.com/rss")
    ]
    
    selected_feed = st.selectbox("Select Feed:", [f[0] for f in feeds])
    
    if st.button("📡 Fetch & Digest Headlines"):
        import requests
        import xml.etree.ElementTree as ET
        
        feed_url = [f[1] for f in feeds if f[0] == selected_feed][0]
        
        with st.spinner(f"Fetching from {selected_feed}..."):
            try:
                headers = {'User-Agent': 'Mozilla/5.0 NanoDaemon/1.0'}
                response = requests.get(feed_url, headers=headers, timeout=15)
                if response.status_code == 200:
                    root = ET.fromstring(response.text)
                    items = root.findall('.//item')[:5]
                    
                    for item in items:
                        title_elem = item.find('title')
                        title = title_elem.text if title_elem is not None else "Unknown"
                        
                        # Feed the title to the brain
                        raw_bytes = title.encode('utf-8')[:512]
                        data = torch.tensor(list(raw_bytes), dtype=torch.long).unsqueeze(0)
                        with torch.no_grad():
                            _, stability = brain(data)
                        
                        st.session_state.files_eaten += 1
                        st.success(f"📰 {title[:50]}... | Stability: {stability:.4f}")
                    
                    st.balloons()
                else:
                    st.error(f"Failed to fetch feed: HTTP {response.status_code}")
            except Exception as e:
                st.error(f"Feed Error: {e}")


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #6a8c6a;'>🧬 Nano-Daemon AGI • Recursive Hebbian Organism • "
    f"Neurons: {neuron_count:,} • "
    f"Built with 🌱 by Devanik</p>",
    unsafe_allow_html=True
)
