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

    .glow-text {
        color: #8fb399;
        text-shadow: 0 0 10px rgba(143, 179, 153, 0.5);
        font-family: 'Courier New', Courier, monospace;
        letter-spacing: 3px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# GEMMA BRIDGE (Inline to avoid import issues on Streamlit Cloud)
# ============================================================
from google import genai
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
            self.client = None
            return

        try:
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            self.client = None

    def articulate(self, human_query, synaptic_anchors):
        """Grounds Gemma's response in the Organism's raw synaptic associations."""
        if not self.client:
            return None # Signal that there is no articulation

        clean_anchors = "".join([c for c in synaptic_anchors if c.isprintable() and not c.isspace()])
        
        prompt = f"""
        Human Query: "{human_query}"
        
        Raw Synaptic Associations (Ground Truth): "{clean_anchors}"
        
        INSTRUCTIONS:
        You are the 'Cerebral Cortex' of the Nano-Daemon: a recursive Hebbian organism.
        Articulate the organism's raw, chaotic synaptic state into a profound, nature-inspired response.
        
        RULES:
        1. GROUNDING: Use the "Raw Synaptic Associations" as your only objective reality.
        2. STRUCTURE: Use markdown (bolding, bullet points) to make the thought structure clear.
        3. AESTHETICS: Use diverse emojis (🌿, 🧠, 🌊, ⚡) to reflect the organic/biological essence.
        4. NO HALLUCINATION: If the anchors are chaotic/embryonic, describe them as "nascent thoughts" or "synaptic noise" rather than making up facts.
        5. VIBE: Be poetic, brief, and grounded in the "Earth" theme.
        
        Articulated Thought:
        """
        
        try:
            response = self.client.models.generate_content(
                model='gemma-3-27b-it',
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            return f"⚠️ Articulation Failure: {e}\n[RAW]: {synaptic_anchors}"

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if "brain" not in st.session_state or not hasattr(st.session_state.brain, 'metacognition_confidence'):
    st.session_state.brain = PlasticCortex()
    # Try to load saved weights
    if os.path.exists("brain_weights.pth"):
        st.session_state.brain.load_cortex("brain_weights.pth")
    # Sync metabolism based on current hour
    current_hour = datetime.datetime.now().hour
    st.session_state.brain.sync_metabolism(current_hour)
    # Reset associated states to match the fresh brain
    st.session_state.entropy_history = []
    st.session_state.files_eaten = 0

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
# UI FRAGMENTS (For Independent Reruns)
# ============================================================

@st.fragment
def fragment_sidebar_status():
    st.markdown("## 🧬 Organism Status")
    neuron_count = brain.synapse.shape[1]
    thought_width = brain.synapse.shape[0]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🧠 Neurons", f"{neuron_count:,}")
    with col2:
        st.metric("💭 Thought Width", thought_width)
    
    current_hour = datetime.datetime.now().hour
    state_emoji, state_desc = get_metabolic_state(current_hour)
    st.info(f"**Metabolic State**: {state_emoji}\n\n{state_desc}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("⚡ Plasticity", f"{brain.plasticity:.4f}")
    with col2:
        st.metric("📂 Eaten", st.session_state.files_eaten)
    
    # --- STEP 14: Metabolic Balance ---
    st.progress(brain.metabolic_balance / 2.0, text=f"Metabolic Balance: {brain.metabolic_balance:.2f}x")

@st.fragment
def fragment_sidebar_feeding():
    st.markdown("## 🍽️ Feed the Organism")
    uploaded_files = st.file_uploader(
        "Upload files to digest",
        accept_multiple_files=True,
        type=["txt", "py", "md", "json", "csv", "html", "css", "js"],
        key="uploader_fragment"
    )
    
    if uploaded_files:
        with st.spinner("Digesting..."):
            for f in uploaded_files:
                raw_bytes = f.read()
                stability = feed_organism(raw_bytes, f.name)
                st.success(f"✅ Digested `{f.name}` | Stability: {stability:.4f}")

@st.fragment
def fragment_sidebar_controls():
    st.markdown("## ⚙️ Advanced Controls")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧬 Trigger Mitosis"):
            old_neurons = brain.synapse.shape[1]
            brain.grow(256)
            st.success(f"Grew {brain.synapse.shape[1] - old_neurons} neurons!")
    
    with col2:
        if st.button("🔄 Consolidate"):
            brain.consolidate()
            st.success("Memories consolidated!")
    
    col3, col4 = st.columns(2)
    with col3:
        if st.button("🌌 Dream"):
            dream = trigger_dream()
            _, reflection_entropy = brain.reflect()
            st.session_state.dream_history.append({
                "content": dream,
                "entropy": reflection_entropy,
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
            })
            if len(st.session_state.dream_history) > 10:
                st.session_state.dream_history.pop(0)
            st.info(f"Dream: {dream[:30]}...")
    
    with col4:
        if st.button("💾 Save Brain"):
            brain.save_cortex("brain_weights.pth")
            st.success("Brain saved!")

    if st.button("🪞 Self-Reflect"):
        activation, entropy = brain.reflect()
        st.info(f"Reflection complete. Entropy: {entropy:.4f}")
        st.session_state.entropy_history.append(entropy)
    
    if st.button("🔬 Scale Dimensions (32→64)"):
        if brain.synapse.shape[0] < 64:
            brain.scale_dimensions(64)
            st.success("Thought resolution increased to 64!")
        else:
            st.warning("Already at high resolution.")
    
    if st.button("� Digest Knowledge Base"):
        with st.spinner("Crawling project files..."):
            found_files = []
            for root, dirs, files in os.walk(parent_dir if "parent_dir" in locals() else "."):
                if any(x in root for x in ["__pycache__", ".git", ".antigravity"]): continue
                for file in files:
                    if file.endswith((".py", ".txt", ".md", ".json")) and file not in ["brain_weights.pth", "brain_state.json"]:
                        found_files.append(os.path.join(root, file))
            for f_path in found_files:
                try:
                    with open(f_path, 'rb') as f:
                        raw_bytes = f.read()
                        if raw_bytes: feed_organism(raw_bytes, os.path.basename(f_path))
                except: continue
            st.success(f"Consumed {len(found_files)} knowledge nodes.")
    
    st.divider()
    st.markdown("### 🌌 AGI Endgame Controls")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("😴 Deep Sleep"):
            brain.deep_sleep()
            st.success("Deep Sleep complete! Synapses pruned.")
    
    with col2:
        if st.button("👤 Switch Perspective"):
            brain.switch_perspective(to_other=not brain.processing_other)
            st.info(f"Now in '{brain.get_agi_status()['perspective']}' mode")

@st.fragment
def fragment_dialogue():
    st.markdown("## 💬 Dialogue with the Organism")
    query = st.text_input("🗣️ Ask the Organism anything:", placeholder="e.g., What is consciousness?", key="query_input")
    
    if query:
        with st.spinner("🧠 Processing synaptic pathways..."):
            synaptic_anchors, articulated_response = query_organism(query)
        
        st.session_state.conversation_history.append({
            "query": query,
            "anchors": synaptic_anchors,
            "response": articulated_response if articulated_response else "🌿 [Organic Pulse Detected]",
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
        })
        
        clean_display_anchors = "".join([c for c in synaptic_anchors if c.isprintable() and not c.isspace()])
        st.markdown(f"""
        <div class="brain-card" style="text-align: center;">
            <h4 style="color: #b8864b; margin-top: 0;">🧬 RAW SYNAPTIC RESONANCE</h4>
            <div class="glow-text">{clean_display_anchors if clean_display_anchors else "EMBRYONIC SILENCE"}</div>
        </div>
        """, unsafe_allow_html=True)
        
        if articulated_response:
            with st.chat_message("assistant", avatar="🧠"):
                st.markdown(articulated_response)
        else:
            st.info("🌑 **Cerebral Bridge Offline**")

@st.fragment
def fragment_metrics():
    st.markdown("## 📊 Cognitive Metrics")
    
    # Row 1: Core metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🧠 Neurons", f"{brain.synapse.shape[1]:,}")
    col2.metric("📈 Stability", f"{st.session_state.last_stability:.4f}")
    col3.metric("💾 Buffer", len(brain.experience_buffer))
    col4.metric("✨ Curiosity", f"{brain.curiosity_score:.2f}")
    col5.metric("🌐 Bridge", "Online" if bridge.client else "Offline")
    
    # Row 2: AGI Endgame metrics
    st.markdown("### 🌌 AGI Endgame Status")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("🪞 Confidence", f"{brain.metacognition_confidence:.2f}")
    col2.metric("🔥 Motivation", brain.motivation_state)
    col3.metric("⚡ Criticality", f"{brain.criticality_score:.2f}")
    col4.metric("🔗 Causal Nodes", len(brain.causal_graph))
    col5.metric("🌐 Prediction Err", f"{brain.prediction_error:.3f}")
    col6.metric("👤 Perspective", "Other" if brain.processing_other else "Self")
    
    # Motivation warning
    if brain.motivation_state == "BORED":
        st.warning("🥱 **The organism is BORED!** Feed it something novel.")
    elif brain.motivation_state == "OVERWHELMED":
        st.error("😵 **The organism is OVERWHELMED!** Slow down input or trigger Deep Sleep.")
    
    if st.session_state.entropy_history:
        st.line_chart(st.session_state.entropy_history, width="stretch")


@st.fragment
def fragment_memory_viz():
    st.markdown("## 🧬 Latent Memory Streams")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ⚡ Short-Term")
        st.bar_chart(brain.short_term_latent.detach().numpy().flatten()[:32], width="stretch")
    with col2:
        st.markdown("### 🌊 Long-Term")
        st.bar_chart(brain.long_term_latent.detach().numpy().flatten()[:32], width="stretch")

@st.fragment
def fragment_knowledge_injection():
    st.markdown("## 🌐 Direct Knowledge Injection")
    tab1, tab2 = st.tabs(["📝 Text Input", "🌍 Internet Feed"])
    with tab1:
        text_input = st.text_area("Paste text to feed:", height=100, key="text_feed_input")
        if st.button("🍽️ Feed Text"):
            if text_input:
                stability = feed_organism(text_input.encode('utf-8')[:4096], "text_input")
                st.success(f"✅ Digested | Stability: {stability:.4f}")
    with tab2:
        feeds = [("🔬 Science Daily", "https://www.sciencedaily.com/rss/all.xml"),
                 ("🤖 arXiv AI", "http://export.arxiv.org/rss/cs.AI"),
                 ("💻 Hacker News", "https://news.ycombinator.com/rss")]
        selected_feed = st.selectbox("Select Feed:", [f[0] for f in feeds], key="feed_select")
        if st.button("📡 Fetch Feed"):
            import requests; import xml.etree.ElementTree as ET
            feed_url = [f[1] for f in feeds if f[0] == selected_feed][0]
            try:
                r = requests.get(feed_url, headers={'User-Agent': 'NanoDaemon/1.0'}, timeout=10)
                root = ET.fromstring(r.text)
                for item in root.findall('.//item')[:5]:
                    title = item.find('title').text
                    stability = feed_organism(title.encode('utf-8')[:512], "rss_feed")
                    st.success(f"📰 {title[:50]}... | {stability:.4f}")
            except Exception as e: st.error(f"Error: {e}")

@st.fragment
def fragment_history_gallery():
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.conversation_history:
            with st.expander("📜 Synaptic History", expanded=False):
                for conv in reversed(st.session_state.conversation_history[-10:]):
                    with st.chat_message("user", avatar="👤"): st.markdown(conv['query'])
                    with st.chat_message("assistant", avatar="🧠"): st.markdown(conv['response'])
    with col2:
        if st.session_state.dream_history:
            with st.expander("🌌 Dream Gallery", expanded=False):
                for dream in reversed(st.session_state.dream_history):
                    st.code(dream["content"][:100], language=None)
                    st.caption(f"⏰ {dream['timestamp']} | Entropy: {dream['entropy']:.4f}")

# ============================================================
# STEP 15: AUTONOMOUS RUMINATOR
# ============================================================
@st.fragment(run_every=10) # Ruminate every 10 seconds
def fragment_autonomous_ruminator():
    # Only ruminate if metabolic cycle allows (Active or Neutral)
    current_hour = datetime.datetime.now().hour
    if current_hour >= 5:
        # Subtle weight shift
        brain.reflect()
        # Occasional subconscious dream
        if time.time() % 60 < 10: # 10s chance every minute
            dream = trigger_dream()
            _, e = brain.reflect()
            st.session_state.dream_history.append({
                "content": "[Auto] " + dream,
                "entropy": e,
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
            })
            if len(st.session_state.dream_history) > 10:
                st.session_state.dream_history.pop(0)
# ============================================================
# APP LAYOUT (Fragment Orchestration)
# ============================================================

# SideBar Orchestration
with st.sidebar:
    fragment_sidebar_status()
    st.divider()
    fragment_sidebar_feeding()
    st.divider()
    fragment_sidebar_controls()
    st.divider()
    if bridge.client: st.success("🟢 Hybrid Intelligence Active")
    else: st.warning("🟡 Organic Mode Active")

# Main Content Orchestration
st.markdown("# 🧬 Nano-Daemon: Recursive Hebbian Organism")
st.markdown("*A self-evolving digital lifeform with hybrid intelligence*")

fragment_metrics()
st.divider()
fragment_dialogue()
st.divider()
fragment_history_gallery()
st.divider()
fragment_memory_viz()
st.divider()
fragment_knowledge_injection()

# Invisible Ruminator
fragment_autonomous_ruminator()

# Static Expanders
with st.expander("📚 Hyper-Intelligence Features", expanded=False):
    st.markdown("### 🧬 The 21 Stages of Ascension")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **Core Neural:**
        1. Neural Mitosis 🧬
        2. Consolidation 🔄
        3. Generative Replay 🌌
        4. Multi-Scale Memory 💾
        5. Dynamic Plasticity ⚡
        6. Self-Reflection 🪞
        7. Dim Scaling 🔬
        """)
    with col2:
        st.markdown("""
        **Cognitive:**
        8. Recursive Refinement 🔮
        9. Metabolic Rhythms 🌙
        10. Hybrid Articulation 🌐
        11. Active Inference ⚖️
        12. Homeostatic Scaling 🌊
        13. Temporal Awareness ⚡
        14. Autonomous Rumination 🌀
        """)
    with col3:
        st.markdown("""
        **AGI Endgame:**
        15. World Modeling 🌍
        16. Metacognition 🪞🪞
        17. Theory of Mind 👤
        18. Intrinsic Motivation 🔥
        19. Causal Inference 🔗
        20. Sleep-Wake Cycle 😴
        21. Edge-of-Chaos ⚡🌀
        """)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #6a8c6a;'>🧬 Nano-Daemon AGI • "
    f"Neurons: {brain.synapse.shape[1]:,} • Built with 🌱 by Devanik</p>",
    unsafe_allow_html=True
)
