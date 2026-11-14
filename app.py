import streamlit as st
import time
import re
import nltk

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

from src.summarizer import LegalSummarizer

# 1. Page Configuration
st.set_page_config(
    page_title="Legal Document Summarizer",
    page_icon="⚖️",
    layout="wide"
)

# 2. Cache the model so it only loads once
@st.cache_resource
def load_summarizer():
    return LegalSummarizer()

summarizer = load_summarizer()

# Initialize session state history for multi-contract tracking
if "history" not in st.session_state:
    st.session_state.history = []

# Helper functions for formatting and cleaning
def sanitize_text(text: str) -> str:
    replacements = {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2014": "-", "\u2013": "-", "\u2026": "..."
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    text = re.sub(r'[\(\[]\s*\*+[\*\s]*[\)\]]', ' [REDACTED] ', text)
    
    # NEW FIX: If there is a paragraph break without punctuation (like a title), add a period so it splits cleanly
    text = re.sub(r'(?<![.\!?:;])(\n\s*){2,}', '. ', text)
    
    # Now flatten the remaining whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_and_deduplicate(sentences):
    unique_bullets = []
    seen_normalized = set()
    for sent in sentences:
        clean_sent = re.sub(r'^[-•*.\s]+', '', sent).strip()
        normalized = re.sub(r'[^a-z0-9]', '', clean_sent.lower())
        
        # Skip sentences that are too short (character count)
        if len(normalized) < 20:
            continue
            
        # NEW FIX: Skip short section headers (must be at least 8 words)
        if len(clean_sent.split()) < 8:
            continue
            
        # Skip incomplete AI sentences that do not end in punctuation
        if not re.search(r'[.!?\"\'\]\)]$', clean_sent):
            continue
            
        # Skip definition clauses
        if re.search(r'\"[A-Za-z\s0-9]+\"\s+(means|mean|shall mean|is defined as)\b', clean_sent, re.IGNORECASE):
            continue
            
        if normalized not in seen_normalized:
            seen_normalized.add(normalized)
            clean_sent = clean_sent[0].upper() + clean_sent[1:]
            unique_bullets.append(clean_sent)
            
    return unique_bullets

def detect_document_title(text: str) -> str:
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if not lines:
        return "UNKNOWN_AGREEMENT"
    
    # Take the first non-empty line as a candidate
    candidate = lines[0]
    
    # Remove special characters but KEEP letters, numbers, spaces, and hyphens
    clean_title = re.sub(r'[^a-zA-Z0-9\s\-]', '', candidate)
    
    # Limit to the first 7 words, join with underscores, and convert to ALL CAPS
    clean_title = "_".join(clean_title.split()[:7]).upper()
    
    return clean_title if len(clean_title) > 3 else "CONTRACT_AGREEMENT"

def detect_legal_focus(text: str) -> str:
    text_lower = text.lower()
    
    # Smarter Auto-Detect keywords to prevent false positives and catch NDAs & Employment docs
    if "limitation of liability" in text_lower or "cap on liability" in text_lower or "consequential damages" in text_lower:
        return "CAP ON LIABILITY"
    elif "non-disclosure" in text_lower or "confidentiality" in text_lower or "trade secret" in text_lower:
        return "CONFIDENTIALITY / NDA"
    elif "employment agreement" in text_lower or "non-compete" in text_lower or "severance" in text_lower or "restrictive covenant" in text_lower:
        return "EMPLOYMENT / NON-COMPETE"
    elif "renewal term" in text_lower or "option to renew" in text_lower or "automatic renewal" in text_lower:
        return "RENEWAL TERM"
    elif "warranty" in text_lower or "guarantee" in text_lower:
        return "WARRANTY DURATION"
    elif "audit rights" in text_lower or "right to audit" in text_lower:
        return "AUDIT RIGHTS"
    elif "assign" in text_lower:
        return "ANTI-ASSIGNMENT"
    elif "governing law" in text_lower or "jurisdiction" in text_lower:
        return "GOVERNING LAW"
    elif "termination for convenience" in text_lower or "terminate for convenience" in text_lower:
        return "TERMINATION FOR CONVENIENCE"
    else:
        return "GENERAL LEGAL TERMS"

# 3. Application UI (Redesigned Header)
st.markdown("""
<div style="background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%); padding: 35px; border-radius: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.15); color: white; margin-bottom: 25px;">
    <h1 style="margin-top: 0; font-size: 34px; font-weight: 800; letter-spacing: 0.5px; color: #ffffff;">⚖️ AI Legal Contract Summarization Brief</h1>
    <h3 style="margin-top: 5px; color: #93c5fd; font-weight: 500; font-size: 22px;">✨ Transform Complex Contracts into Crystal-Clear Insights</h3>
    <p style="margin-top: 15px; font-size: 16px; line-height: 1.6; color: #e2e8f0; margin-bottom: 0;">
        <b>Drop your dense legal text below to instantly generate a laser-focused executive brief.</b><br>
        <i>Every document you process is automatically compiled into a sleek, export-ready HTML master report!</i>
    </p>
</div>
""", unsafe_allow_html=True)

# Put the text area FIRST so Streamlit can read it immediately
contract_text = st.text_area("Legal Document Text", height=250, placeholder="Paste your contract text here...")

# Auto-calculate defaults based on the text area input
if contract_text.strip():
    auto_title = detect_document_title(contract_text)
    auto_focus = detect_legal_focus(contract_text)
else:
    auto_title = "MASTER_SERVICES_AGREEMENT"
    auto_focus = "Auto-Detect"

focus_options = [
    "Auto-Detect", "CONFIDENTIALITY / NDA", "EMPLOYMENT / NON-COMPETE", "WARRANTY DURATION", 
    "AUDIT RIGHTS", "RENEWAL TERM", "ANTI-ASSIGNMENT", "CAP ON LIABILITY", 
    "GOVERNING LAW", "TERMINATION FOR CONVENIENCE", "GENERAL LEGAL TERMS"
]

# Find the index of the auto-detected focus to set the dropdown default
try:
    focus_index = focus_options.index(auto_focus)
except ValueError:
    focus_index = 0

# Inputs for Document Name and Focus override (placed AFTER text area)
col1, col2 = st.columns([2, 1])
with col1:
    contract_title = st.text_input("Document Name / Title", value=auto_title)
with col2:
    manual_focus = st.selectbox("Primary Legal Focus", options=focus_options, index=focus_index)

# 4. Execution Logic
if st.button("Generate Executive Brief & Save", type="primary"):
    if not contract_text.strip():
        st.warning("Please paste a contract into the text box before generating the brief.")
    else:
        with st.spinner("Analyzing legal text and structuring executive brief..."):
            start_time = time.time()
            
            # Use manual focus if user changed it, otherwise use the auto-detected one
            focus_tag = manual_focus if manual_focus != "Auto-Detect" else detect_legal_focus(contract_text)
            
            # Generate summary via model AND pass the targeted tag
            raw_summary = summarizer.summarize_document(contract_text, focus_tag=focus_tag)
            
            # Format into clean sentences and bullets
            sanitized = sanitize_text(raw_summary if len(raw_summary) > 50 else contract_text)
            raw_sentences = nltk.sent_tokenize(sanitized)
            bullets = clean_and_deduplicate(raw_sentences)
            
            # SMART FALLBACK: If the focused summary is too short, pad it with key original clauses
            if len(bullets) < 5:
                orig_sentences = nltk.sent_tokenize(sanitize_text(contract_text))
                orig_clean = clean_and_deduplicate(orig_sentences)
                
                # Look for action-oriented legal keywords to find meaningful clauses
                legal_keywords = ["shall", "terminate", "warrant", "fee", "license", "liability", "renew", "effective", "entered", "between"]
                
                for sent in orig_clean:
                    if len(bullets) >= 5:
                        break
                        
                    # 1. Must be a full sentence (more than 10 words)
                    # 2. Must NOT be an ALL CAPS header
                    # 3. Must contain at least one legal keyword
                    if len(sent.split()) > 10 and not sent.isupper() and any(kw in sent.lower() for kw in legal_keywords):
                        # Ensure we don't accidentally add a slightly modified duplicate of the preamble
                        if not any(sent[:40] in b or b[:40] in sent for b in bullets):
                            bullets.append(sent)
            
            # Enforce exactly 5 bullet points
            bullets = bullets[:5]
            
            if not bullets:
                bullets = [sanitized]
            
            end_time = time.time()
            
            # Save this test into session history automatically
            entry = {
                "title": contract_title.strip(),
                "focus": focus_tag,
                "text": contract_text,
                "bullets": bullets
            }
            st.session_state.history.append(entry)
            
            st.success(f"Brief generated in {round(end_time - start_time, 2)} seconds and added to master history!")

# 5. Render History View and Master Download Option
if st.session_state.history:
    st.divider()
    st.subheader(f"📋 Session History ({len(st.session_state.history)} Contracts Processed)")
    
    for idx, item in enumerate(st.session_state.history, 1):
        # Display each tested contract inside a clean collapsible expander box
        with st.expander(f"Document {idx}: {item['title']} ({item['focus']})", expanded=(idx == len(st.session_state.history))):
            st.markdown(f"### **Document:** {item['title']}")
            st.markdown(f"#### <span style='color:red;'>Primary Legal Focus:</span> **{item['focus']}**", unsafe_allow_html=True)
            st.markdown("##### **Executive Summary:**")
            for bullet in item['bullets']:
                st.markdown(f"- {bullet}")
        
    # Build a gorgeously styled, colorful HTML master report string
    html_report_lines = [
        """<!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Master Legal Contract Summary Report</title>
            <style>
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f1f5f9; color: #1e293b; margin: 0; padding: 40px; }
                .container { max-width: 950px; margin: auto; background: #ffffff; padding: 40px; border-radius: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.08); }
                h1 { color: #0f172a; text-align: center; border-bottom: 4px solid #2563eb; padding-bottom: 20px; margin-bottom: 40px; font-size: 28px; }
                .doc-card { background: #ffffff; border: 1px solid #e2e8f0; border-left: 6px solid #2563eb; border-radius: 10px; padding: 30px; margin-bottom: 35px; box-shadow: 0 4px 6px rgba(0,0,0,0.02); }
                .doc-title { font-size: 22px; font-weight: bold; color: #1e293b; margin-bottom: 8px; }
                .focus-tag { color: #dc2626; font-weight: bold; font-size: 15px; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 20px; }
                .section-heading { font-weight: bold; color: #475569; margin-top: 20px; margin-bottom: 8px; text-transform: uppercase; font-size: 12px; letter-spacing: 1px; }
                .original-text { background: #f8fafc; border: 1px solid #cbd5e1; padding: 15px; font-family: monospace; font-size: 12px; white-space: pre-wrap; max-height: 220px; overflow-y: auto; color: #334155; border-radius: 6px; margin-bottom: 20px; }
                ul { margin: 0; padding-left: 20px; }
                li { margin-bottom: 8px; line-height: 1.6; color: #334155; font-size: 14px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>⚖️ AI Legal Contract Summary Master Report</h1>
        """
    ]

    for idx, item in enumerate(st.session_state.history, 1):
        bullets_html = "".join([f"<li>{b}</li>" for b in item['bullets']])
        html_report_lines.append(f"""
                <div class="doc-card">
                    <div class="doc-title">Document {idx}: {item['title']}</div>
                    <div class="focus-tag">Primary Legal Focus: {item['focus']}</div>
                    
                    <div class="section-heading">Original Contract Text</div>
                    <div class="original-text">{item['text']}</div>
                    
                    <div class="section-heading">Executive Summary</div>
                    <ul>
                        {bullets_html}
                    </ul>
                </div>
        """)

    html_report_lines.append("""
            </div>
        </body>
        </html>
    """)
    
    master_html_data = "".join(html_report_lines)
    
    st.divider()
    
    # 6. Master Download and Clear Buttons
    col_dl1, col_dl2 = st.columns([2, 1])
    with col_dl1:
        st.download_button(
            label="📥 Download Colorful Master Report (.HTML)",
            data=master_html_data,
            file_name="All_Tested_Contracts_Colorful_Report.html",
            mime="text/html",
            type="primary"
        )
    with col_dl2:
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()