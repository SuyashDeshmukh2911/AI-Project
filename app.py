import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Streamlit page settings
st.set_page_config(page_title="Auto Email Draft Assistant", page_icon="📧", layout="wide")

# Cache the model loading to avoid reloading on each interaction
@st.cache_resource
def load_model():
    # Using a better model for text generation
    model_name = "gpt2-large"  # Larger model with better capabilities
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return pipeline('text-generation', model=model, tokenizer=tokenizer)

# Load the model
with st.spinner("Loading the AI model... (this may take a moment)"):
    generator = load_model()

st.title("📧 Auto Email Draft Assistant")
st.markdown("Generate professional email drafts with AI assistance")

# Create columns for a better layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Email Details")
    
    # Input fields with more options
    recipient = st.text_input("Recipient name:", placeholder="John Doe")
    subject = st.text_input("Email subject:", placeholder="Meeting Request")
    purpose = st.text_area("Describe your email purpose:", 
                           placeholder="Apologizing for missing the meeting yesterday due to a family emergency", 
                           height=100)
    
    tone = st.selectbox("Select Tone:", 
                       ["Formal", "Friendly", "Apologetic", "Urgent", "Professional", "Casual"])
    
    # Additional options
    include_greeting = st.checkbox("Include greeting", value=True)
    include_signature = st.checkbox("Include signature", value=True)
    
    # Signature details shown if include_signature is checked
    if include_signature:
        sender_name = st.text_input("Your name:", placeholder="Jane Smith")
        sender_position = st.text_input("Your position/title:", placeholder="Marketing Manager")
    
    # Generate button with loading state
    generate_button = st.button("Generate Email", type="primary")

# Results section
with col2:
    st.subheader("✉️ Generated Email Draft")
    
    if generate_button and purpose:
        with st.spinner("Generating your email..."):
            # Create a detailed prompt template based on selections
            prompt_template = f"""
            Write a {tone.lower()} email
            """
            
            if include_greeting and recipient:
                prompt_template += f" that starts with a greeting to {recipient}"
            
            prompt_template += f" about the following: {purpose}."
            
            if subject:
                prompt_template += f" The email subject is: '{subject}'."
                
            prompt_template += f" The email should be professional, well-structured with clear paragraphs."
            
            if include_signature and sender_name:
                signature = f"{sender_name}"
                if sender_position:
                    signature += f", {sender_position}"
                prompt_template += f" End the email with a proper closing and signature: {signature}."
            
            # Generate the email with higher max_length to ensure complete emails
            try:
                # Use more tokens and lower temperature for more coherent output
                output = generator(prompt_template, 
                                  max_length=500, 
                                  num_return_sequences=1,
                                  do_sample=True,
                                  temperature=0.7,
                                  top_p=0.9,
                                  repetition_penalty=1.2)[0]['generated_text']
                
                # Extract the generated email (remove the prompt part)
                email_text = output.replace(prompt_template, "").strip()
                
                # If we still have issues with generation, add fallback
                if len(email_text) < 50:
                    email_text = f"""
                    {"Dear " + recipient + "," if recipient else "Hello,"}
                    
                    {purpose}
                    
                    {"Best regards," if tone == "Formal" or tone == "Professional" else "Thanks,"}
                    {sender_name if sender_name else ""}
                    {sender_position if sender_position else ""}
                    """
                
                # Display the result in a nice format
                email_container = st.container(border=True)
                with email_container:
                    if subject:
                        st.markdown(f"**Subject:** {subject}")
                    st.markdown(email_text)
                
                # Add copy button
                if st.button("📋 Copy to Clipboard"):
                    st.success("Email copied to clipboard!")
                    
            except Exception as e:
                st.error(f"Error generating email: {str(e)}")
                st.info("Try with a simpler description or different tone.")
    else:
        st.info("Fill in the details on the left and click 'Generate Email' to create your draft.")

# Add helpful tips in the sidebar
with st.sidebar:
    st.title("Tips for Better Emails")
    st.markdown("""
    ### 📝 Email Best Practices
    
    1. **Be clear and concise** - Get to the point quickly
    2. **Use a descriptive subject line** - Help recipients understand the purpose
    3. **Proofread before sending** - Check for errors and clarity
    4. **Maintain appropriate tone** - Match your tone to your audience
    5. **Include a call to action** - Make clear what you expect from the recipient
    """)
