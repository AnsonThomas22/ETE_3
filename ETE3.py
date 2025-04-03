import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import cv2
from PIL import Image
import os
import warnings
import base64
from io import BytesIO

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Set page configuration
st.set_page_config(
    page_title="INBLOOM '25",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for better UI
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .dashboard-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .gallery-image {
        transition: transform 0.3s;
    }
    .gallery-image:hover {
        transform: scale(1.05);
        cursor: pointer;
    }
    .image-container {
        position: relative;
        text-align: center;
    }
    .image-overlay {
        position: absolute;
        bottom: 10px;
        background: rgba(0, 0, 0, 0.6);
        color: white;
        width: 100%;
        transition: .5s ease;
        opacity: 0;
        padding: 5px;
        text-align: center;
        border-radius: 0 0 5px 5px;
    }
    .image-container:hover .image-overlay {
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

# Directory for images
IMAGE_DIR = "event_images/"

def generate_dataset():
    events = ["Music", "Dance", "Drama", "Painting", "Photography", "Debate", "Quiz", "Poetry", "Fashion", "Cooking"]
    colleges = ["ABC College", "XYZ University", "LMN Institute", "PQR College", "JKL University"]
    states = ["Karnataka", "Maharashtra", "Tamil Nadu", "Delhi", "Kerala"]
    feedback_sentences = [
        "An amazing performance full of energy!", "Loved the creativity in the paintings.",
        "The debate was intellectually stimulating.", "Great stage presence by all participants.",
        "The quiz was challenging and fun!", "Music performances were outstanding!",
        "Incredible choreography in the dance event!", "The drama touched our hearts.",
        "Photography contest captured stunning moments.", "Fashion show was glamorous and elegant.",
        "Poetry readings were deeply moving.", "Cooking contest showcased some amazing flavors.",
        "The panel discussion added great insights.", "Amazing teamwork in group events!",
        "Stage lighting and sound were perfect.", "The audience engagement was fantastic!",
        "Unique performances made the event special.", "It was an unforgettable experience!",
        "Talented participants showcased their skills.", "Organized seamlessly with great spirit!"
    ]
    
    # Make the dataset deterministic for consistent filters
    random.seed(42)
    
    data = []
    for i in range(250):
        participation_days = random.randint(0, 5)
        event = random.choice(events)
        college = random.choice(colleges)
        state = random.choice(states)
        day = random.randint(1, 5)
        
        participant = {
            "Participant_ID": i + 1,
            "College": college,
            "State": state,
            "Event": event,
            "Day": day,
            "Participation_Days": participation_days,
            "Feedback": random.choice(feedback_sentences),
            "Participation_Status": "Present" if participation_days > 0 else "Absent",
            "Photo_Path": f"event_{event.lower().replace(' ', '_')}_{day}.jpg"
        }
        data.append(participant)
    
    return pd.DataFrame(data)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = generate_dataset()
    
if 'selected_image' not in st.session_state:
    st.session_state.selected_image = None
    
if 'process_image' not in st.session_state:
    st.session_state.process_image = False

# Load event images
def load_event_images():
    images = []
    if os.path.exists(IMAGE_DIR):
        for file in os.listdir(IMAGE_DIR):
            if file.endswith(("jpg", "png", "jpeg")):
                images.append(os.path.join(IMAGE_DIR, file))
    
    # If no images found, add placeholder images for testing
    if not images:
        # Create a placeholder image for each event and day
        events = ["Music", "Dance", "Drama", "Painting", "Photography"]
        for event in events:
            for day in range(1, 6):
                img_path = f"event_{event.lower()}_{day}.jpg"
                img = Image.new('RGB', (300, 200), color=(random.randint(0, 255), 
                                                        random.randint(0, 255), 
                                                        random.randint(0, 255)))
                images.append(img_path)
                # Save the image if directory exists
                if os.path.exists(IMAGE_DIR):
                    img.save(os.path.join(IMAGE_DIR, img_path))
    
    return images

def show_gallery(images):
    num_cols = 3
    cols = st.columns(num_cols)
    
    for i, img_path in enumerate(images):
        col_idx = i % num_cols
        
        with cols[col_idx]:
            # Use placeholder image if file doesn't exist
            try:
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                else:
                    # Create colored placeholder with text
                    img = Image.new('RGB', (300, 200), color=(random.randint(100, 255), 
                                                            random.randint(100, 255), 
                                                            random.randint(100, 255)))
            except Exception:
                img = Image.new('RGB', (300, 200), color=(random.randint(100, 255), 
                                                        random.randint(100, 255), 
                                                        random.randint(100, 255)))
            
            # Get base image name for display
            img_name = os.path.basename(img_path)
            
            # Create clickable image
            st.markdown(f"""
            <div class="image-container">
                <img src="data:image/png;base64,{image_to_base64(img)}" 
                    class="gallery-image" width="100%" 
                    onclick="handleImageClick('{img_path}')" />
                <div class="image-overlay">{img_name}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Button to process this image
            if st.button(f"Process {img_name}", key=f"btn_{i}"):
                st.session_state.selected_image = img
                st.session_state.process_image = True
                st.info(" The processed image is available in the **Image Processing page**")
                st.rerun()
                

def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def create_word_cloud(feedback_text, title="Feedback Word Cloud"):
    wordcloud = WordCloud(width=800, height=400, background_color="white", 
                         colormap="viridis", max_words=100).generate(feedback_text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=18)
    
    return fig

def process_image(image):
    st.write("### Image Processing Results")

    # Convert image to array for processing
    img_array = np.array(image)

    # Resize image (maintaining aspect ratio)
    new_width = 500  # Set fixed width
    aspect_ratio = img_array.shape[0] / img_array.shape[1]
    new_height = int(new_width * aspect_ratio)
    img_resized = cv2.resize(img_array, (new_width, new_height))

    # Convert to grayscale
    if len(img_array.shape) == 3:  # Color image
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    else:  # Already grayscale
        gray = img_resized

    # Display original and resized image
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)
    with col2:
        st.image(img_resized, caption="Resized Image", use_container_width=True)

    # Apply image sharpening using an unsharp mask
    sharpening_kernel = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]])
    sharpened = cv2.filter2D(img_resized, -1, sharpening_kernel)

    # Apply edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Enhance brightness & contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # Color Enhancement (Boost Saturation)
    if len(img_array.shape) == 3:  # If colored
        img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)
        img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * 1.5, 0, 255)  # Increase saturation
        enhanced_color = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    else:
        enhanced_color = enhanced_gray

    # Display processed images
    col3, col4 = st.columns(2)
    with col3:
        st.image(sharpened, caption="Sharpened Image", use_container_width=True)
        st.image(edges, caption="Edge Detection", use_container_width=True, channels="GRAY")

    with col4:
        st.image(enhanced_gray, caption="Contrast Enhanced (CLAHE)", use_container_width=True, channels="GRAY")
        st.image(enhanced_color, caption="Enhanced Colors", use_container_width=True)


# Add JavaScript for image clicking
st.markdown("""
<script>
function handleImageClick(imgPath) {
    // Send data to Streamlit
    const data = {
        imgPath: imgPath,
        clicked: true
    };
    
    // Use Streamlit's message passing
    window.parent.postMessage({
        type: "streamlit:setComponentValue",
        value: data
    }, "*");
}
</script>
""", unsafe_allow_html=True)

# Main Streamlit App
def main():
    data = st.session_state.data
    
    # Display title with colorful text
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #111 30%, #222 100%);
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 6px 15px rgba(0,0,0,0.5);
        text-align: center;
        margin-bottom: 30px;
        border: 2px solid rgba(255, 255, 255, 0.1);
    ">
        <h1 style="
            font-size: 60px;
            margin-bottom: 10px;
            font-weight: bold;
            text-shadow: 0px 0px 10px rgba(255,255,255,0.8);
        ">
            <span style="color: red; text-shadow: 0px 0px 10px red;">I</span>
            <span style="color: orange; text-shadow: 0px 0px 10px orange;">N</span>
            <span style="color: yellow; text-shadow: 0px 0px 10px yellow;">B</span>
            <span style="color: green; text-shadow: 0px 0px 10px green;">L</span>
            <span style="color: blue; text-shadow: 0px 0px 10px blue;">O</span>
            <span style="color: indigo; text-shadow: 0px 0px 10px indigo;">O</span>
            <span style="color: violet; text-shadow: 0px 0px 10px violet;">M</span>
            <span style="color: white; text-shadow: 0px 0px 10px white;">'25</span>
        </h1>
        <p style="
            color: rgba(255, 255, 255, 0.8);
            font-style: italic;
            font-size: 18px;
            text-shadow: 0px 0px 8px rgba(255,255,255,0.5);
        ">
            Cultural Festival Event Dashboard
        </p>
    </div>
""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📸 Event Gallery", "📝 Feedback Analysis", "🖼️ Image Processing"])

    # Sidebar Filters
    st.sidebar.markdown("## 🔍 Filters")
    
    selected_event = st.sidebar.selectbox("Select Event", ["All"] + sorted(list(data["Event"].unique())))
    selected_college = st.sidebar.selectbox("Select College", ["All"] + sorted(list(data["College"].unique())))
    selected_state = st.sidebar.selectbox("Select State", ["All"] + sorted(list(data["State"].unique())))
    selected_day = st.sidebar.selectbox("Select Day", ["All"] + sorted(list(data["Day"].unique())))
    
    # Apply filters
    filtered_data = data.copy()
    
    if selected_event != "All":
        filtered_data = filtered_data[filtered_data["Event"] == selected_event]
    if selected_college != "All":
        filtered_data = filtered_data[filtered_data["College"] == selected_college]
    if selected_state != "All":
        filtered_data = filtered_data[filtered_data["State"] == selected_state]
    if selected_day != "All":
        filtered_data = filtered_data[filtered_data["Day"] == selected_day]
    
    
    
    # Tab 1: Dashboard
    with tab1:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.markdown("### 📋 Filtered Participation Data")
        st.dataframe(filtered_data, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
            st.markdown("### 📊 Event-wise Participation")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            event_counts = filtered_data["Event"].value_counts()
            sns.barplot(x=event_counts.index, y=event_counts.values, palette="viridis", ax=ax)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
            st.markdown("### 📅 Day-wise Participation")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            day_counts = filtered_data["Day"].value_counts().sort_index()
            sns.barplot(x=day_counts.index, y=day_counts.values, palette="coolwarm", ax=ax)
            plt.xlabel("Day")
            plt.ylabel("Number of Participants")
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
            st.markdown("### 🏫 College-wise Participation")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            college_counts = filtered_data["College"].value_counts()
            sns.barplot(y=college_counts.index, x=college_counts.values, palette="magma", ax=ax)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col4:
            st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
            st.markdown("### 🗺️ State-wise Participation")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            state_counts = filtered_data["State"].value_counts()
            sns.barplot(y=state_counts.index, x=state_counts.values, palette="Set2", ax=ax)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.markdown("### ✅ Participation Status")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        status_counts = filtered_data["Participation_Status"].value_counts()
        
        # Pie chart for participation status
        plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', 
                startangle=90, colors=['#4CAF50', '#F44336'])
        plt.axis('equal')
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Tab 2: Event Gallery
    with tab2:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.markdown("### 📸 Event Gallery")
        st.write("Click on any image to process it or use the button below each image.")
        
        # Filter images by day if day filter is active
        image_files = load_event_images()
        
        # If day filter is applied, filter gallery images
        if selected_day != "All":
            image_files = [img for img in image_files if f"_{selected_day}." in img]
        
        # If event filter is applied, filter gallery images
        if selected_event != "All":
            event_key = selected_event.lower().replace(' ', '_')
            image_files = [img for img in image_files if event_key in img.lower()]
        
        if image_files:
            show_gallery(image_files)
        else:
            st.warning("No images found for the selected filters. Please adjust your filters or upload images to the 'event_images/' folder.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Tab 3: Feedback Analysis
    with tab3:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.markdown("### 💬 Feedback Analysis")
        
        # Overall feedback word cloud
        st.markdown("#### Overall Feedback")
        all_feedback = " ".join(filtered_data["Feedback"])
        if all_feedback.strip():
            fig = create_word_cloud(all_feedback)
            st.pyplot(fig)
        else:
            st.warning("No feedback data available for the selected filters.")
        
        # Event-specific feedback word clouds
        if selected_event == "All" and len(filtered_data["Event"].unique()) > 1:
            st.markdown("#### Event-wise Feedback Comparison")
            selected_events = st.multiselect("Select events to compare",
                                           options=filtered_data["Event"].unique(),
                                           default=list(filtered_data["Event"].unique())[:2])
            
            if selected_events:
                cols = st.columns(min(len(selected_events), 2))
                
                for i, event in enumerate(selected_events):
                    event_data = filtered_data[filtered_data["Event"] == event]
                    event_feedback = " ".join(event_data["Feedback"])
                    
                    if event_feedback.strip():
                        fig = create_word_cloud(event_feedback, title=f"{event} Feedback")
                        cols[i % 2].pyplot(fig)
                    else:
                        cols[i % 2].warning(f"No feedback available for {event}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Tab 4: Image Processing
    with tab4:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.markdown("### 🖼️ Image Processing")
        
        # Option to upload new images
        st.markdown("#### Upload New Images")
        uploaded_file = st.file_uploader("Upload an image to process", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            # Process the uploaded image
            image = Image.open(uploaded_file)
            process_image(image)
        
        # Process selected image if available
        elif st.session_state.process_image and st.session_state.selected_image is not None:
            process_image(st.session_state.selected_image)
            # Reset after processing
            st.session_state.process_image = False
        else:
            st.info("Please upload an image or click on an image from the Event Gallery to process it.")
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()