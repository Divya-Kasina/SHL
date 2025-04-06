import streamlit as st
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import re
from fake_useragent import UserAgent
import time
import random
import json
from fastapi import FastAPI, Query
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import os
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Optional, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app for the API endpoint
app = FastAPI(title="SHL Assessment Recommendation API",
              description="API for recommending SHL assessments based on job descriptions or queries",
              version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for API response
class AssessmentRecommendation(BaseModel):
    title: str
    url: str
    description: str
    remote_testing: str
    adaptive_irt: str
    duration: str
    test_type: str
    score: float

class RecommendationResponse(BaseModel):
    recommendations: List[AssessmentRecommendation]
    query: str

# Streamlit configuration
st.set_page_config(
    page_title="SHL Assessment Recommendation Engine", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced SHL assessment catalog with all required attributes
CATALOG_DATA = [
    {
        "title": "OPQ32",
        "description": "Measures 32 aspects of personality to predict workplace behavior and potential",
        "url": "https://www.shl.com/en/assessments/personality-assessment-opq32/",
        "remote_testing": "Yes",
        "adaptive_irt": "No",
        "duration": "25-35 minutes",
        "test_type": "Personality"
    },
    {
        "title": "Verify Interactive",
        "description": "Interactive cognitive ability test measuring verbal, numerical, and abstract reasoning",
        "url": "https://www.shl.com/en/assessments/verify-interactive/",
        "remote_testing": "Yes",
        "adaptive_irt": "Yes",
        "duration": "30-45 minutes",
        "test_type": "Cognitive"
    },
    {
        "title": "Situational Judgment Test",
        "description": "Measures decision-making skills in workplace scenarios",
        "url": "https://www.shl.com/en/assessments/situational-judgement-tests/",
        "remote_testing": "Yes",
        "adaptive_irt": "No",
        "duration": "20-30 minutes",
        "test_type": "Behavioral"
    },
    {
        "title": "Verify Ability Tests",
        "description": "Measures cognitive abilities including verbal, numerical, and inductive reasoning",
        "url": "https://www.shl.com/en/assessments/cognitive-ability-verify/",
        "remote_testing": "Yes",
        "adaptive_irt": "Yes",
        "duration": "15-25 minutes per module",
        "test_type": "Cognitive"
    },
    {
        "title": "Motivational Questionnaire",
        "description": "Assesses 18 dimensions of motivation to understand what drives employee behavior",
        "url": "https://www.shl.com/en/assessments/motivation-assessment-mq/",
        "remote_testing": "Yes",
        "adaptive_irt": "No",
        "duration": "20-25 minutes",
        "test_type": "Motivation"
    },
    {
        "title": "ADEPT-15",
        "description": "Measures 15 aspects of personality that predict workplace performance",
        "url": "https://www.shl.com/en/assessments/personality-assessment-adept15/",
        "remote_testing": "Yes",
        "adaptive_irt": "Yes",
        "duration": "25 minutes",
        "test_type": "Personality"
    },
    {
        "title": "Coding Assessments",
        "description": "Evaluates programming skills across multiple languages including Java, Python, and JavaScript",
        "url": "https://www.shl.com/en/assessments/coding-assessments/",
        "remote_testing": "Yes",
        "adaptive_irt": "No",
        "duration": "30-60 minutes",
        "test_type": "Technical Skills"
    },
    {
        "title": "SHL Sales Assessment",
        "description": "Evaluates sales capabilities and potential for performance in sales roles",
        "url": "https://www.shl.com/en/assessments/sales-assessment/",
        "remote_testing": "Yes",
        "adaptive_irt": "No",
        "duration": "35 minutes",
        "test_type": "Role Specific"
    },
    {
        "title": "SHL Call Center Assessment",
        "description": "Measures skills specific to customer service and call center roles",
        "url": "https://www.shl.com/en/assessments/contact-centre-assessments/",
        "remote_testing": "Yes",
        "adaptive_irt": "No",
        "duration": "30 minutes",
        "test_type": "Role Specific"
    },
    {
        "title": "Leadership Assessment",
        "description": "Evaluates leadership potential and capabilities across various dimensions",
        "url": "https://www.shl.com/en/assessments/leadership-assessments/",
        "remote_testing": "Yes",
        "adaptive_irt": "No",
        "duration": "40 minutes",
        "test_type": "Leadership"
    },
    {
        "title": "Development Centers",
        "description": "Comprehensive assessment centers for leadership development",
        "url": "https://www.shl.com/en/assessments/development-centres/",
        "remote_testing": "Yes",
        "adaptive_irt": "No",
        "duration": "1-2 days",
        "test_type": "Development"
    },
    {
        "title": "Workplace English Test",
        "description": "Assesses English language proficiency in workplace contexts",
        "url": "https://www.shl.com/en/assessments/workplace-english-test/",
        "remote_testing": "Yes",
        "adaptive_irt": "Yes",
        "duration": "45 minutes",
        "test_type": "Language Skills"
    },
    {
        "title": "Career Guidance Report",
        "description": "Provides insights into career preferences and suitability",
        "url": "https://www.shl.com/en/assessments/career-guidance/",
        "remote_testing": "Yes",
        "adaptive_irt": "No",
        "duration": "30 minutes",
        "test_type": "Career Development"
    },
    {
        "title": "Remote Work Assessment",
        "description": "Evaluates candidate suitability for remote work environments",
        "url": "https://www.shl.com/en/assessments/remote-work-assessment/",
        "remote_testing": "Yes",
        "adaptive_irt": "No",
        "duration": "25 minutes",
        "test_type": "Work Style"
    },
    {
        "title": "SQL Assessment",
        "description": "Evaluates SQL knowledge and database query capabilities",
        "url": "https://www.shl.com/en/assessments/sql-assessment/",
        "remote_testing": "Yes",
        "adaptive_irt": "No",
        "duration": "40 minutes",
        "test_type": "Technical Skills"
    },
    {
        "title": "Java Assessment",
        "description": "Comprehensive evaluation of Java programming skills and knowledge",
        "url": "https://www.shl.com/en/assessments/java-assessment/",
        "remote_testing": "Yes",
        "adaptive_irt": "No",
        "duration": "45 minutes",
        "test_type": "Technical Skills"
    },
    {
        "title": "Python Assessment",
        "description": "Evaluates proficiency in Python programming language",
        "url": "https://www.shl.com/en/assessments/python-assessment/",
        "remote_testing": "Yes",
        "adaptive_irt": "No",
        "duration": "40 minutes",
        "test_type": "Technical Skills"
    },
    {
        "title": "JavaScript Assessment",
        "description": "Tests JavaScript programming capabilities including frameworks",
        "url": "https://www.shl.com/en/assessments/javascript-assessment/",
        "remote_testing": "Yes",
        "adaptive_irt": "No",
        "duration": "40 minutes",
        "test_type": "Technical Skills"
    },
    {
        "title": "DevOps Assessment",
        "description": "Evaluates DevOps concepts, tools, and methodologies",
        "url": "https://www.shl.com/en/assessments/devops-assessment/",
        "remote_testing": "Yes",
        "adaptive_irt": "No",
        "duration": "45 minutes",
        "test_type": "Technical Skills"
    },
    {
        "title": "Agile Assessment",
        "description": "Measures understanding of Agile principles and methodologies",
        "url": "https://www.shl.com/en/assessments/agile-assessment/",
        "remote_testing": "Yes",
        "adaptive_irt": "No",
        "duration": "30 minutes",
        "test_type": "Methodology"
    }
]

# Load the sentence transformer model for semantic search
@st.cache_resource
def load_model():
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        # st.error(f"Error loading model: {str(e)}")
        return None

# Cache the catalog data with embeddings
@st.cache_data(ttl=3600)
def get_catalog_with_embeddings():
    """Get the catalog data with precomputed embeddings"""
    model = load_model()
    df = pd.DataFrame(CATALOG_DATA)
    
    if model:
        # Create combined text for embedding
        df['combined_text'] = df['title'] + " " + df['description'] + " " + df['test_type']
        
        # Generate embeddings
        embeddings = model.encode(df['combined_text'].tolist())
        
        # Convert embeddings to a format suitable for storage
        df['embedding'] = list(embeddings)
    
    return df

@st.cache_data(ttl=3600)
def scrape_shl_catalog():
    """
    Scrape the SHL product catalog to augment our data
    """
    url = "https://www.shl.com/solutions/products/product-catalog/"
    
    try:
        # Generate random user agent and headers
        ua = UserAgent()
        headers = {
            'User-Agent': ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,/;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract products from the table
        products = []
        table = soup.find('table')
        
        if table:
            rows = table.find_all('tr')
            headers = [th.text.strip() for th in rows[0].find_all('th')] if rows else []
            
            for row in rows[1:]:
                cells = row.find_all('td')
                if cells:
                    product_data = {}
                    for i, cell in enumerate(cells):
                        if i < len(headers):
                            product_data[headers[i]] = cell.text.strip()
                    
                    # Extract URL if available
                    link = row.find('a', href=True)
                    if link:
                        product_data['url'] = link['href']
                    
                    products.append(product_data)
        
        # Process scraped data
        if products:
            logger.info(f"Successfully scraped {len(products)} products from SHL catalog")
            
            # Convert to DataFrame
            scraped_df = pd.DataFrame(products)
            
            # Map column names to our format
            if 'Product Name' in scraped_df.columns:
                scraped_df = scraped_df.rename(columns={
                    'Product Name': 'title',
                    'Description': 'description',
                    'Remote Testing': 'remote_testing',
                    'Adaptive/IRT': 'adaptive_irt',
                    'Duration': 'duration',
                    'Assessment Type': 'test_type'
                })
            
            # Merge with our existing data
            base_df = pd.DataFrame(CATALOG_DATA)
            merged_df = pd.concat([base_df, scraped_df]).drop_duplicates(subset=['title']).reset_index(drop=True)
            
            return merged_df
        
        return pd.DataFrame(CATALOG_DATA)
    
    except Exception as e:
        logger.error(f"Error scraping SHL catalog: {str(e)}")
        return pd.DataFrame(CATALOG_DATA)

def get_job_description_from_url(url):
    """
    Extract job description text from a URL
    """
    try:
        # Use newspaper3k to extract content
        article = Article(url)
        article.download()
        article.parse()
        
        # If content is too short, try a different approach
        if len(article.text) < 100:
            response = requests.get(url, headers={'User-Agent': UserAgent().random})
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try to find job description content
            job_content = soup.find(['div', 'section'], id=lambda x: x and ('job' in x.lower() or 'description' in x.lower()))
            if job_content:
                return job_content.get_text(strip=True)
            
            # Look for main content area
            main_content = soup.find(['main', 'article', 'div'], class_=lambda x: x and ('content' in x.lower() or 'main' in x.lower()))
            if main_content:
                return main_content.get_text(strip=True)
            
            # Fallback to body text
            body_text = ' '.join([p.get_text() for p in soup.find_all('p') if len(p.get_text()) > 50])
            if body_text:
                return body_text
        
        return article.text
    except Exception as e:
        logger.error(f"Error extracting job description: {str(e)}")
        return None

def parse_duration_constraint(text):
    """Extract duration constraints from text"""
    duration_pattern = r'(\d+)\s*(?:minute|min|minutes|mins|m)\b'
    matches = re.findall(duration_pattern, text.lower())
    
    if matches:
        # Return the max duration mentioned
        return int(matches[0])
    
    return None

def filter_by_duration(df, query_text):
    """Filter assessments by duration constraint if mentioned"""
    max_duration = parse_duration_constraint(query_text)
    
    if not max_duration:
        return df
    
    # Process the duration strings to extract max minutes
    def extract_max_minutes(duration_str):
        if not isinstance(duration_str, str):
            return 1000  # Very high value for unknown
        
        # Handle ranges like "25-35 minutes"
        range_match = re.search(r'(\d+)-(\d+)', duration_str)
        if range_match:
            return int(range_match.group(2))
        
        # Handle single values like "30 minutes"
        single_match = re.search(r'(\d+)', duration_str)
        if single_match:
            return int(single_match.group(1))
        
        # Handle special cases
        if "day" in duration_str.lower():
            return 1000  # Exclude multi-day assessments
        
        return 1000  # Default high value
    
    # Apply the extraction to create a numeric column
    df['max_minutes'] = df['duration'].apply(extract_max_minutes)
    
    # Filter by max duration
    filtered_df = df[df['max_minutes'] <= max_duration].copy()
    
    # Remove the temporary column
    filtered_df = filtered_df.drop('max_minutes', axis=1)
    
    # If nothing passes the filter, return the top shortest tests
    if filtered_df.empty:
        df['max_minutes'] = df['duration'].apply(extract_max_minutes)
        filtered_df = df.sort_values('max_minutes').head(5).drop('max_minutes', axis=1)
    
    return filtered_df

def extract_skills_keywords(text):
    """Extract potential skills and keywords from job description"""
    common_skills = [
        "leadership", "communication", "analytical", "problem solving", "teamwork", 
        "attention to detail", "critical thinking", "decision making", "time management", 
        "adaptability", "creativity", "emotional intelligence", "negotiation", "project management",
        "verbal", "numerical", "reasoning", "personality", "cognitive", "situational judgment",
        "integrity", "motivation", "strategic", "initiative", "customer service", "sales", 
        "technical", "programming", "coding", "finance", "accounting", "marketing", "management",
        "executive", "entry level", "mid level", "senior level", "supervisor", "director"
    ]
    
    # Add programming languages and technical skills
    tech_skills = [
        "java", "python", "javascript", "js", "sql", "c++", "c#", "ruby", "php", "html", "css",
        "react", "angular", "vue", "node", "express", "django", "flask", "spring", "hibernate",
        "rest", "api", "aws", "azure", "gcp", "devops", "docker", "kubernetes", "jenkins",
        "git", "agile", "scrum", "jira", "confluence", "database", "mongodb", "mysql", "postgresql"
    ]
    
    all_skills = common_skills + tech_skills
    
    # Find skills mentioned in the text
    found_skills = []
    for skill in all_skills:
        if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text.lower()):
            found_skills.append(skill)
    
    return found_skills

def recommend_assessments(query_text, catalog_df, top_n=10):
    """
    Recommend SHL assessments based on query text or job description
    Uses a combination of semantic search and TF-IDF
    """
    if catalog_df.empty:
        return pd.DataFrame()
    
    # Preprocess query text
    query_text = query_text.strip()
    if not query_text:
        return pd.DataFrame()
    
    # Extract skills and additional keywords
    skills_keywords = extract_skills_keywords(query_text)
    enriched_query = query_text + " " + " ".join(skills_keywords)
    
    # Try to use semantic search with sentence transformer
    model = load_model()
    scores = []
    
    if model and 'embedding' in catalog_df.columns:
        # Generate query embedding
        query_embedding = model.encode(enriched_query)
        
        # Calculate cosine similarity with precomputed embeddings
        for idx, row in catalog_df.iterrows():
            if isinstance(row['embedding'], list):
                embedding = np.array(row['embedding'])
                similarity = cosine_similarity([query_embedding], [embedding])[0][0]
                scores.append(similarity)
            else:
                scores.append(0)
        
        catalog_df['score'] = scores
    else:
        # Fallback to TF-IDF if model not available
        docs = [enriched_query] + catalog_df["description"].tolist()
        
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(docs)
        
        query_vec = tfidf_matrix[0]
        cat_vecs = tfidf_matrix[1:]
        
        similarities = cosine_similarity(query_vec, cat_vecs).flatten()
        catalog_df['score'] = similarities
    
    # Apply duration filtering if applicable
    filtered_df = filter_by_duration(catalog_df, query_text)
    
    # Apply skill-specific boosting
    for skill in skills_keywords:
        skill_lower = skill.lower()
        
        # Boost scores for assessments that match found skills
        filtered_df['score'] = filtered_df.apply(
            lambda row: row['score'] * 1.2 
            if skill_lower in row['title'].lower() or skill_lower in row['description'].lower() 
            else row['score'], 
            axis=1
        )
    
    # Get top recommendations
    recommendations = filtered_df.sort_values('score', ascending=False).head(top_n)
    
    return recommendations

# FastAPI endpoint for recommendations
@app.get("/api/recommend", response_model=RecommendationResponse)
async def get_recommendations(
    query: str = Query(..., description="Job description or query text"),
    top_n: int = Query(10, description="Number of recommendations to return", ge=1, le=10)
):
    """API endpoint to get SHL assessment recommendations"""
    # Get catalog data
    catalog_df = pd.DataFrame(CATALOG_DATA)
    
    # Get recommendations
    recommendations = recommend_assessments(query, catalog_df, top_n=top_n)
    
    # Format response
    result = []
    for _, row in recommendations.iterrows():
        result.append(AssessmentRecommendation(
            title=row['title'],
            url=row['url'],
            description=row['description'],
            remote_testing=row['remote_testing'],
            adaptive_irt=row['adaptive_irt'],
            duration=row['duration'],
            test_type=row['test_type'],
            score=float(row['score'])
        ))
    
    return RecommendationResponse(recommendations=result, query=query)

# UI Components
def main():
    # Set up the Streamlit interface
    st.title(" SHL Assessment Recommendation Engine")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Find the perfect SHL assessments for your hiring needs
        Enter a job description or specific query to get personalized assessment recommendations.
        """)
    
    # with col2:
    #     st.markdown("""
    #     ### API Access
    #     Access recommendations programmatically:  
    #     GET /api/recommend?query=your+query+here
    #     """)
    
    # Set up tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["Text Input", "URL Input", "Example Queries"])
    
    # Global variable to store the query
    query_text = ""
    
    with tab1:
        query_text = st.text_area(
            "Enter your query or job description", 
            height=150,
            placeholder="e.g., I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes."
        )
    
    with tab2:
        url = st.text_input(
            "Enter job description URL",
            placeholder="https://example.com/job-description"
        )
        
        if url and st.button("Extract Description", key="extract_btn"):
            with st.spinner("Extracting job description..."):
                extracted_text = get_job_description_from_url(url)
                if extracted_text:
                    st.success("Job description extracted successfully!")
                    query_text = st.text_area("Extracted Job Description", extracted_text, height=150)
                else:
                    st.error("Failed to extract job description. Please try another URL or use text input.")
    
    with tab3:
        example_queries = [
            "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes.",
            "Looking to hire mid-level professionals who are proficient in Python, SQL and Java Script. Need an assessment package that can test all skills with max duration of 60 minutes.",
            "I am hiring for an analyst and wants applications to screen using Cognitive and personality tests, what options are available within 45 mins.",
            "Need to evaluate leadership potential for senior management roles with strong communication skills required."
        ]
        
        selected_example = st.selectbox("Choose an example query", [""] + example_queries)
        if selected_example:
            query_text = selected_example
            st.info(f"Using example query: {selected_example}")
    
    # Options for recommendation
    col1, col2 = st.columns(2)
    with col1:
        num_recommendations = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)
    
    with col2:
        use_semantic_search = st.checkbox("Use semantic search (better results)", value=True)
    
    # Submit button
    if st.button("Get Recommendations", type="primary") and query_text:
        with st.spinner("Analyzing and finding the best matches..."):
            # Get catalog data
            catalog_df = get_catalog_with_embeddings() if use_semantic_search else pd.DataFrame(CATALOG_DATA)
            
            # Get recommendations
            recommendations = recommend_assessments(query_text, catalog_df, top_n=num_recommendations)
            
            if not recommendations.empty:
                st.success(f"Found {len(recommendations)} relevant assessments")
                
                # Display recommendations in a table
                st.subheader("📋 Recommended Assessments")
                
                # Display as a table
                display_df = recommendations[['title', 'description', 'remote_testing', 
                                            'adaptive_irt', 'duration', 'test_type', 'score']]
                display_df['score'] = display_df['score'].round(2)
                display_df = display_df.rename(columns={
                    'title': 'Assessment Name',
                    'description': 'Description',
                    'remote_testing': 'Remote Testing',
                    'adaptive_irt': 'Adaptive/IRT',
                    'duration': 'Duration',
                    'test_type': 'Test Type',
                    'score': 'Relevance Score'
                })
                
                st.dataframe(display_df, use_container_width=True)
                
                # Display detailed cards
                st.subheader("🎯 Detailed Recommendations")
                for idx, row in recommendations.iterrows():
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"### [{row['title']}]({row['url']})")
                            st.markdown(f"*Description:* {row['description']}")
                        with col2:
                            st.markdown(f"*Remote Testing:* {row['remote_testing']}")
                            st.markdown(f"*Adaptive/IRT:* {row['adaptive_irt']}")
                            st.markdown(f"*Duration:* {row['duration']}")
                            st.markdown(f"*Test Type:* {row['test_type']}")
                            st.markdown(f"*Relevance Score:* {row['score']:.2f}")
                        st.markdown("---")
                
                # Analysis of recommendations
                st.subheader("📊 Analysis of Recommendations")
                test_types = recommendations['test_type'].value_counts()
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Assessment Types")
                    st.dataframe(pd.DataFrame({'Count': test_types}))
                with col2:
                    st.markdown("#### Duration Analysis")
                    duration_categories = recommendations['duration'].apply(lambda x: "Under 30 min" if "day" not in x.lower() and int(re.search(r'(\d+)', x).group(1)) < 30 
                                                       else "30-45 min" if "day" not in x.lower() and int(re.search(r'(\d+)', x).group(1)) < 45 
                                                       else "45-60 min" if "day" not in x.lower() and int(re.search(r'(\d+)', x).group(1)) < 60 
                                                       else "Over 60 min").value_counts()
                    st.dataframe(pd.DataFrame({'Count': duration_categories}))
            else:
                st.warning("No matching assessments found. Please try a different query.")
    
    # Display extracted skills if text is provided
    if query_text:
        with st.expander("Query Analysis", expanded=False):
            skills = extract_skills_keywords(query_text)
            duration = parse_duration_constraint(query_text)
            
            st.markdown("#### Extracted Skills & Keywords")
            if skills:
                st.write(", ".join(skills))
            else:
                st.write("No specific skills detected")
            
            st.markdown("#### Time Constraint")
            if duration:
                st.write(f"Maximum duration: {duration} minutes")
            else:
                st.write("No time constraint detected")

# Run the FastAPI app for the API endpoint
def run_api():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Main entry point
if __name__ == "__main__":
    import threading
    
    # Start the API in a separate thread
    api_thread = threading.Thread(target=run_api)
    api_thread.daemon = True
    api_thread.start()
    
    # Run the Streamlit app
    main()
