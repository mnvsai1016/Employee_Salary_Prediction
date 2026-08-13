import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from preprocess import Preprocess, load_model_asset

# -----------------------------------------------------------------------------
# Configuration & Setup
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model_files"

st.set_page_config(
    page_title="Employee Salary Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark / Modern Glassmorphism Theme)
st.markdown("""
<style>
    /* Main Background & Card Styling */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    
    .hero-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    
    .hero-title {
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
    }

    /* Card Containers */
    .result-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
        margin-top: 1.5rem;
    }
    
    .metric-value {
        font-size: 2.8rem;
        font-weight: 800;
        color: #38bdf8;
        text-align: center;
        margin: 1rem 0;
    }
    
    .range-box {
        display: flex;
        justify-content: space-around;
        background: #1e293b;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-top: 1rem;
    }

    /* Form & Input adjustments */
    div[data-baseweb="select"] > div {
        background-color: #1e293b !important;
        border-color: #475569 !important;
        border-radius: 8px !important;
        color: #f8fafc !important;
    }
    
    div[data-baseweb="select"] span {
        color: #f8fafc !important;
    }
    
    ul[role="listbox"] {
        background-color: #1e293b !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Resource Caching
# -----------------------------------------------------------------------------
@st.cache_resource
def load_trained_model():
    """Cache and load the pre-trained salary prediction model."""
    model_path = MODEL_DIR / "salary_model.pkl"
    if not model_path.exists():
        st.error(f"❌ Model file not found at {model_path}. Please ensure model_files.zip is extracted.")
        st.stop()
    with open(model_path, "rb") as f:
        return pickle.load(f)


# Load ML model resource
try:
    salary_model = load_trained_model()
except Exception as err:
    st.error(f"Failed to load salary model: {err}")
    st.stop()


# -----------------------------------------------------------------------------
# Header Section
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-header">
    <div class="hero-title">💼 Tech Salary Predictor</div>
    <div class="hero-subtitle">Predict global developer & tech compensation using machine learning</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Options Data Definitions
# -----------------------------------------------------------------------------
job_options = [
    "Academic researcher", "Blockchain", "Cloud infrastructure engineer",
    "Data or business analyst", "Data engineer", "Data scientist or machine learning specialist",
    "Database administrator", "Designer", "Developer Advocate", "Developer, AI",
    "Developer, back-end", "Developer, desktop or enterprise applications",
    "Developer, embedded applications or devices", "Developer Experience",
    "Developer, front-end", "Developer, full-stack", "Developer, game or graphics",
    "Developer, mobile", "Developer, QA or test", "DevOps specialist", "Educator",
    "Engineer, site reliability", "Engineering manager", "Hardware Engineer",
    "Marketing or sales professional", "Product manager", "Project manager",
    "Research & Development role", "Scientist", "Senior Executive (C-Suite, VP, etc.)",
    "Student", "System administrator", "Security professional", "Other"
]

education_options = [
    "Primary/elementary school",
    "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)",
    "Some college/university study without earning a degree",
    "Associate degree (A.A., A.S., etc.)",
    "Bachelor's degree (B.A., B.S., B.Eng., etc.)",
    "Master's degree (M.A., M.S., M.Eng., MBA, etc.)",
    "Professional degree (JD, MD, Ph.D, Ed.D, etc.)",
    "Something else"
]

country_options = [
    'Pakistan', 'Austria', 'Turkey', 'France', 'United States of America',
    'United Kingdom of Great Britain and Northern Ireland', 'Bulgaria', 'Greece',
    'Brazil', 'Germany', 'Italy', 'Ukraine', 'Russian Federation', 'South Africa',
    'Czech Republic', 'Canada', 'Iran, Islamic Republic of...', 'Dominican Republic',
    'Switzerland', 'Belgium', 'Peru', 'Bolivia', 'Morocco', 'India', 'Luxembourg',
    'Georgia', 'Saudi Arabia', 'Ireland', 'Romania', 'Spain', 'Sweden', 'Cyprus',
    'Paraguay', 'Lithuania', 'Netherlands', 'Slovenia', 'Singapore',
    'Venezuela, Bolivarian Republic of...', 'Japan', 'Latvia', 'Costa Rica',
    'Poland', 'Norway', 'Portugal', 'Finland', 'Israel', 'Nicaragua', 'Serbia',
    'Croatia', 'Hungary', 'Bangladesh', 'Indonesia', 'Denmark',
    'Bosnia and Herzegovina', 'Mexico', 'Philippines', 'Thailand', 'Slovakia',
    'El Salvador', 'Ecuador', 'Argentina', 'Algeria', 'Kazakhstan', 'Malaysia',
    'Zimbabwe', 'Afghanistan', 'Malta', 'Belarus', 'Colombia', 'Egypt',
    'Montenegro', 'Australia', 'Isle of Man', 'New Zealand', 'Palestine', 'Armenia',
    'Maldives', 'United Arab Emirates', 'Nigeria', 'Fiji', 'Guatemala', 'Uganda',
    'Turkmenistan', 'Mauritius', 'Estonia', 'Kenya', 'Gabon', 'South Korea',
    'Chile', 'Uruguay', 'Viet Nam', 'China', 'Ghana', 'Hong Kong (S.A.R.)',
    'Sri Lanka', 'Mongolia', 'Uzbekistan', 'Republic of Korea', 'Nepal', 'Taiwan',
    'Lebanon', 'Benin', 'Democratic Republic of the Congo', 'Syrian Arab Republic',
    'Iraq', 'Namibia', 'Kyrgyzstan', "Lao People's Democratic Republic", 'Tunisia',
    'Burundi', 'Rwanda', 'Iceland', 'Mauritania', 'Sierra Leone', 'Panama', 'Cuba',
    'Guyana', 'Zambia', 'Ethiopia', 'Republic of Moldova', 'Jordan', 'Jamaica',
    'Nomadic', 'Andorra', 'Republic of North Macedonia',
    "Democratic People's Republic of Korea", 'Kuwait', 'Togo', 'Qatar',
    'Tajikistan', 'Albania', 'Sudan', 'Kosovo', 'Angola', "Côte d'Ivoire",
    'Malawi', 'Burkina Faso', 'United Republic of Tanzania', 'Madagascar',
    'Cameroon', 'Yemen', 'Myanmar', 'Oman', 'Azerbaijan', 'Central African Republic',
    'Somalia', 'Suriname', 'Libyan Arab Jamahiriya', 'Cape Verde', 'Bahrain',
    'Bhutan', 'Trinidad and Tobago', 'Niger', 'Mozambique', 'Antigua and Barbuda',
    'Honduras', 'Liechtenstein', 'Senegal', 'Congo, Republic of the...', 'Samoa',
    'Brunei Darussalam', 'Lesotho', 'Cambodia', 'Botswana', 'Barbados', 'Mali',
    'Haiti', 'Swaziland', 'Chad', 'Monaco'
]

age_options = [
    "Under 18 years old", "18-24 years old", "25-34 years old",
    "35-44 years old", "45-54 years old", "55-64 years old",
    "65 years or older", "Prefer not to say"
]

employment_options = [
    "Employed, full-time", "Employed, part-time",
    "Independent contractor, freelancer, or self-employed",
    "Not employed, but looking for work", "Not employed, and not looking for work",
    "Student, full-time", "Student, part-time", "Retired", "I prefer not to say"
]

work_situation_options = ["Remote", "In-person", "Hybrid (some remote, some in-person)"]

programming_languages = [
    "Ada", "Apex", "Assembly", "Bash/Shell (all shells)", "C", "C#", "C++", "Clojure", "Cobol", "Crystal",
    "Dart", "Delphi", "Elixir", "Erlang", "F#", "Fortran", "GDScript", "Go", "Groovy", "Haskell",
    "HTML/CSS", "Java", "JavaScript", "Julia", "Kotlin", "Lisp", "Lua", "MATLAB", "MicroPython", "Nim",
    "Objective-C", "OCaml", "Perl", "PHP", "PowerShell", "Prolog", "Python", "R", "Ruby", "Rust",
    "Scala", "Solidity", "SQL", "Swift", "TypeScript", "VBA", "Visual Basic (.Net)", "Zephyr", "Zig"
]

databases = [
    "BigQuery", "Cassandra", "Clickhouse", "Cloud Firestore", "Cockroachdb", "Cosmos DB", "Couch DB",
    "Couchbase", "Databricks SQL", "Datomic", "DuckDB", "Dynamodb", "Elasticsearch", "EventStoreDB",
    "Firebase Realtime Database", "Firebird", "H2", "IBM DB2", "InfluxDB", "MariaDB", "Microsoft Access",
    "Microsoft SQL Server", "MongoDB", "MySQL", "Neo4J", "Oracle", "PostgreSQL", "Presto", "RavenDB",
    "Redis", "Snowflake", "Solr", "SQLite", "Supabase", "TiDB"
]

cloud_platforms = [
    "Alibaba Cloud", "Amazon Web Services (AWS)", "Cloudflare", "Colocation", "Databricks", "Digital Ocean",
    "Firebase", "Fly.io", "Google Cloud", "Heroku", "Hetzner", "IBM Cloud Or Watson", "Linode",
    "Managed Hosting", "Microsoft Azure", "Netlify", "OpenShift", "OpenStack", "Oracle Cloud Infrastructure (OCI)",
    "OVH", "PythonAnywhere", "Render", "Scaleway", "Supabase", "Vercel", "VMware", "Vultr"
]

web_frameworks = [
    "Angular", "AngularJS", "ASP.NET", "ASP.NET CORE", "Astro", "Blazor", "CodeIgniter", "Deno", "Django",
    "Drupal", "Elm", "Express", "FastAPI", "Fastify", "Flask", "Gatsby", "Htmx", "jQuery", "Laravel",
    "NestJS", "Next.js", "Node.js", "Nuxt.js", "Phoenix", "Play Framework", "React", "Remix",
    "Ruby on Rails", "Solid.js", "Spring Boot", "Strapi", "Svelte", "Symfony", "Vue.js", "WordPress", "Yii 2"
]

embedded_systems = [
    "Arduino", "Boost.Test", "build2", "Catch2", "CMake", "Cargo", "cppunit", "CUTE", "doctest",
    "GNU GCC", "LLVM's Clang", "Meson", "Micronaut", "MSVC", "Ninja", "PlatformIO", "QMake",
    "Rasberry Pi", "SCons", "ZMK"
]

other_frameworks = [
    ".NET (5+)", ".NET Framework (1.0 - 4.8)", ".NET MAUI", "Apache Kafka", "Apache Spark", "Capacitor",
    "Cordova", "CUDA", "DirectX", "Electron", "Flutter", "GTK", "Hadoop", "Hugging Face Transformers",
    "Ionic", "JAX", "Keras", "Ktor", "MFC", "mlflow", "Numpy", "OpenCL", "Opencv", "OpenGL", "Pandas",
    "Qt", "Quarkus", "RabbitMQ", "React Native", "Roslyn", "Ruff", "Scikit-Learn", "Spring Framework",
    "SwiftUI", "Tauri", "TensorFlow", "Tidyverse", "Torch/PyTorch", "Xamarin"
]

developer_tools = [
    "Ansible", "Ant", "APT", "Bun", "Chef", "Chocolatey", "Composer", "Dagger", "Docker", "Godot",
    "Google Test", "Gradle", "Homebrew", "Kubernetes", "Make", "Maven (build tool)", "MSBuild", "Ninja",
    "Nix", "npm", "NuGet", "Pacman", "Pip", "pnpm", "Podman", "Pulumi", "Puppet", "Terraform",
    "Unity 3D", "Unreal Engine", "Visual Studio Solution", "Vite", "Webpack", "Yarn"
]

# -----------------------------------------------------------------------------
# Input Form UI
# -----------------------------------------------------------------------------
st.write("Fill out your developer profile below to estimate your annual market salary.")

tab_demographics, tab_stack, tab_tools = st.tabs([
    "👤 Demographics & Role",
    "💻 Core Tech Stack",
    "🛠️ Tools & Libraries"
])

with tab_demographics:
    col1, col2 = st.columns(2)
    with col1:
        current_job = st.selectbox(
            "Current Job Role *",
            ["Select an option"] + job_options,
            help="Select the role that best describes your primary responsibilities."
        )
        education = st.selectbox(
            "Highest Education Level *",
            ["Select an option"] + education_options
        )
        country = st.selectbox(
            "Work Country *",
            ["Select an option"] + sorted(country_options)
        )

    with col2:
        age = st.selectbox(
            "Age Group *",
            ["Select an option"] + age_options
        )
        experience = st.number_input(
            "Years of Professional Experience",
            min_value=0, max_value=50, value=3, step=1
        )
        work_situation = st.selectbox(
            "Work Setup",
            ["Select an option"] + work_situation_options
        )

    employment_status = st.multiselect(
        "Current Employment Status",
        employment_options,
        default=["Employed, full-time"]
    )

with tab_stack:
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        selected_languages = st.multiselect(
            "Programming / Markup Languages",
            programming_languages,
            default=["Python", "JavaScript", "SQL"] if "Python" in programming_languages else []
        )
        selected_databases = st.multiselect(
            "Database Environments",
            databases
        )
    with col_s2:
        selected_cloud = st.multiselect(
            "Cloud Platforms",
            cloud_platforms
        )
        selected_frameworks = st.multiselect(
            "Web Frameworks & Technologies",
            web_frameworks
        )

with tab_tools:
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        selected_embedded = st.multiselect(
            "Embedded Systems & Build Tools",
            embedded_systems
        )
        selected_other_frameworks = st.multiselect(
            "Other Libraries / Frameworks (e.g. PyTorch, Pandas)",
            other_frameworks
        )
    with col_t2:
        selected_tools = st.multiselect(
            "Developer Tools (e.g. Docker, Git, VS Code)",
            developer_tools
        )

st.markdown("---")

# -----------------------------------------------------------------------------
# Prediction Trigger & Output
# -----------------------------------------------------------------------------
if st.button("🚀 Calculate Estimated Salary", type="primary", use_container_width=True):
    # Validation Check
    if current_job == "Select an option" or education == "Select an option" or country == "Select an option" or age == "Select an option":
        st.warning("⚠️ Please complete all required fields marked with (*) before predicting.")
    else:
        form_data = {
            "DevType": current_job,
            "EdLevel": education,
            "Age": age,
            "Country": country,
            "WorkExp": experience,
            "Employment": employment_status if employment_status else ["Employed, full-time"],
            "RemoteWork": work_situation if work_situation != "Select an option" else "Remote",
            "LanguageHaveWorkedWith": selected_languages,
            "DatabaseHaveWorkedWith": selected_databases,
            "PlatformHaveWorkedWith": selected_cloud,
            "WebframeHaveWorkedWith": selected_frameworks,
            "EmbeddedHaveWorkedWith": selected_embedded,
            "MiscFrameworks": selected_other_frameworks,
            "ToolsHaveWorkedWith": selected_tools
        }

        with st.spinner("Processing profile & computing prediction model..."):
            try:
                # Preprocess input data
                input_df = pd.DataFrame([form_data])
                processed_input = Preprocess(input_df)

                # Inference
                prediction_log = salary_model.predict(processed_input.iloc[0].values.reshape(1, -1))
                predicted_salary = float(np.expm1(prediction_log[0]))

                # Calculate estimated error margin bounds (+/- 10%)
                margin = 0.10
                lower_bound = round(predicted_salary * (1 - margin))
                upper_bound = round(predicted_salary * (1 + margin))

                # Display Results Card
                st.markdown(f"""
                <div class="result-card">
                    <h3 style="text-align: center; margin: 0; color: #94a3b8;">Predicted Annual Base Compensation</h3>
                    <div class="metric-value">${predicted_salary:,.0f} USD</div>
                    <div class="range-box">
                        <div><span style="color: #94a3b8;">Estimated Min:</span> <strong style="color: #22c55e;">${lower_bound:,.0f}</strong></div>
                        <div><span style="color: #94a3b8;">Median Estimate:</span> <strong style="color: #38bdf8;">${predicted_salary:,.0f}</strong></div>
                        <div><span style="color: #94a3b8;">Estimated Max:</span> <strong style="color: #a855f7;">${upper_bound:,.0f}</strong></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.success("🎉 Salary estimation calculated successfully!")

                # View Raw Submitted Features
                with st.expander("📋 View Processed Profile Features"):
                    st.json(form_data)

            except Exception as e:
                st.error(f"❌ Error during salary prediction: {str(e)}")
