import pickle
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from preprocess import Preprocess, ensure_model_files_extracted

# -----------------------------------------------------------------------------
# Page Configuration & Aesthetics
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model_files"

st.set_page_config(
    page_title="Tech Compensation Predictor | AI Salary Insights",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Injectable CSS with Modern Google Fonts & Glassmorphism Aesthetics
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">

<style>
    /* Global Typography & Background */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    .stApp {
        background: radial-gradient(circle at 50% 0%, #1e1b4b 0%, #0f172a 60%, #020617 100%);
        color: #f8fafc;
    }

    /* Glassmorphism Containers */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
    }

    /* Hero Header Styling */
    .hero-container {
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.25) 0%, rgba(147, 51, 234, 0.2) 50%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(129, 140, 248, 0.3);
        border-radius: 24px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 25px 50px -12px rgba(79, 70, 229, 0.25);
        position: relative;
        overflow: hidden;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid rgba(129, 140, 248, 0.4);
        color: #a5b4fc;
        padding: 0.4rem 1rem;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    .hero-title {
        background: linear-gradient(135deg, #ffffff 0%, #c7d2fe 50%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0.5rem 0;
        line-height: 1.2;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.15rem;
        max-width: 650px;
        margin: 0 auto;
        font-weight: 400;
    }

    /* Result Card Highlight */
    .result-hero-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 2px solid rgba(99, 102, 241, 0.5);
        border-radius: 24px;
        padding: 2.5rem;
        text-align: center;
        box-shadow: 0 25px 50px -12px rgba(99, 102, 241, 0.3);
        margin-top: 2rem;
        animation: fadeIn 0.5s ease-in-out;
    }

    .main-salary-display {
        font-size: 3.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0 1rem 0;
        letter-spacing: -0.03em;
    }

    /* Metric Grid */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.2rem;
        margin-top: 1.5rem;
    }

    .sub-metric-box {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.2rem 1rem;
        text-align: center;
    }

    .sub-metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }

    .sub-metric-val {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-top: 0.3rem;
    }

    /* Range Visualizer Bar */
    .range-bar-bg {
        background: rgba(51, 65, 85, 0.5);
        height: 12px;
        border-radius: 10px;
        position: relative;
        margin: 2rem 0 1rem 0;
        overflow: hidden;
    }

    .range-bar-fill {
        background: linear-gradient(90deg, #22c55e 0%, #38bdf8 50%, #a855f7 100%);
        height: 100%;
        border-radius: 10px;
    }

    /* Custom Streamlit Dropdowns & MultiSelect */
    div[data-baseweb="select"] > div {
        background-color: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        color: #f8fafc !important;
    }
    
    div[data-baseweb="select"] span {
        color: #f8fafc !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        padding: 0.8rem 2rem !important;
        border-radius: 14px !important;
        border: none !important;
        box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.5) !important;
        transition: all 0.2s ease !important;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 30px -5px rgba(79, 70, 229, 0.7) !important;
    }

    /* Keyframe Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Resource Caching
# -----------------------------------------------------------------------------
@st.cache_resource
def load_trained_model():
    """Cache and load pre-trained ML salary prediction model."""
    ensure_model_files_extracted()
    model_path = MODEL_DIR / "salary_model.pkl"
    if not model_path.exists():
        st.error(f"❌ Model file not found at {model_path}. Please check repository setup.")
        st.stop()
    with open(model_path, "rb") as f:
        return pickle.load(f)


try:
    salary_model = load_trained_model()
except Exception as err:
    st.error(f"Failed to initialize machine learning model: {err}")
    st.stop()


# -----------------------------------------------------------------------------
# Sidebar Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric-line/100/money-bag-bitcoin.png", width=70)
    st.title("⚙️ Currency & Settings")
    
    currency = st.selectbox(
        "Display Currency",
        ["USD ($)", "EUR (€)", "GBP (£)", "INR (₹)", "CAD ($)", "AUD ($)"],
        index=0
    )
    
    # Currency Rates dictionary
    rates = {
        "USD ($)": (1.0, "$", "USD"),
        "EUR (€)": (0.92, "€", "EUR"),
        "GBP (£)": (0.79, "£", "GBP"),
        "INR (₹)": (83.2, "₹", "INR"),
        "CAD ($)": (1.36, "CA$", "CAD"),
        "AUD ($)": (1.52, "A$", "AUD")
    }
    multiplier, symbol, curr_code = rates[currency]

    st.markdown("---")
    st.markdown("### 🤖 Model Information")
    st.info("""
    **Model Architecture**: XGBoost Regressor  
    **Trained On**: Stack Overflow Developer Data  
    **Features Evaluated**: 50+ Tech Stack Indicators  
    **Prediction Metrics**: Base Compensation
    """)

    st.markdown("---")
    st.markdown("💡 *Tip: Adding more specific technologies improves market alignment.*")


# -----------------------------------------------------------------------------
# Hero Section Header
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">✨ Machine Learning Powered Insights</div>
    <div class="hero-title">Tech Compensation Predictor</div>
    <div class="hero-subtitle">Estimate your market value based on real global developer survey data, experience tier, and technical skill stack.</div>
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
# Interactive Tabbed Form
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "👤  1. Demographics & Experience",
    "💻  2. Core Tech Stack",
    "🛠️  3. Tools, Cloud & Databases"
])

with tab1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Personal & Professional Profile")
    col1, col2 = st.columns(2)
    with col1:
        current_job = st.selectbox(
            "Primary Role *",
            ["Select an option"] + job_options,
            help="Select the option that best reflects your current day-to-day work."
        )
        education = st.selectbox(
            "Highest Formal Education *",
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
            min_value=0, max_value=50, value=4, step=1,
            help="Total years of full-time professional coding/tech experience."
        )
        work_situation = st.selectbox(
            "Work Arrangement",
            ["Select an option"] + work_situation_options
        )

    employment_status = st.multiselect(
        "Current Employment Status",
        employment_options,
        default=["Employed, full-time"]
    )
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Languages & Frameworks")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        selected_languages = st.multiselect(
            "Programming / Scripting Languages",
            programming_languages,
            default=["Python", "JavaScript", "SQL"]
        )
        selected_frameworks = st.multiselect(
            "Web Frameworks & Technologies",
            web_frameworks,
            default=["React", "Node.js"] if "React" in web_frameworks else []
        )
    with col_s2:
        selected_other_frameworks = st.multiselect(
            "ML & Data Frameworks (e.g., PyTorch, Pandas, Scikit-Learn)",
            other_frameworks
        )
        selected_embedded = st.multiselect(
            "Embedded & C++ Systems/Build Tools",
            embedded_systems
        )
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Infrastructure, Databases & Dev Tools")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        selected_cloud = st.multiselect(
            "Cloud Platforms",
            cloud_platforms,
            default=["Amazon Web Services (AWS)"] if "Amazon Web Services (AWS)" in cloud_platforms else []
        )
        selected_databases = st.multiselect(
            "Database Environments",
            databases,
            default=["PostgreSQL"] if "PostgreSQL" in databases else []
        )
    with col_t2:
        selected_tools = st.multiselect(
            "Developer Tools & DevOps (e.g., Docker, Kubernetes, Git)",
            developer_tools,
            default=["Docker", "npm"] if "Docker" in developer_tools else []
        )
    st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Trigger & Results Display
# -----------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)

if st.button("✨ Predict Target Compensation", type="primary"):
    # Validation Check
    if current_job == "Select an option" or education == "Select an option" or country == "Select an option" or age == "Select an option":
        st.warning("⚠️ Please complete all required profile fields marked with (*) in Tab 1 before predicting.")
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

        with st.spinner("⚡ Running XGBoost inference & processing profile vector..."):
            try:
                # Preprocess input data
                input_df = pd.DataFrame([form_data])
                processed_input = Preprocess(input_df)

                # Inference
                prediction_log = salary_model.predict(processed_input.iloc[0].values.reshape(1, -1))
                predicted_salary_usd = float(np.expm1(prediction_log[0]))
                
                # Apply currency conversion
                predicted_salary = predicted_salary_usd * multiplier
                margin = 0.10
                lower_bound = round(predicted_salary * (1 - margin))
                upper_bound = round(predicted_salary * (1 + margin))

                monthly_salary = round(predicted_salary / 12)
                hourly_rate = round(predicted_salary / 2080, 2)

                # Seniority Tier Tag
                if experience < 3:
                    tier = "🌱 Junior / Entry Level"
                elif experience < 7:
                    tier = "🚀 Mid-Level Specialist"
                elif experience < 12:
                    tier = "⭐ Senior Engineer / Lead"
                else:
                    tier = "👑 Executive / Principal Specialist"

                # Render Modern Visual Results Hero Card
                st.markdown(f"""
                <div class="result-hero-card">
                    <div style="font-size: 0.9rem; color: #a5b4fc; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700;">
                        Estimated Annual Compensation Package ({curr_code})
                    </div>
                    <div class="main-salary-display">{symbol}{predicted_salary:,.0f}</div>
                    <div style="display: inline-block; background: rgba(99, 102, 241, 0.2); border: 1px solid rgba(129, 140, 248, 0.4); color: #c7d2fe; padding: 0.4rem 1.2rem; border-radius: 20px; font-weight: 600; font-size: 0.9rem;">
                        {tier}
                    </div>

                    <div class="range-bar-bg">
                        <div class="range-bar-fill"></div>
                    </div>

                    <div class="metric-grid">
                        <div class="sub-metric-box">
                            <div class="sub-metric-label">Estimated Min (10th%)</div>
                            <div class="sub-metric-val" style="color: #22c55e;">{symbol}{lower_bound:,.0f}</div>
                        </div>
                        <div class="sub-metric-box">
                            <div class="sub-metric-label">Monthly Equivalent</div>
                            <div class="sub-metric-val" style="color: #38bdf8;">{symbol}{monthly_salary:,.0f} / mo</div>
                        </div>
                        <div class="sub-metric-box">
                            <div class="sub-metric-label">Estimated Max (90th%)</div>
                            <div class="sub-metric-val" style="color: #c084fc;">{symbol}{upper_bound:,.0f}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.balloons()

                # Collapsible Summary of Inputs
                with st.expander("📋 View Submitted Feature Payload"):
                    st.json(form_data)

            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
