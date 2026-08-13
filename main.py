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
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Injectable CSS for Streamlit Native Elements & Dark Aesthetic
st.markdown("""
<style>
    /* Metric Styling */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: #38bdf8 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
    }
    
    /* Highlight Main Result Card */
    .hero-salary-card {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
        border: 2px solid #6366f1;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 20px 40px rgba(99, 102, 241, 0.2);
    }
    
    .salary-main-text {
        font-size: 3.8rem;
        font-weight: 800;
        color: #38bdf8;
        margin: 0.5rem 0;
        letter-spacing: -0.02em;
    }
    
    .tier-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.25);
        border: 1px solid #818cf8;
        color: #c7d2fe;
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    
    .header-banner {
        background: linear-gradient(135deg, #312e81 0%, #1e1b4b 50%, #0f172a 100%);
        border: 1px solid #4338ca;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Tab Header Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1e293b;
        border-radius: 10px;
        color: #94a3b8;
        font-weight: 600;
        padding: 10px 20px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #4f46e5 !important;
        color: #ffffff !important;
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
    st.title("💼 Settings")
    
    currency = st.selectbox(
        "Display Currency",
        ["USD ($)", "EUR (€)", "GBP (£)", "INR (₹)", "CAD ($)", "AUD ($)"],
        index=0
    )
    
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
    st.subheader("🤖 Model Summary")
    st.markdown("""
    - **Model**: XGBoost Regressor  
    - **Dataset**: Global Developer Survey  
    - **Features Evaluated**: 50+ Tech Stack Indicators  
    - **Target**: Annual Base Compensation
    """)

    st.markdown("---")
    st.caption("💡 *Tip: Selecting your specific skills improves prediction accuracy.*")


# -----------------------------------------------------------------------------
# Hero Header
# -----------------------------------------------------------------------------
st.markdown("""
<div class="header-banner">
    <h1 style="color: #ffffff; margin: 0; font-size: 2.6rem;">💼 Tech Salary Predictor</h1>
    <p style="color: #a5b4fc; font-size: 1.1rem; margin-top: 0.5rem;">Estimate your annual market compensation using machine learning</p>
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
tab1, tab2, tab3 = st.tabs([
    "👤 1. Demographics & Experience",
    "💻 2. Core Tech Stack",
    "🛠️ 3. Cloud, DBs & Tools"
])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        current_job = st.selectbox(
            "Primary Role *",
            ["Select an option"] + job_options
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
            min_value=0, max_value=50, value=4, step=1
        )
        work_situation = st.selectbox(
            "Work Setup",
            ["Select an option"] + work_situation_options
        )

    employment_status = st.multiselect(
        "Employment Status",
        employment_options,
        default=["Employed, full-time"]
    )

with tab2:
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
            "ML & Data Frameworks",
            other_frameworks
        )
        selected_embedded = st.multiselect(
            "Embedded & C++ Systems Tools",
            embedded_systems
        )

with tab3:
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
            "Developer Tools & DevOps",
            developer_tools,
            default=["Docker", "npm"] if "Docker" in developer_tools else []
        )


# -----------------------------------------------------------------------------
# Trigger & Results Display
# -----------------------------------------------------------------------------
st.markdown("---")

if st.button("🚀 Calculate Estimated Salary", type="primary", use_container_width=True):
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

        with st.spinner("⚡ Running model prediction..."):
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

                # Seniority Tier Tag
                if experience < 3:
                    tier = "🌱 Junior / Entry Level"
                elif experience < 7:
                    tier = "🚀 Mid-Level Specialist"
                elif experience < 12:
                    tier = "⭐ Senior Engineer / Lead"
                else:
                    tier = "👑 Executive / Principal Specialist"

                # 1. Main Hero Container
                st.markdown(f"""
                <div class="hero-salary-card">
                    <div style="color: #a5b4fc; font-size: 0.9rem; text-transform: uppercase; font-weight: 700;">
                        Estimated Annual Base Compensation ({curr_code})
                    </div>
                    <div class="salary-main-text">{symbol}{predicted_salary:,.0f}</div>
                    <div class="tier-badge">{tier}</div>
                </div>
                """, unsafe_allow_html=True)

                # 2. Native Streamlit Columns & Metrics (Guaranteed Clean Rendering)
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric(label="Estimated Min (10th%)", value=f"{symbol}{lower_bound:,.0f}")
                with m2:
                    st.metric(label="Monthly Equivalent", value=f"{symbol}{monthly_salary:,.0f} / mo")
                with m3:
                    st.metric(label="Estimated Max (90th%)", value=f"{symbol}{upper_bound:,.0f}")

                st.progress(0.50, text="Estimated Market Salary Percentile Range")

                st.balloons()

                # Collapsible Summary of Inputs
                with st.expander("📋 View Submitted Profile Data"):
                    st.json(form_data)

            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
