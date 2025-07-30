import os
import streamlit as st
import json
import pickle
import numpy as np
import pandas as pd
from preprocess import Preprocess

st.set_page_config(page_title="Employee_salary_predictor", page_icon="💻", layout="wide")

st.title("🔍 Employee Salary prediction")
st.markdown("---")

if 'form_data' not in st.session_state:
    st.session_state.form_data = {}

st.header("Current Job Description")
job_options = [
    "Academic researcher",
    "Blockchain",
    "Cloud infrastructure engineer",
    "Data or business analyst",
    "Data engineer",
    "Data scientist or machine learning specialist",
    "Database administrator",
    "Designer",
    "Developer Advocate",
    "Developer, AI",
    "Developer, back-end",
    "Developer, desktop or enterprise applications",
    "Developer, embedded applications or devices",
    "Developer Experience",
    "Developer, front-end",
    "Developer, full-stack",
    "Developer, game or graphics",
    "Developer, mobile",
    "Developer, QA or test",
    "DevOps specialist",
    "Educator",
    "Engineer, site reliability",
    "Engineering manager",
    "Hardware Engineer",
    "Marketing or sales professional",
    "Product manager",
    "Project manager",
    "Research & Development role",
    "Scientist",
    "Senior Executive (C-Suite, VP, etc.)",
    "Student",
    "System administrator",
    "Security professional",
    "Other"
]

current_job = st.selectbox(
    "Which of the following describes your current job, the one you do most of the time? Please select only one.",
    ["Select an option"] + job_options
)

st.header("Education Level")
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

education = st.selectbox(
    "Which of the following best describes the highest level of formal education that you've completed?",
    ["Select an option"] + education_options
)
st.header("Country")
country_options = ['Pakistan',
 'Austria',
 'Turkey',
 'France',
 'United States of America',
 'United Kingdom of Great Britain and Northern Ireland',
 'Bulgaria',
 'Greece',
 'Brazil',
 'Germany',
 'Italy',
 'Ukraine',
 'Russian Federation',
 'South Africa',
 'Czech Republic',
 'Canada',
 'Iran, Islamic Republic of...',
 'Dominican Republic',
 'Switzerland',
 'Belgium',
 'Peru',
 'Bolivia',
 'Morocco',
 'India',
 'Luxembourg',
 'Georgia',
 'Saudi Arabia',
 'Ireland',
 'Romania',
 'Spain',
 'Sweden',
 'Cyprus',
 'Paraguay',
 'Lithuania',
 'Netherlands',
 'Slovenia',
 'Singapore',
 'Venezuela, Bolivarian Republic of...',
 'Japan',
 'Latvia',
 'Costa Rica',
 'Poland',
 'Norway',
 'Portugal',
 'Finland',
 'Israel',
 'Nicaragua',
 'Serbia',
 'Croatia',
 'Hungary',
 'Bangladesh',
 'Indonesia',
 'Denmark',
 'Bosnia and Herzegovina',
 'Mexico',
 'Philippines',
 'Thailand',
 'Slovakia',
 'El Salvador',
 'Ecuador',
 'Argentina',
 'Algeria',
 'Kazakhstan',
 'Malaysia',
 'Zimbabwe',
 'Afghanistan',
 'Malta',
 'Belarus',
 'Colombia',
 'Egypt',
 'Montenegro',
 'Australia',
 'Isle of Man',
 'New Zealand',
 'Palestine',
 'Armenia',
 'Maldives',
 'United Arab Emirates',
 'Nigeria',
 'Fiji',
 'Guatemala',
 'Uganda',
 'Turkmenistan',
 'Mauritius',
 'Estonia',
 'Kenya',
 'Gabon',
 'South Korea',
 'Chile',
 'Uruguay',
 'Viet Nam',
 'China',
 'Ghana',
 'Hong Kong (S.A.R.)',
 'Sri Lanka',
 'Mongolia',
 'Uzbekistan',
 'Republic of Korea',
 'Nepal',
 'Taiwan',
 'Lebanon',
 'Benin',
 'Democratic Republic of the Congo',
 'Syrian Arab Republic',
 'Iraq',
 'Namibia',
 'Kyrgyzstan',
 "Lao People's Democratic Republic",
 'Tunisia',
 'Burundi',
 'Rwanda',
 'Iceland',
 'Mauritania',
 'Sierra Leone',
 'Panama',
 'Cuba',
 'Guyana',
 'Zambia',
 'Ethiopia',
 'Republic of Moldova',
 'Jordan',
 'Jamaica',
 'Nomadic',
 'Andorra',
 'Republic of North Macedonia',
 "Democratic People's Republic of Korea",
 'Kuwait',
 'Togo',
 'Qatar',
 'Tajikistan',
 'Albania',
 'Sudan',
 'Kosovo',
 'Angola',
 "Côte d'Ivoire",
 'Malawi',
 'Burkina Faso',
 'United Republic of Tanzania',
 'Madagascar',
 'Cameroon',
 'Yemen',
 'Myanmar',
 'Oman',
 'Azerbaijan',
 'Central African Republic',
 'Somalia',
 'Suriname',
 'Libyan Arab Jamahiriya',
 'Cape Verde',
 'Bahrain',
 'Bhutan',
 'Trinidad and Tobago',
 'Niger',
 'Mozambique',
 'Antigua and Barbuda',
 'Honduras',
 'Liechtenstein',
 'Senegal',
 'Congo, Republic of the...',
 'Samoa',
 'Brunei Darussalam',
 'Lesotho',
 'Cambodia',
 'Botswana',
 'Barbados',
 'Mali',
 'Haiti',
 'Swaziland',
 'Chad',
 'Monaco']

country = st.selectbox("Which Country you want to work in?", ["select an option"] + country_options
)

st.header("Age")
age_options = [
    "Under 18 years old",
    "18-24 years old",
    "25-34 years old",
    "35-44 years old",
    "45-54 years old",
    "55-64 years old",
    "65 years or older",
    "Prefer not to say"
]

age = st.selectbox("What is your age?", ["Select an option"] + age_options)


st.header("Professional Experience")
experience = st.number_input("Enter your years of professional experience:", min_value=0, max_value=50, step=1)

st.write("Experience entered:", experience)
st.header("Employment Status")
employment_options = [
    "Employed, full-time",
    "Employed, part-time",
    "Independent contractor, freelancer, or self-employed",
    "Not employed, but looking for work",
    "Not employed, and not looking for work",
    "Student, full-time",
    "Student, part-time",
    "Retired",
    "I prefer not to say"
]

employment_status = st.multiselect(
    "Which of the following best describes your current employment status? Select all that apply.",
    employment_options
)

st.header("Work Situation")
work_situation_options = ["Remote", "In-person", "Hybrid (some remote, some in-person)"]
work_situation = st.selectbox(
    "Which best describes your current work situation?",
    ["Select an option"] + work_situation_options
)

st.header("Programming Languages")
programming_languages = [
    "Ada", "Apex", "Assembly", "Bash/Shell (all shells)", "C", "C#", "C++", "Clojure", "Cobol", "Crystal",
    "Dart", "Delphi", "Elixir", "Erlang", "F#", "Fortran", "GDScript", "Go", "Groovy", "Haskell",
    "HTML/CSS", "Java", "JavaScript", "Julia", "Kotlin", "Lisp", "Lua", "MATLAB", "MicroPython", "Nim",
    "Objective-C", "OCaml", "Perl", "PHP", "PowerShell", "Prolog", "Python", "R", "Ruby", "Rust",
    "Scala", "Solidity", "SQL", "Swift", "TypeScript", "VBA", "Visual Basic (.Net)", "Zephyr", "Zig"
]

selected_languages = st.multiselect(
    "Which programming, scripting, and markup languages have you worked with?",
    programming_languages
)

st.header("Database Environments")
databases = [
    "BigQuery", "Cassandra", "Clickhouse", "Cloud Firestore", "Cockroachdb", "Cosmos DB", "Couch DB",
    "Couchbase", "Databricks SQL", "Datomic", "DuckDB", "Dynamodb", "Elasticsearch", "EventStoreDB",
    "Firebase Realtime Database", "Firebird", "H2", "IBM DB2", "InfluxDB", "MariaDB", "Microsoft Access",
    "Microsoft SQL Server", "MongoDB", "MySQL", "Neo4J", "Oracle", "PostgreSQL", "Presto", "RavenDB",
    "Redis", "Snowflake", "Solr", "SQLite", "Supabase", "TiDB"
]

selected_databases = st.multiselect(
    "Which database environments have you worked with?",
    databases
)

st.header("Cloud Platforms")
cloud_platforms = [
    "Alibaba Cloud", "Amazon Web Services (AWS)", "Cloudflare", "Colocation", "Databricks", "Digital Ocean",
    "Firebase", "Fly.io", "Google Cloud", "Heroku", "Hetzner", "IBM Cloud Or Watson", "Linode",
    "Managed Hosting", "Microsoft Azure", "Netlify", "OpenShift", "OpenStack", "Oracle Cloud Infrastructure (OCI)",
    "OVH", "PythonAnywhere", "Render", "Scaleway", "Supabase", "Vercel", "VMware", "Vultr"
]

selected_cloud = st.multiselect(
    "Which cloud platforms have you worked with?",
    cloud_platforms
)

st.header("Web Frameworks and Technologies")
web_frameworks = [
    "Angular", "AngularJS", "ASP.NET", "ASP.NET CORE", "Astro", "Blazor", "CodeIgniter", "Deno", "Django",
    "Drupal", "Elm", "Express", "FastAPI", "Fastify", "Flask", "Gatsby", "Htmx", "jQuery", "Laravel",
    "NestJS", "Next.js", "Node.js", "Nuxt.js", "Phoenix", "Play Framework", "React", "Remix",
    "Ruby on Rails", "Solid.js", "Spring Boot", "Strapi", "Svelte", "Symfony", "Vue.js", "WordPress", "Yii 2"
]

selected_frameworks = st.multiselect(
    "Which web frameworks and web technologies have you worked with?",
    web_frameworks
)

st.header("Embedded Systems and Technologies")
embedded_systems = [
    "Arduino", "Boost.Test", "build2", "Catch2", "CMake", "Cargo", "cppunit", "CUTE", "doctest",
    "GNU GCC", "LLVM's Clang", "Meson", "Micronaut", "MSVC", "Ninja", "PlatformIO", "QMake",
    "Rasberry Pi", "SCons", "ZMK"
]

selected_embedded = st.multiselect(
    "Which embedded systems and technologies have you worked with?",
    embedded_systems
)

st.header("Other Frameworks and Libraries")
other_frameworks = [
    ".NET (5+)", ".NET Framework (1.0 - 4.8)", ".NET MAUI", "Apache Kafka", "Apache Spark", "Capacitor",
    "Cordova", "CUDA", "DirectX", "Electron", "Flutter", "GTK", "Hadoop", "Hugging Face Transformers",
    "Ionic", "JAX", "Keras", "Ktor", "MFC", "mlflow", "Numpy", "OpenCL", "Opencv", "OpenGL", "Pandas",
    "Qt", "Quarkus", "RabbitMQ", "React Native", "Roslyn", "Ruff", "Scikit-Learn", "Spring Framework",
    "SwiftUI", "Tauri", "TensorFlow", "Tidyverse", "Torch/PyTorch", "Xamarin"
]

selected_other_frameworks = st.multiselect(
    "Which other frameworks and libraries have you worked with?",
    other_frameworks
)

st.header("Developer Tools")
developer_tools = [
    "Ansible", "Ant", "APT", "Bun", "Chef", "Chocolatey", "Composer", "Dagger", "Docker", "Godot",
    "Google Test", "Gradle", "Homebrew", "Kubernetes", "Make", "Maven (build tool)", "MSBuild", "Ninja",
    "Nix", "npm", "NuGet", "Pacman", "Pip", "pnpm", "Podman", "Pulumi", "Puppet", "Terraform",
    "Unity 3D", "Unreal Engine", "Visual Studio Solution", "Vite", "Webpack", "Yarn"
]

selected_tools = st.multiselect(
    "Which developer tools for compiling, building and testing have you worked with?",
    developer_tools
)

st.markdown("---")
if st.button("Predict", type="primary"):
    # Collect all form data
    form_data = {
        "DevType": current_job,
        "EdLevel": education,
        "Age": age,
        "Country": country,
        "WorkExp" : experience,
        "Employment": employment_status,
        "RemoteWork": work_situation,
        "LanguageHaveWorkedWith": selected_languages,
        "DatabaseHaveWorkedWith": selected_databases,
        "PlatformHaveWorkedWith": selected_cloud,
        "WebframeHaveWorkedWith": selected_frameworks,
        "EmbeddedHaveWorkedWith": selected_embedded,
        "MiscFrameworks": selected_other_frameworks,
        "ToolsHaveWorkedWith": selected_tools
    }
    
    st.success("Survey submitted successfully! 🎉")
    
    with st.expander("View Submitted Data"):
        st.json(form_data)

    input_data = Preprocess(pd.DataFrame([form_data]))
    print(input_data.dtypes)
    print(input_data.shape)

    with open(os.path.join('model_files', 'salary_model.pkl'), 'rb') as f:
        model = pickle.load(f)

    prediction = model.predict(input_data.iloc[0].values.reshape(1, -1))
    print("Predicted Salary (log):", prediction[0])

    # Convert from log-salary to actual salary
    predicted_salary = np.expm1(prediction[0])
    margin = 0.10  # ±10% margin

    lower_bound = round(predicted_salary * (1 - margin), 2)
    upper_bound = round(predicted_salary * (1 + margin), 2)

    st.markdown("---")
    st.markdown("## -- Salary Predicted --")

    # For visualization slider position
    range_span = upper_bound - lower_bound
    prediction_position = (predicted_salary - lower_bound) / range_span if range_span > 0 else 0.5

    # Custom HTML for dark mode visualization
    st.markdown(f"""
    <div style="background-color: #1e1e1e; padding: 20px; border-radius: 10px; margin: 20px 0; color: white;">
        <h4 style="text-align: center; margin-bottom: 20px;">💰 Your Salary Prediction</h4>
        <div style="background-color: #333; height: 30px; border-radius: 15px; position: relative; margin: 20px 0;">
            <div style="background: linear-gradient(90deg, #00c853, #ff9800); height: 100%; width: 100%; border-radius: 15px; opacity: 0.4;"></div>
            <div style="position: absolute; top: -5px; left: {prediction_position * 100}%; transform: translateX(-50%); background-color: #2196f3; color: white; padding: 5px 10px; border-radius: 20px; font-weight: bold;">
                ${predicted_salary:,.0f}
            </div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 10px;">
            <span><strong>Min:</strong> ${lower_bound:,.0f}</span>
            <span><strong>Max:</strong> ${upper_bound:,.0f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)





st.markdown("""
<style>
/* Style the background of selectbox and multiselect dropdowns */
div[data-baseweb="select"] > div {
    background-color: #f0f2f6 !important;
    border-radius: 8px !important;
    color: black !important;  /* Ensures selected text is visible */
}

/* Style the selected option text inside selectbox/multiselect */
div[data-baseweb="select"] span {
    color: black !important;
}

/* Optional: fix dropdown items */
ul[role="listbox"] {
    background-color: #f0f2f6 !important;
}
</style>
""", unsafe_allow_html=True)




