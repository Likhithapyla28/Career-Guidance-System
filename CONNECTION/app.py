from flask import Flask, request, render_template, redirect, url_for, make_response,jsonify, session, flash
from flask_mysqldb import MySQL
import urllib.parse
from flask_mail import Mail, Message
import random
import smtplib
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import os
import google.generativeai as genai
from flask import request, jsonify
app = Flask(__name__)
GOOGLE_API_KEY = "API key"  # Keep it secure in prod
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')
chat = model.start_chat()
GREETINGS = ["hi", "hello", "hey", "how are you", "good morning", "good evening", "good afternoon"]
def is_valid_message(question):
    question_lower = question.lower().strip()
    if any(question_lower.startswith(greet) for greet in GREETINGS):
        return "greeting"
    try:
        classification_prompt = f"""
        Classify the following question into one of these categories: 
        1. "educational"
        2. "non-educational"
        3. "greeting"  
        Respond with only one word: educational, non-educational, or greeting.
        Question: {question}
        """
        response_raw = model.generate_content(classification_prompt)
        return response_raw.text.strip().lower()
    except Exception as e:
        print(f"Classification Error: {e}")
        return "non-educational"
@app.route('/chat', methods=['POST'])
def chat_response():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    category = is_valid_message(user_input)
    if category == "greeting":
        return jsonify({"response": "Hello! How can I assist you with your learning today?"})
    if category != "educational":
        return jsonify({"response": "I can only respond to educational topics like math, science, programming, history, etc."})
    try:
        response_raw = chat.send_message(user_input)
        response = response_raw.text if response_raw else "No response from model"
        return jsonify({"response": response})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500
# 🔐 Secret Key for Session Management (Change in Production)
app.secret_key = 'd843fdba37211ebe1f02e160432ca0e8'
# ✅ MySQL Database Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'password'
app.config['MYSQL_DB'] = 'database_name'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
mysql = MySQL(app)
bcrypt = Bcrypt(app)
# ✅ Flask-Mail Configuration
app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USERNAME"] = "user@gmail.com"
app.config["MAIL_PASSWORD"] = "12 letter character"
app.config['MAIL_DEFAULT_SENDER'] = 'user@gmail.com'
mail = Mail(app)
# ✅ Temporary Storage for OTPs (Use Redis/DB in production)
otp_storage = {}
password = urllib.parse.quote("mail password")
app.config['SQLALCHEMY_DATABASE_URI'] = f"mysql+pymysql://root:{password}@127.0.0.1/user_database"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
db = SQLAlchemy(app)
#CORS(app)# Database Models
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
class StudentDetails(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_name = db.Column(db.String(255))
    age = db.Column(db.Integer)
    gender = db.Column(db.String(50))
    email = db.Column(db.String(255))
    phone_number = db.Column(db.String(20))
    native_city = db.Column(db.String(100))
    study_city = db.Column(db.String(100))
    college_name = db.Column(db.String(255))
    academic_level = db.Column(db.String(100))
    branch = db.Column(db.String(100))
    ssc_score = db.Column(db.Float)
    inter_score = db.Column(db.Float)
    current_cgpa = db.Column(db.Float)
    current_semester = db.Column(db.Integer)
    technical_skills = db.Column(db.Text)
    soft_skills = db.Column(db.Text)
    certifications = db.Column(db.Text)
    languages_known = db.Column(db.Text)
    fields_of_interest = db.Column(db.Text)
    work_environment = db.Column(db.Text)
    future_goals = db.Column(db.Text)
    competitions = db.Column(db.Text)
    clubs_societies = db.Column(db.Text)
    internships = db.Column(db.Text)
    part_time_jobs = db.Column(db.Text)
    awards_honors = db.Column(db.Text)
    preferred_industries = db.Column(db.Text)
    preferred_companies = db.Column(db.Text)
    aptitude_ability = db.Column(db.String(100))
    coding_knowledge = db.Column(db.String(100))
    personality_traits = db.Column(db.Text)
    hobbies = db.Column(db.Text)
    passion = db.Column(db.Text)
    career_feedback = db.Column(db.Text)
    career_goals = db.Column(db.Text)
    industry_training = db.Column(db.Text)
    projects = db.Column(db.Text)
    resume_filename = db.Column(db.String(255))
# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
# Routes
@app.route('/')
def index():
    return render_template('home.html')
@app.route('/signup_page')
def signup_page():
    return render_template('signup.html')
@app.route('/login_page')
def login_page():
    return render_template('login.html')
@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')
@app.route('/form')
def form_details():
    if 'user_id' not in session:
        flash("Please login first", "error")
        return redirect(url_for('login_page'))
    return render_template('Form_Details.html')
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        try:
            email = request.form.get('email')
            password = request.form.get('password')
            username = request.form.get('username')
            confirm_password = request.form.get('confirm_password')
            if not all([email, password, username, confirm_password]):
                flash("All fields are required", "danger")
                return render_template('signup.html')
            if User.query.filter_by(email=email).first():
                flash("This email is already registered. Please use a different email or login.", "danger")
                return render_template('signup.html')
            if User.query.filter_by(username=username).first():
                flash("This username is already taken. Please choose a different username.")
                return render_template('signup.html')
            if password != confirm_password:
                flash("Passwords do not match. Please try again.", "danger")
                return render_template('signup.html')
            if len(password) < 6:
                flash("Password must be at least 6 characters long.", "danger")
                return render_template('signup.html')
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
            new_user = User(username=username, email=email, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            flash("Account created successfully! Please login.", "success")
            return redirect(url_for('login_page'))
        except Exception as e:
            db.session.rollback()
            flash(f"Error creating account: {str(e)}", "danger")
            return render_template('signup.html') 
    # GET method
    return redirect(url_for('signup_page'))
@app.route('/login', methods=['POST'])
def login():
    try:
        email = request.form.get('email')
        password = request.form.get('password')
        print(f"Login attempt for email: {email}")  # Debug
        if not email or not password:
            flash("Both email and password are required", "error")
            return redirect(url_for('login_page'))
        user = User.query.filter_by(email=email).first()    
        if not user:
            print("No user found with that email")  # Debug
            flash("No account found with this email. Please check your email or sign up for a new account.", "error")
            return redirect(url_for('login_page'))
        print(f"User found: {user.username}")  # Debug
        print(f"Stored hash: {user.password}")  # Debug 
        if bcrypt.check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['email'] = user.email    
            print("Login successful! Session created")  # Debug
            flash(f"Welcome back, {user.username}!", "success")
            return redirect(url_for('Afterlogin'))
        else:
            print("Password mismatch")  # Debug
            flash("Incorrect password. Please try again.", "error")
            return redirect(url_for('login_page'))       
    except Exception as e:
        print(f"Login error: {str(e)}")  # Debug
        flash("An error occurred during login. Please try again.", "error")
        return redirect(url_for('login_page'))
from internships import recommend_resources
@app.route("/internship", methods=["GET", "POST"])
def internship():
    if request.method == "POST":
        job_role = request.form["job_role"]
        internships = recommend_resources(job_role)
        return render_template("internship_results.html", job_role=job_role, internships=internships["LinkedIn Internships"])
    return render_template("internship_form.html")
@app.route('/feedback')
def feedback_form():
    return render_template('feedback_form.html')
@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    # Here you can handle saving feedback data if needed
    name = request.form['name']
    email = request.form['email']
    experience = request.form['experience']
    interface_rating = request.form['interface_rating']
    recommend = request.form['recommend']
    improvements = request.form.getlist('improve')
    message = request.form['message']
    # You can store this in DB or process it if needed
    print(f"Feedback from {name}: {message}")
    return render_template('thankyou.html')
from flask import Flask, render_template, request, flash, redirect, url_for
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import csv
import os
from datetime import datetime
import json
# Load NLP model
nlp = spacy.load("en_core_web_sm")
# Load SBERT model
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
# Load saved XGBoost model
clf = joblib.load('xgb_model.joblib')
# Load saved PCA model (if you have saved it)
try:
    pca = joblib.load('pca_model.joblib')
except FileNotFoundError:
    # If PCA model isn't saved, load original dataset to fit PCA
    df = pd.read_csv('C:/Users/likhitha/OneDrive/Desktop/4-2 Project/MODEL TRAINING_MY/career_g.csv')  # Load your dataset
    df.columns = df.columns.str.strip()
    df.fillna("", inplace=True)
    # Prepare combined text and numerical features
    text_columns = ["Technical Skills", "Fields of Interest", "Soft Skills", "Preferred Industries"]
    df["combined_text"] = df[text_columns].apply(lambda x: " ".join(x), axis=1)
    X_text = sbert_model.encode(df["combined_text"], convert_to_numpy=True)
    num_columns = ["Age", "Current CGPA"]
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[num_columns])
    X = np.hstack((X_text, X_num))
    # Fit PCA on original dataset
    pca = PCA(n_components=50, random_state=42)
    pca.fit(X)
    # Save PCA model for future use
    joblib.dump(pca, 'pca_model.joblib')
# Function to extract skills using NLP
def extract_skills(text):
    doc = nlp(text.lower())
    skills = [] 
    # Extract noun phrases and named entities
    for chunk in doc.noun_chunks:
        skills.append(chunk.text)
    for ent in doc.ents:
        if ent.label_ in ['ORG', 'PRODUCT', 'TECH']:
            skills.append(ent.text) 
    return list(set(skills))
# Function to find best job match and missing skills
def find_best_match_and_gaps(resume_text):
    job_descriptions = {
        "Machine Learning Engineer": "Machine learning, TensorFlow, PyTorch, deep learning, AI models, data science, NLP, reinforcement learning, big data, computer vision, GANs",
        "Data Scientist": "Data analysis, Python, SQL, statistics, machine learning, big data, AI, data visualization, deep learning, R programming, cloud computing, Apache Spark, Tableau",
        "Web Developer": "HTML, CSS, JavaScript, React, Angular, frontend development, backend development, Node.js, TypeScript, UX/UI design, databases, Next.js, GraphQL",
    }   
    similarities = {}
    missing_skills = {}
    for role, desc in job_descriptions.items():
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([resume_text, desc])
        similarity = cosine_similarity(vectors)[0, 1]
        similarities[role] = similarity
        resume_skills = set(resume_text.split())
        job_skills = set(desc.split(", "))
        missing_skills[role] = list(job_skills - resume_skills)
    best_match = max(similarities, key=similarities.get)
    return best_match, missing_skills[best_match]
# Function to fetch Udemy courses
def fetch_courses_udemy(skill):
    search_url = f"https://www.udemy.com/courses/search/?q={skill}"
    return [("Udemy Courses", search_url)]
# Function to fetch courses from EdX
def fetch_courses_edx(skill):
    search_url = f"https://www.edx.org/search?q={skill}"
    return [("EdX Courses", search_url)]
# Function to fetch courses from Swayam
def fetch_courses_swayam(skill):
    search_url = f"https://swayam.gov.in/explorer?searchText={skill}"
    return [("Swayam Courses", search_url)]
# Function to fetch internships from APSCHE
def fetch_internships_apsc(role):
    search_url = f"https://apprenticeshipindia.gov.in/search-apprenticeship?keyword={role}"
    return [("APSCHE Internships", search_url)]
# Function to fetch relevant blogs
def fetch_blogs(role):
    search_url = f"https://medium.com/search?q={role} career"
    return [("Relevant Blogs", search_url)]
# Function to fetch YouTube learning resources
def fetch_youtube(role):
    search_url = f"https://www.youtube.com/results?search_query={role}+tutorial"
    return [("YouTube Tutorials", search_url)]
# Function to send email
def send_email(to_email, subject, body):
    try:
        # Email configuration
        from_email = "finalyearprojectb9@gmail.com"
        password = "ubph crcn hrwc kios"     
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = from_email
        msg['To'] = to_email      
        # Attach HTML content
        html_part = MIMEText(body, 'html')
        msg.attach(html_part)       
        # Create SMTP session
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(from_email, password)
            server.send_message(msg)            
        print(f"Email sent successfully to {to_email}")
        return True
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False
# Function to save form data to CSV
def save_to_csv(form_data):
    """Save form data to CSV file with timestamp"""
    csv_file = 'career_survey_data.csv'
    file_exists = os.path.isfile(csv_file)    
    # Add timestamp to the data
    form_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')    
    # Convert lists to comma-separated strings
    for key, value in form_data.items():
        if isinstance(value, list):
            form_data[key] = ','.join(value)  
    # Write to CSV
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=form_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(form_data)
@app.route('/internship_form')
def internship_form():
    return render_template('internship_form.html')
@app.route('/job-role-form')
def job_role_form():
    return render_template('jobrole_form.html')
from flask import Flask, request, jsonify
import spacy
import numpy as np
from sentence_transformers import SentenceTransformer, util
import fitz  # PyMuPDF for PDF extraction
from docx import Document  # For DOCX extraction
import os
from werkzeug.utils import secure_filename
# Load advanced NLP models
nlp = spacy.load("en_core_web_sm")
bert_model = SentenceTransformer('all-mpnet-base-v2')  # More advanced than MiniLM
# Job descriptions with required skills
job_descriptions = {
    "Software Engineer": "Develop software using Python, Java, JavaScript, and frameworks like Django or Spring.",
    "Data Scientist": "Analyze big data using Python, machine learning, deep learning, and data visualization tools.",
    "Web Developer": "Build websites with HTML, CSS, JavaScript, React, and backend technologies.",
    "Cybersecurity Analyst": "Ensure security by monitoring systems, performing vulnerability assessments, and implementing firewalls.",
    "Mobile App Developer": "Design and develop mobile applications for Android and iOS using languages like Swift, Kotlin, and Flutter.",
    "DevOps Engineer": "Automate and improve development and release processes, manage CI/CD pipelines, and monitor system performance.",
    "Database Administrator": "Manage and maintain databases, ensure data integrity, and optimize performance using SQL and NoSQL databases.",
    "Machine Learning Engineer": "Design and implement machine learning models, work with big data technologies, and optimize algorithms.",
    "UI/UX Designer": "Create user-friendly interfaces and experiences through wireframes, prototypes, and user testing.",
    "Cloud Engineer": "Manage cloud infrastructure, deploy applications on platforms like AWS, Azure, and Google Cloud, and ensure scalability.",
    "Systems Analyst": "Analyze and design information systems to meet business needs, and improve system efficiency.",
    "Network Engineer": "Design, implement, and manage network infrastructure, including routers, switches, and firewalls.",
    "Game Developer": "Create video games for consoles, PC, and mobile platforms using engines like Unity or Unreal Engine.",
    "Technical Support Engineer": "Provide technical support and troubleshooting for hardware and software issues to clients.",
    "Product Manager": "Oversee product development from conception to launch, collaborating with engineering, design, and marketing teams.",
    "Business Analyst": "Identify business needs, analyze data, and provide insights to help drive business strategy and decision-making."
}
# Skill extraction using Named Entity Recognition (NER)
def extract_skills(text):
    doc = nlp(text)
    skills = [ent.text.lower() for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART"]]
    return list(set(skills))
# Function to match resume with job roles
def match_resume_with_jobs(resume_text):
    extracted_skills = extract_skills(resume_text)
    resume_processed = " ".join(extracted_skills)
    # Compute BERT embeddings
    resume_embedding = bert_model.encode(resume_processed, convert_to_tensor=True)
    job_embeddings = {role: bert_model.encode(desc, convert_to_tensor=True) for role, desc in job_descriptions.items()}
    # Compute similarity scores
    similarity_scores = {role: util.pytorch_cos_sim(resume_embedding, job_emb)[0][0].item() for role, job_emb in job_embeddings.items()}
    # Weighted score (if needed, tweak weight factors)
    skill_weight = 0.6
    desc_weight = 0.4
    final_scores = {role: (skill_weight * similarity_scores[role] + desc_weight * util.pytorch_cos_sim(
        bert_model.encode(resume_text, convert_to_tensor=True), job_emb)[0][0].item())
        for role, job_emb in job_embeddings.items()
    }
    # Find the best job match
    best_match = max(final_scores, key=final_scores.get)
    return best_match, final_scores
# Function to read resume from a PDF file
def read_resume_from_pdf(file_path):
    with fitz.open(file_path) as doc:
        return " ".join(page.get_text() for page in doc)
# Function to read resume from a DOCX file
def read_resume_from_docx(file_path):
    doc = Document(file_path)
    return " ".join(paragraph.text for paragraph in doc.paragraphs)
# Configure upload folder (optional)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    if 'resume' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['resume']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        resume_text = ""
        try:
            if filename.lower().endswith('.pdf'):
                resume_text = read_resume_from_pdf(filepath)
            elif filename.lower().endswith('.docx'):
                resume_text = read_resume_from_docx(filepath)
        except Exception as e:
            os.remove(filepath)  # Clean up the uploaded file
            return jsonify({'error': f'Error reading file: {str(e)}'}), 500
        os.remove(filepath)  # Clean up the uploaded file
        if resume_text:
            best_job, scores = match_resume_with_jobs(resume_text)
            return jsonify({'best_job': best_job, 'scores': {k: f'{v:.4f}' for k, v in scores.items()}})
        else:
            return jsonify({'error': 'Could not extract text from the resume'}), 500
    return jsonify({'error': 'Invalid file format'}), 400
@app.route('/submit', methods=['POST'])
def submit():
    try:
        if 'user_id' not in session:
            flash("Please login first", "error")
            return redirect(url_for('login_page'))
        # Get all form data
        form_data = request.form.to_dict()
        # Handle checkbox groups (multiple selections)
        checkbox_fields = [
            'technicalSkills', 'softSkills', 'languagesKnown',
            'fieldsOfInterest', 'preferredWorkEnvironment', 'futureGoals',
            'preferredIndustries', 'preferredCompanies', 'personalityTraits'
        ]
        for field in checkbox_fields:
            values = request.form.getlist(field + '[]') or request.form.getlist(field)
            other_value = request.form.get(f"{field}Other")
            if other_value:
                values.append(f"Other: {other_value}")
            form_data[field] = ", ".join(values)
        # Handle file upload
        resume = request.files.get('resume')
        resume_filename = None
        if resume and resume.filename != '':
            filename = secure_filename(resume.filename)
            resume_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            resume.save(resume_path)
            resume_filename = filename
            form_data['resume_path'] = resume_path
        # Save to database
        student = StudentDetails(
            student_name=form_data.get('studentName', ''),
            age=int(form_data.get('age', 0)),
            gender=form_data.get('gender', ''),
            email=form_data.get('email', ''),
            phone_number=form_data.get('phoneNumber', ''),
            native_city=form_data.get('cityNative', ''),
            study_city=form_data.get('cityStudy', ''),
            college_name=form_data.get('collegeName', ''),
            academic_level=form_data.get('currentAcademicLevel', ''),
            branch=form_data.get('branch', ''),
            ssc_score=float(form_data.get('sscPercentage', 0)),
            inter_score=float(form_data.get('intermediatePercentage', 0)),
            current_cgpa=float(form_data.get('currentCGPA', 0)),
            current_semester=int(form_data.get('currentSemester', 0)),
            technical_skills=form_data.get('technicalSkills', ''),
            soft_skills=form_data.get('softSkills', ''),
            certifications=form_data.get('certifications', ''),
            languages_known=form_data.get('languagesKnown', ''),
            fields_of_interest=form_data.get('fieldsOfInterest', ''),
            work_environment=form_data.get('preferredWorkEnvironment', ''),
            future_goals=form_data.get('futureGoals', ''),
            competitions=form_data.get('participationInCompetitions', ''),
            clubs_societies=form_data.get('clubsAndSocieties', ''),
            internships=form_data.get('internshipsCompleted', ''),
            part_time_jobs=form_data.get('partTimeJobs', ''),
            awards_honors=form_data.get('awardsAndHonors', ''),
            preferred_industries=form_data.get('preferredIndustries', ''),
            preferred_companies=form_data.get('preferredCompanies', ''),
            aptitude_ability=form_data.get('aptitudeAbility', ''),
            coding_knowledge=form_data.get('codingKnowledge', ''),
            personality_traits=form_data.get('personalityTraits', ''),
            hobbies=form_data.get('hobbies', ''),
            passion=form_data.get('passion', ''),
            career_feedback=form_data.get('feedback', ''),
            career_goals=form_data.get('shortLongTermGoals', ''),
            industry_training=form_data.get('industryOrientedTraining', ''),
            projects=form_data.get('projects', ''),
            resume_filename=resume_filename
        )
        db.session.add(student)
        db.session.commit()
        save_to_csv(form_data)
        # Prediction
        user_input = {
            "Technical Skills": form_data['technicalSkills'],
            "Fields of Interest": form_data['fieldsOfInterest'],
            "Soft Skills": form_data['softSkills'],
            "Preferred Industries": form_data['preferredIndustries'],
            "Age": int(form_data.get("age", 0)),
            "Current CGPA": float(form_data.get("currentCGPA", 0)),
        }
        combined_text = " ".join([
            user_input["Technical Skills"], user_input["Fields of Interest"],
            user_input["Soft Skills"], user_input["Preferred Industries"]
        ])
        X_text = sbert_model.encode([combined_text], convert_to_numpy=True)
        num_features = [[user_input["Age"], user_input["Current CGPA"]]]
        scaler = StandardScaler()
        X_num = scaler.fit_transform(num_features)
        X = np.hstack((X_text, X_num))
        X_pca = pca.transform(X)
        predicted_role = clf.predict(X_pca)[0]
        unique_roles = [
            "Software Engineer", "Data Scientist", "Cybersecurity Analyst",
            "UI/UX Designer", "Cloud Engineer", "AI/ML Engineer",
            "Business Analyst", "Embedded Systems Engineer", "Game Developer", "Network Engineer"
        ]
        suggested_job_role = unique_roles[predicted_role]
        # Skills processing
        resume_text = " ".join([user_input["Technical Skills"], user_input["Soft Skills"]])
        extracted_skills = extract_skills(resume_text)
        best_match, missing_skills = find_best_match_and_gaps(resume_text)
        # Recommendations
        recommended_udemy_courses = []
        recommended_edx_courses = []
        recommended_swayam_courses = []
        for skill in missing_skills:
            recommended_udemy_courses.extend(fetch_courses_udemy(skill))
            recommended_edx_courses.extend(fetch_courses_edx(skill))
            recommended_swayam_courses.extend(fetch_courses_swayam(skill))
        recommended_internships = fetch_internships_apsc(suggested_job_role)
        recommended_blogs = fetch_blogs(suggested_job_role)
        recommended_youtube_tutorials = fetch_youtube(suggested_job_role)
        # Store in session
        session['user_name'] = form_data.get('studentName', '')
        session['user_email'] = form_data.get('email', '')
        session['suggested_job_role'] = suggested_job_role
        session['missing_skills'] = missing_skills
        session['recommended_udemy_courses'] = recommended_udemy_courses
        session['recommended_edx_courses'] = recommended_edx_courses
        session['recommended_swayam_courses'] = recommended_swayam_courses
        session['recommended_internships'] = recommended_internships
        session['recommended_blogs'] = recommended_blogs
        session['recommended_youtube_tutorials'] = recommended_youtube_tutorials
        print("Form data collected:", form_data)
        # Email
        email_body = f"""
    <html>
  <body style="font-family: Arial, sans-serif; color: #333; line-height: 1.6; background-color: #f9f9f9; padding: 20px;">
    <div style="max-width: 700px; margin: auto; background: #fff; border-radius: 10px; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1);">
      <h2 style="text-align: center; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px;">🎯 Career Recommendations for <span style="color: #e67e22;">{form_data.get('studentName')}</span></h2>    
      <p><b>💼 Suggested Role:</b> <span style="color: #27ae60;">{suggested_job_role}</span></p>     
      <p><b>🛠️ Skills to Improve:</b></p>
      <ul>
        {"".join(f"<li>✅ {skill}</li>" for skill in missing_skills)}
      </ul>    
      <p><b>📘 Courses to Learn:</b></p>
      <h3 style="color: #2980b9;">📚 Udemy</h3>
      <ul>
        {"".join(f"<li><b>{c[0]}:</b> <a href='{c[1]}' target='_blank'>{c[1]}</a></li>" for c in recommended_udemy_courses)}
      </ul>
      <h3 style="color: #8e44ad;">📚 EdX</h3>
      <ul>
        {"".join(f"<li><b>{c[0]}:</b> <a href='{c[1]}' target='_blank'>{c[1]}</a></li>" for c in recommended_edx_courses)}
      </ul>
      <h3 style="color: #d35400;">📚 Swayam</h3>
      <ul>
        {"".join(f"<li><b>{c[0]}:</b> <a href='{c[1]}' target='_blank'>{c[1]}</a></li>" for c in recommended_swayam_courses)}
      </ul>
      <p><b>🎓 Internship Opportunity:</b><br>
        <b>{recommended_internships[0][0]}:</b> <a href="{recommended_internships[0][1]}" target="_blank">{recommended_internships[0][1]}</a>
      </p>
      <p><b>📝 Recommended Blog:</b><br>
        <b>{recommended_blogs[0][0]}:</b> <a href="{recommended_blogs[0][1]}" target="_blank">{recommended_blogs[0][1]}</a>
      </p>
      <p><b>🎥 YouTube Tutorial:</b><br>
        <b>{recommended_youtube_tutorials[0][0]}:</b> <a href="{recommended_youtube_tutorials[0][1]}" target="_blank">{recommended_youtube_tutorials[0][1]}</a>
      </p>     
      <p style="text-align:center; margin-top: 30px;">
        🚀 Keep Learning & Growing! Best wishes for your career ahead!
      </p>
    </div>
  </body>
</html>
        """
        send_email(form_data.get('email', ''), f"🌟 Career Recommendations for {form_data.get('studentName')}", email_body)
        flash('Student details submitted and email sent successfully!', 'success')
        print("Encoded text shape:", X_text.shape)
        print("Numerical features:", num_features)
        print("Combined X shape:", X.shape)
        print("PCA transformed shape:", X_pca.shape)
        print("Prediction:", predicted_role)
        print("Missing skills:", missing_skills)
        return redirect(url_for('results'))
    except Exception as e:
        db.session.rollback()
        print(f"Error in submission: {str(e)}")
        print("Encoded text shape:", X_text.shape)
        print("Numerical features:", num_features)
        print("Combined X shape:", X.shape)
        print("PCA transformed shape:", X_pca.shape)
        print("Prediction:", predicted_role)
        print("Missing skills:", missing_skills)
        #flash(f"An error occurred: {str(e)}", 'danger')
        return redirect(url_for('error'))
@app.route('/results', methods=['GET'])
def results():
    # Get user data from session or database
    user_name = session.get('user_name', '')
    user_email = session.get('user_email', '')
    suggested_job_role = session.get('suggested_job_role', '')
    missing_skills = session.get('missing_skills', [])
    recommended_udemy_courses = session.get('recommended_udemy_courses', [])
    recommended_edx_courses = session.get('recommended_edx_courses', [])
    recommended_swayam_courses = session.get('recommended_swayam_courses', [])
    recommended_internships = session.get('recommended_internships', [])
    recommended_blogs = session.get('recommended_blogs', [])
    recommended_youtube_tutorials = session.get('recommended_youtube_tutorials', [])  
    # Render the results template with the data
    return render_template('result.html',
                          user_name=user_name,
                          user_email=user_email,
                          suggested_job_role=suggested_job_role,
                          missing_skills=missing_skills,
                          recommended_udemy_courses=recommended_udemy_courses,
                          recommended_edx_courses=recommended_edx_courses,
                          recommended_swayam_courses=recommended_swayam_courses,
                          recommended_internships=recommended_internships,
                          recommended_blogs=recommended_blogs,
                          recommended_youtube_tutorials=recommended_youtube_tutorials)
# Success and error pages
@app.route('/success')
def success_page():
    if 'user_id' not in session:
        flash("Please login first", "error")
        return redirect(url_for('login_page'))
    return render_template('success.html')
@app.route('/error')
def error_page():
    print(f"Error in submission: {str(e)}")
    return render_template('error.html')
@app.route('/Afterlogin')
def Afterlogin():
    if 'user_id' in session:  # Ensure user is logged in
        return render_template('Afterlogin.html')  # Make sure this file exists
    return redirect(url_for('login_page')) 
@app.route('/logout')
def logout():
    session.clear() 
    flash("You have been logged out.", "success")  # Show flash message
    return redirect(url_for('index'))
# Password Reset Routes
@app.route('/forgot_password')
def forgot_password():
    return render_template('forgot_password.html')
@app.route('/dashboard_onreload')
def dashboard_onreload():
    return render_template('Dashboard_onreload.html')
import traceback
@app.route('/send-otp', methods=['POST'])
def send_otp():
    data = request.json
    email = data.get("email")
    if not email:
        return jsonify({"success": False, "message": "Email is required"}), 400
    try:
        otp = str(random.randint(100000, 999999))
        otp_storage[email] = otp
        msg = Message("Password Reset OTP", sender=app.config['MAIL_USERNAME'], recipients=[email])
        msg.body = f"Your OTP is: {otp}"
        mail.send(msg)  # This may throw an error if SMTP fails
        return jsonify({
    "success": True,
    "message": "Your One-Time Password has been sent successfully. Please check your inbox."
})
    except Exception as e:
        error_msg = str(e)
        print("Error:", error_msg)
        print(traceback.format_exc())  # Prints full error traceback
        return jsonify({"success": False, "message": "Failed to send OTP. Error: " + error_msg}), 500
@app.route('/verify-otp')
def verify_otp_page():
    return render_template('verify-otp.html')  
@app.route('/verify-otp', methods=['GET', 'POST'])
def verify_otp():
    if request.method == 'GET':
        return render_template('verify_otp.html')        
    try:
        data = request.json
        email = data.get("email")
        user_otp = data.get("otp")
        if not email or not user_otp:
            return jsonify({"success": False, "message": "Email and OTP are required"}), 400
        stored_otp = otp_storage.get(email)
        if not stored_otp:
            return jsonify({"success": False, "message": "OTP expired or not found. Please request a new one."}), 400
        if stored_otp == user_otp:
            # OTP verified, allow password reset
            session['verified_email'] = email
            # Remove the used OTP
            otp_storage.pop(email, None)
            return jsonify({"success": True, "message": "OTP verified successfully"}), 200
        else:
            return jsonify({"success": False, "message": "Invalid OTP. Please try again."}), 400
    except Exception as e:
        print(f"Error verifying OTP: {str(e)}")
        return jsonify({"success": False, "message": "An error occurred. Please try again later."}), 500
@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'GET':
        if 'verified_email' not in session:
            flash("Please verify your email first", "error")
            return redirect(url_for('forgot_password'))
        return render_template('reset-password.html')       
    try:
        email = session.get("verified_email")
        if not email:
            return jsonify({"success": False, "message": "Email verification required"}), 400
        data = request.json
        new_password = data.get("new_password")
        confirm_password = data.get("confirm_password")
        if not new_password or not confirm_password:
            return jsonify({"success": False, "message": "Both password fields are required"}), 400
        if new_password != confirm_password:
            return jsonify({"success": False, "message": "Passwords do not match"}), 400
        if len(new_password) < 6:
            return jsonify({"success": False, "message": "Password must be at least 6 characters"}), 400
        # Find user and update password
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"success": False, "message": "User not found"}), 404
        # Hash and update password
        hashed_password = bcrypt.generate_password_hash(new_password).decode('utf-8')
        user.password = hashed_password
        db.session.commit()
        # Clear session
        session.pop("verified_email", None)
        print(f"Password reset successful for {email}")
        return jsonify({"success": True, "message": "Password reset successful! Please login with your new password."}), 200
    except Exception as e:
        db.session.rollback()
        print(f"Error resetting password: {str(e)}")
        return jsonify({"success": False, "message": "An error occurred. Please try again later."}), 500
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
