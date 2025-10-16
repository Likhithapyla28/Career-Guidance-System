from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import UserMixin


db = SQLAlchemy()
bcrypt = Bcrypt()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    email = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(150), nullable=False)
    reset_token = db.Column(db.String(100))  # Ensure this field exists
    


class StudentDetails(db.Model):
    __tablename__ = 'student_details'
    id = db.Column(db.Integer, primary_key=True)
    student_name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    phone_number = db.Column(db.String(15), unique=True, nullable=False)
    native_city = db.Column(db.String(100), nullable=False)
    study_city = db.Column(db.String(100), nullable=False)
    college_name = db.Column(db.String(255), nullable=False)
    academic_level = db.Column(db.String(50), nullable=False)
    branch = db.Column(db.String(100), nullable=False)
    ssc_score = db.Column(db.Float, nullable=False)
    inter_score = db.Column(db.Float, nullable=False)
    current_cgpa = db.Column(db.Float, nullable=False)
    current_semester = db.Column(db.Integer, nullable=False)
    technical_skills = db.Column(db.Text, nullable=True)
    soft_skills = db.Column(db.Text, nullable=True)
    certifications = db.Column(db.Text, nullable=True)
    languages_known = db.Column(db.Text, nullable=True)
    fields_of_interest = db.Column(db.Text, nullable=True)
    work_environment = db.Column(db.Text, nullable=True)
    future_goals = db.Column(db.Text, nullable=True)
    competitions = db.Column(db.Text, nullable=True)
    clubs_societies = db.Column(db.Text, nullable=True)
    internships = db.Column(db.Text, nullable=True)
    part_time_jobs = db.Column(db.Text, nullable=True)
    awards_honors = db.Column(db.Text, nullable=True)
    preferred_industries = db.Column(db.Text, nullable=True)
    preferred_companies = db.Column(db.Text, nullable=True)
    aptitude_ability = db.Column(db.String(100), nullable=False)
    coding_knowledge = db.Column(db.String(100), nullable=False)
    personality_traits = db.Column(db.Text, nullable=True)
    hobbies = db.Column(db.Text, nullable=True)
    passion = db.Column(db.Text, nullable=True)
    career_feedback = db.Column(db.Text, nullable=True)
    career_goals = db.Column(db.Text, nullable=True)
    industry_training = db.Column(db.Text, nullable=True)
    projects = db.Column(db.Text, nullable=True)
    resume_filename = db.Column(db.String(255), nullable=True)


    def __repr__(self):
        return f"<StudentDetails {self.student_name}>"

