from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required
from flask_mail import Message
from itsdangerous import URLSafeTimedSerializer
from models import db, User, bcrypt
from app import app, mail

auth_bp = Blueprint('auth', __name__)

# Token Serializer
s = URLSafeTimedSerializer(app.config['SECRET_KEY'])

@auth_bp.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        user = User.query.filter_by(email=email).first()
        
        if user:
            token = s.dumps(email, salt='email-reset')
            reset_link = url_for('auth.reset_password', token=token, _external=True)

            # Send Email
            msg = Message('Password Reset Request', sender='your_email@gmail.com', recipients=[email])
            msg.body = f"Click the link to reset your password: {reset_link}"
            mail.send(msg)

            flash('Password reset link sent to your email.', 'success')
        else:
            flash('Email not found!', 'danger')

    return render_template('forgot_password.html')

@auth_bp.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    try:
        email = s.loads(token, salt='email-reset', max_age=3600)  # Token valid for 1 hour
    except:
        flash('The reset link is invalid or has expired.', 'danger')
        return redirect(url_for('auth.forgot_password'))

    if request.method == 'POST':
        password = request.form['password']
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        user = User.query.filter_by(email=email).first()
        user.password = hashed_password
        db.session.commit()

        flash('Password reset successful! You can now login.', 'success')
        return redirect(url_for('auth.login'))

    return render_template('reset_password.html')
