import smtplib
from email.message import EmailMessage
from datetime import datetime, timedelta
from typing import Optional
import jwt
from config import MAIL_USERNAME, MAIL_PASSWORD, MAIL_FROM, SECRET_KEY

ALGORITHM = "HS256"

def create_verification_token(email: str, expires_delta: Optional[timedelta] = None):
    to_encode = {"sub": email}
    expire = datetime.utcnow() + (expires_delta or timedelta(hours=24))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def send_verification_email(email: str, token: str):
    msg = EmailMessage()
    msg['Subject'] = 'Email Verification - Flask'
    msg['From'] = MAIL_FROM
    msg['To'] = email

    html_content = f"""
    <html>
        <body>
            <h3>Hello,</h3>
            <p>Thank you for registering. Please verify your email by clicking the link below:</p>
            <p><a href='http://localhost:5000/auth/verify_email?token={token}'>Verify Email</a></p>
            <p>This link will expire in 24 hours.</p>
            <p>Best regards,<br>Your Flask Team</p>
        </body>
    </html>
    """
    msg.set_content("Please view this email in an HTML-compatible client.")
    msg.add_alternative(html_content, subtype='html')

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(MAIL_USERNAME, MAIL_PASSWORD)
            server.send_message(msg)
    except Exception as e:
        raise Exception(f"Error sending email: {e}")