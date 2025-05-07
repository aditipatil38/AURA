import firebase_admin
from firebase_admin import credentials, db
import os
from dotenv import load_dotenv

def init_firebase():
    load_dotenv()
    cred_path = os.getenv("FIREBASE_CREDENTIALS")
    db_url = os.getenv("FIREBASE_DB_URL")
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred, {'databaseURL': db_url})
