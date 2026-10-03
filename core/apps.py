import os
import pickle
import joblib
from django.apps import AppConfig
from django.conf import settings

class CoreConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'core'

    ml_model = None
    ml_encoders = None
    skill_model = None
    skill_tfidf = None
    skill_encoder = None
    branch_options = []
    job_role_classes = []

    def ready(self):
        base_dir = settings.BASE_DIR
        
        # Load profile ML assets
        model_joblib = os.path.join(base_dir, "profile_model.joblib")
        model_pkl = os.path.join(base_dir, "profile_model.pkl")
        encoders_pkl = os.path.join(base_dir, "profile_encoders.pkl")
        
        try:
            if os.path.exists(model_joblib):
                CoreConfig.ml_model = joblib.load(model_joblib)
            elif os.path.exists(model_pkl):
                with open(model_pkl, "rb") as f:
                    CoreConfig.ml_model = pickle.load(f)
                    
            if os.path.exists(encoders_pkl):
                with open(encoders_pkl, "rb") as f:
                    CoreConfig.ml_encoders = pickle.load(f)
        except Exception as e:
            print(f"[WARN] Error loading Profile ML assets: {e}")

        if CoreConfig.ml_encoders:
            CoreConfig.branch_options = list(CoreConfig.ml_encoders['branch'].classes_)
            CoreConfig.job_role_classes = list(CoreConfig.ml_encoders['job_role'].classes_)
        else:
            CoreConfig.branch_options = ['CSE', 'Civil', 'ECE', 'EEE', 'IT', 'Mechanical']
            CoreConfig.job_role_classes = ['Analyst', 'Data Scientist', 'Software Engineer', 'Web Developer']

        # Load skill ML assets
        try:
            CoreConfig.skill_model = joblib.load(os.path.join(base_dir, "job_role_model.pkl"))
            CoreConfig.skill_tfidf = joblib.load(os.path.join(base_dir, "tfidf.pkl"))
            CoreConfig.skill_encoder = joblib.load(os.path.join(base_dir, "label_encoder.pkl"))
        except Exception as e:
            print(f"[WARN] Error loading Skill ML assets: {e}")
