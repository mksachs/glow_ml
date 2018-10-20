PREDICT_ACCIDENT_FEATURES = {
    'numeric': ['risk_score', 'age', 'years_experience', 'hours_worked_per_week'],
    'categorical': ['gender', 'role', 'company', 'state']
}

PREDICT_ACCIDENT_TARGET = 'had_accident'

PREDICT_ACCIDENT_ALGORITHMS = ['lr', 'gb', 'rf']