for pkg in ['sklearn', 'lightgbm', 'xgboost', 'catboost', 'scipy', 'numpy', 'pandas', 'joblib']:
    try:
        __import__(pkg)
        print(f'{pkg}: available')
    except ImportError:
        print(f'{pkg}: NOT available')
