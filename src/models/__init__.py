try:
    from .statistical import arima_model, sarima_model
    from .ml_models import train_random_forest, train_svr, train_xgboost
    from .dl_models import build_lstm_model, build_gru_model

except ImportError:
    from statistical import arima_model, sarima_model
    from ml_models import train_random_forest, train_svr, train_xgboost
    from dl_models import build_lstm_model
