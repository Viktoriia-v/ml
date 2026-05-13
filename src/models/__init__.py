"""Model implementations.

Each module exposes:
    train(train_df, val_df, lang, **cfg) -> trained_model
    predict(model, df, lang) -> np.ndarray of class ids
    predict_proba(model, df, lang) -> np.ndarray of shape (N, 2), optional
"""
