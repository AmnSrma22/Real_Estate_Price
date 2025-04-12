import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def plot_feature_importance(model, X):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=model.feature_importances_, y=X.columns)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    st.image("feature_importance.png")
