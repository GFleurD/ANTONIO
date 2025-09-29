import lime
from lime import lime_text
from lime.lime_text import LimeTextExplainer
def LIME(X_test, y_test):
    class_names = ['medical query', 'not medical query']
    explainer = LimeTextExplainer(class_names=class_names)
    idx = 83
    exp = explainer.explain_instance(X_test.data[idx], )

    return 