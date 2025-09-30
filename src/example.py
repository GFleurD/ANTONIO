from data import load_data, load_embeddings, load_pca, prepare_data_for_training
from perturbations import create_perturbations
from hyperrectangles import load_hyperrectangles
from train import train_base, train_adversarial
from property_parser import parse_properties
from results import calculate_accuracy, calculate_perturbations_accuracy, calculate_cosine_perturbations_filtering #, calculate_marabou_results, calculate_number_of_sentences_inside_the_verified_hyperrectangles
import tensorflow as tf
from tensorflow import keras
import os
import nltk
nltk.download('punkt')
import lime
from lime import lime_text
from lime.lime_text import LimeTextExplainer
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

def get_model(n_components):
    inputs = keras.Input(shape=(n_components,), name="embeddings")
    x = keras.layers.Dense(128, activation="relu", name="dense_1")(inputs)
    outputs = keras.layers.Dense(2, activation="linear", name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    print(model.summary())
    return model


if __name__ == '__main__':
    # Variables to pick data, model etc
    path = 'datasets'
    dataset_names = ['medical']
    #written as dict to easily expand for other models
    encoding_models = {'all-MiniLM-L6-v2': 'sbert22M'}
    og_perturbation_name = 'original'
    perturbation_names = ['character']
    hyperrectangles_names = {'character': ['character']}

    n_components = 30
    batch_size = 64
    seed = 42
    epochs = 30
    pgd_steps = 5 

#by having galse for lots below means it does it all from scratch
    load_saved_embeddings = False
    load_saved_align_mat = False
    load_saved_pca = False
    load_saved_perturbations = False
    load_saved_hyperrectangles = False
    from_logits = True #applies to loss functions in TensorFlow/Keras, so model outputs raw logits (un-normalised scores) not probabilities
    #therefore this expects raw outputs and applies softmax internally

    # Derived variables
    dataset_name = dataset_names[0]
    encoding_model = list(encoding_models.keys())[0]
    encoding_model_name = encoding_models[encoding_model]
    perturbation_name = perturbation_names[0]
    hyperrectangles_name = list(hyperrectangles_names.keys())[0]

    # Load the data and embed them
    data_o = load_data(dataset_name, path=path)
    X_train_pos_embedded_o, X_train_neg_embedded_o, X_test_pos_embedded_o, X_test_neg_embedded_o, y_train_pos_o, y_train_neg_o, y_test_pos_o, y_test_neg_o = load_embeddings(dataset_name, encoding_model, encoding_model_name, og_perturbation_name, load_saved_embeddings, load_saved_align_mat, data_o, path)

    # Create the perturbations and embed them
    data_p = create_perturbations(dataset_name, perturbation_name, data_o, path)
    X_train_pos_embedded_p, X_train_neg_embedded_p, X_test_pos_embedded_p, X_test_neg_embedded_p, y_train_pos_p, y_train_neg_p, y_test_pos_p, y_test_neg_p = load_embeddings(dataset_name, encoding_model, encoding_model_name, perturbation_name, load_saved_perturbations, load_saved_align_mat, data=data_p, path=path)
    
    # Prepare the data for training
    X_train_pos, X_train_neg, X_test_pos, X_test_neg = load_pca(dataset_name, encoding_model_name, load_saved_pca, X_train_pos_embedded_o, X_train_neg_embedded_o, X_test_pos_embedded_o, X_test_neg_embedded_o, n_components, path=path)
    train_dataset, test_dataset = prepare_data_for_training(X_train_pos, X_train_neg, X_test_pos, X_test_neg, y_train_pos_o, y_train_neg_o, y_test_pos_o, y_test_neg_o, batch_size)

    
    # Create the hyper-rectangles
    hyperrectangles = load_hyperrectangles(dataset_name, encoding_model_name, hyperrectangles_name, load_saved_hyperrectangles, path=path)

    # Train and save the base and adversarial models
    model_path = f'{path}/{dataset_name}/models/tf/{encoding_model_name}'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    n_samples = int(len(X_train_pos))

    model = get_model(n_components)
    model = train_base(model, train_dataset, test_dataset, epochs, seed=seed, from_logits=from_logits)
    model.save(f'{model_path}/base_{seed}')

    

    # X_test_FC_strings =  np.concatenate((data_o[2], data_o[3]), axis=0)
    # Y_test_FC_strings = np.concatenate((data_o[6], data_o[7]), axis=0)
    
    # X_testFC, Y_testFC = next(iter(test_dataset.unbatch().batch(len(test_dataset))))
    # X_testFC, Y_testFC = X_testFC.numpy(), Y_testFC.numpy()
    # # print(X_testFC[:3], Y_testFC[:3])
    # X_test_FC_embedded = np.concatenate([X_test_pos, X_test_neg], axis=0)
    
    # #very hard to integrate into ANTONIO pipeline, and so create own classifier function based on internal ANTONIO work
    # def classifier_fn(X_test_FC_strings):
    #     #encoder
    #     encoder = SentenceTransformer('all-MiniLM-L6-v2')
    #     X_test_FC_embed = encoder.encode(X_test_FC_strings, show_progress_bar=False)
    #     #rotate based only on positive strings, and for sake of ease just printed print(data_o[2].shape)
    #     X_test_FC_strings_pos = X_test_FC_strings[:424]
    #     u, s, vh = np.linalg.svd(a=X_test_FC_strings_pos)
    #     align_mat = np.linalg.solve(a=vh, b=np.eye(len(X_test_FC_strings_pos[0])))
    #     X_test_FC_rotated = np.matmul(X_test_FC_embed, align_mat)
    #     #PCA
    #     data_pca = PCA(n_components = 30).fit(X_test_FC_embed)
    #     X_test_FC_PCA= data_pca.transform(X_test_FC_rotated)
    #     proba = model(X_test_FC_PCA, training=False).numpy()
    #     return proba   # return predicted probabilities


    # idx = 3
    # explainer = lime_text.LimeTextExplainer(class_names=['medical query', 'non-medical query'], verbose=True)
    # explanation = explainer.explain_instance(X_test_FC_strings[idx], classifier_fn=classifier_fn, labels=Y_test_FC_strings[idx])


    model = get_model(n_components)
    model = train_adversarial(model, train_dataset, test_dataset, hyperrectangles, epochs, batch_size, n_samples, pgd_steps, seed=seed, from_logits=from_logits)
    model.save(f'{model_path}/{perturbation_name}_{seed}')

#     # Parse properties to VNNlib and Marabou formats
#    # parse_properties(dataset_names, encoding_models, hyperrectangles_names, target='vnnlib', path=path)
#    # parse_properties(dataset_names, encoding_models, hyperrectangles_names, target='marabou', path=path)

    # Results
    calculate_accuracy(dataset_names, encoding_models, batch_size, path=path)
    calculate_perturbations_accuracy(dataset_names, encoding_models, perturbation_names, batch_size, path=path)
    calculate_cosine_perturbations_filtering(dataset_names, encoding_models, perturbation_names, path=path)
