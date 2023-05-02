import numpy as np
from pprint import pprint

import sklearn.datasets
import sklearn.metrics
from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import OneHotEncoder

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import autosklearn.classification
import argparse
import time
import glob
import os



def load_data(path, classA, classB, apply_pca=False):
    images = []
    labels = []
    classes = []

    class_folders = glob.glob(os.path.join(path, '*'))
    class_folders.sort()
    for i, c in enumerate(class_folders):
        classes.append(c.split(os.sep)[-1])
        if (i==classA) or (i==classB):
            for _, ip in enumerate(glob.glob(os.path.join(c, '*'))):
                img = np.load(ip)
                img = img.flatten()
                images.append(img)
                labels.append([i])

    X = np.array(images)
    if apply_pca:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        pca = PCA(n_components = 50, random_state = 631876)
        X = pca.fit_transform(X)

    y = np.array(labels)
    y = OneHotEncoder().fit_transform(y, y=None).toarray()
    y = y.astype(int)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, random_state=1
    )

    return classes, X_train, X_test, y_train, y_test 

def buil_model(X_train, y_train, logging_config, name):
    automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=1200,
            per_run_time_limit=120,
            ensemble_kwargs = {'ensemble_size': 1},
            include={
                'classifier':['bernoulli_nb', 'decision_tree', 'extra_trees', 'gaussian_nb', 'k_nearest_neighbors', 'lda', 'liblinear_svc', 'mlp', 'multinomial_nb', 'passive_aggressive', 'qda', 'random_forest'],
                
                'feature_preprocessor':['densifier', 'extra_trees_preproc_for_classification', 'fast_ica', 'feature_agglomeration', 'kernel_pca', 'kitchen_sinks', 'no_preprocessing', 'nystroem_sampler', 'pca', 
                                        'polynomial', 'random_trees_embedding', 'truncatedSVD']
            },
            # Bellow two flags are provided to speed up calculations
            # Not recommended for a real implementation
            initial_configurations_via_metalearning=0,
            #smac_scenario_args={"runcount_limit": 1},
            logging_config=logging_config,
            # *auto-sklearn* generates temporal files under tmp_folder
            tmp_folder="./tmp_folder/"+name,
            # By default tmp_folder is deleted. We will preserve it
            # for debug purposes
            delete_tmp_folder_after_terminate=False,
            n_jobs = -1,
            #metric = 
            #scoring_functions= autosklearn.metrics.accuracy,
            memory_limit = 8192,
        )
    
    automl.fit(X_train, y_train)
    
    return automl

if __name__ == '__main__':
    logging_config = {
        "version": 1,
        "disable_existing_loggers": True,
        "formatters": {
            "custom": {
                # More format options are available in the official
                # `documentation <https://docs.python.org/3/howto/logging-cookbook.html>`_
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        },
        # Any INFO level msg will be printed to the console
        "handlers": {
            "console": {
                "level": "INFO",
                "formatter": "custom",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "": {  # root logger
                "level": "DEBUG",
            },
            "Client-EnsembleBuilder": {
                "level": "DEBUG",
                "handlers": ["console"],
            },
        },
    }

    parser = argparse.ArgumentParser(description='AutoML')
    

    parser.add_argument('--classA', type=int, required=True)
    parser.add_argument('--classB', type=int, required=True)

    args = parser.parse_args()

    classA = args.classA
    classB = args.classB

    classes, X_train, X_test, y_train, y_test= load_data('EuroSAT_processed_CNN_1', classA, classB, apply_pca=False)
    print(classes, X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    name = '{}vs{}'.format(classes[classA], classes[classB])
    print(name)

    automl = buil_model(X_train, y_train, logging_config, name)

    print(automl.leaderboard())
    pprint(automl.show_models(), indent=4)
    # Print statistics about the auto-sklearn run such as number of
    # iterations, number of models failed with a time out.
    print(automl.sprint_statistics())
    predictions = automl.predict(X_test)
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))
    print('\n\n\n')

    with open('results.txt', 'a') as fi:
        fi.write(name)
        fi.write('\n')
        fi.write(sklearn.metrics.classification_report(y_test, predictions))
        fi.write('\n')
        fi.write('------------------------------------------------------------------\n')