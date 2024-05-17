import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(ROOT_DIR, "models")
HISTORIES_DIR = os.path.join(ROOT_DIR, "histories")
DATA_DIR = os.path.join(ROOT_DIR, "data")
FLICKR30k_DIR = os.path.join(DATA_DIR, 'flickr30k')
TEST_IMAGES_DIR = os.path.join(ROOT_DIR, "test_images")
TEST_RESULTS = os.path.join(ROOT_DIR, "test_results")