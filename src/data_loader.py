import pickle

def load_model(model_path):
    try:
        with open(model_path, 'rb') as file:
           model = pickle.load(file)
        return model
    except FileExistsError:
        print('File not found')
        return None
          