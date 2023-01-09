import os

def save_embeddings_as_file(embeddings, db_folder):
    # ToDo: Build proper header so we can map back from embeddings to initial individuals
    file_header = ""
    file_name = os.path.join(db_folder, "embedding.csv")
    np.savetxt(filename, embeddings, fmt='%s', delimiter=',', header=file_header)