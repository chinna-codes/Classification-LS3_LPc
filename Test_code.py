import tensorflow as tf
import numpy as np
import time
import os
import shutil
import csv
import pandas as pd
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.config.list_physical_devices('GPU')


def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph
    if print_graph == True:
        print("-" * 50)
        print("Frozen model layers: ")
        layers = [op.name for op in import_graph.get_operations()]
        for layer in layers:
            print(layer)
        print("-" * 50)
    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))


def resize_image(image_path, labels):
    '''Perform preprocessing of image using tf. Resize image using tensorflow '''
    raw_image = tf.io.read_file(image_path)
    image_tensor = tf.image.decode_bmp(raw_image)
    image_resized = tf.image.resize(image_tensor, size, method='area')
    image_resized = (image_resized - img_mean) / img_std_dev
    return image_resized, labels, image_path


def get_key(val):
    for key, value in class_label_dict.items():
        if val == value:
            return key
    return "key doesn't exist"


def extract_images(root_dir, move_img=True, logs=True):
    confusion_matrix = np.zeros(shape=(num_of_classes, num_of_classes))

    image_files = [os.path.join(dir, name) for dir, folders, files in os.walk(root_dir) for name in files]
    list_classes = [os.path.split(os.path.split(i)[0])[1] for i in image_files]
    image_labels = [class_label_dict.get(i) for i in list_classes]

    total_num_images = len(image_files)
    print(f"Total number of images present: {total_num_images}")

    dataset = tf.data.Dataset.from_tensor_slices((image_files, image_labels))
    dataset = dataset.map(resize_image, num_parallel_calls=4)
    dataset = dataset.batch(2)
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)

    cnt = 0
    img_cnt = 0
    with tqdm(total=total_num_images, desc="Processing Images") as pbar:
        while cnt < total_num_images:
            image, lbls, paths = iterator.get_next()
            for i in range(image.shape[0]):
                path = paths[i].numpy().decode("utf-8")
                img = tf.expand_dims(image[i], axis=0)

                img_name = path.split('\\')[-1]
                true_label = lbls[i].numpy()
                frozen_graph_predictions = frozen_func(x=tf.constant(img))[0]
                predictions = frozen_graph_predictions[0].numpy()
                predictions = predictions.astype('float32')

                threshold_prediction = [predictions[i] if predictions[i] >= thresholds[i] else 0 for i in
                                            range(len(predictions))]

                predicted_label = []
                if np.count_nonzero(threshold_prediction) == 0:
                    predicted_label.append(np.argmax(predictions))
                else:
                    # 2) when one or more predicted_value > threshold,
                    # Filter out the indices of those classes
                    indices_above_threshold = [i for i, value in enumerate(threshold_prediction) if value > 0]
                    # Find the index with the highest prediction value among these indices
                    if indices_above_threshold:
                        highest_prediction_index = max(indices_above_threshold, key=lambda index: predictions[index])
                        predicted_label.append(highest_prediction_index)

                img_cnt += 1
                # print(f"{img_cnt}th Image, True label: {true_label}, predicted label: {predicted_label}")


                bucket_name = img_name.split(' ')[-1].split('.')[0]

                for i in range(len(predicted_label)):
                    confusion_matrix[true_label][predicted_label[i]] += 1

                    if move_img:
                        true_class_name = os.path.split(path.split('\\')[-2])[-1]
                        predicted_class_name = get_key(predicted_label[i])

                        dst_true_class = os.path.join(move_dir, true_class_name)
                        dst_bucket_name = os.path.join(dst_true_class, bucket_name)
                        os.makedirs(dst_true_class, exist_ok=True)
                        os.makedirs(dst_bucket_name, exist_ok=True)

                        dst_predicted_class = os.path.join(dst_bucket_name, predicted_class_name)
                        os.makedirs(dst_predicted_class, exist_ok=True)

                        dst = os.path.join(dst_predicted_class, img_name)
                        shutil.copy(path, dst)

                        # # Update confusion matrix
                        # for label in predicted_label:
                        #     confusion_matrix[true_label][label] += 1

            cnt += image.shape[0]
            # print(f"Read {img_cnt} images with shape {image.shape[1], image.shape[2]} dtype {image.dtype}\n")
            pbar.update(image.shape[0])
    print(f"Done with reading {img_cnt} Images out of {cnt} Images")
    print(f"Confusion Matrix for {os.path.split(graph_path)[-1]} \n {confusion_matrix}")
    return confusion_matrix


if __name__ == "__main__":
    # np.set_printoptions(precision=5, suppress=True)

    class_label_dict = {'Lens out of FOV': 0,
                        'Low Saline': 1,
                        'Multiple Lens': 2,
                        'No Lens': 3,
                        'No Shell': 4,
                        'Normal Lens': 5,
                        'Shell out of FOV': 6}

    list_class = list(class_label_dict.keys())
    thresholds = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]

    root_dir = "D:/chinna/LPC_classification/data/val" # +folder_names[8]
    graph_path = "D:/chinna/LPC_classification/results/pb_seincep/LPC_287.pb"

    move_imgs = True
    logs = True
    conf_csv = True

    confus_csv = 'D:/chinna/LPC_classification/results/287/Conf_287.csv'
    move_dir = "D:/chinna/LPC_classification/results/287/"

    img_mean = 127.79951954212892
    img_std_dev = 255
    size = (576, 687)

    input_name = "x:0"
    output_name = "Identity:0"

    num_of_classes = len(class_label_dict)
    print("Loading Graph... ")
    start = time.time()

    with tf.io.gfile.GFile(graph_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(f.read())
    end = time.time()
    print(f"Finished Loading graph in {end - start} seconds")

    frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                    inputs=[input_name],
                                    outputs=[output_name],
                                    print_graph=False)

    confusion_matrix = extract_images(root_dir, move_imgs, logs)
    end = time.time()
    print(f"Finished Testing : in {end - start} seconds")

    acc = []
    prec = []

    for i in range(len(list_class)):
        acc_ = confusion_matrix[i][i] / sum(confusion_matrix[i])
        prec_ = confusion_matrix[i][i] / sum(confusion_matrix[:, i])
        acc.append(acc_)
        prec.append(prec_)
        if confusion_matrix[i][i] / sum(confusion_matrix[i]) != 1:
            err = confusion_matrix[i]
            for j, num in enumerate(err):
                if num != confusion_matrix[i][i] and num != 0:
                    print(f"Misclassified as {list_class[j]} with count {confusion_matrix[i][j]}")

    df_conf_mat = pd.DataFrame(confusion_matrix, index=[i for i in list_class], columns=[c for c in list_class])
    df_conf_mat.loc['Precision'] = prec
    df_conf_mat = pd.concat([df_conf_mat, pd.Series(acc, index=[i for i in list_class], name='Accuracy')], axis=1)

    with open(confus_csv, 'a') as f:
        df_conf_mat.to_csv(f)
