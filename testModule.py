import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import os
import glob
import csv
import tensorflow_hub as hub
import argparse



class TestModule:
    model = None

    # Functions to get filenames from image paths
    def get_filename(self,filePath):
        filename = filePath.split("\\")[-1]
        return filename


    def get_filename_without_ext(self,filename):
        filename_without_ext = filename.split(".")[0]
        return filename_without_ext


    def find_pred_type(self,y_actual, y_pred):
        if y_actual == y_pred == 1:
            return "TP"
        elif y_pred == 1 and y_actual != y_pred:
            return "FP"
        elif y_actual == y_pred == 0:
            return "TN"
        elif y_pred == 0 and y_actual != y_pred:
            return "FN"
        else:
            print("ERROR")


    def GT_cat_mapping(self,GT_cat):
        if (GT_cat == "M") or (GT_cat == "I"):
            GT = 1
        else:
            GT = 0
        return GT


    def pred_cat_mapping(self,prediction):
        if prediction == "cataract":
            pred = 1
        else:
            pred = 0
        return pred


    def get_pred_probs(self,prediction, probability):
        if prediction == "cataract":
            cat_prob = probability
            nocat_prob = 1 - float(probability)
        else:
            nocat_prob = probability
            cat_prob = 1 - float(probability)
        return cat_prob, nocat_prob


    def class_prediction(self, img_path):
        img_size = 384
        img = image.image_utils.load_img(img_path, target_size=(img_size, img_size))
        img_array = image.image_utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.  # Normalize pixel values between 0 and 1
        # Make the prediction
        predictions = self.model.predict(img_array)
        # Get the predicted class and probability
        class_index = np.argmax(predictions[0])
        if class_index == 0:
            predicted_class = 0
        else:
            predicted_class = 1
        prob = predictions[0][class_index]
        return predicted_class, prob


    def write_to_csv_file(self, filename, rows):
        with open(filename, 'w', newline='') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(
                ['Filename', 'Patient ID', 'Laterality', 'Image Number', 'Category',
                 'Ground Truth', 'Prediction', 'Pred Type',
                 'Cataract Prob', 'No Cataract Prob']
            )
            for row in rows:
                # writing the data rows
                csvwriter.writerow(row)


    def getTestResult(self, model_path, dataset_path, analyse):
        self.model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
        # self.model = tf.keras.models.load_model(model_path)
        filepaths = glob.glob(dataset_path + '/*.jpg')
        filenames = list(set([self.get_filename_without_ext(self.get_filename(x)) for x in filepaths]))
        rows = []
        for filename in filenames:
            # GT
            GT_cat = filename.split("_")[-1]
            GT = self.GT_cat_mapping(GT_cat)
            image = os.path.join(dataset_path, filename) + '.jpg'
            patient_id = filename.split("_")[0]
            laterality = filename.split("_")[1]
            image_number = filename.split("_")[2]
            category = filename.split("_")[3]

            prediction, probability = self.class_prediction(image)
            pred = self.pred_cat_mapping(prediction)
            pred_type = self.find_pred_type(GT, pred)
            cat_prob, nocat_prob = self.get_pred_probs(prediction, probability)

            print([filename, patient_id, laterality, image_number, category, GT, pred, pred_type, cat_prob, nocat_prob])
            rows.append(
                [filename, patient_id, laterality, image_number, category, GT, pred, pred_type, cat_prob, nocat_prob])
            print(analyse)

            self.write_to_csv_file("report.csv", rows)




#Driver code goes here---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_path", required=True, help="model path")
    ap.add_argument("-d", "--dataset_path", required=True, help="project root directory path")
    args = vars(ap.parse_args())
    model_path = args['model_path']
    dataset_path = args['dataset_path']

    obj = TestModule()
    obj.getTestResult(str(model_path), str(dataset_path), analyse=True)



if __name__ == '__main__':
    main()








