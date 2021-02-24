#!/usr/bin/env python3
""" You only look once """
import tensorflow.keras as K
import numpy as np
import glob
import cv2
import os


class Yolo:
    """ Use the Yolo v3 algorith to perform object detection """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Instantiation method
            model_path is the path to where a Darknet Keras model is s
              tored
            classes_path is the path to where the list of class names
              used for the Darknet model, listed in order of index, can
              be found
            class_t is a float representing the box score threshold for
              the initial filtering step
            nms_t is a float representing the IOU threshold for non-max
               suppression
            anchors is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
              containing all of the anchor boxes:
                  outputs is the number of outputs (predictions) made by the
                    Darknet model
                  anchor_boxes is the number of anchor boxes used for each
                    prediction
                  2 => [anchor_box_width, anchor_box_height]

        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """ calculates sigmoid function """
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Args:
            outputs is a list of numpy.ndarrays containing the predictions
              from the Darknet model for a single image:
                Each output will have the shape (grid_height, grid_width,
                  anchor_boxes, 4 + 1 + classes)
                    grid_height & grid_width => the height and width of the
                      grid used for the output
                    anchor_boxes => the number of anchor boxes used
                    4 => (t_x, t_y, t_w, t_h)
                    1 => box_confidence
                    classes => class probabilities for all classes
            image_size is a numpy.ndarray containing the image’s original
              size [img_h, img_w]
        """
        boxes = []
        for i in range(len(outputs)):
            boxes_i = outputs[i][..., 0:4]
            grid_h_i = outputs[i].shape[0]
            grid_w_i = outputs[i].shape[1]
            anchor_box_i = outputs[i].shape[2]

            for anchor_n in range(anchor_box_i):
                for cy_n in range(grid_h_i):
                    for cx_n in range(grid_w_i):

                        tx_n = outputs[i][cy_n, cx_n, anchor_n, 0:1]
                        ty_n = outputs[i][cy_n, cx_n, anchor_n, 1:2]
                        tw_n = outputs[i][cy_n, cx_n, anchor_n, 2:3]
                        th_n = outputs[i][cy_n, cx_n, anchor_n, 3:4]

                        # size of the anchors
                        pw_n = self.anchors[i][anchor_n][0]
                        ph_n = self.anchors[i][anchor_n][1]

                        # calculating center
                        bx_n = self.sigmoid(tx_n) + cx_n
                        by_n = self.sigmoid(ty_n) + cy_n

                        # calculating hight and width
                        bw_n = pw_n * np.exp(tw_n)
                        bh_n = ph_n * np.exp(th_n)

                        # generating new center
                        new_bx_n = bx_n / grid_w_i
                        new_by_n = by_n / grid_h_i

                        # generating new hight and width
                        new_bh_n = bh_n / self.model.input.shape[2]
                        new_bw_n = bw_n / self.model.input.shape[1]

                        # calculating (cx1, cy1) and (cx2, cy2) coords
                        y1 = (new_by_n - (new_bh_n / 2)) * image_size[0]
                        y2 = (new_by_n + (new_bh_n / 2)) * image_size[0]
                        x1 = (new_bx_n - (new_bw_n / 2)) * image_size[1]
                        x2 = (new_bx_n + (new_bw_n / 2)) * image_size[1]

                        boxes_i[cy_n, cx_n, anchor_n, 0] = x1
                        boxes_i[cy_n, cx_n, anchor_n, 1] = y1
                        boxes_i[cy_n, cx_n, anchor_n, 2] = x2
                        boxes_i[cy_n, cx_n, anchor_n, 3] = y2

            boxes.append(boxes_i)

        # 2. box confidence = dim [4]
        confidence = []
        for i in range(len(outputs)):
            confidence_i = self.sigmoid(outputs[i][..., 4:5])
            confidence.append(confidence_i)

        # 3. box class_probs = dim [5:]
        probs = []
        for i in range(len(outputs)):
            probs_i = self.sigmoid(outputs[i][:, :, :, 5:])
            probs.append(probs_i)

        return (boxes, confidence, probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """

        Args:
            boxes: a list of numpy.ndarrays of shape (grid_height,
              grid_width, anchor_boxes, 4) containing the processed
              boundary boxes for each output, respectively
            box_confidences: a list of numpy.ndarrays of shape
              (grid_height, grid_width, anchor_boxes, 1) containing
              the processed box confidences for each output, respectively
            box_class_probs: a list of numpy.ndarrays of shape
              (grid_height, grid_width, anchor_boxes, classes) containing
              the processed box class probabilities for each output,
              respectively
        """
        scores = []
        classes = []
        box_classes_scores = []
        index_arg_max = []
        box_classes = []

        # 1. Multiply confidence x probs to find real confidence of each class
        for bc_i, probs_j in zip(box_confidences, box_class_probs):
            scores.append(bc_i * probs_j)

        # 2. find temporal indices de clas cajas con los arg mas altos
        for score in scores:
            index_arg_max = np.argmax(score, axis=-1)
            # -1 = last dimension)

            # 3. Flatten each array
            index_arg_max_flat = index_arg_max.flatten()

            # 4. Everything in one single array
            classes.append(index_arg_max_flat)

            # find the values
            score_max = np.max(score, axis=-1)
            score_max_flat = score_max.flatten()
            box_classes_scores.append(score_max_flat)

        boxes = [box.reshape(-1, 4) for box in boxes]
        # (13, 13, 3, 4) ----> (507, 4)

        box_classes = np.concatenate(classes, axis=-1)
        # -1 = add to the end

        box_classes_scores = np.concatenate(box_classes_scores, axis=-1)
        # -1 = add to the end

        boxes = np.concatenate(boxes, axis=0)

        # filtro
        # boxes[box_classes_scores >= self.class_t]
        filtro = np.where(box_classes_scores >= self.class_t)

        return (boxes[filtro], box_classes[filtro], box_classes_scores[filtro])

    def iou(self, x1, x2, y1, y2, pos1, pos2, area):
        """
        Function that
        Args:
            - x1:       xxx
            - x2:       xxx
            - y1        xxx
            - yy2       xxx
            - pos1      xxx
            - pos2      xxx
            - area      xxx
        Returns:
            The intersection over union %
        """

        # find the coordinates
        a = np.maximum(x1[pos1], x1[pos2])
        b = np.maximum(y1[pos1], y1[pos2])

        c = np.minimum(x2[pos1], x2[pos2])
        d = np.minimum(y2[pos1], y2[pos2])

        height = np.maximum(0.0, d - b)
        width = np.maximum(0.0, c - a)

        # overlap ratio betw bounding box
        intersection = (width * height)
        union = area[pos1] + area[pos2] - intersection
        iou = intersection / union

        return iou

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Args:
            - filtered_boxes:   numpy.ndarray of shape (?, 4) containing all
                                of the filtered bounding boxes
            - box_classes:      numpy.ndarray of shape (?,) containing the
                                class number 4 the class that filtered_boxes
                                predicts, respectively
            - box_scores:       numpy.ndarray of shape (?) containing the box
                                scores for each box in filtered_boxes,
                                respectively
        Returns:                Tuple of (box_predictions,
                                          predicted_box_classes,
                                          predicted_box_scores):
                > box_predictions:          numpy.ndarray of shape (?, 4)
                                            containing all of the predicted
                                            bounding boxes ordered by class &
                                            box score
                > predicted_box_classes:    numpy.ndarray of shape (?,)
                                            containing the class number for
                                            box_predictions ordered by class &
                                            box score, respectively
                > predicted_box_scores:     numpy.ndarray of shape (?)
                                            containing the box scores for
                                            box_predictions ordered by class &
                                            box score, respectively
        """

        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for classes in set(box_classes):
            index = np.where(box_classes == classes)

            # function arrays
            filtered = filtered_boxes[index]
            scores = box_scores[index]
            classe = box_classes[index]

            # coordinates of the bounding boxes
            x1 = filtered[:, 0]
            y1 = filtered[:, 1]
            x2 = filtered[:, 2]
            y2 = filtered[:, 3]

            # calculate area of the bounding boxes and sort from high to low
            area = (x2 - x1) * (y2 - y1)
            index_list = np.flip(scores.argsort(), axis=0)

            # loop remaining indexes to hold list of picked indexes
            keep = []
            while (len(index_list) > 0):
                pos1 = index_list[0]
                pos2 = index_list[1:]
                keep.append(pos1)

                # find the intersection over union %
                iou = self.iou(x1, x2, y1, y2, pos1, pos2, area)

                below_threshold = np.where(iou <= self.nms_t)[0]
                index_list = index_list[below_threshold + 1]

            # array of piked indexes
            keep = np.array(keep)

            box_predictions.append(filtered[keep])
            predicted_box_classes.append(classe[keep])
            predicted_box_scores.append(scores[keep])

        box_predictions = np.concatenate(box_predictions)
        predicted_box_classes = np.concatenate(predicted_box_classes)
        predicted_box_scores = np.concatenate(predicted_box_scores)

        return (box_predictions, predicted_box_classes, predicted_box_scores)

    @staticmethod
    def load_images(folder_path):
        """
        Args:
            - folder_path:  a string representing the path to the folder
                            holding all the images to load
        Returns
            - tuple of (images, image_paths):
                > images:       List of images as numpy.ndarrays
                > image_paths:  List of paths of each image in images
        """

        # creating a correct full path argument
        images = []
        image_paths = glob.glob(folder_path + '/*', recursive=False)

        # creating the images list
        for imagepath_i in image_paths:
            images.append(cv2.imread(imagepath_i))

        return(images, image_paths)

    def preprocess_images(self, images):
        """
            Resize the images with inter-cubic interpolation
            Rescale all images to have pixel values in the range [0, 1]
        Args:
            - images:   list of images as numpy.ndarrays
        Returns:       tuple of (pimages, image_shapes):
            > pimages:      numpy.ndarray of shape (ni, input_h, input_w, 3)
                            containing all of the preprocessed images
                @ ni:           the number of images that were preprocessed
                @ input_h:      input height for the Darknet model
                                Note: this can vary by model
                @ input_w:      the input width for the Darknet model
                                Note: this can vary by model
                @ 3:            number of color channels
            > image_shapes:     a numpy.ndarray of shape (ni, 2) containing
                                the original height and width of the images
                @ 2             (image_height, image_width)
        """

        dims = []
        res_images = []

        input_h = self.model.input.shape[1].value
        input_w = self.model.input.shape[2].value
        for image in images:
            dims.append(image.shape[:2])

        dims = np.stack(dims, axis=0)

        newtam = (input_h, input_w)

        interpolation = cv2.INTER_CUBIC
        for image in images:
            resize_img = cv2.resize(image, newtam, interpolation=interpolation)
            resize_img = resize_img / 255
            res_images.append(resize_img)

        res_images = np.stack(res_images, axis=0)

        return (res_images, dims)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Function that displays the image with all boundary boxes, class names,
        and box scores (see example below)
        1  -  Boxes should be drawn as with a blue line of thickness 2
        2  -  Class names and box scores should be drawn above each box in red
        3  -  Box scores should be rounded to 2 decimal places
        4  -  Text should be written 5 pixels above the top left corner of the
              box
        5  -  Text should be written in FONT_HERSHEY_SIMPLEX
        6  -  Font scale should be 0.5
        7  -  Line thickness should be 1
        8  -  You should use LINE_AA as the line type
        9  -  The window name should be the same as file_name
        10 -  If the s key is pressed:
            - The image should be saved in the directory detections, located
              in the current directory
            - If detections does not exist, create it
            - The saved image should have the file name file_name
            - The image window should be closed
            If any key besides s is pressed, the image window should be closed
            without saving
        Args:
            - image:        numpy.ndarray containing an unprocessed image
            - boxes:        numpy.ndarray containing the boundary boxes for
                            the image
            - box_classes:  numpy.ndarray containing the class indices for
                            each box
            - box_scores:   numpy.ndarray containing the box scores for each
                            box
            - file_name:    the file path where the original image is stored
        Returns: Nothing
        """
        # 1. colors for printing
        box_color = (0, 0, 255)
        box_thickness = 2

        # 2. colors for printing
        box_score_color = (255, 0, 0)

        # 5-6. Font
        font_type = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5

        # 7 Line
        line_thickness = 1
        line_type = cv2.LINE_AA

        i = 0
        for box_i in boxes:
            # bounding boxes
            start_point = (int(box_i[0]), int(box_i[1]))
            end_point = (int(box_i[2]), int(box_i[3]))

            # print box
            box_i = cv2.rectangle(img=image,
                                  pt1=start_point,
                                  pt2=end_point,
                                  color=box_score_color,
                                  thickness=box_thickness)

            # print text
            bc_i = box_classes[i]
            bs_i = box_scores[i]
            text = self.class_names[bc_i] + " {:.2f}".format(bs_i)
            org = (start_point[0], start_point[1] - 5)

            box_i = cv2.putText(img=box_i,
                                text=text,
                                org=org,
                                fontFace=font_type,
                                fontScale=font_scale,
                                color=box_color,
                                thickness=line_thickness,
                                lineType=line_type,
                                bottomLeftOrigin=False)
            # show the image
            cv2.imshow(file_name, image)
            i = i + 1

            # wait for key and save if that is the case
            if cv2.waitKey(0) == 's':
                if not os.path.exists('detections'):
                    os.makedirs('detections')
                os.chdir('detections')
                cv2.imwrite(file_name, image)
                os.chdir('../')
            cv2.destroyAllWindows()

    def predict(self, folder_path):
        """
        Function that displays all images using the show_boxes method
        Args:
            - folder_path:  string representing the path to the folder holding
                            all the images to predict
        Note:   - All image windows should be named after the corresponding
                  image filename without its full path(see examples below)
        Returns: - Tuple of (predictions, image_paths):
                    > predictions: list of tuples for each image of (boxes,
                      box_classes, box_scores)
                    > image_paths: list of image paths corresponding to each
                      prediction in predictions without saving
        """

        # load data
        images, image_paths = self.load_images(folder_path)
        preprocess_images, _ = self.preprocess_images(images)
        outs = self.model.predict(preprocess_images)

        predictions_list = []
        i = 0
        for image_i in images:
            outputs = [outs[0][i, ...], outs[1][i, ...], outs[2][i, ...]]
            image_size = np.array([image_i.shape[0], image_i.shape[1]])

            # proces, filter and generate nps
            boxes, confi, c_probs = self.process_outputs(outputs, image_size)
            filt, clas, sc = self.filter_boxes(boxes, confi, c_probs)
            b_pred, c_pred, s_pred = self.non_max_suppression(filt, clas, sc)

            predictions_list.append((b_pred, c_pred, s_pred))

            # get names and save
            names = image_paths[i].split("/")[-1]
            self.show_boxes(image_i, b_pred, c_pred, s_pred, names)
            i = i + 1
        return (predictions_list, image_paths)
