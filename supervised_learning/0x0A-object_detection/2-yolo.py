#!/usr/bin/env python3
""" You only look once """
import tensorflow.keras as K
import numpy as np


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

    def sigmoid_f(self, x):
        """
            sigmoid.
        # Args
            x: Tensor.
        # Returns
            numpy ndarray.
        """

        return (1 / (1 + np.exp(-x)))

    def process_outputs(self, outputs, image_size):
        """
        Function that
        Args:
            - outputs:  list of numpy.ndarrays containing the predictions from
                        the Darknet model for a single image: Each output has
                        shape (grid_height, grid_width, anchor_boxes,
                        4 + 1 + classes)
                    > grid_height:  Height of the grid used for the output
                                    anchor_boxes
                    > grid_width:   Width of the grid used for the output
                                    anchor_boxes
                    > anchor_boxes: Number of anchor boxes used
                    > 4:
                        t_x:    x pos of the center point of the anchor box
                        t_y:    y pos of the center point of the anchor box
                        t_w:    width of the anchor box
                        t_h:    height of the anchor box
                    > 1:            box_confidence
                    > classes:      class probabilities for all classes
            - image_size:   numpy.ndarray containing the image’s original size
                            [image_height, image_width]
        Returns:
            A tuple of (boxes, box_confidences, box_class_probs):
                    > boxes:    List of numpy.ndarrays of shape (grid_height,
                                grid_width, anchor_boxes, 4) containing the
                                processed boundary boxes for each output,
                                respectively:
                            4:  (x1, y1, x2, y2) should represent the boundary
                                box relative to original image
                    > box_confidences:
                                list of numpy.ndarrays of shape (grid_height,
                                grid_width, anchor_boxes, 1) containing the box
                                confidences for each output, respectively
                    > box_class_probs:
                                list of numpy.ndarrays of shape (grid_height,
                                grid_width, anchor_boxes, classes) containing
                                the box’s class probabilities for each output,
                                respectively
        """

        # shape (13,  13,   3,  [t_x, t_y, t_w, t_h],   1    80)
        # Dim   ([0], [1], [2],        [3],           [4]   [5])

        # 1. boxes = dim [2]
        # Procesed according to Fig 2 of paper: https://bit.ly/3emqWp0
        # Adapted from https://bit.ly/2VEZgmZ

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
                        bx_n = self.sigmoid_f(tx_n) + cx_n
                        by_n = self.sigmoid_f(ty_n) + cy_n

                        # calculating hight and width
                        bw_n = pw_n * np.exp(tw_n)
                        bh_n = ph_n * np.exp(th_n)

                        # generating new center
                        new_bx_n = bx_n / grid_w_i
                        new_by_n = by_n / grid_h_i

                        # generating new hight and width
                        new_bh_n = bh_n / self.model.input.shape[2].value
                        new_bw_n = bw_n / self.model.input.shape[1].value

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
            confidence_i = self.sigmoid_f(outputs[i][..., 4:5])
            confidence.append(confidence_i)

        # 3. box class_probs = dim [5:]
        probs = []
        for i in range(len(outputs)):
            probs_i = self.sigmoid_f(outputs[i][:, :, :, 5:])
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
