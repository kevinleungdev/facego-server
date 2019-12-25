# -*- coding: utf-8 -*-

import conf
import dlib
import numpy as np

dlib_face_predictor_model_location = conf.get_prop('dlib', 'dlib_face_predictor_model_location')
dlib_face_recognition_model_location = conf.get_prop('dlib', 'dlib_face_recognition_model_location')


def _rect_to_css(rect):
    return rect.left(), rect.top(), rect.right(), rect.bottom()


def _css_to_rect(css):
    return dlib.rectangle(css[0], css[1], css[2], css[3])


def _trim_css_to_bounds(css, image_shape):
    return max(css[0], 0), max(css[1], 0), min(css[2], image_shape[1]), min(css[3], image_shape[0])


def compare_faces(known_face_encodings, face_encoding_to_check):
    if len(known_face_encodings) == 0:
        return np.empty(0)

    return np.linalg.norm(known_face_encodings - face_encoding_to_check, axis=1)


class FaceApi(object):

    def detect_faces(self, img, *args):
        """
        Return an array of bounding boxes of an image
        :param img: an image
        :type img: numpy.ndarray
        :return A list of tuples of found face locations in css (left, top, right, bottom) order
        """
        raise NotImplementedError()

    def face_encodings(self, img, *args):
        raise NotImplementedError()

    def compare_faces(self, known_face_encodings, face_encoding_to_check):
        raise NotImplemented


class FaceDlibApi(FaceApi):

    def __init__(self):
        # face detector
        self.face_detector = dlib.get_frontal_face_detector()

        # face landmark predictor
        self.face_predictor = dlib.shape_predictor(dlib_face_predictor_model_location)

        # face encoder
        self.face_encoder = dlib.face_recognition_model_v1(dlib_face_recognition_model_location)

    def _raw_face_locations(self, img, number_of_times_upsample=1):
        """
        Return A list of dlib 'rect' objects of found face locations
        """
        return self.face_detector(img, number_of_times_upsample)

    def _raw_face_landmarks(self, img, face_locations=None):
        if face_locations is None:
            face_locations = self._raw_face_locations(img)
        else:
            face_locations = [_css_to_rect(face_location) for face_location in face_locations]

        return [self.face_predictor(img, face_location) for face_location in face_locations]

    def detect_faces(self, img, number_of_times_to_upsample=1):
        """
        :param number_of_times_to_upsample: How many times to upsample the image looking for faces.
                                            Higher numbers find smaller faces.
        :type number_of_times_to_upsample: int
        :return A list of tuples of found face locations in css (left, top, right, bottom) order
        """
        return [_trim_css_to_bounds(_rect_to_css(face), img.shape) for face
                in self._raw_face_locations(img, number_of_times_to_upsample)]

    def face_encodings(self, img, known_face_locations=None, num_jitters=1):
        """
        Given an image, return the 128-dimension face encoding for each face in the image.
        :param img: The image that contains one or more faces
        :param known_face_locations: Optional - the bounding boxes of each face if you already know them.
        :param num_jitters: How many times to re-sample the face when calculating encoding. Higher is more accurate
                            , but slower (i.e. 100 is 100x slower)
        :return: A list of 128-dimentional face encodings (one for each face in the image)
        """
        raw_landmarks = self._raw_face_landmarks(img, known_face_locations)

        return np.array([self.face_encoder.compute_face_descriptor(img, raw_landmark_set, num_jitters)
                         for raw_landmark_set in raw_landmarks])

    def face_distance(self, face_encodings, face_to_compare):
        """
        Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
        for each comparison face. The distance tells you how similar the faces are.
        :param faces: List of face encodings to compare
        :param face_to_compare: A face encoding to compare against
        :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
        """
        if len(face_encodings) == 0:
            return np.empty(0)

        return np.linalg.norm(face_encodings - face_to_compare, axis=1)

    def compare_faces(self, known_face_encodings, face_encoding_to_check):
        """
        Compare a list of face encodings against a candidate encoding to see if they match.
        :param known_face_encodings: A list of known face encodings
        :param face_encoding_to_check: A single face encoding to compare against the list
        :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
        """
        return self.face_distance(known_face_encodings, face_encoding_to_check)


# using dlib api
api = FaceDlibApi()

