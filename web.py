# -*- coding: utf-8 -*-

import json
import base64
import cv2
import tempfile
import os
import numpy as np
import cStringIO as StringIO
import uuid

from PIL import Image
from twisted.web.resource import Resource
from twisted.web import server, static

from db import storage
from face import api


class FaceController(Resource):
    isLeaf = False

    def getChild(self, path, request):
        if path == '':
            return self
        return Resource.getChild(self, path, request)


class NewEmployeeController(Resource):
    isLeaf = True

    def render(self, request):
        request.setHeader('Access-Control-Allow-Origin', '*')
        request.setHeader('Access-Control-Allow-Methods', 'POST')
        request.setHeader('Access-Control-Allow-Headers', 'x-prototype-version,x-requested-with')

        if request.method == 'POST':
            # get image data from request parameter
            img_datas = request.args.get('avatar')

            if img_datas is not None and len(img_datas) > 0:
                img = dataurl2img(img_datas[0], True)

                # detect faces
                face_locations = api.detect_faces(img)

                if len(face_locations) == 0:
                    return json.dumps({'errno': 101, 'message': 'No face detected'})

                if len(face_locations) > 1:
                    return json.dumps({'errno': 102, 'message': 'More than one face detected'})

                # 128-dimensional face features for each face
                face_encodings = api.face_encodings(img, known_face_locations=face_locations)
                if len(face_encodings) == 0:
                    return json.dumps({'errno': 103, 'message': 'Fail to get the face encodings'})

                # change numpy.ndarray to bytes
                f = StringIO.StringIO()
                np.save(f, face_encodings[0])
                f.seek(0)

                if storage.new_employee(request.args, f.read()):
                    return json.dumps({'errno': 0})
                else:
                    return json.dumps({'errno': 300, 'message': 'Internal error'})
            else:
                return json.dumps({'errno': 100, 'message': 'No avatar found'})


class ChangeAvatarController(Resource):
    isLeaf = True

    def render(self, request):
        request.setHeader('Access-Control-Allow-Origin', '*')
        request.setHeader('Access-Control-Allow-Methods', 'POST')
        request.setHeader('Access-Control-Allow-Headers', 'x-prototype-version,x-requested-with')

        if request.method == 'POST':
            # get image data from request parameter
            img_datas = request.args.get('img')

            if img_datas is not None and len(img_datas) > 0:
                img = dataurl2img(img_datas[0], True)

                # detect faces
                face_locations = api.detect_faces(img)

                if len(face_locations) == 0:
                    return json.dumps({'errno': 101, 'message': 'No face detected'})

                if len(face_locations) > 1:
                    return json.dumps({'errno': 102, 'message': 'More than one face detected'})

                # 128-dimensional face features for each face
                face_encodings = api.face_encodings(img, known_face_locations=face_locations)
                if len(face_encodings) == 0:
                    return json.dumps({'errno': 103, 'message': 'Fail to get the face encodings'})

                # change numpy.ndarray to bytes
                f = StringIO.StringIO()
                np.save(f, face_encodings[0])
                f.seek(0)

                if storage.change_avatar(request.args, f.read()):
                    return json.dumps({'errno': 0})
                else:
                    return json.dumps({'errno': 300, 'message': 'Internal error'})
            else:
                return json.dumps({'errno': 100, 'message': 'No avatar found'})


def dataurl2img(data_url, save_temp_file=False):
    img_data_header = 'data:image/jpeg;base64,'

    img_data = base64.b64decode(data_url[len(img_data_header):])

    # get file_like object
    img_file = StringIO.StringIO()

    # write data
    img_file.write(img_data)
    img_file.seek(0)

    # convert file_like object to array_like object
    img = Image.open(img_file)

    buf = np.asarray(img)

    if save_temp_file:
        # RGB to BGR
        temp = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
        # write to temp file
        cv2.imwrite(os.path.join(tempfile.gettempdir(), str(uuid.uuid1()) + '.jpeg'), temp)

    return buf.copy()


def get_site(root_dir=os.path.expandvars('mego')):
    root = static.File(root_dir)

    # init controllers
    face_controller = FaceController()
    face_controller.putChild('new_employee', NewEmployeeController())
    face_controller.putChild('change_employee_avatar', ChangeAvatarController())

    root.putChild("face", face_controller)

    site = server.Site(root)
    return site
