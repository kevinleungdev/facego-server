# -*- coding: utf-8 -*-

import base64
import cStringIO as StringIO
import json
import sys
import click
import uuid
import numpy as np
import conf
import cPickle as pickle

from twisted.internet import ssl
from autobahn.twisted.websocket import WebSocketServerFactory, WebSocketServerProtocol, listenWS
from PIL import Image
from skimage import io

from db import storage
from face import api
from web import get_site

import txaio
txaio.use_twisted()


# load classifier model
classifier_filename = conf.get_prop('classifier', 'model_location')
with open(classifier_filename, 'rb') as infile:
    (classifier_model, classes) = pickle.load(infile)
    print 'Loaded classifier model from file "%s"' % classifier_filename

# load all employees info [ employee_no, employee_fullname ]
all_employees = storage.load_all_employees()
print 'There are %d employees in total' % len(all_employees)


class Session:

    def __init__(self, meeting_id, tolerance=0.6):
        """
        :param meeting_id: the unique identity of meeting
        :type meeting_id: int
        """
        assert meeting_id is not None
        self.meeting_id = meeting_id

        self.tolerance = tolerance

        # load all participants of a meeting
        self.attendants = storage.load_attendants_by_meeting_id(meeting_id)

        # generate unique session id
        self.id = str(uuid.uuid1())

        print("Meeting session created! session_id: {}, meeting_id: {}, participants: {}"
              .format(self.id, self.meeting_id, self.attendants))


class FaceServerProtocol(WebSocketServerProtocol):

    def __init__(self):
        super(FaceServerProtocol, self).__init__()

        # meeting sessions
        self.sessions = {}

    def onConnect(self, request):
        print('Client connecting: {}'.format(request.peer))

    def onOpen(self):
        print('WebSocket connection open.')

    def onMessage(self, payload, is_binary):
        raw = payload.decode('utf8')
        msg = json.loads(raw)

        print("Received {} message of length {}.".format(msg['type'], len(raw)))

        if msg['type'] == 'OPEN':
            self.open_session(msg['meeting_id'])
        elif msg['type'] == 'CLOSE':
            self.close_session(msg['session_id'])
        elif msg['type'] == 'PROCESSING':
            self.process_frame(msg['session_id'], msg['data_url'])
        else:
            print('Unknown type: {}', msg['type'])

    def onClose(self, was_clean, code, reason):
        print('WebSocket connection closed {}.'.format(reason))

    def open_session(self, meeting_id):
        """
        open a meeting session
        :param meeting_id: identity of meeting
        :return:
        """
        if meeting_id not in self.sessions:
            # create a new meeting session
            sess = Session(meeting_id)

            # cache current session
            self.sessions[sess.id] = sess

        # send to client
        self.sendMessage(json.dumps({
            'type': 'OPENED',
            'message': 'The session of meeting {} is opened'.format(meeting_id),
            'session_id': sess.id})
        )

    def close_session(self, sess_id):
        """
        Close a meeting session
        :param sess_id: session identity of meeting
        :return:
        """
        if sess_id in self.sessions:
            self.sessions.pop(sess_id)
            print('remove session[{}]'.format(sess_id))
        else:
            print('session[{}] not found')

        self.sendMessage(json.dumps({
            'type': 'CLOSED',
            'message': 'Session[{}] removed on server'.format(sess_id)
        }))

    def process_frame(self, sess_id, data, save_data=False):
        """
        faces recognition
        :param sess_id: session identifier
        :param data: data_url
        :param save_data: save data on the disk if true
        :return:
        """
        assert sess_id is not None
        assert data is not None

        if sess_id not in self.sessions:
            print("process frame failed! Because session {} not found.")
            return

        sess = self.sessions[sess_id]
        attendants = sess.attendants

        # data_url head
        data_url_head = 'data:image/jpeg;base64,'

        # deserialize from data_url
        file_like = StringIO.StringIO(base64.b64decode(data[len(data_url_head):]))

        # convert file_like object to array_like object
        img = np.asarray(Image.open(file_like)).copy()

        if save_data:
            import os
            import tempfile

            filename = os.path.join(tempfile.gettempdir(), str(uuid.uuid1()) + ".jpg")
            io.imsave(filename, img)

        # nos, labels, locations and scores
        attendants_detected = []

        know_face_locations = api.detect_faces(img)
        if len(know_face_locations) == 0:
            print 'No face detected in the current frame'
        else:
            embeddings = api.face_encodings(img, known_face_locations=know_face_locations)

            predictions = classifier_model.predict_proba(embeddings)

            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
            for i in range(len(best_class_indices)):
                employee_no = classes[best_class_indices[i]]

                if employee_no == '-1':
                    # unknown person
                    attendants_detected.append({
                        'id': -1,
                        'no': -1,
                        'name': 'Unknown',
                        'score': best_class_probabilities[i],
                        'face_location': know_face_locations[i],
                        'is_attendant': False
                    })
                else:
                    label = all_employees.get(employee_no, 'No name')
                    print '%s %s: %.3f' % (employee_no, label, best_class_probabilities[i])

                    e = all_employees.get(employee_no)
                    if e is None:
                        print 'Employee `{}` not found'.format(employee_no)
                        continue
                    else:
                        attendants_detected.append({
                            'id': e['id'],
                            'no': employee_no,
                            'name': e['fullname'],
                            'english_name': e['english_name'],
                            'score': best_class_probabilities[i],
                            'face_location': know_face_locations[i],
                            'is_attendant': e['id'] in attendants
                        })

        # send the reply to client
        self.sendMessage(json.dumps({
            'type': 'PROCESSED',
            'data': attendants_detected
        }), isBinary=False)


class FaceServerFactory(WebSocketServerFactory):
    protocol = FaceServerProtocol


@click.command()
@click.option('--iface', default='127.0.0.1',
              help='Listen on interface when tcp client connection coming. Default is \'127.0.0.1\'')
@click.option('--port', default=9000,
              help='Listen on port when tcp client connection coming. Default is 9000')
@click.option('--web-port', default=8080, help='Web server port')
@click.option('--web-dir', default=None, help='Web site directory')
@click.option('--enable-ssl/--disable-ssl', default=False)
@click.option('--ssl-key', default=None)
@click.option('--ssl-crt', default=None)
def main(iface, port, web_port, web_dir, enable_ssl, ssl_key, ssl_crt):

    from twisted.python import log

    # start logging
    log.startLogging(sys.stdout)

    # choose the "best" available Twisted reactor
    from autobahn.twisted.choosereactor import install_reactor

    reactor = install_reactor()
    print("Running reactor on {}".format(reactor))

    # get web site
    site = get_site(web_dir)

    if enable_ssl:
        import os

        if ssl_key is None or not os.path.exists(ssl_key):
            raise Exception("ssl key file not found: {}" % ssl_key)

        if ssl_crt is None or not os.path.exists(ssl_crt):
            raise Exception("ssl certificate not found: {}" % ssl_crt)

        # construct ssl context factory
        ssl_factory = ssl.DefaultOpenSSLContextFactory(ssl_key, ssl_crt)

        # construct web factory
        ws_factory = FaceServerFactory("wss://{}:{}".format(iface, port))
        ws_factory.setProtocolOptions(allowedOrigins="*")

        # listen on ssl connection
        listenWS(ws_factory, ssl_factory)

        # listen on web ssl connection
        reactor.listenSSL(web_port, site, ssl_factory)
        print('Secure web server listen on {}: {}'.format(web_port, web_dir))
    else:
        # start a WebSocket server
        ws_factory = FaceServerFactory("ws://{}:{}".format(iface, port))
        ws_factory.setProtocolOptions(allowedOrigins="*")

        # listen on port when tcp client connection coming
        reactor.listenTCP(port, ws_factory)

        # listen on web port
        reactor.listenTCP(web_port, site)
        print('Web server listen on {}: {}'.format(port, web_dir))

    # start reactor
    print 'Staring the reactor.'
    reactor.run()


if __name__ == '__main__':
    main(sys.argv[1:])

