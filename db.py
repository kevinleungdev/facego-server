# -*- coding: utf-8 -*-
import cStringIO as StringIO
import numpy as np

from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, BLOB, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

import conf

default_db_host = conf.get_prop('db', 'host')
default_db_name = conf.get_prop('db', 'name')
default_db_user = conf.get_prop('db', 'user')
default_db_password = conf.get_prop('db', 'password')


class Storage(object):
    pass


Base = declarative_base()


class MeetingSchedule(Base):

    __tablename__ = 'MG_MEETING_SCHEDULE'

    s_id = Column('SCHEDULE_ID', Integer, autoincrement=True, primary_key=True)
    s_attendant_id = Column('ATTENDANTS', String)


class Employee(Base):

    __tablename__ = 'MG_EMPLOYEE'

    id = Column('EMPLOYEE_ID', Integer, autoincrement=True, primary_key=True)
    no = Column('EMPLOYEE_NO', String, unique=True)
    firstname = Column('FIRST_NAME', String)
    lastname = Column('LAST_NAME', String)
    engname = Column('ENGLISH_NAME', String)
    title = Column('JOB_TITLE', String)
    group = Column('GROUP', String)
    gender = Column('GENDER', Integer)
    email = Column('EMAIL', String)

    @property
    def fullname(self):
        return self.firstname + ' ' + self.lastname

    @fullname.setter
    def fullname(self, val):
        names = val.split(' ')

        if len(names) == 2:
            self.firstname = names[0]
            self.lastname = names[1]


class EmployeeInfo(Base):

    __tablename__ = 'MG_EMPLOYEE_INFO'

    employee_id = Column('EMPLOYEE_ID', Integer, primary_key=True)
    avatar = Column('AVATAR', Text)
    avatar_thumbnail = Column('AVATAR_THUMBNAIL', Text)


class EmployeeReps(Base):

    __tablename__ = 'MG_EMPLOYEE_REPS'

    employee_id = Column('EMPLOYEE_ID', Integer, primary_key=True)
    face_reps = Column('FACE_REPS', BLOB)


class DBStorage(Storage):

    def __init__(self, db_type='mysql', host='localhost', port=3306, user=None, passwd=None, db_name=None):
        """
        Database storage
        :param db_type: database type, e.g. 'mysql', 'sqlite' etc.
        :type db_type: str.
        :param host: db server host, default is 'localhost'.
        :type host: str.
        :param port: db server port, default is 3306.
        :type port: int.
        :param user: db user name.
        :type user: str.
        :param passwd: db user password.
        :type passwd: int.
        :param db_name: database name.
        :type db_name: str.
        """
        assert user is not None
        assert passwd is not None

        conn_url = '{}://{}:{}@{}:{}/{}'.format(db_type, user, passwd, host, port, '' if db_name is None else db_name)
        print 'Database connection url: ', conn_url

        self.engine = create_engine(conn_url)
        self.Session = sessionmaker(bind=self.engine)

    def load_all_employees(self):
        employees = {}

        sess = self.Session()
        for e in sess.query(Employee):
            employees[e.no] = { 'id': e.id, 'fullname': e.fullname, 'english_name': e.engname }

        return employees

    def load_attendants(self, meeting_id):
        """
        :param meeting_id: meeting id
        :return: all participants of a meeting
        :rtype: list(user_name), list(face_reps)
        """
        a_ids = []
        a_names = []
        a_reps = []

        sess = self.Session()
        # fetch one meeting schedule record
        rec = sess.query(MeetingSchedule).filter(MeetingSchedule.s_id == meeting_id).one()

        if rec is not None and rec.s_attendant_id is not None:

            for up, u in sess.query(EmployeeReps, Employee)\
                    .filter(EmployeeReps.employee_id == Employee.id)\
                    .filter(EmployeeReps.employee_id.in_(rec.s_attendant_id.split(','))):

                # append employee id
                a_ids.append(u.id)

                # append employee full name
                a_names.append(u.fullname)

                # change binary to numpy.ndarray
                f = StringIO.StringIO()
                f.write(up.face_reps)
                f.seek(0)

                a_reps.append(np.load(f))

        return a_ids, a_names, a_reps

    def load_attendants_by_meeting_id(self, meeting_id):
        """
        :param meeting_id: meeting id
        :return: all participants of a meeting
        :rtype: list(user_name), list(face_reps)
        """
        ids = []

        sess = self.Session()
        # fetch one meeting schedule record
        rec = sess.query(MeetingSchedule).filter(MeetingSchedule.s_id == meeting_id).one()

        if rec is not None and rec.s_attendant_id is not None:

            for u in sess.query(Employee)\
                    .filter(Employee.id.in_(rec.s_attendant_id.split(','))):
                # append employee
                ids.append(u.id)

        return ids

    def new_employee(self, params, reps):
        session = self.Session()

        try:
            employee = Employee(
                no=params.get('employee_no')[0],
                firstname=params.get('firstname')[0],
                lastname=params.get('lastname')[0],
                engname=params.get('engname')[0],
                title=params.get('title')[0],
                group=params.get('group')[0],
                gender=int(params.get('gender')[0]),
                email=params.get('email')[0]
            )
            session.add(employee)
            session.flush()

            employee_info = EmployeeInfo(employee_id=employee.id, avatar=params.get('avatar')[0])
            session.add(employee_info)

            employee_reps = EmployeeReps(employee_id=employee.id, face_reps=reps)
            session.add(employee_reps)

            session.commit()

            return True
        except Exception as e:
            print e
            session.rollback()

            return False
        finally:
            session.close()

    def change_avatar(self, params, reps):
        session = self.Session()

        try:
            eid = int(params.get('id')[0])

            session.query(EmployeeInfo)\
                .filter_by(employee_id=eid)\
                .update({'avatar': params.get('img')[0]})

            session.query(EmployeeReps)\
                .filter_by(employee_id=eid)\
                .update({'face_reps': reps})

            session.commit()

            return True
        except Exception as e:
            print e
            session.rollback()

            return False
        finally:
            session.close()


# Using databases storage
storage = DBStorage(db_name=default_db_name, host=default_db_host, user=default_db_user, passwd=default_db_password)