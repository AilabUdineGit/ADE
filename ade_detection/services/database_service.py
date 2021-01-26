#!/usr/bin/env python3
# coding: utf-8

'''DB manager based on SQL Alchemy engine'''

__author__ = None
__version__ = '1.0'
__license__ = None
__copyright__ = None

import json
import logging

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool

import ade_detection.utils.localizations as loc
from ade_detection.services import Base as BASE
from ade_detection.utils.env import Env

LOGGER = logging.getLogger(__name__)


class DatabaseService():

    '''Exposes some primitives for the DB access'''

    __engine: object

    def __init__(self):
        try:
            # Disable connection pooling
            if Env.get_value(Env.DB) == Env.INTEGRATION_TEST_DB:
                db_path = loc.INTEGRATION_TEST_DB_CONNECTION_STRING  
            elif Env.get_value(Env.DB) == Env.TEST_DB:
                db_path = loc.TEST_DB_CONNECTION_STRING  
            else:
                db_path = loc.DB_CONNECTION_STRING
    
            self.__engine = create_engine(db_path, poolclass=NullPool)
        except Exception as error:
            LOGGER.critical(loc.FAIL_ENGINE_CREATION_EXCEPTION, error)
        self.create_all()


    def get_engine(self) -> Engine:
        '''Return the current SQL Alchemy engine instance'''
        return self.__engine


    def new_session(self) -> Session:
        '''Create a new DB session'''
        Session = sessionmaker(bind=self.__engine)
        return Session()


    def create(self, tables) -> None:
        '''Create the tables (tables can be ORM entity instances or classes)'''
        BASE.metadata.create_all(self.__engine, tables=[
                                 table.__table__ for table in tables])


    def create_all(self) -> None:
        '''Create all tables'''
        BASE.metadata.create_all(self.__engine)


    def drop(self, tables) -> None:
        '''Drop the tables (table can be ORM entity instances or classes)'''
        BASE.metadata.drop_all(self.__engine, tables=[
                               table.__table__ for table in tables])