from typing import Optional

class GenericException(Exception):
    def __init__(
        self, name: str, message: Optional[str] = None, info: Optional[str] = None
    ):
        self.name = name
        self.message = message
        self.info = info


class NotFoundException(Exception):
    def __init__(
        self, name: str, message: Optional[str] = None, info: Optional[str] = None
    ):
        self.name = name
        self.message = message
        self.info = info


class ImportException(ImportError):
    def __init__(
        self, name: str, message: Optional[str] = None, info: Optional[str] = None
    ):
        self.name = name
        self.message = message
        self.info = info


class ForbiddenException(Exception):
    def __init__(
        self, name: str, message: Optional[str] = None, info: Optional[str] = None
    ):
        self.name = name
        self.message = message
        self.info = info


class NotImplementedException(GenericException):
    pass


class PathNotExistsException(NotFoundException):
    pass


class PathEmptyException(NotFoundException):
    pass
