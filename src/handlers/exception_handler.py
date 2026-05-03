from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from domain.exceptions.global_exceptions import *


def add_exception_handlers(app: FastAPI):

    @app.exception_handler(GenericException)
    async def generic_exception_handler(request: Request, exc: GenericException):
        return JSONResponse(
            status_code=400,
            content={
                "Message": exc.name,
                "Details": exc.message,
                "Additional Info": exc.info,
            },
        )

    @app.exception_handler(NotFoundException)
    async def not_found_exception_handler(request: Request, exc: NotFoundException):
        return JSONResponse(
            status_code=404,
            content={
                "Message": exc.name,
                "Details": exc.message,
                "Additional Info": exc.info,
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):

        return JSONResponse(
            status_code=500,
            content={"Message": f"Error", "Details": str(exc)},
        )