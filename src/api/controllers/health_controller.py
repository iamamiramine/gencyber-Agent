"""
Endpoint for checking the health of the API services.

The prefix endpoint is '/health'
"""

from fastapi import APIRouter
from starlette.responses import Response

router = APIRouter()


@router.get("")
def health():
    """
    Health check endpoint that returns a 200 OK response if the API is running.
    
    This endpoint can be used by monitoring tools and load balancers to verify
    that the API service is up and responding to requests.

    Returns:
        Response: A Starlette Response object with status code 200
        
    Example:
        >>> response = health()
        >>> response.status_code
        200
    """
    return Response(status_code=200)