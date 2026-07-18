from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models import Client
from app.schemas import ClientCreate, ClientOut

router = APIRouter(prefix="/clients", tags=["clients"])


@router.post("", response_model=ClientOut, status_code=201)
def create_client(payload: ClientCreate, db: Session = Depends(get_db)):
    settings = get_settings()
    client = Client(
        name=payload.name,
        rate_limit_per_minute=payload.rate_limit_per_minute or settings.default_rate_limit_per_minute,
    )
    db.add(client)
    db.commit()
    db.refresh(client)
    return client


@router.get("", response_model=list[ClientOut])
def list_clients(db: Session = Depends(get_db)):
    return db.query(Client).order_by(Client.id).all()
