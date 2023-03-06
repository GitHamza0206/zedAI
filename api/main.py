from fastapi import FastAPI,Depends,HTTPException,status,File, UploadFile 
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext

import os
import sys 


sys.path.append('../notebooks/001-hello-world')
sys.path.append('../notebooks/205-vision-background-removal') 
### AI MODULES ###
import hello_world as model001
import background_removal as model205 

# Open a file



SECRET_KEY="998ce96ac4d4acd89eadcb13e9d05ff4feba8b6b650caeee116182c531f00ad6" ##run command : openssl rand -hex 32
ALGORITHM="HS256"
ACCESS_TOKEN_EXPIRE_MINUTES=30
 
app=FastAPI()
 
db = {
    "hamza":{
        "username":"hamza",
        "full_name":"hamza zerouali",
        "email": "hamzazerouali@gmail.com",
        "hashed_password":"$2b$12$RU63YdFIcPb7ESVISUTfxu9eI/oLmfDxhIUeB9VDL.ONq32C.QOlK",
        "disabled":False
    }

}

class Token(BaseModel):
    access_token:str
    token_type:str

class TokenData(BaseModel):
    username:str or None=None

class User(BaseModel):
    username:str
    email:str or None=None
    full_name:str or None=None
    disabled:bool or None=None

class UserInDB(User):
    hashed_password:str

class HelloDetectionModel(BaseModel):
    classname : str 
    precision : float

class BgRemovalModel(BaseModel):

    mask : bytes
    result : bytes 
    

pwd_context = CryptContext(schemes=['bcrypt'], deprecated="auto")
oauth_2_scheme=OAuth2PasswordBearer(tokenUrl="token")

 

def verify_password(plain_password,hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password) 

def get_user(db, username: str):
    if username in db:
        user_data = db[username]
        return UserInDB(**user_data) # **user_data c'est comme is on avait mis : username="hamza", email=hamza@gmail.com etc...

def authenticate_user(db,username:str, password:str):
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data:dict, expires_delta:timedelta or None = None):
    to_encode=data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)

    to_encode.update({
        "exp": expire
    })
    encoded_jwt=jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

 

async def get_current_user(token:str=Depends(oauth_2_scheme)) :
    credential_excpetion= HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials", headers={"WWW-Authenticate":'Bearer'})
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get('sub')
        if username is None:
            raise credential_excpetion
        token_data = TokenData(username=username)
    except JWTError:
        raise credential_excpetion
    user = get_user(db, username=token_data.username)
    if user is None:
        raise credential_excpetion
    return user

 

async def get_current_active_user(current_user:UserInDB=Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400,detail="Inactive user")
    return current_user

 

@app.post('/token', response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm=Depends()):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user :
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password", headers={"WWW-Authenticate":'Bearer'})
    access_token_expires= timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token=create_access_token(data={"sub": user.username},expires_delta=access_token_expires)
    return{"access_token":access_token, "token_type":"bearer"}

 

@app.get('/users/me',response_model=User)
async def read_users_me(current_user: User=Depends(get_current_active_user)) :
    return current_user

   

@app.get('/users/me/items')
async def read_own_items(current_user: User=Depends(get_current_active_user)) :
    return [{"item_id":1,"owner": current_user}]

@app.post('/hello-detection', response_model=HelloDetectionModel)
async def hello_detection(input_file: UploadFile):

    return model001.predict(input_file.file)


@app.post('/background-removal')
async def background_removal(input_file: UploadFile): 
    return model205.predict(input_file.file)



## run command
#uvicorn main:app --reload