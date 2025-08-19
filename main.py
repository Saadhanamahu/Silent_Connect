from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import base64
from io import BytesIO
from PIL import Image
import numpy as np

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Index page
@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Learning page
@app.get("/learning", response_class=HTMLResponse)
async def learning_page(request: Request):
    return templates.TemplateResponse("learning.html", {"request": request})

# Practice page
@app.get("/practice", response_class=HTMLResponse)
async def practice_page(request: Request):
    return templates.TemplateResponse("practice.html", {"request": request})

# Testing page
@app.get("/testing", response_class=HTMLResponse)
async def testing_page(request: Request):
    return templates.TemplateResponse("testing.html", {"request": request})


#Websocket
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            frame = await websocket.receive_text()

            image_data = frame.split(",")[1]  
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))

            img_array = np.array(image)
            #MODEL WILL BE INTEGRATED HERE(numpy array cause YOLO?)!!!

            print("Received and processed a frame")

            # ONLY FOR TESTING REMOVE LATER!!!
            await websocket.send_text("Frame processed")

    except WebSocketDisconnect:
        print("Client disconnected")