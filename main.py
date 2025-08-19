from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import os
import inference
import cv2
import time


app = FastAPI()

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


# MODEL
os.environ["ROBOFLOW_API_KEY"] = "fXcN6XgFI4bggMDMzPHY"
model = inference.get_model("yolov5-dem-jwkci/1")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def create_class_folders(classes):
    for cls in classes:
        os.makedirs(f"detected_signs/{cls}", exist_ok=True)
unique_classes = ['hello','thankyou','yes','no','love']
create_class_folders(unique_classes)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            frame = await websocket.receive_text()
            image_data = frame.split(",")[1] 

            try:
                # Decode base64 image
                image_bytes = base64.b64decode(image_data)
                nparr = np.frombuffer(image_bytes, np.uint8)
                cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # Perform inference using Roboflow
                results = model.infer(image=image_bytes)
                predictions = results[0].predictions
                
                processed_results = {
                    "detections": [],
                    "confidence_threshold": 0.75
                }
                
                for pred in predictions:
                    if pred.confidence > 0.75:
                        # Extract bounding box info
                        x, y, width, height = int(pred.x), int(pred.y), int(pred.width), int(pred.height)
                        x = int(pred.x)
                        y = int(pred.y)
                        w = int(pred.width)
                        h = int(pred.height)
            
                        x1 = x - w // 2
                        y1 = y - h // 2
                        x2 = x + w // 2
                        y2 = y + h // 2
        
                        label = pred.class_name
                        confidence = pred.confidence
        
                        cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(cv_image, f"{label} ({confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                        classname = pred.class_name
                        
                        # Generate unique filename
                        annotated_filename = f"static/detected_signs/{classname}/{classname}_{int(time.time() * 1000)}.jpg"
                        cv2.imwrite(annotated_filename, cv_image)

                        processed_results['detections'].append({
                            "class": pred.class_name,
                            "confidence": pred.confidence,
                            "bbox": {
                                "x": x,
                                "y": y,
                                "width": width,
                                "height": height
                            }
                        })

                await websocket.send_json(processed_results)
            except Exception as inference_error:
                print(f"Inference Error: {inference_error}")
                await websocket.send_json({"error": str(inference_error)})

    except WebSocketDisconnect:
        print("WebSocket connection closed")
    except Exception as e:
        print(f"WebSocket Error: {e}")
        await websocket.close()
