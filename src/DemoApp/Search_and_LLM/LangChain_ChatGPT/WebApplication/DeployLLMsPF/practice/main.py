from fastapi import FastAPI

app = FastAPI()

# @app.get("/sample")
# def read_root():
#     return {"テスト"}

@app.get("/items/{item_id}")
def read_item(item_id):
    return {"item_id": item_id, "item_name": "T-shirts"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
    # uvicorn.run(app, host="127.0.0.1", port=8000)