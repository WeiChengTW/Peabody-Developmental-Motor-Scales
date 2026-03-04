from flask import Flask, Response, render_template
import cv2
from main import camera
import threading

app = Flask(__name__)

# 啟動攝影機串流的執行緒
camera_thread = threading.Thread(target=camera.run)
camera_thread.daemon = True
camera_thread.start()


@app.route("/")
def index():
    return render_template("main.html")


def generate_frames():
    while True:
        try:
            frame = camera.get_frame()
            if frame is not None:
                ret, buffer = cv2.imencode(".jpg", frame)
                if ret:
                    frame = buffer.tobytes()
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                    )
        except Exception as e:
            print(f"Error generating frame: {e}")
            continue


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
