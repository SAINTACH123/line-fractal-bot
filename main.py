from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, ImageMessage, TextSendMessage
import tempfile
import cv2
import numpy as np
import os

app = Flask(__name__)

# โหลดค่าจาก Environment Variables
LINE_CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET')

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

def fractal_dimension(image_path):
    img = cv2.imread(image_path, 0)
    if img is None:
        raise ValueError("ไม่สามารถเปิดภาพได้")

    img = cv2.resize(img, (512, 512))
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    Z = binary / 255

    sizes = 2 ** np.arange(1, int(np.log2(min(Z.shape))), 1)

    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
            np.arange(0, Z.shape[1], k), axis=1)
        return len(np.where(S > 0)[0])

    counts = [boxcount(Z, size) for size in sizes]
    coeffs = np.polyfit(np.log(1/sizes), np.log(counts), 1)
    return -coeffs[0]

def crack_severity(fd):
    if fd < 1.2:
        return "🟢 รอยแตกร้าวระดับเล็ก (Minor)"
    elif fd < 1.5:
        return "🟡 รอยแตกร้าวปานกลาง (Moderate)"
    else:
        return "🔴 รอยแตกร้าวรุนแรง (Severe)"

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers.get('X-Line-Signature')
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except Exception as e:
        print("Webhook error:", e)
        abort(400)

    return 'OK'

@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    message_content = line_bot_api.get_message_content(event.message.id)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tf:
        for chunk in message_content.iter_content():
            tf.write(chunk)
        temp_image_path = tf.name

    try:
        fd = fractal_dimension(temp_image_path)
        level = crack_severity(fd)
        reply = f"✅ วิเคราะห์สำเร็จ\n📈 Fractal Dimension: {fd:.3f}\n📊 ระดับความรุนแรง: {level}"
    except Exception as e:
        reply = f"❌ เกิดข้อผิดพลาด: {str(e)}"
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))

@app.route("/", methods=["GET"])
def home():
    return "LINE Fractal Bot is running."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
