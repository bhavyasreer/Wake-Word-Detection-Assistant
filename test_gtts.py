try:
    from gtts import gTTS
    print("gTTS import SUCCESSFUL!")
except Exception as e:
    print("gTTS import FAILED:", e)