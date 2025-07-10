import asyncio
import edge_tts

# 要朗读的文本
text = "你好，我是你的语音助手。现在是下午四点。"

# 选择语音（你可以换成 'zh-CN-XiaoxiaoNeural'）
voice = "zh-CN-XiaoxiaoNeural"

# 设置说话速度（100% 正常，-50% 变慢，+50% 变快）
rate = "+0%"

async def speak():
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.play()

# 异步运行
asyncio.run(speak())
