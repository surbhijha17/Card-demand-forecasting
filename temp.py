import asyncio
from six.moves import queue  # You can replace this with `asyncio.Queue`
import numpy as np
from fastapi.websockets import WebSocketDisconnect
from app.utils.language_support.whisper_enums import code_to_language_enum
import torch
import json
import time
import uuid
from app.utils.logging import logger
from app.core.db.services.transcriber import MediaTranslationLiveModelCreateSchema

class WhisperTranscoder(object):
    def __init__(self, model, websocket, media_translation_service):
        self.start_time = time.time()
        self.buff = asyncio.Queue()  # Changed to asyncio.Queue
        self.closed = True
        self.transcript = None
        self.whisper_model = model
        self.language_flag = False
        self.language_detection_chunk = np.array([], dtype=np.int16)
        self.final_transcript = ''
        self.transcribed_language = ''
        self.request_id = str(uuid.uuid4())
        self.media_translation_service = media_translation_service
        self.websocket = websocket
        self.process_task = None  # Task for processing transcription

    async def start(self):
        # Use asyncio.create_task to run process concurrently
        self.process_task = asyncio.create_task(self.process())

    async def detect_language(self):
        if len(self.language_detection_chunk) >= 16000 * 10:
            print("Detecting language...")
            segments, info = self.whisper_model.transcribe(self.language_detection_chunk)
            try:
                self.transcribed_language = code_to_language_enum[info.language].value
            except Exception:
                self.transcribed_language = info.language
                logger.warning("New language added into whisper support, Whisper Enum update required")
            text = ''
            for segment in segments:
                text = text + ' ' + segment.text
            self.transcript = text
            self.language_flag = True
            self.final_transcript += self.transcript

    async def process(self):
        audio_generator = self.stream_generator()
        print("Inside process")
        async for content in audio_generator:
            audio_data = np.frombuffer(content, dtype=np.int16)
            if not self.language_flag and len(self.language_detection_chunk) < 16000 * 10:
                self.language_detection_chunk = np.append(self.language_detection_chunk, audio_data)
                await self.detect_language()  # Language detection should also be async
                print("Inside language chunk")
            else:
                print("Inside transcription loop")
                segments, _ = self.whisper_model.transcribe(audio_data)
                text = ''
                for segment in segments:
                    text = text + segment.text
                self.transcript = text
                self.final_transcript += self.transcript
                print("Transcription completed:", time.time())

    async def stream_generator(self):
        while not self.closed:
            chunk = await self.buff.get()  # Non-blocking async queue get
            if chunk is None:
                return
            data = [chunk]
            while True:
                try:
                    chunk = self.buff.get_nowait()  # Non-blocking async queue get_nowait
                    if chunk is None:
                        break
                    data.append(chunk)
                except asyncio.QueueEmpty:
                    break
            yield b''.join(data)

    async def write(self, data):
        await self.buff.put(data)  # Non-blocking async put
        print("Chunk received: ", time.time())

    async def whisper_transcribe(self):
        try:
            while True:
                data = await self.websocket.receive_bytes()
                await self.write(data)  # Write the received data asynchronously
                if self.closed:
                    self.closed = False
                    await self.start()  # Start the transcription process

                if self.language_flag:
                    await self.websocket.send_json({'type': 'language', 'language': self.transcribed_language})
                    print(f"Language Detected: {self.transcribed_language}")
                    self.language_flag = None

                if self.transcript:
                    print(self.transcript)
                    await self.websocket.send_json({'type': 'transcript', 'transcript': self.transcript})
                    self.transcript = None
        except WebSocketDisconnect:
            self.closed = True
            self.buff = asyncio.Queue()  # Reset the buffer
            await self.media_translation_service.create(
                MediaTranslationLiveModelCreateSchema(
                    uuid=self.request_id,
                    duration=time.time() - self.start_time,
                    audio_language=self.transcribed_language,
                    service_type='Whisper',
                    token_length=len(self.final_transcript)
                )
            )
            await self.media_translation_service.session.commit()
            logger.warning(f"Client: {self.request_id} Disconnected Unexpectedly")
        except Exception as e:
            logger.warning(f"Error: {e}")
