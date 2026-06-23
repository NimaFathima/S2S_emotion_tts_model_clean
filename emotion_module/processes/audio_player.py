"""
audio_player.py
Signet Aid — Process 3: Audio Hardware Playback

Non-blocking double-buffered audio playback via sounddevice.
Receives numpy audio chunks from audio_consumer via audio_queue.
Plays each chunk immediately as it arrives (streaming playback).
Handles silence gracefully when queue is empty.
"""

def audio_player(audio_queue, run_flag):
    """
    Process 3: Audio Player.
    Streams audio chunks from audio_queue to the speaker in real time.
    Uses sounddevice non-blocking output with a double buffer.
    """
    import time, logging
    import numpy as np

    log = logging.getLogger("AudioPlayer")

    try:
        import sounddevice as sd
    except ImportError:
        log.error("sounddevice not installed. Run: pip install sounddevice")
        return

    SAMPLE_RATE    = 24000      # Chatterbox output sample rate
    SILENCE_CHUNK  = np.zeros(int(SAMPLE_RATE * 0.02), dtype=np.float32)
    # 20ms silence buffer — prevents click artifacts during queue underrun

    log.info("Audio player started.")

    while run_flag.value:

        if audio_queue.empty():
            # Play silence to keep the audio stream warm
            # Prevents click when audio resumes
            sd.play(SILENCE_CHUNK, samplerate=SAMPLE_RATE, blocking=False)
            time.sleep(0.015)
            continue

        chunk = audio_queue.get()

        # None sentinel signals end of one sentence — brief pause
        if chunk is None:
            time.sleep(0.05)   # 50ms natural pause between sentences
            continue

        if not isinstance(chunk, np.ndarray) or len(chunk) == 0:
            continue

        # Ensure correct dtype and shape
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)
        if chunk.ndim > 1:
            chunk = chunk.squeeze()

        # Clip to prevent clipping distortion
        chunk = np.clip(chunk, -1.0, 1.0)

        # Non-blocking play — returns immediately while audio plays in background
        # sd.wait() is NOT called — next chunk starts before this one finishes
        # This creates a seamless double-buffer effect
        sd.play(chunk, samplerate=SAMPLE_RATE, blocking=False)

        # Wait approximately the duration of this chunk before fetching next
        # This prevents audio_queue reads from racing ahead of playback
        chunk_duration = len(chunk) / SAMPLE_RATE
        time.sleep(chunk_duration * 0.85)   # 85% of chunk duration — slight overlap

    # Clean shutdown
    sd.stop()
    log.info("Audio player shut down.")
