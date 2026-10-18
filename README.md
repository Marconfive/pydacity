# Pydacity 🎙️

**Pydacity** is a personalized Python audio processing pipeline designed to analyze and clean recordings of university lectures and conferences (such as iPhone Voice Memos), transforming echoey and muffled room audio into crisp, intelligible speech.

### Key Features

- **Dual Noise Profiling (Automatic & Manual Selection):**
  - **Intelligent Auto-Detection:** Analyzes low-energy stationary segments to automatically isolate classroom hum and room murmur without manual intervention.
  - **Interactive Waveform Selector:** Renders an interactive waveform window (`Matplotlib` + `SpanSelector`) where you can visually select one or more noise-only intervals with the mouse.
- **Artifact-Free Spectral Denoising:** Employs calibrated spectral gating (`noisereduce`) to attenuate steady room noise, HVAC hum, and distant chatter while preserving vocal warmth.
- **Tailored Vocal DSP Chain (`pedalboard`):**
  - **High-Pass Filter (115 Hz):** Eliminates low-frequency desk vibrations, mic bumps, and air conditioning rumble.
  - **Dynamic Compressor:** Smooths out speech dynamics by lifting quiet explanations and taming sudden volume spikes.
  - **Speech-Optimized Parametric EQ:** Carves out muddy room frequencies (300 Hz & 800 Hz) and boosts consonant presence (3 kHz) for improved intelligibility.
  - **Gentle Low-Pass Filter (8 kHz):** Eliminates harsh high-frequency hiss.
- **Universal Input & MP3 Export:** Seamlessly reads any common format (`.m4a`, `.mp3`, `.wav`, `.flac`, `.ogg`, `.aac`, `.opus`) with automatic FFmpeg fallback, and exports lightweight, high-fidelity VBR MP3s via LAME.
- **Flexible Workflow & Batch Processing:** Execute via an interactive step-by-step wizard (with Tkinter file/folder dialogs), CLI arguments, or JSON config files.