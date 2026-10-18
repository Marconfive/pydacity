#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline Automatica e Semi-Automatica per Pulizia Audio Lezioni Universitarie e Conferenze
Ottimizzata per registrazioni da iPhone (.m4a, .wav, .mp3, .flac) in aule affollate.

Dipendenze su Ubuntu 24.04:
    sudo apt update && sudo apt install -y ffmpeg python3-tk
    pip install numpy soundfile noisereduce pedalboard matplotlib
"""

import os
import sys
import json
import shutil
import tempfile
import subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
import noisereduce as nr
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from pedalboard import (
    Pedalboard,
    Compressor,
    HighpassFilter,
    LowpassFilter,
    PeakFilter,
)

# Estensioni audio supportate (incluso .m4a nativo di iPhone Memo Vocali)
SUPPORTED_EXTENSIONS = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".aac", ".mp4", ".opus"}


# ==============================================================================
# 1. LETTURA E SCRITTURA AUDIO (tramite FFmpeg + SoundFile)
# ==============================================================================

def load_audio_any_format(filepath: str):
    """
    Carica qualunque formato audio. Prova prima la lettura diretta con SoundFile;
    se il codec non è supportato da libsndfile (es. .ogg con codec Opus, .m4a, .mp3, .aac),
    usa automaticamente FFmpeg per decodificarlo in WAV float32 temporaneo.
    """
    try:
        audio, sr = sf.read(filepath, dtype="float32")
    except Exception:
        # Fallback universale tramite FFmpeg per .ogg (Opus), .m4a, .mp3, .aac o file rinominati
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_path = tmp_wav.name
        try:
            cmd = [
                "ffmpeg", "-y", "-v", "error",
                "-i", filepath,
                "-vn", "-acodec", "pcm_f32le",
                tmp_path
            ]
            subprocess.run(cmd, check=True)
            audio, sr = sf.read(tmp_path, dtype="float32")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # Converti in mono se la registrazione è stereo (ottimale per parlato/lezioni)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    return audio, sr


def export_to_mp3(audio: np.ndarray, sr: int, output_mp3_path: str, vbr_quality: int = 2):
    """
    Esporta l'array audio in formato MP3 usando l'encoder LAME tramite FFmpeg.
    - vbr_quality = 2 corrisponde allo standard LAME -V2 (bit rate variabile 170-210 kbps,
      qualità trasparente e leggerissima senza artefatti sulle consonanti).
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        tmp_wav_path = tmp_wav.name

    try:
        # Scriviamo prima in WAV float32 temporaneo
        sf.write(tmp_wav_path, audio, sr, subtype="FLOAT")

        # Scriviamo su un MP3 temporaneo per sicurezza (nel caso input e output coincidano)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
            tmp_mp3_path = tmp_mp3.name

        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", tmp_wav_path,
            "-codec:a", "libmp3lame",
            "-qscale:a", str(vbr_quality),  # -q:a 2 => range standard 170-210 kbps
            tmp_mp3_path
        ]
        subprocess.run(cmd, check=True)
        shutil.move(tmp_mp3_path, output_mp3_path)
    finally:
        if os.path.exists(tmp_wav_path):
            os.remove(tmp_wav_path)


# ==============================================================================
# 2. SELEZIONE PROFILO RUMORE (AUTOMATICA PER AULA O MANUALE CON SPANSELECTOR)
# ==============================================================================

def find_classroom_noise_auto(audio: np.ndarray, sr: int, clip_duration: float = 1.2, top_k: int = 3):
    """
    Ricerca automatica ottimizzata per registrazioni iPhone in aula universitaria:
    - Divide l'audio in finestre da `clip_duration` secondi.
    - Scarta i silenzi digitali assoluti o i click iniziali/finali (primi e ultimi 2 secondi).
    - Valuta sia il basso volume (RMS) sia la stazionarietà interna della finestra
      (evita di selezionare per sbaglio colpi di tosse o bisbigli forti isolati).
    - Unisce i migliori `top_k` segmenti di brusio di fondo per creare un profilo solido.
    """
    win_len = int(clip_duration * sr)
    step = int(0.25 * sr)
    margin = int(2.0 * sr)

    start_idx = margin if len(audio) > 4 * margin else 0
    end_idx = len(audio) - margin if len(audio) > 4 * margin else len(audio)

    if end_idx - start_idx <= win_len:
        return audio

    candidates = []
    global_rms = np.sqrt(np.mean(audio ** 2)) + 1e-9
    min_valid_rms = global_rms * 0.01  # Scarta muting digitale totale

    for s in range(start_idx, end_idx - win_len, step):
        seg = audio[s:s + win_len]
        rms = np.sqrt(np.mean(seg ** 2))
        if rms < min_valid_rms:
            continue

        # Dividiamo la finestra in 4 sotto-blocchi per verificare che sia brusio costante
        sub_blocks = np.array_split(seg, 4)
        sub_rms = [np.sqrt(np.mean(b ** 2)) for b in sub_blocks]
        crest_stability = np.std(sub_rms) / (rms + 1e-9)

        # Punteggio combinato: cerchiamo bassa energia + alta stabilità (brusio costante)
        score = rms * (1.0 + 1.5 * crest_stability)
        candidates.append((score, s))

    if not candidates:
        return audio[:win_len]

    candidates.sort(key=lambda x: x[0])

    # Prendiamo i top_k segmenti non sovrapposti
    selected_segments = []
    used_indices = []
    for _, s in candidates:
        if all(abs(s - u) >= win_len for u in used_indices):
            selected_segments.append(audio[s:s + win_len])
            used_indices.append(s)
            if len(selected_segments) >= top_k:
                break

    return np.concatenate(selected_segments)


def select_noise_intervals_gui(audio: np.ndarray, sr: int, filename: str):
    """
    Apre una finestra grafica leggera con Matplotlib e SpanSelector.
    Permette di selezionare col mouse 1, 2 o più intervalli di rumore sulla forma d'onda.
    - Trascina col tasto sinistro del mouse per selezionare un intervallo.
    - Premi 'r' sulla tastiera per resettare le selezioni se hai sbagliato.
    - Premi 'Invio' o chiudi la finestra per confermare e avviare il denoise.
    """
    # Sottocampioniamo solo per il disegno a schermo (max ~150.000 punti) -> zero lag!
    max_plot_points = 150_000
    step = max(1, len(audio) // max_plot_points)
    plot_audio = audio[::step]
    time_axis = np.arange(len(plot_audio)) * (step / sr)

    selected_intervals = []
    highlight_patches = []

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.canvas.manager.set_window_title(f"Selezione Rumore - {filename}")
    ax.plot(time_axis, plot_audio, color="#2b5c8f", linewidth=0.6, alpha=0.85)
    ax.set_xlim(0, time_axis[-1])
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Tempo (secondi)", fontsize=11)
    ax.set_ylabel("Ampiezza", fontsize=11)

    title_text = (
        f"File: {filename}\n"
        "Trascina col mouse per selezionare 1 o 2 intervalli di SOLO RUMORE DI FONDO.\n"
        "[Premi 'R' per cancellare le selezioni | Premi 'INVIO' o chiudi la finestra per confermare]"
    )
    ax.set_title(title_text, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)

    status_box = ax.text(
        0.01, 0.95, "Intervalli selezionati: 0 (Se chiudi senza selezionare verrà usato il modo AUTO)",
        transform=ax.transAxes, fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="#fff8dc", alpha=0.9)
    )

    def update_status():
        if not selected_intervals:
            status_box.set_text("Intervalli selezionati: 0 (Se chiudi senza selezionare verrà usato il modo AUTO)")
        else:
            desc = ", ".join([f"[{a:.1f}s - {b:.1f}s]" for a, b in selected_intervals])
            status_box.set_text(f"Intervalli selezionati ({len(selected_intervals)}): {desc}")
        fig.canvas.draw_idle()

    def onselect(xmin, xmax):
        if xmax - xmin < 0.15:
            return
        selected_intervals.append((xmin, xmax))
        patch = ax.axvspan(xmin, xmax, color="crimson", alpha=0.35)
        highlight_patches.append(patch)
        update_status()

    def on_key(event):
        if event.key in ("r", "R", "backspace"):
            selected_intervals.clear()
            for p in highlight_patches:
                p.remove()
            highlight_patches.clear()
            update_status()
        elif event.key in ("enter", "return"):
            plt.close(fig)

    span = SpanSelector(
        ax, onselect, "horizontal", useblit=True,
        props=dict(alpha=0.4, facecolor="crimson"),
        interactive=True, drag_from_anywhere=True
    )
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout()
    plt.show()

    if not selected_intervals:
        print("   [!] Nessun intervallo selezionato manualmente: passo alla selezione AUTOMATICA.")
        return find_classroom_noise_auto(audio, sr)

    clips = []
    for start_sec, end_sec in selected_intervals:
        s = max(0, int(start_sec * sr))
        e = min(len(audio), int(end_sec * sr))
        if e > s:
            clips.append(audio[s:e])

    return np.concatenate(clips) if clips else find_classroom_noise_auto(audio, sr)


# ==============================================================================
# 3. CATENA DI ELABORAZIONE AUDIO (DENOISE -> NORM 1 -> COMP -> EQ -> NORM 2)
# ==============================================================================

def normalize_peak(audio: np.ndarray, target_db: float = -1.0) -> np.ndarray:
    """Normalizza il picco massimo dell'audio a `target_db` (default -1.0 dBFS)."""
    peak = np.max(np.abs(audio))
    if peak < 1e-9:
        return audio
    target_linear = 10.0 ** (target_db / 20.0)
    return audio * (target_linear / peak)


def reduce_classroom_noise(audio: np.ndarray, sr: int, noise_clip: np.ndarray) -> np.ndarray:
    """
    Applica la riduzione del rumore calibrata per aula universitaria + iPhone.
    """
    denoised = nr.reduce_noise(
        y=audio,
        sr=sr,
        y_noise=noise_clip,
        stationary=True,
        prop_decrease=0.88,          # Riduce dell'88% il rumore (evita artefatti robotici)
        n_fft=2048,                  # Ottima risoluzione sulle frequenze vocali
        time_mask_smooth_ms=64,      # Smoothing temporale per smussare il brusio d'aula
        freq_mask_smooth_hz=400      # Smoothing in frequenza contro l'effetto "musical noise"
    )
    return denoised.astype(np.float32)


def build_university_lecture_chain() -> Pedalboard:
    """
    Catena DSP per Lezioni Universitarie e Conferenze.
    """
    return Pedalboard([
        # Passa-Alto: taglia rimbombi sub-bassi e colpi sulla cattedra
        HighpassFilter(cutoff_frequency_hz=115.0),

        # Compressore: schiaccia i picchi e sostiene le frasi dette a bassa voce
        Compressor(
            threshold_db=-22.0,
            ratio=4.0,
            attack_ms=8.0,
            release_ms=140.0
        ),

        # Equalizzatore per Lezioni Universitarie e Conferenze
        PeakFilter(cutoff_frequency_hz=300.0, gain_db=-4.5, q=1.2),   # Anti-rimbombo aula
        PeakFilter(cutoff_frequency_hz=800.0, gain_db=-2.0, q=1.5),   # Anti-effetto scatola
        PeakFilter(cutoff_frequency_hz=3000.0, gain_db=4.0, q=1.0),   # Chiarezza consonanti
        LowpassFilter(cutoff_frequency_hz=8000.0),                    # Taglio fruscio acuto (hiss)
    ])


def process_single_file(
    input_path: str,
    output_dir: str = None,
    noise_mode: str = "auto",
    mp3_vbr_quality: int = 2
):
    """
    Esegue l'intera pipeline su un singolo file audio ed esporta in MP3 (170-210 kbps)
    mantenendo il nome originale.
    """
    input_path_obj = Path(input_path).resolve()
    base_name = input_path_obj.stem  # Nome originale senza estensione

    # Di default salva nella stessa cartella del file originale
    target_dir = Path(output_dir).resolve() if output_dir else input_path_obj.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    output_mp3_path = target_dir / f"{base_name}.mp3"

    # Sicurezza: se l'originale era già .mp3 e salviamo nella stessa cartella, facciamo backup
    if input_path_obj == output_mp3_path:
        backup_dir = target_dir / "backup_originali"
        backup_dir.mkdir(exist_ok=True)
        backup_path = backup_dir / input_path_obj.name
        if not backup_path.exists():
            shutil.copy2(input_path_obj, backup_path)
            print(f"   [i] File originale MP3 salvato per sicurezza in: {backup_path}")

    print(f"\n============================================================")
    print(f"In elaborazione: {input_path_obj.name}")
    print(f"============================================================")

    print(" -> [1/6] Caricamento e conversione in Mono 32-bit float...")
    audio, sr = load_audio_any_format(str(input_path_obj))

    if noise_mode == "manual":
        print(" -> [2/6] Apertura grafico per selezione manuale del rumore...")
        noise_clip = select_noise_intervals_gui(audio, sr, input_path_obj.name)
    else:
        print(" -> [2/6] Ricerca automatica del brusio di fondo dell'aula...")
        noise_clip = find_classroom_noise_auto(audio, sr)

    print(" -> [3/6] Applicazione Noise Reduction (Spectral Gating per aula)...")
    audio_denoised = reduce_classroom_noise(audio, sr, noise_clip)

    print(" -> [4/6] Prima Normalizzazione a -1.0 dB (calibrazione ingresso compressore)...")
    audio_norm1 = normalize_peak(audio_denoised, target_db=-1.0)

    print(" -> [5/6] Applicazione High-Pass, Compressore ed Equalizzatore Lezioni...")
    board = build_university_lecture_chain()
    audio_eq_comp = board(audio_norm1, sr)

    print(" -> [6/6] Seconda Normalizzazione finale a -1.0 dB ed esportazione MP3 (170-210 kbps)...")
    audio_final = normalize_peak(audio_eq_comp, target_db=-1.0)

    export_to_mp3(audio_final, sr, str(output_mp3_path), vbr_quality=mp3_vbr_quality)
    print(f" [OK] Completato! File salvato in: {output_mp3_path}")


# ==============================================================================
# 4. HELPER PER RACCOGLIERE FILE E CARTELLE (DA CODICE O CON FINESTRA GRAFICA)
# ==============================================================================

def collect_audio_files(sources) -> list:
    if isinstance(sources, (str, Path)):
        sources = [sources]

    collected = []
    for src in sources:
        p = Path(src).expanduser().resolve()
        if p.is_dir():
            for child in sorted(p.iterdir()):
                if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS:
                    collected.append(str(child))
        elif p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            collected.append(str(p))
        else:
            print(f" [!] Attenzione: percorso ignorato o non trovato -> {src}")
    return collected


def interactive_wizard():
    print("\n" + "=" * 65)
    print("   PULIZIA AUDIO LEZIONI UNIVERSITARIE & CONFERENZE (iPhone/Aula)")
    print("=" * 65)

    print("\nCome vuoi selezionare gli intervalli di rumore di fondo?")
    print("  [1] AUTOMATICO (Consigliato: trova da solo il brusio costante dell'aula)")
    print("  [2] MANUALE col Mouse (Apre il grafico della forma d'onda con SpanSelector)")
    scelta_noise = input("Scegli [1/2, default=1]: ").strip()
    noise_mode = "manual" if scelta_noise == "2" else "auto"

    print("\nCosa vuoi elaborare?")
    print("  [1] Un'intera cartella di registrazioni")
    print("  [2] Uno o più file specifici selezionati singolarmente")
    scelta_input = input("Scegli [1/2, default=1]: ").strip()

    use_gui_dialog = True
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
    except Exception:
        use_gui_dialog = False

    files_to_process = []
    if scelta_input == "2":
        if use_gui_dialog:
            print(" -> Seleziona uno o più file nella finestra che si è aperta...")
            selected = filedialog.askopenfilenames(
                title="Seleziona le registrazioni da pulire",
                filetypes=[("File Audio", "*.m4a *.mp3 *.wav *.flac *.ogg *.aac *.mp4"), ("Tutti i file", "*.*")]
            )
            files_to_process = collect_audio_files(list(selected))
        else:
            raw = input("Inserisci i percorsi dei file separati da virgola: ")
            paths = [x.strip().strip("'\"") for x in raw.split(",") if x.strip()]
            files_to_process = collect_audio_files(paths)
    else:
        if use_gui_dialog:
            print(" -> Seleziona la cartella delle registrazioni nella finestra che si è aperta...")
            folder = filedialog.askdirectory(title="Seleziona la cartella con le registrazioni")
            if folder:
                files_to_process = collect_audio_files(folder)
        else:
            folder = input("Inserisci il percorso della cartella: ").strip().strip("'\"")
            files_to_process = collect_audio_files(folder)

    if not files_to_process:
        print("\n[!] Nessun file audio selezionato o trovato. Uscita.")
        return

    print(f"\nTrovati {len(files_to_process)} file audio da elaborare.")
    print("Dove vuoi esportare i file MP3 puliti?")
    print("  [1] DEFAULT: Nella stessa cartella da cui proviene ciascun file originale")
    print("  [2] Scegli una cartella di esportazione specifica")
    scelta_out = input("Scegli [1/2, default=1]: ").strip()

    output_folder = None
    if scelta_out == "2":
        if use_gui_dialog:
            print(" -> Seleziona la cartella di esportazione nella finestra...")
            chosen_out = filedialog.askdirectory(title="Seleziona la cartella di esportazione")
            output_folder = chosen_out if chosen_out else None
        else:
            chosen_out = input("Inserisci il percorso della cartella di output: ").strip().strip("'\"")
            output_folder = chosen_out if chosen_out else None

    for filepath in files_to_process:
        process_single_file(
            input_path=filepath,
            output_dir=output_folder,
            noise_mode=noise_mode,
            mp3_vbr_quality=2  # Standard LAME V2: 170-210 kbps
        )

    print("\n============================================================")
    print(" TUTTE LE REGISTRAZIONI SONO STATE ELABORATE CON SUCCESSO!")
    print("============================================================\n")


# ==============================================================================
# 5. GESTIONE CONFIGURAZIONE (.json), ARGOMENTI CLI E AVVIO
# ==============================================================================


def ask_audio_files_gui() -> list:
    """
    Apre solo la richiesta (GUI/terminale) per selezionare i file o la cartella audio,
    usata quando avvii lo script passando SOLO il file .json da terminale.
    """
    print("\nCosa vuoi elaborare con la configurazione caricata?")
    print("  [1] Un'intera cartella di registrazioni")
    print("  [2] Uno o più file specifici selezionati singolarmente")
    scelta_input = input("Scegli [1/2, default=1]: ").strip()

    use_gui_dialog = True
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
    except Exception:
        use_gui_dialog = False

    if scelta_input == "2":
        if use_gui_dialog:
            print(" -> Seleziona uno o più file nella finestra che si è aperta...")
            selected = filedialog.askopenfilenames(
                title="Seleziona le registrazioni da pulire",
                filetypes=[("File Audio", "*.m4a *.mp3 *.wav *.flac *.ogg *.aac *.mp4"), ("Tutti i file", "*.*")]
            )
            return collect_audio_files(list(selected))
        else:
            raw = input("Inserisci i percorsi dei file separati da virgola: ")
            paths = [x.strip().strip("'\"") for x in raw.split(",") if x.strip()]
            return collect_audio_files(paths)
    else:
        if use_gui_dialog:
            print(" -> Seleziona la cartella delle registrazioni nella finestra che si è aperta...")
            folder = filedialog.askdirectory(title="Seleziona la cartella con le registrazioni")
            return collect_audio_files(folder) if folder else []
        else:
            folder = input("Inserisci il percorso della cartella: ").strip().strip("'\"")
            return collect_audio_files(folder)


def parse_cli_arguments(argv: list):
    """
    Smista gli argomenti passati da terminale in base all'estensione:
    - .json -> file di configurazione
    - file con estensione in SUPPORTED_EXTENSIONS o cartelle -> sorgenti audio
    """
    config_file = None
    audio_sources = []

    for arg in argv:
        p = Path(arg).expanduser()
        ext = p.suffix.lower()

        if ext == ".json" and p.is_file():
            config_file = p.resolve()
        elif p.is_dir() or (p.is_file() and ext in SUPPORTED_EXTENSIONS):
            audio_sources.append(str(p.resolve()))
        # Eventuali altri file (es. .py o .txt presi usando '*') vengono ignorati

    return config_file, audio_sources


# ==============================================================================
# 6. AVVIO DEL PROGRAMMA (CONFIGURAZIONE, ARGOMENTI CLI O MENU INTERATTIVO)
# ==============================================================================

if __name__ == "__main__":

    # Impostazioni di default del file Python (usate se non passi un file .json)
    DEFAULT_CONFIG = {
        "use_interactive_menu": True,
        "noise_mode": "auto",             # "auto" oppure "manual" (SpanSelector)
        "input_sources": ["./registrazioni_lezioni"],
        "output_folder": None,            # None = stessa cartella dei file originali
        "mp3_vbr_quality": 2              # 2 = LAME V2 (170-210 kbps)
    }

    # 2. Configurazione attiva (parte dai default, ma può essere aggiornata dal .json)
    config = DEFAULT_CONFIG.copy()

    cli_args = sys.argv[1:]
    
    # CASO 1: Nessun argomento da terminale
    if not cli_args:
        if config["use_interactive_menu"]:
            interactive_wizard()
        else:
            files = collect_audio_files(config["input_sources"])
            if not files:
                print("[!] Nessun file audio trovato in 'input_sources'.")
                sys.exit(1)
            for f in files:
                process_single_file(
                    input_path=f,
                    output_dir=config["output_folder"],
                    noise_mode=config["noise_mode"],
                    mp3_vbr_quality=config["mp3_vbr_quality"]
                )
        sys.exit(0)

    # Smistamento argomenti CLI (.json vs file audio/cartelle)
    json_path, cli_audio_sources = parse_cli_arguments(cli_args)

    # Se è stato passato un file .json, aggiorna le impostazioni di default
    if json_path:
        print(f" [i] Caricamento configurazione da: {json_path.name}")
        with open(json_path, "r", encoding="utf-8") as jf:
            user_cfg = json.load(jf)
            config.update(user_cfg)

    # CASO 2: Passato SOLO il file .json (nessun audio da CLI) -> apre la GUI per scegliere gli audio
    if json_path and not cli_audio_sources:
        files_to_process = ask_audio_files_gui()
    # CASO 3 e 4: Passati file audio da CLI (con o senza .json) -> raccoglie gli audio passati
    else:
        files_to_process = collect_audio_files(cli_audio_sources)

    if not files_to_process:
        print("\n[!] Nessun file audio valido trovato o selezionato. Uscita.")
        sys.exit(1)

    # Esecuzione su tutti i file selezionati con le impostazioni attive (da JSON o da Python)
    for f in files_to_process:
        process_single_file(
            input_path=f,
            output_dir=config["output_folder"],
            noise_mode=config["noise_mode"],
            mp3_vbr_quality=config["mp3_vbr_quality"]
        )





#
## ==============================================================================
## 5. AVVIO DEL PROGRAMMA (CONFIGURAZIONE RAPIDA O MENU INTERATTIVO)
## ==============================================================================
#
#if __name__ == "__main__":
#    # --------------------------------------------------------------------------
#    # MODALITA' DI AVVIO:
#    # - Se lasci USE_INTERACTIVE_MENU = True, all'avvio si aprirà un comodo menu
#    #   (con finestre grafiche di Ubuntu) per scegliere auto/manuale, file/cartella
#    #   e cartella di esportazione.
#    # - Se imposti USE_INTERACTIVE_MENU = False, lo script userà direttamente
#    #   le variabili qui sotto senza fare domande!
#    # --------------------------------------------------------------------------
#    USE_INTERACTIVE_MENU = True
#
#    # --- CONFIGURAZIONE DIRETTA (usata se USE_INTERACTIVE_MENU = False) ---
#    # 1. Modalità rumore: "auto" (automatico per aula) oppure "manual" (grafico SpanSelector)
#    NOISE_MODE = "auto"
#
#    # 2. Cosa elaborare: puoi mettere il percorso di UNA CARTELLA intera
#    #    oppure una LISTA DI FILE SINGOLI (o un mix di entrambi):
#    #    Esempio cartella: INPUT_SOURCES = "/home/tuonome/Documenti/Lezioni_Fisica"
#    #    Esempio file singoli: INPUT_SOURCES = ["/home/tuonome/lez1.m4a", "/home/tuonome/lez2.m4a"]
#    INPUT_SOURCES = [
#        "./registrazioni_lezioni"
#    ]
#
#    # 3. Cartella di esportazione:
#    #    - Lascia None per esportare di DEFAULT nella stessa cartella dei file originali
#    #    - Oppure scrivi un percorso, es: OUTPUT_FOLDER = "./lezioni_pulite_mp3"
#    OUTPUT_FOLDER = None
#
#    # 4. Qualità MP3: 2 corrisponde a LAME V2 (bit rate standard 170-210 kbps)
#    MP3_VBR_QUALITY = 2
#    # --------------------------------------------------------------------------
#
#    if USE_INTERACTIVE_MENU:
#        interactive_wizard()
#    else:
#        files = collect_audio_files(INPUT_SOURCES)
#        if not files:
#            print("[!] Nessun file audio trovato nei percorsi specificati in INPUT_SOURCES.")
#            sys.exit(1)
#
#        for f in files:
#            process_single_file(
#                input_path=f,
#                output_dir=OUTPUT_FOLDER,
#                noise_mode=NOISE_MODE,
#                mp3_vbr_quality=MP3_VBR_QUALITY
#            )