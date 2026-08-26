import io
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.io import wavfile
from scipy.signal import correlate, correlation_lags, resample_poly


st.set_page_config(
    page_title="SP18 Audio Resampler",
    page_icon="🎧",
    layout="wide",
)


def pcm_to_float(audio: np.ndarray) -> np.ndarray:
    """Convert WAV PCM/float samples to float64 in approximately [-1, 1]."""
    if np.issubdtype(audio.dtype, np.floating):
        return np.clip(audio.astype(np.float64), -1.0, 1.0)

    info = np.iinfo(audio.dtype)
    values = audio.astype(np.float64)
    if np.issubdtype(audio.dtype, np.unsignedinteger):
        midpoint = (info.max + 1) / 2
        return (values - midpoint) / midpoint

    scale = max(abs(info.min), info.max)
    return values / scale


def float_to_int16(audio: np.ndarray) -> np.ndarray:
    """Create browser-compatible 16-bit PCM audio."""
    return np.round(np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)


def make_wav(audio: np.ndarray, rate: int) -> bytes:
    output = io.BytesIO()
    wavfile.write(output, rate, float_to_int16(audio))
    return output.getvalue()


def analysis_channel(audio: np.ndarray) -> np.ndarray:
    """Use a mono mix only for measurements; conversion preserves all channels."""
    return audio if audio.ndim == 1 else np.mean(audio, axis=1)


def align_signals(reference: np.ndarray, test: np.ndarray, rate: int):
    """Estimate and remove any small delay before calculating errors."""
    # Ten seconds is enough for stable delay estimation and avoids huge correlations.
    limit = min(len(reference), len(test), rate * 10)
    ref_part = reference[:limit] - np.mean(reference[:limit])
    test_part = test[:limit] - np.mean(test[:limit])

    correlation = correlate(test_part, ref_part, mode="full", method="fft")
    lags = correlation_lags(len(test_part), len(ref_part), mode="full")
    lag = int(lags[np.argmax(np.abs(correlation))])

    if lag > 0:
        test = test[lag:]
    elif lag < 0:
        reference = reference[-lag:]

    length = min(len(reference), len(test))
    return reference[:length], test[:length], lag


def distortion_metrics(reference: np.ndarray, reconstructed: np.ndarray, rate: int):
    reference, reconstructed, lag = align_signals(reference, reconstructed, rate)

    # Remove tiny DC differences before signal-error measurements.
    reference = reference - np.mean(reference)
    reconstructed = reconstructed - np.mean(reconstructed)
    error = reference - reconstructed

    signal_power = float(np.mean(reference**2))
    noise_power = float(np.mean(error**2))
    mse = noise_power
    rmse = math.sqrt(mse)
    peak_error = float(np.max(np.abs(error)))

    if noise_power == 0:
        snr = float("inf")
    elif signal_power == 0:
        snr = float("-inf")
    else:
        snr = 10 * math.log10(signal_power / noise_power)

    denominator = np.linalg.norm(reference) * np.linalg.norm(reconstructed)
    correlation_value = (
        float(np.dot(reference, reconstructed) / denominator)
        if denominator > 0
        else 0.0
    )

    return {
        "reference": reference,
        "reconstructed": reconstructed,
        "error": error,
        "lag": lag,
        "snr": snr,
        "mse": mse,
        "rmse": rmse,
        "peak_error": peak_error,
        "correlation": correlation_value,
    }


def spectrum(audio: np.ndarray, rate: int):
    """Return a normalized single-sided spectrum for display."""
    maximum_samples = min(len(audio), 262_144)
    data = audio[:maximum_samples]
    if len(data) < 2:
        return np.array([0.0]), np.array([-120.0])

    data = data - np.mean(data)
    window = np.hanning(len(data))
    magnitude = np.abs(np.fft.rfft(data * window))
    magnitude /= max(float(np.max(magnitude)), 1e-12)
    magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-8))
    frequencies = np.fft.rfftfreq(len(data), 1 / rate)
    return frequencies, magnitude_db


def active_window_start(audio: np.ndarray, window_samples: int) -> int:
    """Find a high-energy region so the waveform plot is informative."""
    if len(audio) <= window_samples:
        return 0

    block_count = len(audio) // window_samples
    blocks = audio[: block_count * window_samples].reshape(
        block_count, window_samples
    )
    energies = np.mean(blocks**2, axis=1)
    return int(np.argmax(energies) * window_samples)


def interpretation(snr: float, correlation_value: float) -> str:
    if snr >= 60 and correlation_value >= 0.999:
        return "Excellent: the in-band reconstruction is extremely close to the original."
    if snr >= 35 and correlation_value >= 0.99:
        return "Good: low measured round-trip distortion."
    if snr >= 20 and correlation_value >= 0.95:
        return "Moderate: some difference is measurable or audible."
    return "Large difference: this can be expected when the new rate removes substantial high-frequency content."


st.title("Multi-Rate Audio Resampler")
st.caption("SP18 · Rational resampling and distortion evaluation")
st.write(
    "Upload a WAV file, play it, choose a new sampling rate, convert it, "
    "then play, analyse, and download the result."
)

uploaded_file = st.file_uploader("1. Upload the original WAV audio", type=["wav"])

if uploaded_file is None:
    st.info("Select a WAV file to begin.")
    st.stop()

input_bytes = uploaded_file.getvalue()

try:
    input_rate, original_pcm = wavfile.read(io.BytesIO(input_bytes))
except Exception as error:
    st.error(f"The WAV file could not be read: {error}")
    st.stop()

original = pcm_to_float(original_pcm)
channels = 1 if original.ndim == 1 else original.shape[1]
duration = len(original) / input_rate

st.subheader("2. Play the original audio")
st.audio(input_bytes, format="audio/wav")

info_1, info_2, info_3, info_4 = st.columns(4)
info_1.metric("Detected input rate", f"{input_rate:,} Hz")
info_2.metric("Duration", f"{duration:.2f} s")
info_3.metric("Channels", channels)
info_4.metric("Samples/channel", f"{len(original):,}")

with st.expander("Original WAV details"):
    st.write(f"File name: **{uploaded_file.name}**")
    st.write(f"Original data type: **{original_pcm.dtype}**")
    st.write(f"Normalized minimum: **{np.min(original):.6f}**")
    st.write(f"Normalized maximum: **{np.max(original):.6f}**")

st.subheader("3. Enter the required output rate")
preset_rates = [8000, 16000, 20000, 22050, 24000, 32000, 44100, 48000, 96000]
default_rate = 20000 if input_rate != 20000 else 16000

output_rate = int(
    st.number_input(
        "Output sampling rate (Hz)",
        min_value=1000,
        max_value=192000,
        value=default_rate,
        step=1000,
        help="Common values include: " + ", ".join(f"{rate:,}" for rate in preset_rates),
    )
)

divisor = math.gcd(int(input_rate), output_rate)
up_factor = output_rate // divisor
down_factor = int(input_rate) // divisor
st.write(f"Rational conversion ratio: **L/M = {up_factor}/{down_factor}**")

if output_rate == input_rate:
    st.warning("Choose an output rate different from the input rate.")

if st.button(
    "4. Convert and evaluate distortion",
    type="primary",
    disabled=output_rate == input_rate,
    use_container_width=True,
):
    with st.spinner("Resampling and analysing the audio…"):
        start_time = time.perf_counter()

        # resample_poly performs rational polyphase resampling and anti-alias filtering.
        converted = resample_poly(
            original,
            up=up_factor,
            down=down_factor,
            axis=0,
        )
        processing_time = time.perf_counter() - start_time
        converted_pcm = float_to_int16(converted)
        converted_bytes = make_wav(converted, output_rate)

        # Read the actual generated WAV bytes so evaluation includes PCM quantization.
        saved_rate, saved_pcm = wavfile.read(io.BytesIO(converted_bytes))
        saved_float = pcm_to_float(saved_pcm)

        # A fair error comparison requires both signals at the same sampling rate.
        # Resample the output back to the input rate, then align it with the original.
        reconstructed = resample_poly(
            saved_float,
            up=down_factor,
            down=up_factor,
            axis=0,
        )
        metrics = distortion_metrics(
            analysis_channel(original),
            analysis_channel(reconstructed),
            int(input_rate),
        )

    st.success("Conversion and analysis completed.")

    st.subheader("5. Play and download the converted audio")
    st.audio(converted_bytes, format="audio/wav")

    output_name = f"{uploaded_file.name.rsplit('.', 1)[0]}_{output_rate}Hz.wav"
    st.download_button(
        "Download converted WAV",
        data=converted_bytes,
        file_name=output_name,
        mime="audio/wav",
        use_container_width=True,
    )

    st.subheader("6. Distortion evaluation")
    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    snr_text = "∞ dB" if math.isinf(metrics["snr"]) and metrics["snr"] > 0 else f"{metrics['snr']:.2f} dB"
    metric_1.metric("Round-trip SNR", snr_text)
    metric_2.metric("Mean squared error", f"{metrics['mse']:.3e}")
    metric_3.metric("RMSE", f"{metrics['rmse']:.3e}")
    metric_4.metric("Correlation", f"{metrics['correlation']:.6f}")

    more_1, more_2, more_3, more_4 = st.columns(4)
    more_1.metric("Peak absolute error", f"{metrics['peak_error']:.3e}")
    more_2.metric("Alignment offset", f"{metrics['lag']} samples")
    more_3.metric("Processing time", f"{processing_time * 1000:.1f} ms")
    more_4.metric("Output samples/channel", f"{len(converted):,}")

    detail_1, detail_2, detail_3 = st.columns(3)
    detail_1.metric("Compared samples", f"{len(metrics['reference']):,}")
    detail_2.metric("Output duration", f"{len(converted) / output_rate:.6f} s")
    detail_3.metric("Output size", f"{len(converted_bytes) / 1024:.2f} KiB")

    st.info(interpretation(metrics["snr"], metrics["correlation"]))
    st.caption(
        "SNR, MSE, RMSE, and correlation use a round-trip comparison: the converted "
        "signal is resampled back to the input rate and aligned with the original. "
        "When downsampling, frequencies above the new Nyquist limit are intentionally "
        "removed by the anti-alias filter; that expected bandwidth loss contributes to the error."
    )

    st.subheader("Waveform and reconstruction-error comparison")
    window_samples = max(1, int(input_rate * 0.05))
    start = active_window_start(metrics["reference"], window_samples)
    end = min(start + window_samples, len(metrics["reference"]))
    time_axis = np.arange(end - start) / input_rate * 1000

    waveform_figure, waveform_axes = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True
    )
    waveform_axes[0].plot(
        time_axis,
        metrics["reference"][start:end],
        label="Original",
        linewidth=1.3,
    )
    waveform_axes[0].plot(
        time_axis,
        metrics["reconstructed"][start:end],
        label="Round-trip reconstructed",
        linewidth=1,
        alpha=0.8,
    )
    waveform_axes[0].set_ylabel("Normalized amplitude")
    waveform_axes[0].set_title("Original and round-trip reconstructed waveforms")
    waveform_axes[0].grid(alpha=0.2)
    waveform_axes[0].legend()

    waveform_axes[1].plot(
        time_axis,
        metrics["error"][start:end],
        color="crimson",
        linewidth=1,
    )
    waveform_axes[1].set_xlabel("Time within selected 50 ms window (ms)")
    waveform_axes[1].set_ylabel("Error amplitude")
    waveform_axes[1].set_title("Reconstruction error")
    waveform_axes[1].grid(alpha=0.2)
    waveform_figure.tight_layout()
    st.pyplot(waveform_figure)
    plt.close(waveform_figure)

    st.caption(
        f"Automatically selected active interval: "
        f"{start / input_rate:.3f} to {end / input_rate:.3f} seconds."
    )

    st.subheader("Frequency-spectrum comparison")
    original_frequency, original_db = spectrum(analysis_channel(original), int(input_rate))
    converted_frequency, converted_db = spectrum(analysis_channel(saved_float), saved_rate)
    spectrum_figure, spectrum_axis = plt.subplots(figsize=(12, 5))
    spectrum_axis.plot(original_frequency, original_db, label=f"Original ({input_rate:,} Hz)", linewidth=1.2)
    spectrum_axis.plot(converted_frequency, converted_db, label=f"Converted ({output_rate:,} Hz)", linewidth=1.2)
    spectrum_axis.axvline(output_rate / 2, color="crimson", linestyle="--", linewidth=1, label="Output Nyquist limit")
    spectrum_axis.set_xlim(0, min(input_rate / 2, 24000))
    spectrum_axis.set_ylim(-120, 5)
    spectrum_axis.set_xlabel("Frequency (Hz)")
    spectrum_axis.set_ylabel("Relative magnitude (dB)")
    spectrum_axis.set_title("Normalized magnitude spectra")
    spectrum_axis.grid(alpha=0.2)
    spectrum_axis.legend()
    spectrum_figure.tight_layout()
    st.pyplot(spectrum_figure)
    plt.close(spectrum_figure)

    with st.expander("Conversion details"):
        st.write(f"Input file: **{uploaded_file.name}**")
        st.write(f"Input rate: **{input_rate:,} Hz**")
        st.write(f"Output rate: **{output_rate:,} Hz**")
        st.write(f"Rational factors: **L = {up_factor}, M = {down_factor}**")
        st.write(f"Channels preserved: **{channels}**")
        st.write(f"Input samples/channel: **{len(original):,}**")
        st.write(f"Output samples/channel: **{len(converted):,}**")
        st.write(f"Input duration: **{len(original) / input_rate:.6f} seconds**")
        st.write(f"Output duration: **{len(converted) / output_rate:.6f} seconds**")
        st.write(f"Compared samples: **{len(metrics['reference']):,}**")
        st.write("Output encoding: **16-bit PCM WAV**")
        st.write("Resampling method: **polyphase FIR (`scipy.signal.resample_poly`)**")
