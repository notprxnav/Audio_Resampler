# SP18 Multi-Rate Audio Resampler

A Python and Streamlit web application for converting WAV audio from one sampling rate to another using rational polyphase resampling and evaluating the resulting round-trip reconstruction error.

This project was developed for the **EE4999 Signals and Systems Project** as part of the **BS in Electronic Systems programme at IIT Madras**.

## Author

- **Name:** Pranav Arakkal
- **Student ID:** ES23F3000047
- **Programme:** BS in Electronic Systems
- **Institution:** Indian Institute of Technology Madras
- **Email:** 23f3000047@es.study.iitm.ac.in

## Project Objective

The application converts an uploaded WAV audio file from its detected input sampling rate to a user-selected output sampling rate.

For example:

```text
Input sampling rate:  44,100 Hz
Output sampling rate: 20,000 Hz
Rational ratio:       200/441
```

The application performs anti-aliased rational resampling, generates a new 16-bit PCM WAV file, and evaluates the difference between the original and converted signals.

## Features

- Upload a WAV audio file
- Play the original audio
- Automatically detect the input sampling rate
- Display audio duration, channels, sample count and data type
- Enter a required output sampling rate
- Calculate the reduced rational ratio \(L/M\)
- Perform polyphase FIR resampling
- Apply anti-alias filtering during downsampling
- Preserve mono and stereo audio channels
- Play the converted audio
- Download the converted 16-bit PCM WAV
- Calculate round-trip SNR
- Calculate mean squared error
- Calculate root mean squared error
- Calculate peak absolute error
- Calculate normalized correlation
- Estimate the signal-alignment offset
- Measure processing time
- Display the number of compared samples
- Display input and output durations
- Automatically select an active 50 ms waveform interval
- Compare the original and reconstructed waveforms
- Display a separate reconstruction-error plot
- Compare the original and converted frequency spectra
- Display the output Nyquist-frequency limit

## Rational Resampling Algorithm

The sampling-rate conversion ratio is:

\[
\frac{f_{\mathrm{out}}}{f_{\mathrm{in}}}
=
\frac{L}{M}
\]

The reduced integer factors are calculated using the greatest common divisor:

```python
divisor = math.gcd(input_rate, output_rate)

L = output_rate // divisor
M = input_rate // divisor
```

The conversion is performed using:

```python
converted = resample_poly(
    original,
    up=L,
    down=M,
    axis=0
)
```

The polyphase operation conceptually performs:

1. Interpolation by \(L\)
2. FIR low-pass filtering
3. Decimation by \(M\)

The low-pass filter prevents avoidable aliasing during downsampling.

## Distortion Evaluation

The original and converted signals cannot be compared directly because they have different sampling rates and sample counts.

The application therefore uses a round-trip comparison:

```text
Original signal
      ↓
Converted to selected rate
      ↓
Converted back to original rate
      ↓
Aligned with the original
      ↓
Reconstruction measurements calculated
```

The reported results are round-trip reconstruction measurements. They are not THD or THD+N measurements.

When downsampling, frequencies above the output Nyquist limit are intentionally removed. This expected bandwidth reduction contributes to the measured reconstruction error.

## Technologies Used

- Python
- Streamlit
- NumPy
- SciPy
- Matplotlib
- GitHub
- Streamlit Community Cloud

## Project Files

```text
audio-resampler/
├── streamlit_app.py
├── requirements.txt
├── README.md
└── SP18_Multi_Rate_Audio_Resampler.ipynb
```

The Jupyter Notebook is optional for deployment but is included to demonstrate the development and validation of the underlying algorithm.

## Local Installation

### 1. Clone or download the repository

```bash
git clone YOUR-GITHUB-REPOSITORY-URL
cd audio-resampler
```

Alternatively, download the repository as a ZIP file and extract it.

### 2. Create a virtual environment

On Windows:

```powershell
py -m venv .venv
```

### 3. Install the required packages

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Run the application

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

The application should open at:

```text
http://localhost:8501
```

## Using the Application

1. Upload a WAV file.
2. Play the original audio.
3. Check the detected input sampling rate.
4. Enter the required output sampling rate.
5. Click **Convert and evaluate distortion**.
6. Play the converted audio.
7. Download the converted WAV file.
8. Examine the measurements and comparison plots.

## Deployment

The application can be deployed using Streamlit Community Cloud.

1. Upload the project files to GitHub.
2. Sign in to Streamlit Community Cloud using GitHub.
3. Create a new application.
4. Select this repository.
5. Select the `main` branch.
6. Enter `streamlit_app.py` as the main file path.
7. Click **Deploy**.

GitHub stores the source code, while Streamlit Community Cloud executes the Python application.

## Supported Platforms

The deployed application can be accessed using:

- Desktop web browsers
- Android web browsers
- iPhone and iPad web browsers

The current implementation is a responsive web application rather than a separately compiled native Android or iOS application.

## Input Limitations

- The current version accepts WAV audio files.
- Large audio files require more memory and processing time.
- Browser playback depends on browser support for PCM WAV.
- SNR and related metrics represent round-trip reconstruction error.
- Downsampling intentionally removes frequencies above the new Nyquist limit.

## Future Work

- Additional audio formats
- Batch conversion
- Selectable filter quality
- Spectrogram comparison
- Channel-wise measurements
- Controlled THD and THD+N testing
- Improved long-file processing
- Native Android and iOS packaging
