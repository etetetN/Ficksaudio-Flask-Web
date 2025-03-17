import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf
from typing import Tuple, Dict, Any, Optional
from scipy import signal
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from skimage.metrics import structural_similarity as compare_ssim
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, Conv2DTranspose, Permute, AveragePooling2D ,MaxPooling2D, Flatten ,Cropping2D, Add, BatchNormalization, Dropout, LayerNormalization, MultiHeadAttention, Reshape, ZeroPadding2D, Concatenate, UpSampling2D, Reshape, Embedding, Lambda, RepeatVector, Layer, Conv1D
)
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
import soundfile as sf
import gc
import random
import math
import requests
import tempfile
from flask import current_app, session
import time
import shutil
import uuid
import threading

#Track availability of optional dependencies
_TF_AVAILABLE = False
_DIFFUSION_AVAILABLE = False

#Initialize global progress tracking dictionary
_progress = {
    'message': '',
    'current': 0,
    'total': 0,
    'done': False
}

#Use a threading lock to prevent race conditions when updating progress
_progress_lock = threading.RLock()

try:
    #Attempt to import TensorFlow and enable optimizations
    import tensorflow as tf
    _TF_AVAILABLE = True
    #Enable mixed precision to improve performance on compatible hardware
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("TensorFlow successfully imported with mixed precision enabled")
except ImportError:
    print("TensorFlow not available")


try:
    from tensorflow.keras.layers import GroupNormalization
except Exception:
    from tensorflow_addons.layers import GroupNormalization


#Set mixed precision for better performance
tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
#Enable eager execution for simplified debugging and development
tf.config.run_functions_eagerly(True)

def cross_attention_block_with_groupnorm(input_layer, cross_layer, num_heads, embedding_dim, num_groups=32):
    """Cross-attention block with group normalization and residual connections."""
    height, width, channels = input_layer.shape[1], input_layer.shape[2], input_layer.shape[3]
    cross_height, cross_width, cross_channels = cross_layer.shape[1], cross_layer.shape[2], cross_layer.shape[3]

    #Calculate pool size for converting the cross layer spatial dimensions to input layer
    kernel_height = cross_height // height
    kernel_width = cross_width // width

    cross_processed = MaxPooling2D(pool_size=(kernel_height, kernel_width))(cross_layer)
    cross_processed = Conv2D(channels, (1, 1), activation='silu')(cross_processed)
    #Flatten spatial dimensions
    x = Reshape((height * width, channels))(input_layer)
    x = GroupNormalization(groups=num_groups)(x)

    #Flatten cross layer spatial dimensions
    cross_x = Reshape((cross_processed.shape[1] * cross_processed.shape[2], channels))(cross_processed)
    cross_x = GroupNormalization(groups=num_groups)(cross_x)

    #Compute queries from input, keys and values from cross layer
    q = Dense(channels)(x)
    k = Dense(channels)(cross_x)
    v = Dense(channels)(cross_x)

    #Cross multi-head attention
    attention_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embedding_dim,
        dtype=tf.bfloat16
    )(q, k, v)

    #Reshape back to spatial dimensions
    attention_output = Reshape((height, width, channels))(attention_output)

    #Residual connection with GroupNorm
    normalized = GroupNormalization(groups=num_groups)(attention_output)
    output = Add()([input_layer, normalized])

    return output


def EncoderBlock(input, features):
  conv = Conv2D(features, (3, 3), padding='same', activation='silu')(input)
  pool = AveragePooling2D(pool_size=(2, 2), padding='same')(conv)
  pool = Conv2D(features, (1, 1), padding='same', activation='silu')(pool)

  return conv, pool

def DecoderBlock(input, features, cropping, encoder_input):
  up = Conv2DTranspose(features, (3, 3), strides=(2, 2), padding='valid', activation='silu')(input)
  up = Cropping2D(cropping=cropping)(up)
  up = Conv2D(features, (1, 1), padding='same', activation='silu')(up)
  up = Add()([up, encoder_input])

  return up

def create_unet(input_shape=(368, 862, 15)):

    input_image = Input(input_shape)
    coordinate_mult_image = Input((input_image.shape[1], input_image.shape[2], 1))

    coord_mult_process = Reshape((input_image.shape[1] * input_image.shape[2], 1))(coordinate_mult_image)
    coord_mult_process = Dense(1, activation='sigmoid')(coord_mult_process)
    coord_mult_process = Reshape((input_image.shape[1], input_image.shape[2], 1))(coord_mult_process)

    x = Concatenate()([input_image, coord_mult_process]) #(368, 862, 16)
    x = ZeroPadding2D(padding=((0, 0), (1, 1)))(x)

    #=== Encoder ===
    conv1, pool1 = EncoderBlock(x, 32) #(368, 864, 32)

    conv2, pool2 = EncoderBlock(pool1, 64)

    conv3, pool3 = EncoderBlock(pool2, 128)

    conv4, pool4 = EncoderBlock(pool3, 256)

    conv5, pool5 = EncoderBlock(pool4, 512)

    #=== Bottleneck with Multi-Head Attention ===
    bottleneck = Conv2D(1024, (3, 3), padding='same', activation='silu')(pool5) #(12, 27, 1024)

    attn_output  = cross_attention_block_with_groupnorm(bottleneck, x, 16, 512, 32)
    attn_output2 = cross_attention_block_with_groupnorm(bottleneck, x, 12, 512, 32)

    bottleneck = LayerNormalization()(Concatenate()([attn_output, attn_output2]))
    bottleneck = Conv2D(1024, (1, 1), padding='same', activation='silu')(bottleneck)

    #=== Decoder ===
    up1 = DecoderBlock(bottleneck, 512, ((1, 1), (1, 0)), conv5)

    up2 = DecoderBlock(up1, 256, ((1, 0), (1, 0)), conv4)

    up3 = DecoderBlock(up2, 128, ((1, 0), (1, 0)), conv3)

    up4 = DecoderBlock(up3, 64, ((1, 0), (1, 0)), conv2)

    up5 = DecoderBlock(up4, 32, ((1, 0), (1, 0)), conv1)

    up5 = Cropping2D(cropping=((0, 0), (1, 1)))(up5)

    #=== Output Layer ===
    output_tensor = Conv2D(1, (1, 1), activation='linear', padding='valid', dtype=tf.float32)(up5)

    return Model(inputs=[input_image, coordinate_mult_image], outputs=output_tensor)


def extremely_noisy_processing(spectrogram,
                                offset_range=(10.0, 30.0),
                                noise_std=15.0,
                                clip_min=-80.0,
                                clip_max=40.0):
    """
    Takes a spectrogram in [-80, 0] dB, then adds a big random offset
    plus large normal noise to every bin. The result is clipped to a
    wider range so the model sees artificially 'loud' and messy values.

    Args:
        spectrogram (np.ndarray): Input spectrogram in dB, shape (...).
        offset_range (tuple): Range of random offset (dB) added to entire spectrogram.
                             e.g. (10, 30) means add between +10 dB and +30 dB.
        noise_std (float): Standard deviation of normal noise. The bigger, the noisier.
        clip_min (float): Minimum dB after corruption.
        clip_max (float): Maximum dB after corruption.

    Returns:
        np.ndarray: A heavily corrupted spectrogram (still in dB scale).
    """
    #1. Pick a random offset within offset_range
    offset = np.random.uniform(*offset_range)

    #2. Generate large noise
    noise = np.random.normal(loc=0.0, scale=noise_std, size=spectrogram.shape)

    #3. Add offset + noise to the entire spectrogram
    corrupted = spectrogram + offset + noise

    #4. Clip to keep final values within [clip_min, clip_max]
    corrupted = np.clip(corrupted, clip_min, clip_max)

    return corrupted


class SHDMAudioProcessor:
    
    def __init__(self):
        """
        Initialize the audio processor with an optional model path.
        """
        #Don't load TensorFlow dependencies until needed
        self.model = None
        self.tf_loaded = False

        #Initialize progress tracking
        self._reset_progress()

        self.timesteps = 1000
        self.beta_start = 0.0
        self.beta_end = 1.0
        self.betas = np.linspace(self.beta_start, self.beta_end, self.timesteps)
        self.alphas = 1 - self.betas

        #Check if running on PythonAnywhere
        self.is_pythonanywhere = os.environ.get('PYTHONANYWHERE', 'false') == 'true'
        
        #Set up model paths
        home_dir = os.path.expanduser('~')
        if self.is_pythonanywhere:
            self.model_dir = os.path.join(home_dir, 'Ficksaudio-Flask-Web', 'model_files')
            self.model_weights_path = os.path.join(self.model_dir, 'model_export')
            self.model_coord_mult_path = os.path.join(self.model_dir, 'coord_mult.npy')
            self.model_fourier_features_path = os.path.join(self.model_dir, 'fourier_features.npy')
        else:
            #Local development - use paths from .env file or default to models_data directory
            self.model_weights_path = os.environ.get('MODEL_WEIGHTS_PATH', 'models_data/best_model.weights.h5')
            self.model_coord_mult_path = os.environ.get('MODEL_COORD_MULT_PATH', 'models_data/coord_mult.npy')
            self.model_fourier_features_path = os.environ.get('MODEL_FOURIER_FEATURES_PATH', 'models_data/fourier_features.npy')
        
        self.sample_rate = 44100

        self.fourier_feature_timestep_freqs = tf.Variable(
            initial_value=np.random.normal(size=(368, 862, 7)) * 12.0,
            trainable=False,
            dtype=tf.float32,
            name="fourier_feature_timestep_freqs"
            )

        self.x_coords = np.arange(862) / 862
        self.y_coords = np.arange(368) / 368
        self.xv, self.yv = np.meshgrid(self.x_coords, self.y_coords)
        self.img_coordinates = self.xv * self.yv + 0.5 * (self.xv + self.yv)
        self.img_coordinates = np.reshape(self.img_coordinates, (1, 368, 862, 1))

        self.coord_mult = tf.Variable(initial_value=1.0,
                                      dtype=tf.float32,
                                      trainable=True,
                                      name="image_coord_multiplier"
                                      )

        #Create cache directories for the temporary data such as waveforms and spectrograms
        if self.is_pythonanywhere:
            self.cache_dir = os.path.join('/tmp', 'ficksaudio_cache')
        else:
            self.cache_dir = os.path.join('static', 'cache')
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        #Load the model
        self._load_model()
    
    def _reset_progress(self):
        """Reset progress tracking"""
        global _progress
        with _progress_lock:
            _progress = {
                'message': 'Initializing...',
                'current': 0,
                'total': 0,
                'done': False
            }
        
        try:
            # Attempt to store in Flask session if available
            if session is not None:
                session['progress_data'] = {'status': _progress}
                session.modified = True
        except (RuntimeError, ImportError):
            # Either we're outside a request context or Flask isn't available
            pass
            
    def _update_progress(self, message=None, current=None, total=None, done=False):
        """Update global progress information
        
        Args:
            message: Status message to display
            current: Current progress step
            total: Total number of steps
            done: Whether processing is complete
        """
        global _progress
        with _progress_lock:
            if message is not None:
                _progress['message'] = message
            if current is not None:
                _progress['current'] = current
            if total is not None:
                _progress['total'] = total
            
            # Only set done=True, never back to False
            if done:
                _progress['done'] = True
            
        # Log progress update for debugging
        print(f"Progress update: {current}/{total} - {message} (done: {done})")
        
        try:
            # Attempt to store in Flask session if available
            if session is not None:
                session['progress_data'] = {'status': _progress}
                session.modified = True
        except (RuntimeError, ImportError):
            # Either we're outside a request context or Flask isn't available
            pass
            
    def get_progress(self):
        """Return current progress information"""
        global _progress
        with _progress_lock:
            return _progress.copy()
    
    def _download_file(self, url: str, local_path: str) -> bool:
        """Download a file from a URL to a local path."""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception as e:
            print(f"Error downloading file: {e}")
            return False
    
    def _load_model(self):
        """Load the model and its weights."""
        if not self.tf_loaded:
            # Lazy load TensorFlow dependencies
            global tf
            import tensorflow as tf
            from tensorflow.keras.models import Model
            tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
            tf.config.run_functions_eagerly(True)
            self.tf_loaded = True
            
        try:
            print(f"Loading model from {self.model_weights_path}")
            
            if self.is_pythonanywhere:
                # For PythonAnywhere: use tf.saved_model.load
                if os.path.exists(self.model_weights_path):
                    print("Loading saved model from directory...")
                    self.model = tf.saved_model.load(self.model_weights_path)
                    print("Model loaded successfully!")
                else:
                    raise FileNotFoundError(f"Model directory not found at {self.model_weights_path}")
            else:
                # For local development: use create_unet and load_weights
                self.model = create_unet()
                
                if os.path.isdir(self.model_weights_path):
                    # If it's a directory, load as SavedModel
                    print("Loading as SavedModel...")
                    saved_model = tf.keras.models.load_model(self.model_weights_path)
                    # Copy weights from saved model to our model
                    self.model.set_weights(saved_model.get_weights())
                elif os.path.exists(self.model_weights_path):
                    # If it's a file, load as weights file
                    self.model.load_weights(self.model_weights_path)
                else:
                    raise FileNotFoundError(f"Model weights not found at {self.model_weights_path}")
            
            # Load auxiliary model data
            if os.path.exists(self.model_coord_mult_path):
                print(f"Loading coord_mult from {self.model_coord_mult_path}")
                self.coord_mult.assign(np.load(self.model_coord_mult_path))
            else:
                print(f"Warning: coord_mult file not found at {self.model_coord_mult_path}")
                
            if os.path.exists(self.model_fourier_features_path):
                print(f"Loading fourier_features from {self.model_fourier_features_path}")
                self.fourier_feature_timestep_freqs.assign(np.load(self.model_fourier_features_path))
            else:
                print(f"Warning: fourier_features file not found at {self.model_fourier_features_path}")
                
            print("Model loading complete!")
                
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def create_time_channels(self, x, t):
        """Create time channels for each pixel"""

        #Reshape t and broadcast it to match the spatial dimensions of x
        t = tf.reshape(t, [-1, 1, 1, 1])
        t = tf.broadcast_to(t, tf.shape(x))

        time_channel = tf.cast(t, tf.float32)

        #Fourier feature
        time_proj = time_channel * math.tau * self.fourier_feature_timestep_freqs

        fourier_features = tf.concat([tf.sin(time_proj), tf.cos(time_proj)], axis=-1)

        return fourier_features
    
    @tf.function
    def sample(self, x_input, num_inference_steps=60):
        """Multi-step inference process for iterative alpha deblending"""
        image = x_input

        t_mult = self.timesteps // (num_inference_steps)

        for t in reversed(range(num_inference_steps)):
          gc.collect()

          x_t = image

          x_t_coord_mult = self.img_coordinates + (x_t * self.coord_mult)

          #Inference steps won't match the timesteps range the model is trained on, so we have to map it properly
          t_for_timesteps = t * t_mult

          alpha_start = t / num_inference_steps
          alpha_end = (t + 1) / num_inference_steps

          #Create time tensor for the singular image
          t_batch = np.full(1, t_for_timesteps)

          time_channel = self.create_time_channels(x_t, t_batch)
          x_t_with_time = tf.concat([x_t, time_channel], axis=-1)

          #Model predicts y = x_1 - x_output
          y_pred = self.model([x_t_with_time, x_t_coord_mult])

          final_velocity = y_pred

          image = image + final_velocity * (alpha_end-alpha_start)


        return image
    
    def _load_audio(self, audio_path: str, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """
        Load an audio file and return the audio data and sample rate.
        
        Args:
            audio_path: Path to the audio file
            sr: Optional sample rate to resample to (defaults to self.sample_rate)
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            #Use the provided sample rate or default to self.sample_rate
            target_sr = sr if sr is not None else self.sample_rate
            
            # Load audio file
            audio_data, sample_rate = librosa.load(audio_path, sr=target_sr)
            return audio_data, sample_rate
        except Exception as e:
            raise ValueError(f"Error loading audio file: {str(e)}")
        
    def process_audio(self, audio_path: str, inference_steps: int = 50, enable_noise: bool = False, 
                 noise_offset: float = 20.0, noise_std: float = 15.0) -> Dict[str, Any]:
        """
        Process an audio file and return the enhanced version along with visualizations.
        
        Args:
            audio_path: Path to the input audio file
            inference_steps: Number of inference steps (20-100)
            enable_noise: Whether to apply noisy processing to the input
            noise_offset: Offset range for noisy processing (10-30)
            noise_std: Standard deviation for noise (5-25)
            
        Returns:
            Dictionary containing:
                - enhanced_audio_path: Path to the enhanced audio file
                - original_waveform: Waveform data of the original audio
                - enhanced_waveform: Waveform data of the enhanced audio
                - original_spectrogram: Spectrogram data of the original audio
                - enhanced_spectrogram: Spectrogram data of the enhanced audio
        """
        #Store inference steps
        self.inference_steps = max(20, min(100, inference_steps))
        
        try:
            #Load and preprocess the audio
            audio_data, sample_rate = self._load_audio(audio_path)
            
            #Generate visualizations for the original audio
            original_waveform = self._generate_waveform(audio_data)
            original_spectrogram_data = self.convert_audio_to_spectrogram(audio_data)
            
            #Apply noisy processing if enabled
            if enable_noise:
                print(f"Applying noisy processing with offset={noise_offset}, std={noise_std}")
                original_spectrogram_data = extremely_noisy_processing(
                    original_spectrogram_data,
                    offset_range=(noise_offset, noise_offset + 10.0),
                    noise_std=noise_std,
                    clip_min=-80.0,
                    clip_max=40.0
                )
            
            original_spectrogram = self._process_spectrogram_for_visualization(original_spectrogram_data)
            
            #Apply the model to enhance the audio
            enhanced_audio, temp_path = self._enhance_audio(audio_data)
            
            #Generate visualizations for the enhanced audio
            enhanced_waveform = self._generate_waveform(enhanced_audio)
            enhanced_spectrogram_data = self.convert_audio_to_spectrogram(enhanced_audio)
            
            #Process spectrogram for visualization
            enhanced_spectrogram = self._process_spectrogram_for_visualization(enhanced_spectrogram_data)
            
            #Save the enhanced audio
            enhanced_audio_path = self._save_enhanced_audio(enhanced_audio, audio_path, temp_path)
            
            #Convert the enhanced_audio_path to a relative URL path for the web interface
            if self.is_pythonanywhere:
                #For PythonAnywhere, we need to create a URL that points to the media URL
                enhanced_filename = os.path.basename(enhanced_audio_path)
                enhanced_url_path = f'/media/enhanced/{enhanced_filename}'
            elif os.environ.get('VERCEL_ENV') == 'production':
                #For Vercel, we use the /tmp URL pattern
                enhanced_filename = os.path.basename(enhanced_audio_path)
                enhanced_url_path = f'/tmp/enhanced/{enhanced_filename}'
            else:
                #For local development
                if enhanced_audio_path.startswith(os.path.join('static', 'cache')):
                    #Already a relative path within static
                    enhanced_url_path = enhanced_audio_path.replace('\\', '/').replace('static/', '')
                else:
                    #Full path, extract filename and use default location
                    enhanced_filename = os.path.basename(enhanced_audio_path)
                    enhanced_url_path = f'cache/enhanced/{enhanced_filename}'
            
            print(f"Enhanced audio URL path: {enhanced_url_path}")
            
            return {
                'enhanced_audio_path': enhanced_url_path,
                'original_waveform': original_waveform,
                'enhanced_waveform': enhanced_waveform,
                'original_spectrogram': original_spectrogram,
                'enhanced_spectrogram': enhanced_spectrogram
            }
        except Exception as e:
            print(f"Error in process_audio: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def get_progress(self):
        """Return current progress information"""
        global _progress
        with _progress_lock:
            return _progress.copy()
    
    def _generate_waveform(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Generate waveform data for visualization.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Dictionary with waveform data
        """
        # For simplicity, we'll downsample the audio data to reduce size
        # In a real application, you might want to use a more sophisticated approach
        if len(audio_data) > 1000:
            indices = np.linspace(0, len(audio_data) - 1, 1000, dtype=int)
            downsampled_data = audio_data[indices]
        else:
            downsampled_data = audio_data
        
        return {
            'data': downsampled_data.tolist(),
            'max': float(np.max(np.abs(downsampled_data))),
            'min': float(np.min(downsampled_data))
        }
    
    def convert_audio_to_spectrogram(self, audio, sr=44100, n_fft=2048, hop_length=512):
        """
        Takes a single audio file and turns it into a visual representation called a spectrogram.
        Returns the spectrogram array directly for use, rather than saving to disk.
        This function is designed to work with multiple processes at once for speed.
        
        Args:
            audio: Audio data as numpy array
            sr: Sample rate (default: 44100)
            n_fft: FFT window size (default: 2048)
            hop_length: Hop length for STFT (default: 512)
            
        Returns:
            Spectrogram as numpy array with shape (368, time_steps, 1)
        """
        try:
            #Ensure audio is a numpy array
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio)
            
            #Ensure audio is not empty
            if len(audio) == 0:
                raise ValueError("Audio data is empty")
            
            fmax = sr / 2  #Nyquist frequency
            
            #Create mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=368, 
                window="hann", norm='slaney', fmax=fmax
            )

            #Convert to decibel scale (logarithmic)
            spectrogram = librosa.power_to_db(mel_spec, ref=np.max)

            #Reshape to (368, time_steps, 1) for model input
            spectrogram = spectrogram.reshape(spectrogram.shape[0], spectrogram.shape[1], 1)
            
            print(f"Generated spectrogram with shape: {spectrogram.shape}")

            return spectrogram
        except Exception as e:
            print(f"Error converting audio to spectrogram: {str(e)}")
            raise ValueError(f"Error converting audio to spectrogram: {str(e)}")
    
    def convert_spectrogram_to_audio(self, spectrogram, output_path, sr=44100, n_fft=2048, hop_length=512, n_iter=128):
        """
        Converts a spectrogram back to audio and saves it to disk.
        """
        try:
            #Ensure the spectrogram is 2D (mel bins x time)
            if len(spectrogram.shape) > 2:
                print(f"Warning: Spectrogram has shape {spectrogram.shape}, expected 2D. Squeezing dimensions.")
                spectrogram = np.squeeze(spectrogram)
            
            print(f"Converting spectrogram with shape {spectrogram.shape} to audio")
            
            #Convert from decibels back to power spectrogram
            power_spec = librosa.db_to_power(spectrogram)

            fmax = sr / 2  #Nyquist frequency

            #Reconstruct audio from mel spectrogram
            stft_spec = librosa.feature.inverse.mel_to_stft(
                power_spec, sr=sr, n_fft=n_fft, norm='slaney', fmax=fmax
            )

            #Use Griffin-Lim algorithm to approximate the phase
            audio = librosa.griffinlim(
                stft_spec, n_iter=n_iter, hop_length=hop_length, win_length=n_fft
            )

            #Normalize audio to prevent clipping
            audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio

            #Ensure the directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            #Write to output file
            sf.write(output_path, audio, sr)

            return True

        except Exception as e:
            print(f"Error converting spectrogram to audio: {str(e)}")
            return False
    
    def _process_spectrogram_for_visualization(self, spectrogram_data):
        """
        Process the spectrogram data for visualization in the web interface.
        
        Args:
            spectrogram_data: The raw spectrogram data, which can be 2D, 3D, or 4D
            
        Returns:
            Dictionary with processed spectrogram data for visualization
        """
        #Handle different input shapes
        print(f"Processing spectrogram for visualization with shape: {spectrogram_data.shape}")
        
        #Extract the 2D spectrogram from the array (handle different dimensions)
        if len(spectrogram_data.shape) == 4:  # (batch, height, width, channels)
            spec_2d = spectrogram_data[0, :, :, 0]
        elif len(spectrogram_data.shape) == 3:  # (height, width, channels)
            spec_2d = spectrogram_data[:, :, 0]
        elif len(spectrogram_data.shape) == 2:  # (height, width)
            spec_2d = spectrogram_data
        else:
            raise ValueError(f"Unexpected spectrogram shape: {spectrogram_data.shape}")
        
        print(f"Extracted 2D spectrogram with shape: {spec_2d.shape}")
        
        #Downsample for visualization if needed
        if spec_2d.shape[1] > 200:
            indices = np.linspace(0, spec_2d.shape[1] - 1, 200, dtype=int)
            spec_2d = spec_2d[:, indices]
        
        if spec_2d.shape[0] > 200:
            indices = np.linspace(0, spec_2d.shape[0] - 1, 200, dtype=int)
            spec_2d = spec_2d[indices, :]
        
        #Ensure we don't have NaN or Inf values
        spec_2d = np.nan_to_num(spec_2d, nan=0.0, posinf=0.0, neginf=0.0)
        
        return {
            'data': spec_2d.tolist(),
            'max': float(np.max(spec_2d)),
            'min': float(np.min(spec_2d))
        }
    
    def _enhance_audio(self, audio_data: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Process audio in 10-second segments using the sample function.
        Process multiple segments in parallel for improved performance.
        Concatenates spectrograms along the time axis and converts back to audio.
        
        Returns:
            Tuple of (enhanced_audio_data, temp_file_path)
        """
        #Initialize progress tracking
        self._update_progress(message="Loading model...")

        #Calculate segment parameters
        segment_length = 10 * self.sample_rate  #10 seconds of audio
        
        #Calculate number of segments
        total_segments = math.ceil(len(audio_data) / segment_length)
        self._update_progress(message=f"Processing {total_segments} segments...", total=total_segments)
        
        #Initialize array for enhanced spectrograms
        all_spectrograms = []
        
        #First, get the original full spectrogram to determine final dimensions
        original_full_spectrogram = self.convert_audio_to_spectrogram(audio_data)
        original_shape = original_full_spectrogram.shape
        print(f"Original full spectrogram shape: {original_shape}")
        
        #Determine batch size for parallel processing
        #If segments <= 3, process all at once, otherwise use 3 at a time
        batch_size = min(3, total_segments)
        
        #Process segments in batches
        for batch_idx in range(0, total_segments, batch_size):
            batch_spectrograms = []
            batch_end = min(batch_idx + batch_size, total_segments)
            batch_segments = range(batch_idx, batch_end)
            
            # Update progress
            self._update_progress(message=f"Processing segments {batch_idx+1}-{batch_end} of {total_segments}", 
                                 current=batch_idx+1)
            
            # Process each segment in this batch
            for i in batch_segments:
                #Extract segment
                start_idx = i * segment_length
                end_idx = min((i + 1) * segment_length, len(audio_data))
                segment = audio_data[start_idx:end_idx]
                
                #Pad last segment if needed
                if len(segment) < segment_length:
                    padding = np.zeros(segment_length - len(segment))
                    segment = np.concatenate([segment, padding])
                
                #Convert to spectrogram
                spec = self.convert_audio_to_spectrogram(segment)

                #Add batch dimension for model input
                spec = np.expand_dims(spec, axis=0)

                print(f"Input spectrogram shape: {spec.shape}")

                #Convert to tensor
                spec = tf.convert_to_tensor(spec, dtype=tf.float32)
                
                #Sample the spectrogram
                enhanced_spec = self.sample(
                    x_input=spec,
                    num_inference_steps=self.inference_steps
                )
                
                #Convert from tensor to numpy
                if isinstance(enhanced_spec, tf.Tensor):
                    enhanced_spec = enhanced_spec.numpy()
                
                print(f"Enhanced spectrogram shape: {enhanced_spec.shape}")
                
                #Store prediction
                batch_spectrograms.append(enhanced_spec)
                
                # Update progress for this specific segment
                self._update_progress(
                    message=f"Processed segment {i+1} of {total_segments}",
                    current=i+1
                )
            
            # Add completed batch spectrograms to main list
            all_spectrograms.extend(batch_spectrograms)
            
            # Clean up TensorFlow memory between batches
            tf.keras.backend.clear_session()
            gc.collect()
        
        #Concatenate all spectrograms along the time axis (axis=2)
        #Each spectrogram has shape (1, 368, 862, 1)
        print(f"Number of spectrograms to concatenate: {len(all_spectrograms)}")
        
        if len(all_spectrograms) == 1:
            final_spectrogram = all_spectrograms[0]
        else:
            #Concatenate along the time axis (axis=2)
            try:
                final_spectrogram = np.concatenate(all_spectrograms, axis=2)
                print(f"Concatenated spectrogram shape: {final_spectrogram.shape}")
            except Exception as e:
                print(f"Error concatenating spectrograms: {str(e)}")
                #Fallback: use the first spectrogram if concatenation fails
                final_spectrogram = all_spectrograms[0]
        
        #Remove batch dimension and channel dimension for audio conversion
        final_spectrogram = np.squeeze(final_spectrogram)
        print(f"Final spectrogram shape after squeeze: {final_spectrogram.shape}")
        
        #Trim or pad the final spectrogram to match the original spectrogram's time dimension
        if final_spectrogram.shape[1] > original_shape[1]:
            print(f"Trimming final spectrogram from {final_spectrogram.shape} to match original width {original_shape[1]}")
            final_spectrogram = final_spectrogram[:, :original_shape[1]]
        elif final_spectrogram.shape[1] < original_shape[1]:
            print(f"Padding final spectrogram from {final_spectrogram.shape} to match original width {original_shape[1]}")
            pad_width = original_shape[1] - final_spectrogram.shape[1]
            final_spectrogram = np.pad(final_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
            print(f"Padded spectrogram shape: {final_spectrogram.shape}")
        
        #Update progress for conversion back to audio
        self._update_progress(message="Converting enhanced spectrogram to audio...")
        
        #Convert concatenated spectrogram back to audio
        temp_path = os.path.join(self.cache_dir, 'temp_enhanced.wav')
        
        #Ensure the cache directory exists
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        
        conversion_success = self.convert_spectrogram_to_audio(final_spectrogram, temp_path)
        
        if not conversion_success or not os.path.exists(temp_path):
            print(f"Failed to convert spectrogram to audio or file not found at {temp_path}")
            #Return original audio as fallback
            return audio_data, ""
        
        # Mark processing as complete
        self._update_progress(message="Processing complete!", done=True)
        
        #Load the converted audio but DO NOT delete the temp file
        try:
            enhanced_audio, _ = librosa.load(temp_path, sr=self.sample_rate)
            return enhanced_audio, temp_path
        except Exception as e:
            print(f"Error loading enhanced audio: {e}")
            return audio_data, ""
    
    def _save_enhanced_audio(self, enhanced_audio: np.ndarray, original_path: str, temp_path: str = "") -> str:
        """
        Save the enhanced audio to a file.
        
        Args:
            enhanced_audio: Enhanced audio data
            original_path: Path to the original audio file
            temp_path: Path to the temporary enhanced audio file (if available)
            
        Returns:
            Path to the saved enhanced audio file
        """
        try:
            #Determine output directory based on environment
            if self.is_pythonanywhere:
                output_dir = os.path.join('/tmp', 'ficksaudio_enhanced')
            else:
                output_dir = os.path.join('static', 'cache', 'enhanced')
            
            #Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            #Generate output filename
            filename = os.path.basename(original_path)
            name, ext = os.path.splitext(filename)
            enhanced_filename = f"{name}_enhanced{ext}"
            enhanced_path = os.path.join(output_dir, enhanced_filename)
            
            #If we have a valid temp file, just copy/move it instead of reprocessing
            if temp_path and os.path.exists(temp_path):
                print(f"Using existing temp file: {temp_path}")
                
                #Copy the temp file to the final location
                import shutil
                shutil.copy2(temp_path, enhanced_path)
                
                #Clean up the temp file after copying
                try:
                    os.remove(temp_path)
                    print(f"Temp file removed: {temp_path}")
                except Exception as e:
                    print(f"Warning: Could not remove temp file {temp_path}: {str(e)}")
                
                print(f"Enhanced audio saved to {enhanced_path}")
                return enhanced_path
            
            #If no temp file, process the audio data directly
            #Ensure audio is not empty and has no NaN values
            if len(enhanced_audio) == 0:
                raise ValueError("Enhanced audio data is empty")
            
            enhanced_audio = np.nan_to_num(enhanced_audio, nan=0.0)
            
            #Normalize audio to prevent clipping
            if np.max(np.abs(enhanced_audio)) > 0:
                enhanced_audio = enhanced_audio / np.max(np.abs(enhanced_audio))
            
            #Save the enhanced audio
            sf.write(enhanced_path, enhanced_audio, self.sample_rate)
            
            print(f"Enhanced audio saved to {enhanced_path}")
            return enhanced_path
        except Exception as e:
            print(f"Error saving enhanced audio: {str(e)}")
            #Return original path as fallback
            return original_path 