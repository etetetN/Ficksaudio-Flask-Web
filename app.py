from flask import Flask, render_template, request, jsonify, url_for, redirect, flash, Response
import os
from werkzeug.utils import secure_filename
from audio_processor import SHDMAudioProcessor
import json
import time
from werkzeug.middleware.proxy_fix import ProxyFix

#Load environment variables from .env file if it exists
try:
    from load_env import load_dotenv
    load_dotenv()
except ImportError:
    pass  #No .env file or load_env.py not found, continue with environment variables as is

# Set PythonAnywhere environment variable if not already set
os.environ['PYTHONANYWHERE'] = os.environ.get('PYTHONANYWHERE', 'false')

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
app.secret_key = os.environ.get('SECRET_KEY', 'ficksaudio_secret_key')  

#Check if running on PythonAnywhere
is_pythonanywhere = os.environ.get('PYTHONANYWHERE') == 'true'

#Configure upload folder - using appropriate directory based on environment
if is_pythonanywhere:
    home_dir = os.path.expanduser('~')
    UPLOAD_FOLDER = os.path.join(home_dir, 'ficksaudio_uploads')
    ENHANCED_FOLDER = os.path.join(home_dir, 'ficksaudio_enhanced')
else:
    UPLOAD_FOLDER = '/tmp/uploads' if os.environ.get('VERCEL_ENV') == 'production' else os.path.join('static', 'cache', 'uploads')
    ENHANCED_FOLDER = '/tmp/enhanced' if os.environ.get('VERCEL_ENV') == 'production' else os.path.join('static', 'cache', 'enhanced')

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # Default: 16MB max file size

#Create cache directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ENHANCED_FOLDER, exist_ok=True)

#Initialize the audio processor
audio_processor = SHDMAudioProcessor()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_old_files():
    """Clean up files older than 1 hour"""
    current_time = time.time()
    for folder in [UPLOAD_FOLDER, ENHANCED_FOLDER]:
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if os.path.getmtime(filepath) < current_time - 3600:  #1 hour
                try:
                    os.remove(filepath)
                except:
                    pass

@app.route('/')
def home():
    cleanup_old_files()  #Clean up old files on home page visits
    return render_template('index.html', title='Home')

@app.route('/process', methods=['GET', 'POST'])
def process():
    if request.method == 'POST':
        cleanup_old_files()  #Clean up before processing new file
        
        #Check if the post request has the file part
        if 'audio_file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        
        file = request.files['audio_file']
        inference_steps = int(request.form.get('inference_steps', 50))
        
        #Get noisy processing parameters
        enable_noise = request.form.get('enable_noise') == 'on'
        noise_offset = float(request.form.get('noise_offset', 20))
        noise_std = float(request.form.get('noise_std', 15))
        
        #If user does not select file, browser also submits an empty part
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                #Process the audio file
                result = audio_processor.process_audio(
                    filepath, 
                    inference_steps=inference_steps,
                    enable_noise=enable_noise,
                    noise_offset=noise_offset,
                    noise_std=noise_std
                )
                
                #Get the paths for the original and enhanced audio
                #Use forward slashes for URLs, regardless of OS
                original_audio = 'cache/uploads/' + filename
                enhanced_audio = result['enhanced_audio_path']
                
                return render_template('results.html', 
                                      title='Results',
                                      original_audio=original_audio,
                                      fixed_audio=enhanced_audio,
                                      original_waveform=result['original_waveform'],
                                      enhanced_waveform=result['enhanced_waveform'],
                                      original_spectrogram=result['original_spectrogram'],
                                      enhanced_spectrogram=result['enhanced_spectrogram'])
            except Exception as e:
                flash(f'Error processing audio: {str(e)}', 'error')
                return redirect(request.url)
        else:
            flash('File type not allowed. Please upload WAV, MP3, OGG, or FLAC files.', 'error')
            return redirect(request.url)
    
    return render_template('process.html', title='Fix Audio')

@app.route('/progress')
def progress():
    def generate():
        while True:
            progress = audio_processor.get_progress()
            yield f"data: {json.dumps(progress)}\n\n"
            if progress['done']:
                break
            time.sleep(0.1)#Update every 100ms
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    if os.environ.get('PYTHONANYWHERE') == 'true':
        #It will be run by the WSGI server
        pass
    elif os.environ.get('VERCEL_ENV') == 'production':
        app.run()
    else:
        app.run(debug=True) 