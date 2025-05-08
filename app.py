from flask import Flask, render_template, request, jsonify, url_for, redirect, flash, Response, current_app, session
import os
from werkzeug.utils import secure_filename
from audio_processor import SHDMAudioProcessor
import json
import time
from werkzeug.middleware.proxy_fix import ProxyFix
from pathlib import Path
import sys
import uuid
import threading

#Attempt to load environment variables from .env file using our custom dotenv loader
try:
    from load_env import load_dotenv
    #Load environment from .env file - the function will search in multiple locations including PythonAnywhere paths
    if load_dotenv():
        print("Environment variables loaded from .env file")
    else:
        print("No .env file found, using existing environment variables")
except ImportError:
    print("load_env.py not found, continuing with environment variables as is")

#Ensure PYTHONANYWHERE environment variable exists with a default value of 'false'
os.environ['PYTHONANYWHERE'] = os.environ.get('PYTHONANYWHERE', 'false')

app = Flask(__name__)
#Add ProxyFix middleware to handle proxy headers correctly
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
#Set secret key for session management from environment or use default
app.secret_key = os.environ.get('SECRET_KEY', 'ficksaudio_secret_key')  

#Determine if we're running on PythonAnywhere to configure appropriate paths
is_pythonanywhere = os.environ.get('PYTHONANYWHERE') == 'true'

#Configure upload and enhanced folders based on environment
if is_pythonanywhere:
    #Use /tmp directory on PythonAnywhere to avoid permission issues with file operations
    UPLOAD_FOLDER = os.path.join('/tmp', 'ficksaudio_uploads')
    ENHANCED_FOLDER = os.path.join('/tmp', 'ficksaudio_enhanced')
else:
    #For local development, use subdirectories in the static/cache folder
    UPLOAD_FOLDER = os.path.join('static', 'cache', 'uploads')
    ENHANCED_FOLDER = os.path.join('static', 'cache', 'enhanced')

#Set allowed audio file extensions for uploads
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#Set maximum upload size to 16MB by default, or use environment variable if specified
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))

#Initialize audio processor at startup and store as a global variable
audio_processor = SHDMAudioProcessor()

#Store background tasks with their status
background_tasks = {}

def get_audio_processor():
    """Return the global audio processor instance"""
    global audio_processor
    return audio_processor

def allowed_file(filename):
    """Check if a file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#Ensure upload directories exist
for folder in [UPLOAD_FOLDER, ENHANCED_FOLDER]:
    os.makedirs(folder, exist_ok=True)

@app.route('/')
def index():
    """Render the homepage"""
    cleanup_old_files()  # Perform cleanup on home page visits to maintain disk space
    return render_template('index.html', title='Home')

@app.route('/process', methods=['GET', 'POST'])
def process():
    if request.method == 'POST':
        cleanup_old_files()  #Clean temporary files before processing new ones
        
        #Verify the file was included in the request
        if 'audio_file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        
        file = request.files['audio_file']
        #Get inference steps parameter (controls quality vs. speed tradeoff)
        inference_steps = int(request.form.get('inference_steps', 50))
        
        #Extract optional noise processing parameters from the form
        enable_noise = request.form.get('enable_noise') == 'on'
        noise_offset = float(request.form.get('noise_offset', 20))
        noise_std = float(request.form.get('noise_std', 15))
        
        #Handle case where user submits form without selecting a file
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            #Secure the filename to prevent directory traversal attacks
            filename = secure_filename(file.filename)
            
            #Generate a unique task ID
            task_id = str(uuid.uuid4())
            
            #Store the task ID in the session
            session['task_id'] = task_id
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            #Save the uploaded file to the server
            file.save(filepath)
            
            #Initialize task in background_tasks
            background_tasks[task_id] = {
                'status': 'starting',
                'message': 'Preparing to process',
                'current': 0,
                'total': 0,
                'result': None,
                'error': None,
                'done': False,
                'filepath': filepath
            }
            
            #Start processing in a background thread
            processing_thread = threading.Thread(
                target=process_audio_task,
                args=(task_id, filepath, inference_steps, enable_noise, noise_offset, noise_std)
            )
            processing_thread.daemon = True
            processing_thread.start()
            
            #Redirect to processing page
            return redirect(url_for('process_page'))
        else:
            #Inform user about allowed file types if they upload an unsupported format
            flash('File type not allowed. Please upload WAV, MP3, OGG, or FLAC files.', 'error')
            return redirect(request.url)
    
    #For GET request, show the processing form page
    return render_template('process.html', title='Fix Audio')

def process_audio_task(task_id, filepath, inference_steps, enable_noise, noise_offset, noise_std):
    """Process audio in a background thread"""
    try:
        # Get processor reference at the start
        processor = get_audio_processor()
        
        # Update task status and ensure it's in the dictionary
        if task_id in background_tasks:
            background_tasks[task_id]['status'] = 'processing'
            background_tasks[task_id]['message'] = 'Starting audio processing'
            background_tasks[task_id]['filepath'] = filepath  # Store the input filepath
            print(f"Beginning processing for task {task_id}")
        
        # Set up a function to sync processor progress to task status
        def sync_progress():
            if task_id in background_tasks:
                try:
                    progress = processor.get_progress()
                    background_tasks[task_id].update({
                        'message': progress.get('message', 'Processing...'),
                        'current': progress.get('current', 0),
                        'total': progress.get('total', 0),
                        'done': progress.get('done', False)
                    })
                    # Print status for server logs
                    if os.environ.get('PYTHONANYWHERE') == 'true':
                        print(f"Task {task_id} progress: {progress.get('current', 0)}/{progress.get('total', 0)} - {progress.get('message', 'Processing...')}")
                except Exception as e:
                    print(f"Error syncing progress: {e}")
        
        # Set up a timer to regularly sync progress (every 2 seconds)
        # This ensures background_tasks is updated even if no client is requesting progress
        sync_timer = None
        
        def start_sync_timer():
            def sync_and_reschedule():
                sync_progress()
                # Only reschedule if task is still active
                if task_id in background_tasks and not background_tasks[task_id].get('done', False):
                    start_sync_timer()
            
            nonlocal sync_timer
            if sync_timer:
                try:
                    sync_timer.cancel()
                except:
                    pass
            sync_timer = threading.Timer(2.0, sync_and_reschedule)
            sync_timer.daemon = True
            sync_timer.start()
        
        # Start the sync timer
        start_sync_timer()
        
        # Process the audio file with the configured parameters
        try:
            # Initial sync before processing starts
            sync_progress()
            
            result = processor.process_audio(
                filepath, 
                inference_steps=inference_steps,
                enable_noise=enable_noise,
                noise_offset=noise_offset,
                noise_std=noise_std
            )
            
            # Final sync after processing
            sync_progress()
            
            # Store result for later retrieval
            if task_id in background_tasks:
                background_tasks[task_id]['result'] = result
                background_tasks[task_id]['status'] = 'completed'
                background_tasks[task_id]['message'] = 'Processing complete'
                background_tasks[task_id]['done'] = True
                print(f"Task {task_id} completed successfully")
            
        except Exception as e:
            # Handle processing errors
            error_msg = f"Error processing audio: {str(e)}"
            if task_id in background_tasks:
                background_tasks[task_id]['status'] = 'error'
                background_tasks[task_id]['message'] = error_msg
                background_tasks[task_id]['error'] = str(e)
                background_tasks[task_id]['done'] = True
            print(f"Error in processing task {task_id}: {error_msg}")
        
        # Cancel the timer when done
        if sync_timer:
            sync_timer.cancel()
        
    except Exception as e:
        # Handle unexpected errors
        print(f"Unexpected error in background task {task_id}: {str(e)}")
        if task_id in background_tasks:
            background_tasks[task_id]['status'] = 'error'
            background_tasks[task_id]['message'] = f"Unexpected error: {str(e)}"
            background_tasks[task_id]['error'] = str(e)
            background_tasks[task_id]['done'] = True

@app.route('/progress')
def progress():
    """Stream progress updates to the client using Server-Sent Events (SSE)"""
    task_id = session.get('task_id')
    
    # PythonAnywhere has a 60-second timeout for requests
    # We'll limit the SSE duration to 50 seconds to be safe
    max_duration = 50  # seconds
    
    def generate():
        try:
            last_data = None
            start_time = time.time()
            
            while True:
                try:
                    # Check if we've exceeded the max duration
                    if time.time() - start_time > max_duration:
                        print("SSE max duration reached, ending stream")
                        yield f"data: {json.dumps({'message': 'Stream timeout - please reconnect for updates', 'reconnect': True, 'timestamp': int(time.time() * 1000)})}\n\n"
                        break
                    
                    # Get task progress
                    if task_id and task_id in background_tasks:
                        progress = background_tasks[task_id].copy()
                        
                        # If task has no progress data, get it from processor
                        if progress['current'] == 0 and progress['total'] == 0 and not progress.get('result'):
                            processor_progress = get_audio_processor().get_progress()
                            progress.update({
                                'message': processor_progress.get('message', progress.get('message', '')),
                                'current': processor_progress.get('current', 0),
                                'total': processor_progress.get('total', 0),
                                'done': processor_progress.get('done', False) or progress.get('done', False)
                            })
                    else:
                        # Fall back to processor progress if no task
                        progress = get_audio_processor().get_progress()
                    
                    #Only send data if it's changed since last update to reduce bandwidth
                    current_data = json.dumps(progress)
                    if current_data != last_data:
                        last_data = current_data
                        #Add timestamp to prevent browser caching and help with debugging
                        progress['timestamp'] = int(time.time() * 1000)
                        yield f"data: {json.dumps(progress)}\n\n"
                        
                        #Force buffer flush to ensure immediate delivery to client
                        sys.stdout.flush()
                        
                    # Exit the generator when processing completes
                    if progress.get('done', False) or progress.get('status') == 'completed':
                        yield f"data: {json.dumps({'message': 'Complete!', 'done': True, 'timestamp': int(time.time() * 1000)})}\n\n"
                        break
                        
                    #Sleep to reduce server load (500ms provides good balance between responsiveness and resource usage)
                    time.sleep(0.5)
                    
                except Exception as e:
                    # Report errors to both server log and client
                    print(f"Error getting progress: {str(e)}")
                    yield f"data: {json.dumps({'message': f'Error: {str(e)}', 'error': True, 'timestamp': int(time.time() * 1000)})}\n\n"
                    time.sleep(1)
                    
        except (GeneratorExit, BrokenPipeError, ConnectionError) as e:
            # Handle client disconnection gracefully without error
            print(f"Client disconnected from progress stream: {str(e)}")
            return
            
    #Configure response with appropriate headers for SSE streaming
    response = Response(generate(), mimetype='text/event-stream')
    #Prevent caching of event stream data
    response.headers['Cache-Control'] = 'no-cache, no-transform'
    #Disable buffering in nginx for immediate delivery
    response.headers['X-Accel-Buffering'] = 'no'
    #Keep connection alive for continuous updates
    response.headers['Connection'] = 'keep-alive'
    #Use chunked encoding for streaming
    response.headers['Transfer-Encoding'] = 'chunked'
    return response

@app.route('/progress_status')
def progress_status():
    """Polling endpoint to check progress (fallback for when SSE doesn't work)"""
    task_id = session.get('task_id')
    
    # Add timestamp to response for cache busting
    timestamp = int(time.time() * 1000)
    
    if not task_id or task_id not in background_tasks:
        # No task ID or task not found
        return jsonify({
            'status': 'unknown',
            'message': 'No active processing task found',
            'current': 0,
            'total': 0,
            'done': False,
            'timestamp': timestamp
        })
    
    task_data = background_tasks[task_id].copy()
    
    # If there's no specific data from background task, try to get it from audio processor
    if task_data['current'] == 0 and task_data['total'] == 0 and not task_data.get('result'):
        try:
            progress_data = get_audio_processor().get_progress()
            # Don't overwrite 'status' or 'result' fields from the task
            task_data.update({
                'message': progress_data.get('message', task_data.get('message')),
                'current': progress_data.get('current', 0),
                'total': progress_data.get('total', 0),
                'done': progress_data.get('done', False) or task_data.get('done', False)
            })
        except Exception as e:
            print(f"Error getting processor progress: {e}")
    
    # Add timestamp to prevent caching
    task_data['timestamp'] = timestamp
    
    # Additional debugging for PythonAnywhere
    if os.environ.get('PYTHONANYWHERE') == 'true':
        print(f"Progress update: {task_data['current']}/{task_data['total']} - {task_data['message']} (done: {task_data['done']})")
    
    return jsonify(task_data)

@app.route('/task_result')
def task_result():
    """Get the result of a completed task"""
    task_id = session.get('task_id')
    
    if not task_id or task_id not in background_tasks:
        flash('Task not found or has expired', 'error')
        return redirect(url_for('process'))
    
    task_data = background_tasks[task_id]
    
    if task_data['status'] == 'error':
        flash(f"Error processing audio: {task_data['error']}", 'error')
        return redirect(url_for('process'))
    
    if task_data['status'] != 'completed':
        flash('Processing is not complete yet', 'warning')
        return redirect(url_for('process_page'))
    
    result = task_data['result']
    if not result:
        flash('Processing result not found', 'error')
        return redirect(url_for('process'))
    
    # Clean up task data from memory after retrieving result
    # but keep it around for a bit in case of page refreshes
    # Use a separate thread to clean it up after a delay
    def cleanup_task():
        time.sleep(300)  # Keep result for 5 minutes
        if task_id in background_tasks:
            del background_tasks[task_id]
    
    cleanup_thread = threading.Thread(target=cleanup_task)
    cleanup_thread.daemon = True
    cleanup_thread.start()
    
    # Prepare file paths for template rendering
    # Get the original filepath that was processed for this task
    original_file_path = task_data.get('filepath', '')
    if original_file_path:
        original_audio = os.path.basename(original_file_path)
        original_audio = f"cache/uploads/{original_audio}"
    else:
        # Fallback if filepath wasn't stored
        original_audio = "cache/uploads/unknown_original.wav" 
    enhanced_audio = result['enhanced_audio_path']
    
    # Render results page with all processed audio data and visualizations
    return render_template('results.html', 
                          title='Results',
                          original_audio=original_audio,
                          fixed_audio=enhanced_audio,
                          original_waveform=result['original_waveform'],
                          enhanced_waveform=result['enhanced_waveform'],
                          original_spectrogram=result['original_spectrogram'],
                          enhanced_spectrogram=result['enhanced_spectrogram'])

@app.route('/process_page')
def process_page():
    """Show processing page with progress updates"""
    return render_template('process.html', title='Processing Audio', processing=True)

@app.route('/enhance', methods=['POST'])
def enhance_audio():
    """
    Enhance uploaded audio file and return processed file path
    """
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['audio_file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        #Generate a unique filename using UUID to prevent collisions
        filename = secure_filename(file.filename)
        _, ext = os.path.splitext(filename)
        unique_filename = f"{uuid.uuid4()}{ext}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        #Ensure the upload directory exists before saving
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        #Save the uploaded file to disk
        file.save(file_path)
        
        #Store file path in session for later reference
        session['original_file'] = file_path
        
        #Initialize progress tracking in session
        session['progress_data'] = {
            'status': {
                'message': 'File uploaded successfully. Starting processing...',
                'current': 0,
                'total': 0,
                'done': False
            }
        }
        session.modified = True
        
        #Process audio in background thread to avoid blocking response
        def process_audio_task():
            try:
                audio_processor = get_audio_processor()
                enhanced_path = audio_processor.process_audio(file_path)
                
                #Store the result path in session for the results page
                session['enhanced_file'] = enhanced_path
                
                #Update progress data in session to indicate completion
                session['progress_data']['status'] = {
                    'message': 'Processing complete!',
                    'current': 1,
                    'total': 1,
                    'percent': 1.0,
                    'done': True
                }
                session.modified = True
                
                print(f"Audio processing complete: {enhanced_path}")
            except Exception as e:
                #Handle and log processing errors
                error_msg = f"Error processing audio: {str(e)}"
                print(error_msg)
                session['progress_data']['status'] = {
                    'message': error_msg,
                    'error': True,
                    'done': True
                }
                session.modified = True
                
        #Launch processing in background thread
        processing_thread = threading.Thread(target=process_audio_task)
        processing_thread.daemon = True
        processing_thread.start()
        
        #Redirect to the processing page that shows progress
        return redirect(url_for('process_page'))
    
    return jsonify({'error': 'File type not allowed'}), 400

def cleanup_old_files():
    """Clean up files older than 1 hour to prevent disk space issues"""
    current_time = time.time()
    for folder in [UPLOAD_FOLDER, ENHANCED_FOLDER]:
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            #Delete files older than 1 hour (3600 seconds)
            if os.path.getmtime(filepath) < current_time - 3600:
                try:
                    os.remove(filepath)
                except:
                    pass

if __name__ == '__main__':
    if os.environ.get('PYTHONANYWHERE') == 'true':
        #Skip direct execution when on PythonAnywhere - the app will be run by the WSGI server
        pass
    else:
        #Run development server with debug mode enabled when executed directly
        app.run(debug=True) 