import os
import uuid
import torch
import logging
from flask import Flask, request, jsonify, send_file, render_template, url_for
from werkzeug.utils import secure_filename
from TTS.api import TTS
from pydub import AudioSegment
import tempfile

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['SECRET_KEY'] = 'voicecloningsecretkey'

# Initialize the TTS models (lazy loading)
tts_model = None
hindi_tts_model = None

def get_tts_model():
    global tts_model
    if tts_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing YourTTS model on {device}...")
        # Using YourTTS model which is good for voice cloning
        tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", 
                        progress_bar=False).to(device)
    return tts_model

def get_hindi_model():
    global hindi_tts_model
    if hindi_tts_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing Hindi TTS model on {device}...")
        # First try a dedicated Hindi model if available
        try:
            # Try to load a Hindi-specific model first
            hindi_tts_model = TTS(model_name="tts_models/hi/fairseq/vits", 
                                progress_bar=False).to(device)
            logger.info("Hindi-specific model loaded successfully")
        except Exception as e:
            logger.warning(f"Hindi-specific model not available: {str(e)}")
            # Fall back to universal model
            hindi_tts_model = get_tts_model()
            logger.info("Using YourTTS model for Hindi as fallback")
    return hindi_tts_model

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_wav(input_file):
    """Convert any audio file to WAV format if needed"""
    file_ext = os.path.splitext(input_file)[1].lower()
    
    if file_ext == '.wav':
        return input_file
    
    try:
        audio = AudioSegment.from_file(input_file)
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        wav_path = temp_wav.name
        temp_wav.close()
        
        audio.export(wav_path, format="wav")
        logger.info(f"Converted {file_ext} to WAV format")
        return wav_path
    except Exception as e:
        logger.error(f"Error converting audio: {str(e)}")
        return input_file

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/clone_voice', methods=['POST'])
def clone_voice():
    # Check if both text and voice file are provided
    if 'text' not in request.form:
        return jsonify({'error': 'No text provided'}), 400
    
    if 'voice_sample' not in request.files:
        return jsonify({'error': 'No voice sample file provided'}), 400
    
    file = request.files['voice_sample']
    if file.filename == '':
        return jsonify({'error': 'No voice sample file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Use one of: {ALLOWED_EXTENSIONS}'}), 400
    
    # Save the uploaded voice sample
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    # Convert to WAV if needed
    converted_path = convert_to_wav(file_path)
    
    # Get parameters from the request
    text = request.form['text']
    speaker_wav = converted_path
    language = request.form.get('language', 'en')
    
    # Generate unique output filename
    output_filename = f"{uuid.uuid4()}.wav"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    
    try:
        if language == 'hi':
            # Use special Hindi handling
            logger.info("Processing Hindi text")
            model = get_hindi_model()
            
            # Try different approaches for Hindi
            try:
                # Try with Hindi language code if model supports it
                if 'hi' in model.languages:
                    model.tts_to_file(
                        text=text,
                        file_path=output_path,
                        speaker_wav=speaker_wav,
                        language='hi'
                    )
                    logger.info("Hindi TTS completed with 'hi' language code")
                else:
                    # Try with English language code as fallback
                    model.tts_to_file(
                        text=text,
                        file_path=output_path,
                        speaker_wav=speaker_wav,
                        language='en'
                    )
                    logger.info("Hindi TTS completed with 'en' language code fallback")
            except Exception as hindi_error:
                logger.error(f"Error processing Hindi: {str(hindi_error)}")
                
                # Final fallback to YourTTS with English
                fallback_model = get_tts_model()
                fallback_model.tts_to_file(
                    text=text,
                    file_path=output_path,
                    speaker_wav=speaker_wav,
                    language='en'
                )
                logger.info("Used YourTTS fallback for Hindi text")
        else:
            # Normal processing for other languages
            model = get_tts_model()
            model.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=speaker_wav,
                language=language
            )
            logger.info(f"Generated TTS for language: {language}")
        
        # Return the output file for download
        return send_file(
            output_path,
            mimetype="audio/wav",
            as_attachment=True,
            download_name=output_filename
        )
    
    except Exception as e:
        logger.error(f"Error generating TTS: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500
    finally:
        # Clean up the uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)
        # Clean up converted file if different from original
        if converted_path != file_path and os.path.exists(converted_path):
            os.remove(converted_path)

@app.route('/api/clone_voice', methods=['POST'])
def api_clone_voice():
    """API endpoint for programmatic access"""
    # Check if both text and voice file are provided
    if 'text' not in request.form:
        return jsonify({'error': 'No text provided'}), 400
    
    if 'voice_sample' not in request.files:
        return jsonify({'error': 'No voice sample file provided'}), 400
    
    file = request.files['voice_sample']
    if file.filename == '':
        return jsonify({'error': 'No voice sample file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Use one of: {ALLOWED_EXTENSIONS}'}), 400
    
    # Save the uploaded voice sample
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    # Convert to WAV if needed
    converted_path = convert_to_wav(file_path)
    
    # Get parameters from the request
    text = request.form['text']
    speaker_wav = converted_path
    language = request.form.get('language', 'en')
    
    # Generate unique output filename
    output_filename = f"{uuid.uuid4()}.wav"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    
    try:
        if language == 'hi':
            # Use special Hindi handling
            logger.info("Processing Hindi text via API")
            model = get_hindi_model()
            
            # Try different approaches for Hindi
            try:
                # Try with Hindi language code if model supports it
                if 'hi' in model.languages:
                    model.tts_to_file(
                        text=text,
                        file_path=output_path,
                        speaker_wav=speaker_wav,
                        language='hi'
                    )
                else:
                    # Try with English language code as fallback
                    model.tts_to_file(
                        text=text,
                        file_path=output_path,
                        speaker_wav=speaker_wav,
                        language='en'
                    )
            except Exception as hindi_error:
                logger.error(f"API Error processing Hindi: {str(hindi_error)}")
                
                # Final fallback to YourTTS with English
                fallback_model = get_tts_model()
                fallback_model.tts_to_file(
                    text=text,
                    file_path=output_path,
                    speaker_wav=speaker_wav,
                    language='en'
                )
        else:
            # Normal processing for other languages
            model = get_tts_model()
            model.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=speaker_wav,
                language=language
            )
        
        # Return success JSON with the download URL
        return jsonify({
            'success': True,
            'download_url': url_for('download_file', filename=output_filename, _external=True)
        })
    
    except Exception as e:
        logger.error(f"API Error generating TTS: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500
    finally:
        # Clean up the uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)
        # Clean up converted file if different from original
        if converted_path != file_path and os.path.exists(converted_path):
            os.remove(converted_path)

@app.route('/download/<filename>')
def download_file(filename):
    """Endpoint to download generated files"""
    return send_file(
        os.path.join(app.config['OUTPUT_FOLDER'], filename),
        mimetype="audio/wav",
        as_attachment=True,
        download_name=filename
    )

@app.route('/health', methods=['GET'])
def health_check():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return jsonify({'status': 'ok', 'device': device})

@app.route('/languages', methods=['GET'])
def get_languages():
    """Return supported languages"""
    languages = {
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'pl': 'Polish',
        'tr': 'Turkish',
        'ru': 'Russian',
        'nl': 'Dutch',
        'cs': 'Czech',
        'ar': 'Arabic',
        'zh-cn': 'Chinese (Simplified)',
        'hi': 'Hindi'
    }
    return jsonify(languages)

@app.route('/model_info', methods=['GET'])
def get_model_info():
    """Return detailed information about the loaded models"""
    # Get the models
    yourTTS = get_tts_model()
    hindi_model = get_hindi_model()
    
    # Get model information
    yourTTS_info = {
        'name': yourTTS.model_name,
        'languages': yourTTS.languages if hasattr(yourTTS, 'languages') else [],
        'is_multi_lingual': hasattr(yourTTS, 'languages') and len(yourTTS.languages) > 1,
        'speakers': yourTTS.speakers if hasattr(yourTTS, 'speakers') else []
    }
    
    hindi_model_info = {
        'name': hindi_model.model_name,
        'languages': hindi_model.languages if hasattr(hindi_model, 'languages') else [],
        'is_multi_lingual': hasattr(hindi_model, 'languages') and len(hindi_model.languages) > 1,
        'speakers': hindi_model.speakers if hasattr(hindi_model, 'speakers') else []
    }
    
    return jsonify({
        'yourTTS': yourTTS_info,
        'hindi_model': hindi_model_info
    })

if __name__ == '__main__':
    print(f"Starting Voice Cloning Service...")
    app.run(host='0.0.0.0', port=5000, debug=True)