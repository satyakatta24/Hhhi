##Content Detection Project
#App Link
🔗https://contentshield-1.preview.emergentagent.com/
#Prototype Link 
 https://vscode-036395ba-d4e6-425a-bf13-fb2358bae235.preview.emergentagent.com/
#For the access the prototype file
Passkey :d8871dc1

## 📄 BACKEND FILES

### /backend/server.py

python
from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import datetime
import tempfile
import shutil
import mimetypes
from emergentintegrations.llm.chat import LlmChat, UserMessage, FileContentWithMimeType

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix=\"/api\")

# Models
class ContentAnalysisResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    content_type: str
    file_size: int
    analysis_result: dict
    is_appropriate: bool
    confidence_score: float
    detected_issues: List[str]
    timestamp: datetime = Field(default_factory=lambda: datetime.now())

class AnalysisRequest(BaseModel):
    text: str

class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now())

class StatusCheckCreate(BaseModel):
    client_name: str

# Initialize LLM Chat
EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY')

def get_analysis_prompt(content_type: str) -> str:
    \"\"\"Generate analysis prompt based on content type\"\"\"
    base_prompt = \"\"\"You are a content moderation AI. Analyze the provided content and determine if it's appropriate or inappropriate.
    
Categories to check:
- NSFW/Adult content
- Violence and graphic content
- Hate speech and harassment
- Harmful or dangerous content

Provide your response in the following JSON format:
{
    \"is_appropriate\": true/false,
    \"confidence_score\": 0.0-1.0,
    \"detected_issues\": [\"list of specific issues found\"],
    \"reasoning\": \"detailed explanation of your analysis\",
    \"severity\": \"low/medium/high\",
    \"categories_violated\": [\"list of violated categories\"]
}\"\"\"
    
    if content_type.startswith('image'):
        return base_prompt + \"\n\nAnalyze the provided image for inappropriate content.\"
    elif content_type.startswith('audio'):
        return base_prompt + \"\n\nAnalyze the provided audio file for inappropriate content. First transcribe the audio, then analyze the transcript.\"
    elif content_type.startswith('video'):
        return base_prompt + \"\n\nAnalyze the provided video file for inappropriate content. Consider both visual and audio elements.\"
    else:
        return base_prompt + \"\n\nAnalyze the provided text for inappropriate content.\"

async def analyze_content_with_ai(file_path: str, content_type: str, text_content: str = None) -> dict:
    \"\"\"Analyze content using AI\"\"\"
    try:
        # Determine which model to use based on content type
        if content_type.startswith('image'):
            # Use OpenAI for image analysis
            chat = LlmChat(
                api_key=EMERGENT_LLM_KEY,
                session_id=f\"image_analysis_{uuid.uuid4()}\",
                system_message=\"You are a content moderation specialist.\"
            ).with_model(\"openai\", \"gpt-4o\")
            
            # For images, we need to use file attachment
            file_content = FileContentWithMimeType(
                file_path=file_path,
                mime_type=content_type
            )
            
            message = UserMessage(
                text=get_analysis_prompt(content_type),
                file_contents=[file_content]
            )
            
        elif content_type.startswith('audio') or content_type.startswith('video'):
            # Use OpenAI Whisper + GPT for audio/video
            chat = LlmChat(
                api_key=EMERGENT_LLM_KEY,
                session_id=f\"audio_analysis_{uuid.uuid4()}\",
                system_message=\"You are a content moderation specialist.\"
            ).with_model(\"openai\", \"gpt-4o\")
            
            # For audio/video, include file for transcription
            file_content = FileContentWithMimeType(
                file_path=file_path,
                mime_type=content_type
            )
            
            message = UserMessage(
                text=get_analysis_prompt(content_type),
                file_contents=[file_content]
            )
            
        else:
            # Use GPT for text analysis
            chat = LlmChat(
                api_key=EMERGENT_LLM_KEY,
                session_id=f\"text_analysis_{uuid.uuid4()}\",
                system_message=\"You are a content moderation specialist.\"
            ).with_model(\"openai\", \"gpt-4o\")
            
            message = UserMessage(
                text=f\"{get_analysis_prompt(content_type)}\n\nContent to analyze: {text_content or 'See attached file'}\"
            )
            
            if file_path:
                file_content = FileContentWithMimeType(
                    file_path=file_path,
                    mime_type=content_type
                )
                message.file_contents = [file_content]
        
        response = await chat.send_message(message)
        
        # Parse JSON response
        import json
        try:
            # Extract JSON from response
            response_text = response.strip()
            if response_text.startswith('json'):
                response_text = response_text[7:-3]
            elif response_text.startswith(''):
                response_text = response_text[3:-3]
            
            result = json.loads(response_text)
            return result
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                \"is_appropriate\": True,
                \"confidence_score\": 0.5,
                \"detected_issues\": [],
                \"reasoning\": response,
                \"severity\": \"low\",
                \"categories_violated\": []
            }
            
    except Exception as e:
        logging.error(f\"Error in AI analysis: {str(e)}\")
        return {
            \"is_appropriate\": True,
            \"confidence_score\": 0.0,
            \"detected_issues\": [f\"Analysis error: {str(e)}\"],
            \"reasoning\": \"Unable to complete analysis due to technical error\",
            \"severity\": \"unknown\",
            \"categories_violated\": []
        }

# API Routes
@api_router.get(\"/\")
async def root():
    return {\"message\": \"Content Moderation API\"}

@api_router.post(\"/analyze-text\")
async def analyze_text(request: AnalysisRequest):
    \"\"\"Analyze text content\"\"\"
    try:
        analysis = await analyze_content_with_ai(None, \"text/plain\", request.text)
        
        result = ContentAnalysisResult(
            filename=\"text_input\",
            content_type=\"text/plain\",
            file_size=len(request.text.encode('utf-8')),
            analysis_result=analysis,
            is_appropriate=analysis.get(\"is_appropriate\", True),
            confidence_score=analysis.get(\"confidence_score\", 0.0),
            detected_issues=analysis.get(\"detected_issues\", [])
        )
        
        # Store in database
        await db.content_analysis.insert_one(result.dict())
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f\"Analysis failed: {str(e)}\")

@api_router.post(\"/analyze-file\")
async def analyze_file(file: UploadFile = File(...)):
    \"\"\"Analyze uploaded file (image, audio, video, text)\"\"\"
    try:
        # Validate file type
        allowed_types = [
            'image/jpeg', 'image/png', 'image/gif', 'image/webp',
            'audio/mpeg', 'audio/wav', 'audio/ogg', 'audio/mp4',
            'video/mp4', 'video/mpeg', 'video/quicktime', 'video/webm',
            'text/plain', 'application/pdf'
        ]
        
        if file.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail=f\"Unsupported file type: {file.content_type}\")
        
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        try:
            # Analyze content
            analysis = await analyze_content_with_ai(temp_path, file.content_type)
            
            result = ContentAnalysisResult(
                filename=file.filename,
                content_type=file.content_type,
                file_size=file.size,
                analysis_result=analysis,
                is_appropriate=analysis.get(\"is_appropriate\", True),
                confidence_score=analysis.get(\"confidence_score\", 0.0),
                detected_issues=analysis.get(\"detected_issues\", [])
            )
            
            # Store in database
            await db.content_analysis.insert_one(result.dict())
            
            return result
            
        finally:
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f\"Analysis failed: {str(e)}\")

@api_router.get(\"/analysis-history\", response_model=List[ContentAnalysisResult])
async def get_analysis_history():
    \"\"\"Get analysis history\"\"\"
    try:
        analyses = await db.content_analysis.find().sort(\"timestamp\", -1).limit(50).to_list(50)
        return [ContentAnalysisResult(**analysis) for analysis in analyses]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f\"Failed to retrieve history: {str(e)}\")

# Legacy routes
@api_router.post(\"/status\", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.dict()
    status_obj = StatusCheck(**status_dict)
    _ = await db.status_checks.insert_one(status_obj.dict())
    return status_obj

@api_router.get(\"/status\", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**status_check) for status_check in status_checks]

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=[\"*\"],
    allow_headers=[\"*\"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event(\"shutdown\")
async def shutdown_db_client():
    client.close()


### /backend/requirements.txt

txt
aiohappyeyeballs==2.6.1
aiohttp==3.12.15
aiosignal==1.4.0
annotated-types==0.7.0
anyio==4.11.0
attrs==25.3.0
black==24.2.0
boto3==1.34.140
cachetools==5.5.2
certifi==2025.8.3
charset-normalizer==3.4.3
click==8.3.0
cryptography==43.0.3
distro==1.9.0
email-validator==2.2.0
emergentintegrations==0.1.0
fastapi==0.110.1
fastuuid==0.13.5
filelock==3.19.1
flake8==7.0.0
frozenlist==1.7.0
fsspec==2025.9.0
google-ai-generativelanguage==0.6.15
google-api-core==2.25.1
google-api-python-client==2.183.0
google-auth==2.40.3
google-auth-httplib2==0.2.0
google-genai==1.39.1
google-generativeai==0.8.5
googleapis-common-protos==1.70.0
grpcio==1.75.1
grpcio-status==1.71.2
h11==0.16.0
hf-xet==1.1.10
httpcore==1.0.9
httplib2==0.31.0
httpx==0.28.1
huggingface-hub==0.35.1
idna==3.10
importlib-metadata==8.7.0
isort==5.13.2
jinja2==3.1.6
jiter==0.11.0
jq==1.6.0
jsonschema==4.25.1
jsonschema-specifications==2025.9.1
litellm==1.77.4
madoka==0.7.1
MarkupSafe==3.0.2
motor==3.3.1
multidict==6.6.4
mypy==1.8.0
numpy==2.2.1
openai==1.99.9
packaging==25.0
pandas==2.2.3
passlib==1.7.4
Pillow==11.3.0
pondpond==1.4.1
propcache==0.3.2
proto-plus==1.26.1
protobuf==5.29.5
pyasn1==0.6.1
pyasn1-modules==0.4.2
pydantic==2.11.9
pydantic-core==2.33.2
pyjwt==2.10.1
pymongo==4.5.0
pyparsing==3.2.5
pytest==8.0.0
python-dotenv==1.1.1
python-jose==3.3.0
python-multipart==0.0.20
pyyaml==6.0.3
referencing==0.36.2
regex==2025.9.18
requests==2.32.5
requests-oauthlib==2.0.0
rpds-py==0.27.1
rsa==4.9.1
sniffio==1.3.1
starlette==0.37.2
stripe==12.5.1
tenacity==9.1.2
tiktoken==0.11.0
tokenizers==0.22.1
tqdm==4.67.1
typer==0.9.4
typing-extensions==4.15.0
typing-inspection==0.4.1
tzdata==2024.2
uritemplate==4.2.0
urllib3==2.5.0
uvicorn==0.25.0
websockets==15.0.1
yarl==1.20.1
zipp==3.23.0


### /backend/.env

env
MONGO_URL=\"mongodb://localhost:27017\"
DB_NAME=\"contentshield_db\"
CORS_ORIGINS=\"*\"
EMERGENT_LLM_KEY=sk-emergent-2B4Da83FcAdE1E2900


---

## 📄 FRONTEND FILES

### /frontend/src/App.js

jsx
import React, { useState, useEffect } from 'react';
import '@/App.css';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Upload, FileText, Image, Music, Video, Shield, AlertTriangle, CheckCircle, Clock, Eye } from 'lucide-react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  const [textContent, setTextContent] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [analysisHistory, setAnalysisHistory] = useState([]);
  const [dragActive, setDragActive] = useState(false);

  useEffect(() => {
    fetchAnalysisHistory();
  }, []);

  const fetchAnalysisHistory = async () => {
    try {
      const response = await axios.get(`${API}/analysis-history`);
      setAnalysisHistory(response.data);
    } catch (error) {
      console.error('Failed to fetch history:', error);
    }
  };

  const analyzeText = async () => {
    if (!textContent.trim()) return;
    
    setIsAnalyzing(true);
    try {
      const response = await axios.post(`${API}/analyze-text`, {
        text: textContent
      });
      setAnalysisResult(response.data);
      fetchAnalysisHistory();
    } catch (error) {
      console.error('Text analysis failed:', error);
      alert('Analysis failed. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const analyzeFile = async () => {
    if (!selectedFile) return;
    
    setIsAnalyzing(true);
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    try {
      const response = await axios.post(`${API}/analyze-file`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      setAnalysisResult(response.data);
      fetchAnalysisHistory();
    } catch (error) {
      console.error('File analysis failed:', error);
      alert('Analysis failed. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setSelectedFile(e.dataTransfer.files[0]);
    }
  };

  const getFileIcon = (contentType) => {
    if (contentType?.startsWith('image/')) return <Image className=\"w-5 h-5\" />;
    if (contentType?.startsWith('audio/')) return <Music className=\"w-5 h-5\" />;
    if (contentType?.startsWith('video/')) return <Video className=\"w-5 h-5\" />;
    return <FileText className=\"w-5 h-5\" />;
  };

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'high': return 'bg-red-100 text-red-800 border-red-200';
      case 'medium': return 'bg-orange-100 text-orange-800 border-orange-200';
      case 'low': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className=\"min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100\">
      {/* Header */}
      <header className=\"bg-white/80 backdrop-blur-md border-b border-slate-200 sticky top-0 z-10\">
        <div className=\"max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4\">
          <div className=\"flex items-center justify-between\">
            <div className=\"flex items-center space-x-3\">
              <div className=\"p-2 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl\">
                <Shield className=\"w-8 h-8 text-white\" />
              </div>
              <div>
                <h1 className=\"text-2xl font-bold text-slate-900\">ContentShield</h1>
                <p className=\"text-sm text-slate-600\">AI-Powered Multimedia Content Moderation</p>
              </div>
            </div>
            <Badge variant=\"outline\" className=\"bg-green-50 text-green-700 border-green-200\">
              <CheckCircle className=\"w-3 h-3 mr-1\" />
              Ready
            </Badge>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <div className=\"bg-gradient-to-r from-indigo-50 via-purple-50 to-pink-50 border-b border-slate-200\">
        <div className=\"max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12\">
          <div className=\"text-center\">
            <h2 className=\"text-4xl font-bold text-slate-900 mb-4\">
              Advanced Multimedia Content Moderation
            </h2>
            <p className=\"text-xl text-slate-600 mb-8 max-w-3xl mx-auto\">
              AI-powered analysis of images, audio, and video content to detect inappropriate material with high accuracy and detailed reasoning.
            </p>
            <div className=\"grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto\">
              <div className=\"bg-white/70 backdrop-blur-sm rounded-xl p-6 border border-slate-200\">
                <div className=\"w-12 h-12 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl flex items-center justify-center mb-4 mx-auto\">
                  <Image className=\"w-6 h-6 text-white\" />
                </div>
                <h3 className=\"text-lg font-semibold text-slate-900 mb-2\">Image Analysis</h3>
                <p className=\"text-sm text-slate-600\">
                  Detect NSFW content, violence, hate symbols, and inappropriate imagery with confidence scoring
                </p>
              </div>
              <div className=\"bg-white/70 backdrop-blur-sm rounded-xl p-6 border border-slate-200\">
                <div className=\"w-12 h-12 bg-gradient-to-br from-purple-500 to-pink-600 rounded-xl flex items-center justify-center mb-4 mx-auto\">
                  <Music className=\"w-6 h-6 text-white\" />
                </div>
                <h3 className=\"text-lg font-semibold text-slate-900 mb-2\">Audio Analysis</h3>
                <p className=\"text-sm text-slate-600\">
                  Transcribe and analyze audio for hate speech, harmful content, and inappropriate language
                </p>
              </div>
              <div className=\"bg-white/70 backdrop-blur-sm rounded-xl p-6 border border-slate-200\">
                <div className=\"w-12 h-12 bg-gradient-to-br from-pink-500 to-red-500 rounded-xl flex items-center justify-center mb-4 mx-auto\">
                  <Video className=\"w-6 h-6 text-white\" />
                </div>
                <h3 className=\"text-lg font-semibold text-slate-900 mb-2\">Video Analysis</h3>
                <p className=\"text-sm text-slate-600\">
                  Comprehensive analysis of both visual and audio elements in video content
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className=\"max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8\">
        <div className=\"grid grid-cols-1 lg:grid-cols-3 gap-8\">
          {/* Main Analysis Panel */}
          <div className=\"lg:col-span-2 space-y-6\">
            <Tabs defaultValue=\"text\" className=\"w-full\">
              <TabsList className=\"grid w-full grid-cols-2\">
                <TabsTrigger value=\"text\" data-testid=\"text-tab\">Text Analysis</TabsTrigger>
                <TabsTrigger value=\"file\" data-testid=\"file-tab\">File Analysis</TabsTrigger>
              </TabsList>
              
              <TabsContent value=\"text\" className=\"space-y-4\">
                <Card>
                  <CardHeader>
                    <CardTitle className=\"flex items-center space-x-2\">
                      <FileText className=\"w-5 h-5\" />
                      <span>Text Content Analysis</span>
                    </CardTitle>
                    <CardDescription>
                      Paste or type text content to analyze for inappropriate material
                    </CardDescription>
                  </CardHeader>
                  <CardContent className=\"space-y-4\">
                    <Textarea
                      data-testid=\"text-input\"
                      placeholder=\"Enter text content to analyze...\"
                      value={textContent}
                      onChange={(e) => setTextContent(e.target.value)}
                      className=\"min-h-[120px] resize-none\"
                    />
                    <Button 
                      data-testid=\"analyze-text-btn\"
                      onClick={analyzeText}
                      disabled={!textContent.trim() || isAnalyzing}
                      className=\"w-full bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700\"
                    >
                      {isAnalyzing ? (
                        <>
                          <Clock className=\"w-4 h-4 mr-2 animate-spin\" />
                          Analyzing...
                        </>
                      ) : (
                        <>
                          <Eye className=\"w-4 h-4 mr-2\" />
                          Analyze Text
                        </>
                      )}
                    </Button>
                  </CardContent>
                </Card>
              </TabsContent>
              
              <TabsContent value=\"file\" className=\"space-y-4\">
                <Card>
                  <CardHeader>
                    <CardTitle className=\"flex items-center space-x-2\">
                      <Upload className=\"w-5 h-5\" />
                      <span>Multimedia Content Analysis</span>
                    </CardTitle>
                    <CardDescription>
                      Upload <strong>images, audio, video</strong> files for AI-powered content moderation
                    </CardDescription>
                  </CardHeader>
                  <CardContent className=\"space-y-4\">
                    <div
                      data-testid=\"file-drop-zone\"
                      className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                        dragActive 
                          ? 'border-indigo-400 bg-indigo-50' 
                          : 'border-slate-300 bg-slate-50 hover:border-slate-400'
                      }`}
                      onDragEnter={handleDrag}
                      onDragLeave={handleDrag}
                      onDragOver={handleDrag}
                      onDrop={handleDrop}
                    >
                      {selectedFile ? (
                        <div className=\"space-y-2\">
                          <div className=\"flex items-center justify-center space-x-2\">
                            {getFileIcon(selectedFile.type)}
                            <span className=\"font-medium\">{selectedFile.name}</span>
                          </div>
                          <p className=\"text-sm text-slate-600\">
                            {formatFileSize(selectedFile.size)} • {selectedFile.type}
                          </p>
                        </div>
                      ) : (
                        <div className=\"space-y-2\">
                          <div className=\"flex justify-center space-x-2 mb-3\">
                            <Image className=\"w-8 h-8 text-indigo-400\" />
                            <Music className=\"w-8 h-8 text-purple-400\" />
                            <Video className=\"w-8 h-8 text-pink-400\" />
                          </div>
                          <p className=\"text-slate-600 font-medium\">Drop your multimedia file here or click to browse</p>
                          <p className=\"text-sm text-slate-500\">
                            <strong>Supported:</strong> Images (JPG, PNG, GIF, WebP) • Audio (MP3, WAV, OGG) • Video (MP4, WebM, MOV)
                          </p>
                        </div>
                      )}
                      <input
                        data-testid=\"file-input\"
                        type=\"file\"
                        className=\"absolute inset-0 w-full h-full opacity-0 cursor-pointer\"
                        onChange={(e) => setSelectedFile(e.target.files[0])}
                        accept=\"image/*,audio/*,video/*,text/plain,application/pdf\"
                      />
                    </div>
                    
                    <Button 
                      data-testid=\"analyze-file-btn\"
                      onClick={analyzeFile}
                      disabled={!selectedFile || isAnalyzing}
                      className=\"w-full bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700\"
                    >
                      {isAnalyzing ? (
                        <>
                          <Clock className=\"w-4 h-4 mr-2 animate-spin\" />
                          Analyzing...
                        </>
                      ) : (
                        <>
                          <Eye className=\"w-4 h-4 mr-2\" />
                          Analyze File
                        </>
                      )}
                    </Button>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>

            {/* Analysis Results */}
            {analysisResult && (
              <Card data-testid=\"analysis-results\">
                <CardHeader>
                  <CardTitle className=\"flex items-center justify-between\">
                    <span>Analysis Results</span>
                    <Badge 
                      variant={analysisResult.is_appropriate ? \"default\" : \"destructive\"}
                      className={analysisResult.is_appropriate ? \"bg-green-100 text-green-800\" : \"\"}
                    >
                      {analysisResult.is_appropriate ? (
                        <>
                          <CheckCircle className=\"w-3 h-3 mr-1\" />
                          Appropriate
                        </>
                      ) : (
                        <>
                          <AlertTriangle className=\"w-3 h-3 mr-1\" />
                          Inappropriate
                        </>
                      )}
                    </Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent className=\"space-y-4\">
                  <div className=\"grid grid-cols-2 gap-4\">
                    <div>
                      <p className=\"text-sm font-medium text-slate-600\">Confidence Score</p>
                      <div className=\"mt-1\">
                        <Progress value={analysisResult.confidence_score * 100} className=\"h-2\" />
                        <p className=\"text-sm text-slate-500 mt-1\">
                          {(analysisResult.confidence_score * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>
                    <div>
                      <p className=\"text-sm font-medium text-slate-600\">File Info</p>
                      <p className=\"text-sm text-slate-800\">{analysisResult.filename}</p>
                      <p className=\"text-xs text-slate-500\">
                        {formatFileSize(analysisResult.file_size)} • {analysisResult.content_type}
                      </p>
                    </div>
                  </div>
                  
                  {analysisResult.detected_issues?.length > 0 && (
                    <Alert className=\"border-orange-200 bg-orange-50\">
                      <AlertTriangle className=\"h-4 w-4 text-orange-600\" />
                      <AlertDescription className=\"text-orange-800\">
                        <strong>Detected Issues:</strong>
                        <ul className=\"mt-1 list-disc list-inside text-sm\">
                          {analysisResult.detected_issues.map((issue, index) => (
                            <li key={index}>{issue}</li>
                          ))}
                        </ul>
                      </AlertDescription>
                    </Alert>
                  )}
                  
                  {analysisResult.analysis_result?.reasoning && (
                    <div>
                      <p className=\"text-sm font-medium text-slate-600 mb-2\">Analysis Details</p>
                      <div className=\"bg-slate-50 p-3 rounded-lg\">
                        <p className=\"text-sm text-slate-700\">{analysisResult.analysis_result.reasoning}</p>
                      </div>
                    </div>
                  )}
                  
                  {analysisResult.analysis_result?.severity && (
                    <div className=\"flex items-center space-x-2\">
                      <span className=\"text-sm font-medium text-slate-600\">Severity:</span>
                      <Badge className={getSeverityColor(analysisResult.analysis_result.severity)}>
                        {analysisResult.analysis_result.severity.toUpperCase()}
                      </Badge>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}
          </div>

          {/* History Panel */}
          <div className=\"space-y-6\">
            <Card>
              <CardHeader>
                <CardTitle className=\"text-lg\">Analysis History</CardTitle>
                <CardDescription>Recent content moderation results</CardDescription>
              </CardHeader>
              <CardContent>
                <ScrollArea className=\"h-[600px]\">
                  <div className=\"space-y-3\">
                    {analysisHistory.length === 0 ? (
                      <p className=\"text-sm text-slate-500 text-center py-8\">
                        No analysis history yet
                      </p>
                    ) : (
                      analysisHistory.map((item, index) => (
                        <div key={item.id} className=\"border rounded-lg p-3 bg-white/50\">
                          <div className=\"flex items-center justify-between mb-2\">
                            <div className=\"flex items-center space-x-2\">
                              {getFileIcon(item.content_type)}
                              <span className=\"text-sm font-medium truncate max-w-[120px]\">
                                {item.filename}
                              </span>
                            </div>
                            <Badge 
                              variant={item.is_appropriate ? \"default\" : \"destructive\"}
                              className={`text-xs ${item.is_appropriate ? \"bg-green-100 text-green-700\" : \"\"}`}
                            >
                              {item.is_appropriate ? \"✓\" : \"⚠\"}
                            </Badge>
                          </div>
                          <div className=\"flex justify-between items-center text-xs text-slate-500\">
                            <span>{(item.confidence_score * 100).toFixed(0)}% confidence</span>
                            <span>{new Date(item.timestamp).toLocaleDateString()}</span>
                          </div>
                          {index < analysisHistory.length - 1 && (
                            <Separator className=\"mt-3\" />
                          )}
                        </div>
                      ))
                    )}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;


### /frontend/src/App.css

css
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  background-color: #f8fafc;
}

.App {
  min-height: 100vh;
}

/* Custom scrollbar */
::-webkit-scrollbar {
  width: 6px;
}

::-webkit-scrollbar-track {
  background: #f1f5f9;
}

::-webkit-scrollbar-thumb {
  background: #cbd5e1;
  border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
  background: #94a3b8;
}

/* Smooth transitions for interactive elements */
button, .card, .badge {
  transition: all 0.2s ease-in-out;
}

/* Custom gradient backgrounds */
.gradient-bg {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

/* Animation for loading states */
@keyframes pulse {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.5;
  }
}

.animate-pulse {
  animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}

/* File upload drop zone styles */
.drag-active {
  border-color: #6366f1 !important;
  background-color: #eef2ff !important;
}

/* Custom focus styles */
input:focus, textarea:focus, button:focus {
  outline: 2px solid #6366f1;
  outline-offset: 2px;
}

/* Glass morphism effect */
.glass {
  background: rgba(255, 255, 255, 0.25);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.18);
}

/* Custom card hover effects */
.card:hover {
  transform: translateY(-2px);
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
}

/* Progress bar custom styling */
.progress-bar {
  background: linear-gradient(90deg, #10b981, #059669);
}

/* Badge animations */
.badge {
  position: relative;
  overflow: hidden;
}

.badge::before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
  transition: left 0.5s;
}

.badge:hover::before {
  left: 100%;
}

/* Mobile responsive adjustments */
@media (max-width: 768px) {
  .max-w-7xl {
    padding-left: 1rem;
    padding-right: 1rem;
  }
  
  .grid-cols-1.lg\:grid-cols-3 {
    grid-template-columns: 1fr;
  }
}

/* Dark mode support */
@media (prefers-color-scheme: dark) {
  body {
    background-color: #0f172a;
    color: #e2e8f0;
  }
}

/* Success and error states */
.success {
  color: #059669;
  background-color: #d1fae5;
  border-color: #a7f3d0;
}

.error {
  color: #dc2626;
  background-color: #fee2e2;
  border-color: #fecaca;
}

.warning {
  color: #d97706;
  background-color: #fef3c7;
  border-color: #fde68a;
}

/* Loading spinner */
.spinner {
  border: 2px solid #f3f4f6;
  border-top: 2px solid #6366f1;
  border-radius: 50%;
  width: 20px;
  height: 20px;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

/* File type icons styling */
.file-icon {
  padding: 8px;
  border-radius: 8px;
  background: #f1f5f9;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}

/* Typography improvements */
h1, h2, h3, h4, h5, h6 {
  line-height: 1.2;
  letter-spacing: -0.025em;
}

p {
  line-height: 1.6;
}

/* Custom shadows */
.shadow-soft {
  box-shadow: 0 2px 15px rgba(0, 0, 0, 0.08);
}

.shadow-medium {
  box-shadow: 0 4px 25px rgba(0, 0, 0, 0.12);
}

.shadow-hard {
  box-shadow: 0 8px 35px rgba(0, 0, 0, 0.15);
}


### /frontend/package.json

json
{
  \"name\": \"contentshield-frontend\",
  \"version\": \"1.0.0\",
  \"private\": true,
  \"dependencies\": {
    \"@hookform/resolvers\": \"^5.0.1\",
    \"@radix-ui/react-accordion\": \"^1.2.8\",
    \"@radix-ui/react-alert-dialog\": \"^1.1.11\",
    \"@radix-ui/react-aspect-ratio\": \"^1.1.4\",
    \"@radix-ui/react-avatar\": \"^1.1.7\",
    \"@radix-ui/react-checkbox\": \"^1.2.3\",
    \"@radix-ui/react-collapsible\": \"^1.1.8\",
    \"@radix-ui/react-context-menu\": \"^2.2.12\",
    \"@radix-ui/react-dialog\": \"^1.1.11\",
    \"@radix-ui/react-dropdown-menu\": \"^2.1.12\",
    \"@radix-ui/react-hover-card\": \"^1.1.11\",
    \"@radix-ui/react-label\": \"^2.1.4\",
    \"@radix-ui/react-menubar\": \"^1.1.12\",
    \"@radix-ui/react-navigation-menu\": \"^1.2.10\",
    \"@radix-ui/react-popover\": \"^1.1.11\",
    \"@radix-ui/react-progress\": \"^1.1.4\",
    \"@radix-ui/react-radio-group\": \"^1.3.4\",
    \"@radix-ui/react-scroll-area\": \"^1.2.6\",
    \"@radix-ui/react-select\": \"^2.2.2\",
    \"@radix-ui/react-separator\": \"^1.1.4\",
    \"@radix-ui/react-slider\": \"^1.3.2\",
    \"@radix-ui/react-slot\": \"^1.2.0\",
    \"@radix-ui/react-switch\": \"^1.2.2\",
    \"@radix-ui/react-tabs\": \"^1.1.9\",
    \"@radix-ui/react-toast\": \"^1.2.11\",
    \"@radix-ui/react-toggle\": \"^1.1.6\",
    \"@radix-ui/react-toggle-group\": \"^1.1.7\",
    \"@radix-ui/react-tooltip\": \"^1.2.4\",
    \"axios\": \"^1.8.4\",
    \"class-variance-authority\": \"^0.7.1\",
    \"clsx\": \"^2.1.1\",
    \"cmdk\": \"^1.1.1\",
    \"cra-template\": \"1.2.0\",
    \"date-fns\": \"^4.1.0\",
    \"embla-carousel-react\": \"^8.6.0\",
    \"input-otp\": \"^1.4.2\",
    \"lucide-react\": \"^0.507.0\",
    \"next-themes\": \"^0.4.6\",
    \"react\": \"^19.0.0\",
    \"react-day-picker\": \"8.10.1\",
    \"react-dom\": \"^19.0.0\",
    \"react-hook-form\": \"^7.56.2\",
    \"react-resizable-panels\": \"^3.0.1\",
    \"react-router-dom\": \"^7.5.1\",
    \"react-scripts\": \"5.0.1\",
    \"sonner\": \"^2.0.3\",
    \"tailwind-merge\": \"^3.2.0\",
    \"tailwindcss-animate\": \"^1.0.7\",
    \"vaul\": \"^1.1.2\",
    \"zod\": \"^3.24.4\"
  },
  \"scripts\": {
    \"start\": \"craco start\",
    \"build\": \"craco build\",
    \"test\": \"craco test\"
  },
  \"browserslist\": {
    \"production\": [
      \">0.2%\",
      \"not dead\",
      \"not op_mini all\"
    ],
    \"development\": [
      \"last 1 chrome version\",
      \"last 1 firefox version\",
      \"last 1 safari version\"
    ]
  },
  \"devDependencies\": {
    \"@craco/craco\": \"^7.1.0\",
    \"@eslint/js\": \"9.23.0\",
    \"autoprefixer\": \"^10.4.20\",
    \"eslint\": \"9.23.0\",
    \"eslint-plugin-import\": \"2.31.0\",
    \"eslint-plugin-jsx-a11y\": \"6.10.2\",
    \"eslint-plugin-react\": \"7.37.4\",
    \"globals\": \"15.15.0\",
    \"postcss\": \"^8.4.49\",
    \"tailwindcss\": \"^3.4.17\"
  }
}


---

## 🚀 Setup Instructions

### Backend Setup

1. *Create virtual environment:*
   bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   

2. *Install dependencies:*
   bash
   pip install -r requirements.txt
   

3. *Install emergentintegrations:*
   bash
   pip install emergentintegrations --extra-index-url https://d33sy5i8bnduwe.cloudfront.net/simple/
   

4. *Set up environment variables:*
   Create .env file with:
   env
   MONGO_URL=\"mongodb://localhost:27017\"
   DB_NAME=\"contentshield_db\"
   CORS_ORIGINS=\"*\"
   EMERGENT_LLM_KEY=your_emergent_key_here
   

5. *Start the server:*
   bash
   uvicorn server:app --host 0.0.0.0 --port 8001 --reload
   

### Frontend Setup

1. *Install dependencies:*
   bash
   yarn install
   

2. *Set up environment variables:*
   Create .env file with:
   env
   REACT_APP_BACKEND_URL=http://localhost:8001
   

3. *Start the development server:*
   bash
   yarn start
   

### Database Setup

1. *Install MongoDB:*
   - Follow MongoDB installation guide for your OS
   - Start MongoDB service

2. *Database will be automatically created* when you first run the application

---

## 📡 API Endpoints

### Content Analysis
- POST /api/analyze-text - Analyze text content
- POST /api/analyze-file - Analyze uploaded files (images, audio, video)
- GET /api/analysis-history - Get analysis history

### Health Check
- GET /api/ - API status
- POST /api/status - Create status check
- GET /api/status - Get status checks

---

## 🎯 Features Overview

### Supported File Types
- *Images*: JPEG, PNG, GIF, WebP
- *Audio*: MP3, WAV, OGG, MP4
- *Video*: MP4, MPEG, QuickTime, WebM
- *Text*: Plain text, PDF

### Analysis Categories  
- NSFW/Adult content
- Violence and graphic content
- Hate speech and harassment
- Harmful or dangerous content

### Analysis Output
- Appropriate/Inappropriate classification
- Confidence score (0-1)
- Detailed reasoning
- Severity level (low/medium/high)
- Specific detected issues
- Category violations

---

## 🔧 Configuration

### AI Model Configuration
The system uses OpenAI GPT-4o via the emergentintegrations library. You can modify the model in server.py:

python
chat = LlmChat(
    api_key=EMERGENT_LLM_KEY,
    session_id=f\"analysis_{uuid.uuid4()}\",
    system_message=\"You are a content moderation specialist.\"
).with_model(\"openai\", \"gpt-4o\")  # Change model here


### File Upload Limits
Modify allowed file types in server.py:

python
allowed_types = [
    'image/jpeg', 'image/png', 'image/gif', 'image/webp',
    'audio/mpeg', 'audio/wav', 'audio/ogg', 'audio/mp4',
    'video/mp4', 'video/mpeg', 'video/quicktime', 'video/webm',
    'text/plain', 'application/pdf'
]


---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

## 📞 Support

For questions or support, please:
- Open an issue on GitHub
- Contact the development team
- Check the documentation

---

*Built with ❤ using FastAPI, React, and OpenAI GPT-4o*"
Observation: Create successful: /app/ContentShield_Complete_Project.md
