# ITI Chatbot Application

An advanced AI-powered chatbot application designed for Industrial Training Institutes (ITIs) in India, providing comprehensive assistance for students, instructors, and administrators.

## 🚀 Features

### Core Features
- **AI-Powered Chatbot**: Utilizes Google's Gemini 2.0 Flash model for natural conversations
- **Progress Tracking System**: Visual progress bars for single and multiple tasks
- **Voice Interaction**: Speech-to-text input and text-to-speech output
- **Multilingual Support**: Multiple language translations with 20+ supported languages

### ITI-Specific Modules
- **Admission Management**: Information about admission processes and requirements
- **Course Finder**: Find suitable ITI courses based on interests and qualifications
- **Exam Preparation**: Study materials, practice tests, and exam resources
- **Industry Connections**: Industry information, trends, and job opportunities
- **Mentorship Resources**: Connect with mentors and get career guidance
- **Practical Assessment**: Skill assessment tools and practical exercises
- **Scholarship Management**: Information about available scholarships
- **Trade Comparison**: Compare different ITI trades and their career prospects

### Additional Features
- **Document Processing**: Extract and summarize text from PDFs and images
- **Web Search Integration**: Real-time information retrieval from the web
- **YouTube Integration**: Relevant educational video recommendations
- **Study Material Generation**: Dynamic creation of study content

## 📋 Requirements

- Python 3.9 or higher
- Internet connection (for AI features and web search)
- Minimum 4GB RAM recommended
- API Keys:
  - Gemini API key (required for AI features)
  - SERP API key (optional for web search)
  - YouTube API key (optional for video features)

## 🔧 Installation

### Windows
1. Extract the package to a directory of your choice
2. Double-click `install.bat` or run from command prompt
3. Follow the on-screen instructions

### Linux/macOS
1. Extract the package to a directory of your choice
2. Open terminal in the extracted directory
3. Run: `chmod +x install.sh`
4. Run: `./install.sh`
5. Follow the on-screen instructions

## 🔑 API Keys Setup

1. Open the `.env` file in the installation directory
2. Update the following API keys:
   - `GEMINI_API_KEY`: Get from [Google MakerSuite](https://makersuite.google.com/)
   - `SERP_API_KEY`: Get from [SERP API](https://serpapi.com/)
   - `YOUTUBE_API_KEY`: Get from [Google Cloud Console](https://console.cloud.google.com/)

## 🚀 Running the Application

### Windows
1. Open command prompt in the installation directory
2. Run: `venv\Scripts\activate.bat`
3. Run: `python run_iti_app.py`

### Linux/macOS
1. Open terminal in the installation directory
2. Run: `source venv/bin/activate`
3. Run: `python run_iti_app.py`

## 📚 Available Commands

- `!help` - Show help message
- `!clear` - Clear conversation history
- `!exit` - Exit application
- `!voice` - Toggle voice interaction
- `!language <code>` - Set language (e.g., !language hi for Hindi)
- `!progress` - Show progress tracking demo
- `!admission` - Show admission information
- `!courses` - Find suitable ITI courses
- `!exams` - Get exam preparation resources
- `!industry` - Explore industry connections
- `!mentorship` - Access mentorship resources
- `!assessment` - Practice assessment tools
- `!scholarship` - Find scholarship opportunities
- `!trades` - Compare different ITI trades

## 🌐 Supported Languages

The application supports 20+ languages including:
- English (en)
- Hindi (hi)
- Bengali (bn)
- Tamil (ta)
- Telugu (te)
- Marathi (mr)
- Gujarati (gu)
- Kannada (kn)
- Malayalam (ml)
- Punjabi (pa)
- And many more...

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Support

For support, please contact the development team or open an issue on the project repository. 